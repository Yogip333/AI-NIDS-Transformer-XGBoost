# Adapted from concepts by:
# Nussbaum, Z. et al. (2024) 'Nomic Embed: Training a Reproducible Long Context
#   Text Embedder', arXiv:2402.01613.
# Reimers, N. and Gurevych, I. (2019) 'Sentence-BERT: Sentence Embeddings using
#   Siamese BERT-Networks', Proceedings of EMNLP-IJCNLP 2019, pp. 3982-3992.
# Kusupati, A. et al. (2022) 'Matryoshka Representation Learning',
#   Advances in Neural Information Processing Systems 35.

"""Text embedding wrapper used by the retrieval-augmented enrichment layer.

The default backbone is Nomic Embed Text v1.5 (Nussbaum et al., 2024),
loaded through the sentence-transformers framework (Reimers and Gurevych,
2019). It was chosen because it runs fully locally, has a permissive
licence and supports Matryoshka Representation Learning (Kusupati et al.,
2022), which lets the same model serve both high-dimensional and truncated
retrieval at no extra training cost. The all-MiniLM-L6-v2 model is used
as a fallback if the nomic weights are unavailable on the host.
"""
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Union

import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")

# Authenticate with Hugging Face if token is available
_hf_token = os.getenv("HUGGINGFACE_TOKEN")
if _hf_token:
    try:
        from huggingface_hub import login
        login(token=_hf_token, add_to_git_credential=False)
        logging.getLogger(__name__).info("Authenticated with Hugging Face")
    except Exception as e:
        logging.getLogger(__name__).warning("HuggingFace login failed: %s", e)

logger = logging.getLogger(__name__)

_NOMIC_MODEL_ID   = "nomic-ai/nomic-embed-text-v1.5"
_FALLBACK_MODEL_ID = "all-MiniLM-L6-v2"

# Prompt prefix required by nomic-embed-text for retrieval tasks
NOMIC_SEARCH_PREFIX    = "search_query: "
NOMIC_DOCUMENT_PREFIX  = "search_document: "


class EmbeddingModel:
    """
    Wraps sentence-transformers for embedding text.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.  Defaults to nomic-embed-text-v1.5.
    device : str
        'cuda' | 'cpu' | 'auto'
    """

    def __init__(
        self,
        model_id: str = _NOMIC_MODEL_ID,
        device: str = "cpu",
        matryoshka_dim: int | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.device = device
        self.matryoshka_dim = matryoshka_dim  # None = full 768-dim
        self._is_nomic = "nomic" in model_id.lower()

        logger.info("Loading embedding model: %s", model_id)
        try:
            self._model = SentenceTransformer(
                model_id,
                device=device,
                trust_remote_code=True,
            )
            logger.info("Embedding model loaded successfully (dim=%d)", self.embedding_dim)
        except Exception as e:
            logger.warning("Failed to load %s: %s — falling back to %s",
                           model_id, e, _FALLBACK_MODEL_ID)
            self._model = SentenceTransformer(_FALLBACK_MODEL_ID, device=device)
            self._is_nomic = False
            self.model_id = _FALLBACK_MODEL_ID

    @property
    def embedding_dim(self) -> int:
        if self.matryoshka_dim:
            return self.matryoshka_dim
        return self._model.get_sentence_embedding_dimension()

    def _add_prefix(self, texts: list[str], is_query: bool) -> list[str]:
        """Add nomic task prefix if using nomic model."""
        if not self._is_nomic:
            return texts
        prefix = NOMIC_SEARCH_PREFIX if is_query else NOMIC_DOCUMENT_PREFIX
        return [f"{prefix}{t}" for t in texts]

    def embed_documents(
        self, texts: list[str], batch_size: int = 64, show_progress: bool = False
    ) -> np.ndarray:
        """
        Embed a list of documents for storage in the vector DB.

        Returns
        -------
        np.ndarray : shape (len(texts), embedding_dim)
        """
        prefixed = self._add_prefix(texts, is_query=False)
        embeddings = self._model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        if self.matryoshka_dim and embeddings.shape[1] > self.matryoshka_dim:
            embeddings = embeddings[:, : self.matryoshka_dim]
        return embeddings.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query string for similarity search.

        Returns
        -------
        np.ndarray : shape (embedding_dim,)
        """
        prefixed = self._add_prefix([text], is_query=True)
        emb = self._model.encode(
            prefixed, normalize_embeddings=True
        )[0]
        if self.matryoshka_dim and emb.shape[0] > self.matryoshka_dim:
            emb = emb[: self.matryoshka_dim]
        return emb.astype(np.float32)

    def embed_batch(
        self,
        texts: list[str],
        is_query: bool = False,
        batch_size: int = 64,
    ) -> np.ndarray:
        """General-purpose batch embedding."""
        if is_query:
            return np.vstack([self.embed_query(t).reshape(1, -1) for t in texts])
        return self.embed_documents(texts, batch_size=batch_size)


@lru_cache(maxsize=1)
def get_default_embedding_model(device: str = "cpu") -> EmbeddingModel:
    """Singleton loader — call once and reuse."""
    return EmbeddingModel(model_id=_NOMIC_MODEL_ID, device=device)
