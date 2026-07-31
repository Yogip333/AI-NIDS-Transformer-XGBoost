# Adapted from concepts by:
# Lewis, P. et al. (2020) 'Retrieval-Augmented Generation for Knowledge-Intensive
#   NLP Tasks', Advances in Neural Information Processing Systems 33.
# Nussbaum, Z. et al. (2024) 'Nomic Embed: Training a Reproducible Long Context
#   Text Embedder', arXiv:2402.01613.
# Malkov, Y. A. and Yashunin, D. A. (2020) 'Efficient and Robust Approximate
#   Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs',
#   IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(4), pp. 824-836.
# Grattafiori, A. et al. (2024) 'The Llama 3 Herd of Models', arXiv:2407.21783.

"""Retrieval-augmented enrichment layer for alert generation.

The module follows the Retrieval-Augmented Generation pattern of Lewis et
al. (2020): a dense retriever — Nomic Embed Text (Nussbaum et al., 2024)
over a ChromaDB collection that uses HNSW indexing (Malkov and Yashunin,
2020) — fetches the most similar entries in the hand-curated threat
knowledge base, and a generator (optionally Groq-hosted Llama 3.3, see
Grattafiori et al., 2024) turns the retrieved context into a short
analyst-style narrative. The classifier's verdict itself is never
modified by this module; enrichment is strictly a post-hoc explanation.
"""
import logging
import os
from typing import Any, Optional

import chromadb
from chromadb.config import Settings

from src.data.loader import ID_TO_LABEL
from src.rag.embeddings import EmbeddingModel, get_default_embedding_model
from src.rag.groq_analyst import GroqAnalyst
from src.rag.knowledge_base import (
    THREAT_KNOWLEDGE_BASE,
    ThreatEntry,
    get_all_texts,
    get_threat_by_id,
)

logger = logging.getLogger(__name__)


class ThreatRAG:
    """
    Retrieval-Augmented Generation pipeline for threat intelligence.

    Parameters
    ----------
    db_path       : directory for ChromaDB persistence
    collection    : name of the ChromaDB collection
    embedding_model : EmbeddingModel instance (nomic by default)
    top_k         : number of results to retrieve per query
    """

    def __init__(
        self,
        db_path: str = "data/rag_db",
        collection: str = "threat_intelligence",
        embedding_model: Optional[EmbeddingModel] = None,
        top_k: int = 5,
        groq_analyst: Optional[GroqAnalyst] = None,
    ):
        self.db_path    = db_path
        self.collection_name = collection
        self.top_k      = top_k
        self._emb_model = embedding_model or get_default_embedding_model()
        self._groq      = groq_analyst  # None = enrichment disabled

        os.makedirs(db_path, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' ready (%d documents)",
                    collection, self._collection.count())

        # ── Detect stale HNSW index (dimension mismatch) ─────────────────────
        # Occurs when the DB was built with a different embedding model than the
        # one currently loaded (e.g. nomic 768-d vs MiniLM 384-d fallback).
        # Probe with a cheap test query; if ChromaDB raises a dimension error,
        # mark for auto-rebuild so the next build_knowledge_base() call purges
        # the stale segment files completely.
        self._needs_dimension_rebuild = False
        if self._collection.count() > 0:
            try:
                test_vec = self._emb_model.embed_query("test connectivity")
                self._collection.query(
                    query_embeddings=[test_vec.tolist()], n_results=1
                )
            except Exception as probe_err:
                if "dimension" in str(probe_err).lower():
                    logger.warning(
                        "HNSW dimension mismatch detected (%s). "
                        "RAG will be force-rebuilt on next build_knowledge_base() call.",
                        probe_err,
                    )
                    self._needs_dimension_rebuild = True

    # ── Knowledge base population ─────────────────────────────────────────────

    def _hard_reset_db(self) -> None:
        """
        Completely wipe the ChromaDB directory and reinitialise the client.

        Required when the HNSW segment files are stale (left over from a
        previous build with a different embedding dimension). A simple
        delete_collection() does NOT remove the on-disk HNSW files reliably
        across all ChromaDB versions, so we delete the entire db_path tree.
        """
        import shutil
        logger.warning(
            "Hard-resetting RAG database at '%s' to fix stale HNSW index…",
            self.db_path,
        )
        try:
            self._client = None   # release file handles
        except Exception:
            pass
        shutil.rmtree(self.db_path, ignore_errors=True)
        os.makedirs(self.db_path, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._needs_dimension_rebuild = False
        logger.info("RAG database hard-reset complete.")

    def build_knowledge_base(self, force_rebuild: bool = False) -> None:
        """
        Embed all threat KB entries and upsert into ChromaDB.
        Skips if already populated (unless force_rebuild=True or a dimension
        mismatch was detected at init time).
        """
        # Auto-force rebuild when stale HNSW dimension mismatch was detected
        if self._needs_dimension_rebuild:
            force_rebuild = True

        existing = self._collection.count()
        if existing > 0 and not force_rebuild:
            logger.info("Knowledge base already populated (%d entries). Skipping.", existing)
            return

        if force_rebuild and existing > 0:
            # Hard reset: wipes stale HNSW segment files from disk
            self._hard_reset_db()
        elif force_rebuild:
            # Collection is empty but a rebuild was explicitly requested — clear anyway
            self._hard_reset_db()

        logger.info("Building threat knowledge base…")
        texts_and_meta = get_all_texts()

        texts    = [t for t, _ in texts_and_meta]
        metas    = [m for _, m in texts_and_meta]
        ids      = [f"attack_{m['attack_id']}" for m in metas]

        embeddings = self._emb_model.embed_documents(texts, show_progress=True)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metas,
        )
        logger.info("Knowledge base built: %d entries", self._collection.count())

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        filter_severity: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve top-k relevant threat entries for a free-text query.

        Parameters
        ----------
        query_text      : natural language or structured description of detected behaviour
        top_k           : override default top-k
        filter_severity : optional filter ('HIGH', 'CRITICAL', 'MEDIUM', 'LOW')

        Returns
        -------
        list of dicts with keys: id, document, metadata, distance
        """
        k = top_k or self.top_k
        query_emb = self._emb_model.embed_query(query_text)

        where = {"severity": filter_severity} if filter_severity else None

        results = self._collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=min(k, self._collection.count()),
            where=where,
        )

        out = []
        for i in range(len(results["ids"][0])):
            out.append({
                "id":       results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return out

    def query_by_attack_id(self, attack_id: int) -> Optional[ThreatEntry]:
        """Directly fetch the knowledge base entry for a known attack type."""
        return get_threat_by_id(attack_id)

    # ── Alert generation ──────────────────────────────────────────────────────

    def format_alert(
        self,
        attack_id: int,
        probabilities: list[float],
        session_stats: dict,
        session_idx: int = 0,
        config: str = "c",
    ) -> dict:
        """
        Build a structured threat alert combining model output and RAG context.

        Parameters
        ----------
        attack_id     : predicted attack class
        probabilities : per-class probability vector
        session_stats : dict of statistical features for the session
        session_idx   : session index for logging

        Returns
        -------
        dict : structured alert with description, indicators, response actions, IOCs
        """
        attack_name = ID_TO_LABEL.get(attack_id, f"Unknown ({attack_id})")
        confidence  = float(probabilities[attack_id]) if attack_id < len(probabilities) else 0.0

        # Get direct KB entry
        kb_entry = self.query_by_attack_id(attack_id)

        # Build RAG query from session statistics
        query_text = self._build_rag_query(attack_name, session_stats)
        retrieved  = self.query(query_text, top_k=3)

        top_preds = self._top_predictions(probabilities, n=3)

        alert = {
            "session_idx":    session_idx,
            "attack_id":      attack_id,
            "attack_name":    attack_name,
            "confidence":     round(confidence, 4),
            "severity":       kb_entry.severity if kb_entry else "UNKNOWN",
            "model_config":   config.upper(),
            "description":    kb_entry.description if kb_entry else "",
            "mitre_tactics":  kb_entry.mitre_tactics if kb_entry else [],
            "techniques":     kb_entry.techniques[:5] if kb_entry else [],
            "zeek_indicators": kb_entry.zeek_indicators if kb_entry else [],
            "network_iocs":   kb_entry.network_iocs if kb_entry else [],
            "response_actions": kb_entry.response_actions if kb_entry else [],
            "tags":           kb_entry.tags if kb_entry else [],
            "rag_context":    [r["document"][:400] for r in retrieved[:2]],
            "session_stats":  session_stats,
            "top_predictions": top_preds,
        }

        # ── Groq LLM enrichment (optional) ───────────────────────────────────
        if self._groq is not None and self._groq.available and attack_id != 0:
            try:
                kb_context_text = (
                    kb_entry.to_text() if kb_entry else
                    "\n".join(r["document"] for r in retrieved[:2])
                )
                groq_result = self._groq.analyze(
                    attack_name=attack_name,
                    confidence=confidence,
                    session_stats=session_stats,
                    kb_context=kb_context_text,
                    top_predictions=top_preds,
                )
                if groq_result:
                    alert["groq_analysis"] = groq_result
                    # Promote Groq's severity context and fp assessment to top level
                    alert["fp_likelihood"]  = groq_result.get("fp_likelihood", "UNKNOWN")
                    alert["fp_reasoning"]   = groq_result.get("fp_reasoning", "")
                    # Prefer Groq's contextual summary over static KB description
                    if groq_result.get("threat_summary"):
                        alert["description"] = groq_result["threat_summary"]
                    logger.debug("Groq enrichment applied for alert %s (fp=%s)",
                                 attack_name, alert["fp_likelihood"])
            except Exception as e:
                logger.warning("Groq enrichment failed (non-fatal): %s", e)

        return alert

    def _build_rag_query(self, attack_name: str, stats: dict) -> str:
        """Build a natural-language query from session stats for RAG retrieval."""
        parts = [f"Detected attack type: {attack_name}"]
        if stats.get("stat_syn_ratio", 0) > 0.5:
            parts.append("High SYN flag ratio suggesting SYN flood")
        if stats.get("stat_port_entropy", 0) > 3.0:
            parts.append("High destination port entropy suggesting port scanning")
        if stats.get("stat_unique_dst_ports", 0) > 50:
            parts.append("Many unique destination ports accessed")
        if stats.get("stat_short_flow_ratio", 0) > 0.8:
            parts.append("Mostly very short flows suggesting rapid connection attempts")
        if stats.get("stat_avg_flow_bps", 0) > 1e6:
            parts.append("High bandwidth suggesting volumetric attack")
        if stats.get("stat_rst_ratio", 0) > 0.3:
            parts.append("High RST flag count suggesting connection refusals")
        return ". ".join(parts)

    def _top_predictions(self, probs: list[float], n: int = 3) -> list[dict]:
        """Return top-n predicted classes with names and probabilities."""
        indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:n]
        return [
            {"attack_id": i, "attack_name": ID_TO_LABEL.get(i, str(i)),
             "probability": round(p, 4)}
            for i, p in indexed
        ]

    # ── Context retrieval for dashboard ──────────────────────────────────────

    def get_threat_intelligence(self, attack_name: str) -> dict:
        """Return full intelligence report for a named attack type."""
        results = self.query(f"Attack: {attack_name}", top_k=5)

        # Find direct match
        direct = next(
            (r for r in results if attack_name.lower() in r["metadata"]["attack_name"].lower()),
            results[0] if results else None,
        )

        if direct:
            attack_id = direct["metadata"]["attack_id"]
            entry = get_threat_by_id(attack_id)
            if entry:
                return {
                    "attack_name":    entry.attack_name,
                    "category":       entry.category,
                    "severity":       entry.severity,
                    "description":    entry.description,
                    "mitre_tactics":  entry.mitre_tactics,
                    "techniques":     entry.techniques,
                    "zeek_indicators": entry.zeek_indicators,
                    "network_iocs":   entry.network_iocs,
                    "response_actions": entry.response_actions,
                    "tags":           entry.tags,
                    "similar_threats": [
                        r["metadata"]["attack_name"]
                        for r in results if r != direct
                    ][:3],
                }
        return {"error": f"No intelligence found for: {attack_name}"}
