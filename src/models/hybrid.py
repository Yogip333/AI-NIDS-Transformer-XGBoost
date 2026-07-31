"""Hybrid detector — combines the Transformer encoder with an XGBoost head.

This module is the orchestration layer for the ablation study. It extracts
CLS-token embeddings from the fine-tuned classifier, concatenates them with
the 26-dimensional session statistics where appropriate, and trains the
three XGBoost configurations that Chapter 4 of the thesis analyses:

    Config A — XGBoost on session statistics only (no Transformer).
    Config B — XGBoost on Transformer embeddings only.
    Config C — XGBoost on embeddings concatenated with session statistics.

Config C is the primary production path. Config B is kept as a graceful
fall-back when the session-statistics vector is unavailable at inference
time (for example, when streaming partial sessions).
"""
import logging
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.detection.contracts import (
    EXPECTED_EMBEDDING_DIM,
    EXPECTED_HYBRID_INPUT_DIM,
    EXPECTED_NUM_CLASSES,
    EXPECTED_STAT_FEATURES,
)
from src.detection.exceptions import InferenceError, ModelNotLoadedError
from src.models.transformer import TransformerClassifier
from src.models.xgboost_model import ZeekXGBoost, compute_class_weights

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    classifier: TransformerClassifier,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the Transformer encoder over all sessions and collect CLS embeddings.

    Returns
    -------
    embeddings : (N, d_model)
    labels     : (N,)
    """
    classifier.eval()
    all_embeddings = []
    all_labels = []

    iterator = tqdm(dataloader, desc="Extract embeddings", leave=False, disable=(len(dataloader) == 0))
    for batch in iterator:
        features     = batch["features"].to(device)
        log_types    = batch["log_types"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        emb = classifier.get_embedding(features, log_types, padding_mask)
        all_embeddings.append(emb.cpu().numpy())

        if "label" in batch:
            all_labels.append(batch["label"].numpy())

    embeddings = np.vstack(all_embeddings)
    labels     = np.concatenate(all_labels) if all_labels else np.array([])
    return embeddings.astype(np.float32), labels.astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid model orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class HybridIDS:
    """
    Orchestrates all three evaluation configurations:

      config_a : XGBoost on statistical features only
      config_b : XGBoost on Transformer embeddings only
      config_c : XGBoost on (Transformer embeddings ‖ statistical features)

    Parameters
    ----------
    cfg       : top-level project config dict
    classifier : fine-tuned TransformerClassifier
    """

    def __init__(
        self,
        cfg: dict,
        classifier: TransformerClassifier,
        num_classes: int = 15,
    ):
        self.cfg = cfg
        self.classifier = classifier
        self.num_classes = num_classes
        self.device = next(classifier.parameters()).device

        self.xgb_a: Optional[ZeekXGBoost] = None  # stat only
        self.xgb_b: Optional[ZeekXGBoost] = None  # embed only
        self.xgb_c: Optional[ZeekXGBoost] = None  # hybrid

    def _combine(self, embeddings: np.ndarray, stats: np.ndarray) -> np.ndarray:
        return np.hstack([embeddings, stats])

    # ── Training ──────────────────────────────────────────────────────────────

    def train_all(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_stats: np.ndarray,
        val_stats: np.ndarray,
        stat_feature_names: Optional[list[str]] = None,
        embed_feature_names: Optional[list[str]] = None,
    ) -> None:
        logger.info("Extracting Transformer embeddings (train set)…")
        train_emb, train_labels = extract_embeddings(self.classifier, train_loader, self.device)

        logger.info("Extracting Transformer embeddings (val set)…")
        val_emb, val_labels = extract_embeddings(self.classifier, val_loader, self.device)

        sw = compute_class_weights(train_labels, self.num_classes)

        # ── Config A: statistical features only ──────────────────────────────
        logger.info("Training Config-A (stat only)…")
        self.xgb_a = ZeekXGBoost(self.cfg, self.num_classes)
        self.xgb_a.fit(train_stats, train_labels,
                       X_val=val_stats, y_val=val_labels,
                       feature_names=stat_feature_names,
                       sample_weight=sw)

        # ── Config B: embeddings only ─────────────────────────────────────────
        logger.info("Training Config-B (embedding only)…")
        e_names = embed_feature_names or [f"emb_{i}" for i in range(train_emb.shape[1])]
        self.xgb_b = ZeekXGBoost(self.cfg, self.num_classes)
        self.xgb_b.fit(train_emb, train_labels,
                       X_val=val_emb, y_val=val_labels,
                       feature_names=e_names,
                       sample_weight=sw)

        # ── Config C: hybrid ──────────────────────────────────────────────────
        logger.info("Training Config-C (hybrid)…")
        hybrid_names = (e_names or []) + (stat_feature_names or [])
        self.xgb_c = ZeekXGBoost(self.cfg, self.num_classes)
        self.xgb_c.fit(
            self._combine(train_emb, train_stats),
            train_labels,
            X_val=self._combine(val_emb, val_stats),
            y_val=val_labels,
            feature_names=hybrid_names,
            sample_weight=sw,
        )

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        test_loader: DataLoader,
        test_stats: np.ndarray,
        config: str = "c",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        config : 'a' | 'b' | 'c'

        Returns
        -------
        predictions  : (N,)
        probabilities: (N, num_classes)
        """
        test_emb, test_labels = extract_embeddings(self.classifier, test_loader, self.device)

        if config == "a":
            model = self.xgb_a
            X = test_stats
        elif config == "b":
            model = self.xgb_b
            X = test_emb
        else:
            model = self.xgb_c
            X = self._combine(test_emb, test_stats)

        if model is None:
            raise ModelNotLoadedError(
                f"Config-{config.upper()} XGBoost",
                self.cfg.get("paths", {}).get("models_dir", "models/checkpoints"),
            )

        return model.predict(X), model.predict_proba(X)

    def predict_single_session(
        self,
        session_features: np.ndarray,
        session_log_types: np.ndarray,
        session_stats: np.ndarray,
        config: str = "c",
    ) -> dict:
        """
        Real-time prediction for a single session.

        Parameters
        ----------
        session_features : (L, F)
        session_log_types: (L,)
        session_stats    : (num_stat_features,)

        Returns
        -------
        dict with 'prediction', 'probabilities', 'embedding'
        """
        from src.data.sessionizer import SessionDataset

        # Create a single-sample batch
        feat_t = torch.from_numpy(session_features).unsqueeze(0).to(self.device)
        lt_t   = torch.from_numpy(session_log_types.astype(np.int64)).unsqueeze(0).to(self.device)
        # Build padding mask
        L = session_features.shape[0]
        max_len = self.classifier.encoder.pos_enc.pe.shape[1] - 1
        pad_len = max(0, max_len - L)
        if pad_len > 0:
            pad_feat = torch.zeros(1, pad_len, session_features.shape[1], device=self.device)
            feat_t   = torch.cat([feat_t, pad_feat], dim=1)
            pad_lt   = torch.zeros(1, pad_len, dtype=torch.long, device=self.device)
            lt_t     = torch.cat([lt_t, pad_lt], dim=1)
        mask = torch.zeros(1, max(L, max_len), dtype=torch.bool, device=self.device)
        if pad_len > 0:
            mask[0, L:] = True

        # Trim to max_len if needed
        feat_t = feat_t[:, :max_len, :]
        lt_t   = lt_t[:, :max_len]
        mask   = mask[:, :max_len]

        self.classifier.eval()
        with torch.no_grad():
            emb = self.classifier.get_embedding(feat_t, lt_t, mask).cpu().numpy()

        stats = session_stats.reshape(1, -1)

        if config == "b":
            model = self.xgb_b
            X = emb
        else:
            # Default: config == "c" (hybrid — primary production path)
            model = self.xgb_c
            X = self._combine(emb, stats)

        if model is None:
            raise ModelNotLoadedError(
                f"Config-{config.upper()} XGBoost",
                os.path.join(
                    self.cfg.get("paths", {}).get("models_dir", "models/checkpoints"),
                    "xgb_hybrid.joblib" if config != "b" else "xgb_embed_only.joblib",
                ),
            )

        # Shape contract: hybrid input must be (1, EXPECTED_HYBRID_INPUT_DIM) or (1, emb_dim)
        expected_cols = EXPECTED_HYBRID_INPUT_DIM if config != "b" else EXPECTED_EMBEDDING_DIM
        if X.shape[1] != expected_cols:
            raise InferenceError(
                config.upper(),
                ValueError(
                    f"XGBoost input shape mismatch: "
                    f"expected (1, {expected_cols}), got {X.shape}"
                ),
            )

        try:
            pred  = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
        except Exception as exc:
            raise InferenceError(config.upper(), exc) from exc

        if len(proba) != EXPECTED_NUM_CLASSES:
            raise InferenceError(
                config.upper(),
                ValueError(
                    f"XGBoost output class count mismatch: "
                    f"expected {EXPECTED_NUM_CLASSES}, got {len(proba)}"
                ),
            )

        return {"prediction": int(pred), "probabilities": proba.tolist(), "embedding": emb[0].tolist()}

    def predict_with_fallback(
        self,
        session_features: np.ndarray,
        session_log_types: np.ndarray,
        session_stats: np.ndarray,
    ) -> dict:
        """
        Real-time single-session prediction.

        Tries Transformer-backed XGBoost heads in priority order:

            C  (hybrid: embeddings + stats)   →  primary, 90.94% acc
            B  (embeddings only)              →  90.94% acc

        Config A (stat-only, bypasses the Transformer entirely) is intentionally
        excluded. The Transformer encoder is a hard architectural requirement; if
        neither C nor B can run, the caller must handle the failure explicitly.

        Returns the same dict shape as ``predict_single_session`` plus a
        ``'config_used'`` key ('C' or 'B').

        Raises
        ------
        ModelNotLoadedError
            If both Config-C and Config-B checkpoints are absent.
        InferenceError
            If the loaded checkpoint produces a shape / class-count mismatch.
        """
        # Only C (hybrid) and B (embed-only) are production-valid paths.
        # Config A bypasses the Transformer and is excluded from production inference.
        chain = [
            ("c", self.xgb_c),
            ("b", self.xgb_b),
        ]
        last_err: Optional[Exception] = None
        for config_name, model in chain:
            if model is None:
                continue
            try:
                result = self.predict_single_session(
                    session_features, session_log_types, session_stats,
                    config=config_name,
                )
                result["config_used"] = config_name.upper()
                return result
            except (InferenceError, ModelNotLoadedError) as e:
                last_err = e
                logger.warning(
                    "predict_with_fallback: Config-%s failed (%s) — trying next",
                    config_name.upper(), e,
                )

        model_dir = self.cfg.get("paths", {}).get("models_dir", "models/checkpoints")
        raise ModelNotLoadedError(
            "Config-C and Config-B XGBoost (both Transformer-backed heads absent)",
            model_dir,
        )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save_all(self, model_dir: str) -> None:
        os.makedirs(model_dir, exist_ok=True)
        if self.xgb_a: self.xgb_a.save(os.path.join(model_dir, "xgb_stat_only.joblib"))
        if self.xgb_b: self.xgb_b.save(os.path.join(model_dir, "xgb_embed_only.joblib"))
        if self.xgb_c: self.xgb_c.save(os.path.join(model_dir, "xgb_hybrid.joblib"))
        logger.info("All XGBoost models saved to %s", model_dir)

    def load_all(self, model_dir: str) -> None:
        p_a = os.path.join(model_dir, "xgb_stat_only.joblib")
        p_b = os.path.join(model_dir, "xgb_embed_only.joblib")
        p_c = os.path.join(model_dir, "xgb_hybrid.joblib")
        if os.path.exists(p_a): self.xgb_a = ZeekXGBoost.load(p_a)
        if os.path.exists(p_b): self.xgb_b = ZeekXGBoost.load(p_b)
        if os.path.exists(p_c): self.xgb_c = ZeekXGBoost.load(p_c)
        logger.info("Loaded XGBoost models from %s", model_dir)
