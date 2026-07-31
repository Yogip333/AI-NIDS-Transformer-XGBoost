# Adapted from concepts by:
# Lundberg, S. M. and Lee, S.-I. (2017) 'A Unified Approach to Interpreting
#   Model Predictions', Advances in Neural Information Processing Systems 30.
# Lundberg, S. M. et al. (2020) 'From Local Explanations to Global Understanding
#   with Explainable AI for Trees', Nature Machine Intelligence, 2(1), pp. 56-67.
# Abnar, S. and Zuidema, W. (2020) 'Quantifying Attention Flow in Transformers',
#   Proceedings of ACL 2020, pp. 4190-4197.

"""Interpretability utilities for the hybrid detector.

Two complementary views are exposed. Feature attribution on the XGBoost
head uses TreeSHAP as implemented in the ``shap`` package, which
materialises the exact Shapley values for tree ensembles in polynomial
time (Lundberg and Lee, 2017; Lundberg et al., 2020). Attention analysis
on the Transformer encoder provides per-layer, per-head self-attention
weights and an attention-rollout score in the style of Abnar and Zuidema
(2020) so that the CLS-token verdict can be traced back to specific
flows inside a session.
"""
import logging
from typing import Optional

import numpy as np
import torch

from src.data.feature_engineer import STAT_FEATURE_NAMES
from src.data.loader import ID_TO_LABEL

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SHAP Explainer for XGBoost
# ─────────────────────────────────────────────────────────────────────────────

class SHAPExplainer:
    """TreeSHAP wrapper around a fitted ``ZeekXGBoost`` instance.

    Implementation detail: ``shap.TreeExplainer`` implements the polynomial-
    time TreeSHAP algorithm of Lundberg et al. (2020), which gives exact
    Shapley values for a gradient-boosted tree ensemble without the Monte
    Carlo sampling used by the kernel explainer.
    """

    def __init__(self, xgb_model, feature_names: Optional[list[str]] = None):
        import shap
        self.explainer = shap.TreeExplainer(xgb_model.model)
        self.feature_names = feature_names or xgb_model._feature_names or []

    def explain_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for a batch of samples.

        Returns
        -------
        shap_values : (N, n_features, n_classes) or list of (N, n_features) arrays
        """
        import shap
        sv = self.explainer.shap_values(X)
        return sv

    def explain_single(self, x: np.ndarray) -> dict:
        """
        Explain a single prediction.

        Parameters
        ----------
        x : (n_features,) array

        Returns
        -------
        dict with feature_names, shap_values (per class), top_features
        """
        import shap
        sv = self.explainer.shap_values(x.reshape(1, -1))

        # sv is (n_classes, 1, n_features) or list length n_classes
        if isinstance(sv, list):
            sv_matrix = np.array([s[0] for s in sv]).T   # (n_features, n_classes)
        else:
            sv_matrix = sv[0]  # (n_features, n_classes)

        # Magnitude across classes
        sv_magnitude = np.abs(sv_matrix).mean(axis=-1) if sv_matrix.ndim > 1 else np.abs(sv_matrix)

        sorted_idx = np.argsort(sv_magnitude)[::-1]
        top_n = min(15, len(sorted_idx))

        top_features = []
        for i in sorted_idx[:top_n]:
            name = self.feature_names[i] if i < len(self.feature_names) else f"feat_{i}"
            top_features.append({
                "feature": name.strip(),
                "importance": float(sv_magnitude[i]),
                "value": float(x[i]),
            })

        return {
            "feature_names": self.feature_names,
            "shap_magnitude": sv_magnitude.tolist(),
            "top_features": top_features,
        }

    def global_importance(self, X: np.ndarray, n_top: int = 20) -> list[dict]:
        """
        Compute global feature importance via mean |SHAP| over a dataset.

        Returns sorted list of {feature, importance}.
        """
        import shap
        sv = self.explainer.shap_values(X)
        if isinstance(sv, list):
            sv_matrix = np.abs(np.array(sv)).mean(axis=(0, 1))  # (n_features,)
        else:
            sv_matrix = np.abs(sv).mean(axis=0)
            if sv_matrix.ndim > 1:
                sv_matrix = sv_matrix.mean(axis=-1)

        sorted_idx = np.argsort(sv_matrix)[::-1][:n_top]
        return [
            {
                "feature":    (self.feature_names[i] if i < len(self.feature_names) else f"feat_{i}").strip(),
                "importance": float(sv_matrix[i]),
            }
            for i in sorted_idx
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Attention Analysis for the Transformer
# ─────────────────────────────────────────────────────────────────────────────

class AttentionAnalyser:
    """
    Extracts and analyses self-attention weights from the Transformer encoder.
    """

    def __init__(self, classifier, device: torch.device):
        self.classifier = classifier
        self.device = device

    @torch.no_grad()
    def get_attention_maps(
        self,
        features: np.ndarray,
        log_types: np.ndarray,
        padding_mask: np.ndarray,
    ) -> list[np.ndarray]:
        """
        Get per-layer, per-head attention weights for a single session.

        Parameters
        ----------
        features     : (L, F)
        log_types    : (L,)
        padding_mask : (L,) bool

        Returns
        -------
        list of (n_heads, L+1, L+1) arrays, one per encoder layer
        """
        self.classifier.eval()
        feat_t = torch.from_numpy(features).unsqueeze(0).to(self.device)
        lt_t   = torch.from_numpy(log_types.astype(np.int64)).unsqueeze(0).to(self.device)
        pm_t   = torch.from_numpy(padding_mask).unsqueeze(0).to(self.device)

        attn_list = self.classifier.encoder.get_attention_weights(feat_t, lt_t, pm_t)
        return [a.squeeze(0).cpu().numpy() for a in attn_list]   # list of (n_heads, L+1, L+1)

    def cls_attention_over_events(
        self,
        attention_maps: list[np.ndarray],
        layer: int = -1,
    ) -> np.ndarray:
        """
        Extract CLS → events attention weights (averaged over heads) from a given layer.

        Returns
        -------
        np.ndarray : (L,) attention weights for each event position
        """
        attn = attention_maps[layer]        # (n_heads, L+1, L+1)
        cls_attn = attn[:, 0, 1:]          # CLS row, skip CLS column → (n_heads, L)
        return cls_attn.mean(axis=0)        # (L,)

    def identify_anomalous_events(
        self,
        attention_weights: np.ndarray,
        top_n: int = 5,
    ) -> list[dict]:
        """
        Identify the events most attended to by the CLS token.

        Returns
        -------
        list of {event_idx, attention_weight} sorted by weight descending
        """
        from src.data.zeek_mapper import LOG_TYPE_NAMES
        sorted_idx = np.argsort(attention_weights)[::-1][:top_n]
        return [
            {"event_idx": int(i), "attention_weight": float(attention_weights[i])}
            for i in sorted_idx
        ]

    def layer_wise_relevance(
        self, attention_maps: list[np.ndarray]
    ) -> np.ndarray:
        """Attention rollout across encoder layers, Abnar and Zuidema (2020).

        Residual-adjusted attention matrices are multiplied through the
        stack to approximate the total information flow from each input
        position to the CLS token, which is a more faithful attribution
        than a single final-layer attention map.
        """
        n_layers = len(attention_maps)
        L1 = attention_maps[0].shape[-1]
        rollout = np.eye(L1)

        for attn in attention_maps:
            # Average over heads and add residual
            avg = attn.mean(axis=0)             # (L+1, L+1)
            avg = avg + np.eye(L1)
            avg = avg / avg.sum(axis=-1, keepdims=True)
            rollout = rollout @ avg

        return rollout


# ─────────────────────────────────────────────────────────────────────────────
# Combined explainability report
# ─────────────────────────────────────────────────────────────────────────────

def generate_explanation_report(
    shap_result: dict,
    attention_events: list[dict],
    attack_name: str,
    confidence: float,
) -> str:
    """Generate a human-readable explanation combining SHAP and attention."""
    lines = [
        f"=== EXPLANATION REPORT ===",
        f"Detected: {attack_name}  (confidence: {confidence:.1%})",
        "",
        "--- Top 10 XGBoost Feature Contributions (SHAP) ---",
    ]
    for item in shap_result.get("top_features", [])[:10]:
        lines.append(f"  {item['feature']:40s}  SHAP={item['importance']:+.4f}  value={item['value']:.4f}")

    lines += [
        "",
        "--- Most Attended Network Events (Transformer Attention) ---",
    ]
    for item in attention_events[:5]:
        lines.append(f"  Event index {item['event_idx']:3d}  attention={item['attention_weight']:.4f}")

    lines.append("")
    return "\n".join(lines)
