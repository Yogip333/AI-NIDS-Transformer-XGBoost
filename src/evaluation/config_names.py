"""
Canonical configuration naming for the Mini AI-SOC ablation study.

Final thesis-ready mapping (source of truth):
  Config A — XGBoost on 26 statistical features only         (baseline)
  Config B — XGBoost on 256-dim Transformer embeddings       (embedding classifier)
  Config C — XGBoost on Transformer embeddings + stats       (full hybrid)
  Config D — Transformer encoder + CLS → MLP classifier head (Transformer-only)

Internal HybridIDS attribute mapping:
  Config A → hybrid.xgb_a  / internal id "a"  / xgb_stat_only.joblib
  Config B → hybrid.xgb_b  / internal id "b"  / xgb_embed_only.joblib
  Config C → hybrid.xgb_c  / internal id "c"  / xgb_hybrid.joblib
  Config D → classifier_final.pt  (standalone, no XGBoost)
"""

# ── Thesis-facing labels ─────────────────────────────────────────────────────

CONFIG_A = "Config A"
CONFIG_B = "Config B"
CONFIG_C = "Config C"
CONFIG_D = "Config D"

CONFIG_DESCRIPTIONS = {
    CONFIG_A: "XGBoost on statistical features only (baseline)",
    CONFIG_B: "XGBoost on Transformer embeddings",
    CONFIG_C: "XGBoost on Transformer embeddings + statistical features (full hybrid)",
    CONFIG_D: "Transformer classifier (CLS -> MLP head)",
}

# ── Internal-to-thesis mapping ────────────────────────────────────────────────
# HybridIDS uses single-letter ids a/b/c.  Config D is standalone (no internal id).

INTERNAL_TO_THESIS = {
    "a": CONFIG_A,
    "b": CONFIG_B,
    "c": CONFIG_C,
}

THESIS_TO_INTERNAL = {v: k for k, v in INTERNAL_TO_THESIS.items()}

# ── Model artifact filenames ──────────────────────────────────────────────────

MODEL_ARTIFACTS = {
    CONFIG_A: "xgb_stat_only.joblib",
    CONFIG_B: "xgb_embed_only.joblib",
    CONFIG_C: "xgb_hybrid.joblib",
    CONFIG_D: "classifier_final.pt",
}

# ── SHAP eligibility (XGBoost-based configs only) ────────────────────────────

SHAP_CONFIGS = [CONFIG_A, CONFIG_B, CONFIG_C]

# ── Attention eligibility (Transformer-based configs) ─────────────────────────

ATTENTION_CONFIGS = [CONFIG_D]
