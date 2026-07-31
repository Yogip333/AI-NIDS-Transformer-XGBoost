"""
Architectural contracts for the Hybrid Transformer → XGBoost detection pipeline.

These constants encode the invariants that must hold between training
(``scripts/preprocess.py``, ``scripts/train_model.py``) and inference
(``src/api/routes.py``, ``scripts/live_demo.py``).

Changing any value here without re-training from scratch will silently corrupt
inference — treat them as an interface contract between the training pipeline
and the runtime.

Reference:
    trained schema  = 77 CICIDS + 4 Zeek columns = 81 per-flow features
    hybrid input    = 256-dim CLS embedding + 26 session stats = 282 dims
"""

# ── Per-flow feature schema ──────────────────────────────────────────────────

CICIDS_RAW_COLS: int = 78
"""Number of raw CICIDS2017 columns in FEATURE_COLS (as defined in loader.py).
Includes 'Subflow Bwd Packets' at index 64, which is absent from the CICIDS2017
CSV files and therefore dropped during trained-schema alignment."""

SUBFLOW_BWD_PKTS_IDX: int = 64
"""0-based index of 'Subflow Bwd Packets' in FEATURE_COLS. Dropped because the
CICIDS2017 CSVs emit this column as ' Subflow Bwd Packets' (leading space), which
does not match the name in FEATURE_COLS, so the column was never included during
training. Dropping it at inference reproduces that training condition."""

TRAINED_NUM_FEATURES: int = 81
"""Per-flow feature count after trained-schema alignment:
  77 CICIDS cols (FEATURE_COLS minus the missing 'Subflow Bwd Packets')
+ 4 Zeek-derived cols  (log_type, zeek_proto, zeek_conn_state, zeek_service)
= 81
This is the input dimension for both the FeatureScaler and the Transformer encoder."""

ZEEK_EXTRA_COLS: tuple[str, ...] = (
    "log_type",
    "zeek_proto",
    "zeek_conn_state",
    "zeek_service",
)
"""Ordered names of the four Zeek-derived columns appended at positions 77-80."""

# ── Session window ───────────────────────────────────────────────────────────

EXPECTED_SESSION_SIZE: int = 20
"""Number of flows per session window. Matches ``data.session_size`` in
config.yaml and the Transformer's positional encoding budget."""

# ── Statistical features ─────────────────────────────────────────────────────

EXPECTED_STAT_FEATURES: int = 26
"""Number of session-level statistical features produced by
``compute_session_stats()`` — identical to ``NUM_STAT_FEATURES`` in
feature_engineer.py.  Reproduced here so model-layer code can validate
without importing feature_engineer."""

# ── Transformer encoder ──────────────────────────────────────────────────────

EXPECTED_EMBEDDING_DIM: int = 256
"""d_model of the Transformer encoder and dimensionality of the CLS embedding
extracted by ``TransformerClassifier.get_embedding()``."""

# ── Hybrid XGBoost ───────────────────────────────────────────────────────────

EXPECTED_HYBRID_INPUT_DIM: int = EXPECTED_EMBEDDING_DIM + EXPECTED_STAT_FEATURES  # 282
"""Width of the feature vector fed to the Config-C XGBoost head:
  256 CLS embedding dims + 26 statistical features = 282."""

EXPECTED_NUM_CLASSES: int = 15
"""Number of output classes produced by XGBoost / Transformer classifier.
Must match ``NUM_CLASSES`` in loader.py and the 15-row ID_TO_LABEL map."""
