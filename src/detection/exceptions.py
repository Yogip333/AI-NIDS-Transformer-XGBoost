"""
Typed exceptions for the Hybrid Transformer → XGBoost detection pipeline.

Every exception carries enough context to:
  1. Identify exactly which pipeline stage failed.
  2. Produce an actionable log message.
  3. Map to an appropriate HTTP status code in the REST layer.

Hierarchy
---------
DetectionError (base)
├── ModelNotLoadedError        — checkpoint missing or wrong architecture
├── SchemaError (base)
│   ├── FeatureDimensionError  — wrong column count
│   ├── SessionSizeError       — wrong number of flows in window
│   └── NaNInfError            — NaN / Inf in input tensor
├── InferenceError             — model forward pass failed for a known reason
└── EnrichmentError            — RAG / Groq failed (non-fatal; detection remains valid)
"""
from __future__ import annotations


class DetectionError(RuntimeError):
    """Base class for all detection-pipeline errors."""


# ── Model loading ─────────────────────────────────────────────────────────────

class ModelNotLoadedError(DetectionError):
    """
    Raised when a required model checkpoint is absent or cannot be loaded.

    Parameters
    ----------
    component : human-readable name ('Config-C XGBoost', 'FeatureScaler', …)
    path      : expected filesystem path of the missing checkpoint
    """

    def __init__(self, component: str, path: str | None = None) -> None:
        self.component = component
        self.path = path
        msg = f"{component} is not loaded"
        if path:
            msg += f" (expected at: {path})"
        msg += ". Run scripts/train_model.py to produce the checkpoint."
        super().__init__(msg)


# ── Schema validation ─────────────────────────────────────────────────────────

class SchemaError(DetectionError):
    """Base class for schema-validation failures."""


class FeatureDimensionError(SchemaError):
    """
    Raised when a feature matrix does not have the expected number of columns.

    Parameters
    ----------
    stage    : pipeline stage where the check was made ('realign', 'scaler', …)
    expected : expected column count
    actual   : actual column count
    """

    def __init__(self, stage: str, expected: int, actual: int) -> None:
        self.stage = stage
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"[{stage}] Feature dimension mismatch: "
            f"expected {expected} columns, got {actual}. "
            f"Verify the input follows the CICIDS2017 FEATURE_COLS ordering."
        )


class SessionSizeError(SchemaError):
    """
    Raised when a session window contains the wrong number of flows.

    Parameters
    ----------
    expected : expected number of flows (EXPECTED_SESSION_SIZE = 20)
    actual   : actual number of flows received
    """

    def __init__(self, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Session size mismatch: expected {expected} flows, got {actual}. "
            f"The Transformer positional encoding and statistical features "
            f"were trained with sessions of exactly {expected} flows."
        )


class NaNInfError(SchemaError):
    """
    Raised when NaN or ±Inf values are detected in the feature matrix
    after cleaning. This indicates a malformed input that would corrupt
    the scaler transform and Transformer forward pass.

    Parameters
    ----------
    stage  : pipeline stage where the check was made
    n_bad  : number of bad values found
    """

    def __init__(self, stage: str, n_bad: int) -> None:
        self.stage = stage
        self.n_bad = n_bad
        super().__init__(
            f"[{stage}] {n_bad} NaN/Inf value(s) found in feature matrix. "
            f"Ensure all flow features are valid finite numbers before "
            f"submitting to the detection pipeline."
        )


# ── Inference ─────────────────────────────────────────────────────────────────

class InferenceError(DetectionError):
    """
    Raised when the model forward pass fails for a recoverable but known reason
    (e.g. shape mismatch between the stored checkpoint and the current input).

    Parameters
    ----------
    config  : detection config that failed ('C', 'B', …)
    cause   : underlying exception
    """

    def __init__(self, config: str, cause: Exception) -> None:
        self.config = config
        self.cause = cause
        super().__init__(
            f"Config-{config} inference failed: {cause}. "
            f"Possible causes: checkpoint trained with a different feature schema, "
            f"scaler not fitted, or corrupted model file."
        )


# ── Enrichment ────────────────────────────────────────────────────────────────

class EnrichmentError(DetectionError):
    """
    Raised when RAG retrieval or Groq LLM enrichment fails.

    This is explicitly *non-fatal*: the detection result (attack_id,
    probabilities) remains valid.  The alert should still be emitted with
    enrichment fields missing rather than suppressing the detection entirely.

    Parameters
    ----------
    stage  : 'rag' or 'groq'
    cause  : underlying exception or error description
    """

    def __init__(self, stage: str, cause: str | Exception) -> None:
        self.stage = stage
        self.cause = cause
        super().__init__(
            f"Enrichment failure ({stage}): {cause}. "
            f"The detection result is valid; alert will be emitted without "
            f"full threat intelligence context."
        )
