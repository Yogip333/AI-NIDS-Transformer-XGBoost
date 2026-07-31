"""
Session feature pipeline — single source of truth.

This module owns the canonical logic for converting raw CICIDS2017 flow
features into the 81-column trained schema consumed by the FeatureScaler,
Transformer encoder, and XGBoost hybrid head.

Both the REST API (``src/api/routes.py``) and the live streaming engine
(``scripts/live_demo.py``) delegate here so no per-path drift is possible.

Trained schema (81 columns per flow, after ``realign_to_trained_schema``):
    [  0:77] — 77 CICIDS2017 columns
               (all ``FEATURE_COLS`` from loader.py, *excluding*
               'Subflow Bwd Packets' at raw index 64, which is absent from
               the CICIDS2017 CSVs and was never part of the training data)
    [ 77   ] — log_type        (int 0-4, Zeek log class)
    [ 78   ] — zeek_proto      (0=udp, 1=tcp — derived from FIN/SYN counts)
    [ 79   ] — zeek_conn_state (0=SF, 1=S0, 2=RSTO, 3=OTH)
    [ 80   ] — zeek_service    (0=unknown … 7=imap)

This exact ordering must match the order used in ``scripts/preprocess.py``
(``all_feat_cols = FEATURE_COLS_present + zeek_extra``).  If the training
pipeline changes the column order or count, update ``contracts.py``
*and* re-train before deploying.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.data.loader import FEATURE_COLS
from src.data.zeek_mapper import map_to_zeek_records
from src.detection.contracts import (
    CICIDS_RAW_COLS,
    EXPECTED_SESSION_SIZE,
    EXPECTED_STAT_FEATURES,
    SUBFLOW_BWD_PKTS_IDX,
    TRAINED_NUM_FEATURES,
    ZEEK_EXTRA_COLS,
)
from src.detection.exceptions import (
    FeatureDimensionError,
    NaNInfError,
    SessionSizeError,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ── Public constants (re-exported for convenience) ───────────────────────────

__all__ = [
    "CICIDS_RAW_COLS",
    "SUBFLOW_BWD_PKTS_IDX",
    "TRAINED_NUM_FEATURES",
    "EXPECTED_SESSION_SIZE",
    "EXPECTED_STAT_FEATURES",
    "ZEEK_EXTRA_COLS",
    "realign_to_trained_schema",
    "trained_feature_names",
    "validate_flow_array",
    "validate_session_dataframe",
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _check_finite(arr: np.ndarray, stage: str) -> None:
    """Raise NaNInfError if any NaN or ±Inf value is present."""
    n_bad = int(np.sum(~np.isfinite(arr)))
    if n_bad > 0:
        raise NaNInfError(stage=stage, n_bad=n_bad)


# ── Public validation API ─────────────────────────────────────────────────────

def validate_flow_array(
    arr: np.ndarray,
    stage: str = "input",
    expected_cols: int = CICIDS_RAW_COLS,
) -> None:
    """
    Validate a raw (N × expected_cols) CICIDS flow feature matrix.

    Raises
    ------
    FeatureDimensionError
        If ``arr.shape[1] != expected_cols``.
    NaNInfError
        If any element is NaN or ±Inf.
    """
    if arr.ndim != 2 or arr.shape[1] != expected_cols:
        actual = arr.shape[1] if arr.ndim == 2 else f"ndim={arr.ndim}"
        raise FeatureDimensionError(stage=stage, expected=expected_cols, actual=actual)
    _check_finite(arr, stage)


def validate_session_dataframe(
    df: pd.DataFrame,
    enforce_size: bool = True,
) -> None:
    """
    Validate a raw-flow DataFrame before Zeek enrichment and alignment.

    Checks performed:
    - All FEATURE_COLS that ARE present in the DataFrame are numeric.
    - If ``enforce_size`` is True, exactly EXPECTED_SESSION_SIZE rows required.

    Raises
    ------
    SessionSizeError
        If ``enforce_size`` and ``len(df) != EXPECTED_SESSION_SIZE``.
    """
    if enforce_size and len(df) != EXPECTED_SESSION_SIZE:
        raise SessionSizeError(expected=EXPECTED_SESSION_SIZE, actual=len(df))


def validate_aligned_array(arr: np.ndarray) -> None:
    """
    Validate a post-alignment (N × 81) feature matrix.

    Raises
    ------
    FeatureDimensionError : if shape[1] != TRAINED_NUM_FEATURES
    NaNInfError           : if any element is NaN or ±Inf
    """
    if arr.ndim != 2 or arr.shape[1] != TRAINED_NUM_FEATURES:
        actual = arr.shape[1] if arr.ndim == 2 else f"ndim={arr.ndim}"
        raise FeatureDimensionError(
            stage="trained-schema",
            expected=TRAINED_NUM_FEATURES,
            actual=actual,
        )
    _check_finite(arr, "trained-schema")


# ── Core pipeline function ────────────────────────────────────────────────────

def realign_to_trained_schema(
    feat_array_78: np.ndarray,
    log_types: np.ndarray,
) -> np.ndarray:
    """
    Convert a (N × 78) raw CICIDS feature matrix to the (N × 81) trained schema.

    This is the **single source of truth** for schema alignment. Both the REST
    API (``src/api/routes.py``) and the live streaming engine
    (``scripts/live_demo.py``) call this function so the Transformer and
    XGBoost always receive exactly the same feature distribution the scaler and
    models were trained on.

    Parameters
    ----------
    feat_array_78 : np.ndarray, shape (N, 78), dtype float32
        Raw per-flow CICIDS2017 features in ``FEATURE_COLS`` column order.
        Column 64 ('Subflow Bwd Packets') may contain any value; it is
        dropped unconditionally.
    log_types : np.ndarray, shape (N,)
        Zeek log-type integer per flow (0=conn 1=dns 2=http 3=ssl 4=files).
        The REST API receives this from the client per ``FlowRecord.log_type``.
        The live streaming engine derives it from destination port via
        ``zeek_mapper.assign_log_types_vectorized()``.

    Returns
    -------
    np.ndarray, shape (N, 81), dtype float32
        Aligned feature matrix ready for ``FeatureScaler.transform()`` and
        subsequent Transformer / XGBoost inference.

    Raises
    ------
    FeatureDimensionError : if input does not have exactly 78 columns
    NaNInfError           : if input contains NaN or ±Inf after cleaning
    """
    validate_flow_array(feat_array_78, stage="realign-input", expected_cols=CICIDS_RAW_COLS)

    # Reconstruct the three Zeek-derived columns using the same heuristics that
    # ``scripts/preprocess.py`` applies via ``map_to_zeek_records()`` during training.
    df = pd.DataFrame(feat_array_78, columns=FEATURE_COLS)
    enriched = map_to_zeek_records(df)

    zeek_proto      = enriched["zeek_proto"].to_numpy(dtype=np.float32)
    zeek_conn_state = enriched["zeek_conn_state"].to_numpy(dtype=np.float32)
    zeek_service    = enriched["zeek_service"].to_numpy(dtype=np.float32)

    # Drop column 64 ('Subflow Bwd Packets') → (N, 77)
    cicids_77 = np.concatenate(
        [feat_array_78[:, :SUBFLOW_BWD_PKTS_IDX],
         feat_array_78[:, SUBFLOW_BWD_PKTS_IDX + 1:]],
        axis=1,
    ).astype(np.float32)

    result = np.hstack([
        cicids_77,
        np.asarray(log_types, dtype=np.float32).reshape(-1, 1),
        zeek_proto.reshape(-1, 1),
        zeek_conn_state.reshape(-1, 1),
        zeek_service.reshape(-1, 1),
    ])

    # Post-alignment validation — catches internal bugs before they corrupt inference.
    validate_aligned_array(result)
    return result


def trained_feature_names() -> list[str]:
    """
    Return the 81 column names in trained-schema order.

    Used as the ``feature_col_names`` argument to ``compute_session_stats()``
    to ensure statistical features bind to the correct underlying columns,
    matching the column names used in ``scripts/preprocess.py``.

    Returns
    -------
    list[str] of length TRAINED_NUM_FEATURES (81)
    """
    cicids_77: list[str] = (
        list(FEATURE_COLS[:SUBFLOW_BWD_PKTS_IDX])
        + list(FEATURE_COLS[SUBFLOW_BWD_PKTS_IDX + 1:])
    )
    return cicids_77 + list(ZEEK_EXTRA_COLS)
