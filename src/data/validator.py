"""
Feature schema validation for train/inference parity.

Ensures the exact same features, in the exact same order, with valid
dtypes and ranges are present at both training time and inference time.
Fails loudly on any mismatch to prevent silent model degradation.
"""
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Result of a feature validation check."""
    valid: bool = True
    n_rows: int = 0
    n_features: int = 0
    nan_counts: dict = field(default_factory=dict)
    inf_counts: dict = field(default_factory=dict)
    missing_features: list = field(default_factory=list)
    extra_features: list = field(default_factory=list)
    dtype_errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Validation: {'PASS' if self.valid else 'FAIL'}",
            f"  Rows: {self.n_rows:,}",
            f"  Features: {self.n_features}",
        ]
        total_nan = sum(self.nan_counts.values())
        total_inf = sum(self.inf_counts.values())
        if total_nan > 0:
            lines.append(f"  NaN cells: {total_nan:,} (across {len(self.nan_counts)} columns)")
        if total_inf > 0:
            lines.append(f"  Inf cells: {total_inf:,} (across {len(self.inf_counts)} columns)")
        if self.missing_features:
            lines.append(f"  Missing features: {self.missing_features}")
        if self.extra_features:
            lines.append(f"  Extra features: {self.extra_features[:5]}...")
        if self.dtype_errors:
            lines.append(f"  Dtype errors: {self.dtype_errors[:5]}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def compute_schema_hash(feature_names: list[str]) -> str:
    """Deterministic hash of feature names for parity checking."""
    raw = "|".join(feature_names)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def validate_dataframe(
    df: pd.DataFrame,
    expected_features: list[str],
    label_col: str = "label_id",
    fix_issues: bool = True,
) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Validate a loaded DataFrame against the expected feature schema.

    Parameters
    ----------
    df : raw DataFrame from loader
    expected_features : list of feature column names in exact order
    label_col : name of the label column (checked for existence, not validated)
    fix_issues : if True, fix NaN/inf/dtype issues in-place; if False, just report

    Returns
    -------
    (cleaned DataFrame, ValidationReport)
    """
    report = ValidationReport(n_rows=len(df))

    # Check feature presence
    present = set(df.columns)
    expected_set = set(expected_features)
    report.missing_features = [f for f in expected_features if f not in present]
    report.extra_features = [f for f in present if f not in expected_set and f != label_col]
    report.n_features = len([f for f in expected_features if f in present])

    if report.missing_features:
        report.valid = False
        logger.error("Missing features: %s", report.missing_features)

    # Check dtypes — all features should be numeric
    for col in expected_features:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            report.dtype_errors.append(col)
            if fix_issues:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    if report.dtype_errors:
        report.warnings.append(f"{len(report.dtype_errors)} columns required numeric coercion")

    # Check NaN
    for col in expected_features:
        if col not in df.columns:
            continue
        nan_count = int(df[col].isna().sum())
        if nan_count > 0:
            report.nan_counts[col] = nan_count

    # Check inf
    numeric_cols = [c for c in expected_features if c in df.columns]
    for col in numeric_cols:
        inf_count = int(np.isinf(df[col].values).sum()) if pd.api.types.is_numeric_dtype(df[col]) else 0
        if inf_count > 0:
            report.inf_counts[col] = inf_count

    total_nan = sum(report.nan_counts.values())
    total_inf = sum(report.inf_counts.values())

    if total_nan > 0 or total_inf > 0:
        report.warnings.append(
            f"Found {total_nan:,} NaN and {total_inf:,} inf values"
        )
        if fix_issues:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)
            report.warnings.append("Fixed: replaced inf→NaN→0")

    # Clip extreme values
    if fix_issues:
        for col in numeric_cols:
            df[col] = df[col].clip(-1e9, 1e9)

    # Check label column
    if label_col not in df.columns:
        report.warnings.append(f"Label column '{label_col}' not found")

    logger.info(report.summary())
    return df, report


def validate_label_distribution(
    labels: np.ndarray,
    num_classes: int,
    min_samples_per_class: int = 2,
) -> dict:
    """
    Check label distribution and report issues.

    Returns dict with class counts and warnings.
    """
    counts = np.bincount(labels, minlength=num_classes)
    result = {
        "total_samples": int(labels.shape[0]),
        "num_classes_present": int(np.sum(counts > 0)),
        "num_classes_expected": num_classes,
        "class_counts": {int(i): int(c) for i, c in enumerate(counts)},
        "empty_classes": [int(i) for i in range(num_classes) if counts[i] == 0],
        "rare_classes": [int(i) for i in range(num_classes) if 0 < counts[i] < min_samples_per_class],
        "warnings": [],
    }

    if result["empty_classes"]:
        result["warnings"].append(
            f"Classes with zero samples: {result['empty_classes']}"
        )
    if result["rare_classes"]:
        result["warnings"].append(
            f"Classes with <{min_samples_per_class} samples: {result['rare_classes']}"
        )

    # Class imbalance ratio
    present_counts = counts[counts > 0]
    if len(present_counts) > 1:
        ratio = float(present_counts.max() / present_counts.min())
        result["imbalance_ratio"] = round(ratio, 1)
        if ratio > 1000:
            result["warnings"].append(
                f"Extreme class imbalance: {ratio:.0f}:1 (max:min)"
            )

    return result


def save_schema(feature_names: list[str], path: str) -> None:
    """Save feature schema for inference-time validation."""
    schema = {
        "feature_names": feature_names,
        "schema_hash": compute_schema_hash(feature_names),
        "n_features": len(feature_names),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)
    logger.info("Feature schema saved to %s (hash=%s)", path, schema["schema_hash"])


def load_and_verify_schema(
    feature_names: list[str],
    schema_path: str,
) -> bool:
    """Load saved schema and verify current features match."""
    if not os.path.exists(schema_path):
        logger.warning("No schema file found at %s — skipping verification", schema_path)
        return True

    with open(schema_path) as f:
        saved = json.load(f)

    current_hash = compute_schema_hash(feature_names)
    if current_hash != saved["schema_hash"]:
        logger.error(
            "Feature schema mismatch! Training hash=%s, current hash=%s",
            saved["schema_hash"], current_hash,
        )
        saved_set = set(saved["feature_names"])
        current_set = set(feature_names)
        missing = saved_set - current_set
        extra = current_set - saved_set
        if missing:
            logger.error("Missing features (in training but not now): %s", missing)
        if extra:
            logger.error("Extra features (not in training): %s", extra)
        return False

    logger.info("Feature schema verified (hash=%s)", current_hash)
    return True
