"""
Data loader for CICIDS2017 dataset.
Converts CIC flow-level CSV features into structured records for Zeek log simulation.
"""
import os
import glob
import logging
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Canonical label mapping (normalised names → int ID)
LABEL_MAP: dict[str, int] = {
    "BENIGN": 0,
    "Bot": 1,
    "DDoS": 2,
    "DoS GoldenEye": 3,
    "DoS Hulk": 4,
    "DoS Slowhttptest": 5,
    "DoS slowloris": 6,
    "FTP-Patator": 7,
    "Heartbleed": 8,
    "Infiltration": 9,
    "PortScan": 10,
    "SSH-Patator": 11,
    "Web Attack \x96 Brute Force": 12,  # em-dash variant in raw CSV
    "Web Attack \x96 Sql Injection": 13,
    "Web Attack \x96 XSS": 14,
    "Web Attack  Brute Force": 12,       # double-space variant
    "Web Attack  Sql Injection": 13,
    "Web Attack  XSS": 14,
    "Web Attack - Brute Force": 12,
    "Web Attack - Sql Injection": 13,
    "Web Attack - XSS": 14,
    "Web Attack ï¿½ Brute Force": 12,   # latin-1 misread of UTF-8 BOM (pandas latin-1 on Thursday CSV)
    "Web Attack ï¿½ Sql Injection": 13,
    "Web Attack ï¿½ XSS": 14,
}

ID_TO_LABEL: dict[int, str] = {
    0: "BENIGN",
    1: "Bot",
    2: "DDoS",
    3: "DoS GoldenEye",
    4: "DoS Hulk",
    5: "DoS Slowhttptest",
    6: "DoS slowloris",
    7: "FTP-Patator",
    8: "Heartbleed",
    9: "Infiltration",
    10: "PortScan",
    11: "SSH-Patator",
    12: "Web Attack Brute Force",
    13: "Web Attack Sql Injection",
    14: "Web Attack XSS",
}

NUM_CLASSES = 15

# All 76 numerical features from CICIDS2017 (excluding label)
FEATURE_COLS = [
    " Destination Port", " Flow Duration",
    " Total Fwd Packets", " Total Backward Packets",
    "Total Length of Fwd Packets", " Total Length of Bwd Packets",
    " Fwd Packet Length Max", " Fwd Packet Length Min",
    " Fwd Packet Length Mean", " Fwd Packet Length Std",
    "Bwd Packet Length Max", " Bwd Packet Length Min",
    " Bwd Packet Length Mean", " Bwd Packet Length Std",
    "Flow Bytes/s", " Flow Packets/s",
    " Flow IAT Mean", " Flow IAT Std", " Flow IAT Max", " Flow IAT Min",
    "Fwd IAT Total", " Fwd IAT Mean", " Fwd IAT Std", " Fwd IAT Max", " Fwd IAT Min",
    "Bwd IAT Total", " Bwd IAT Mean", " Bwd IAT Std", " Bwd IAT Max", " Bwd IAT Min",
    "Fwd PSH Flags", " Bwd PSH Flags", " Fwd URG Flags", " Bwd URG Flags",
    " Fwd Header Length", " Bwd Header Length",
    "Fwd Packets/s", " Bwd Packets/s",
    " Min Packet Length", " Max Packet Length",
    " Packet Length Mean", " Packet Length Std", " Packet Length Variance",
    "FIN Flag Count", " SYN Flag Count", " RST Flag Count",
    " PSH Flag Count", " ACK Flag Count", " URG Flag Count",
    " CWE Flag Count", " ECE Flag Count",
    " Down/Up Ratio", " Average Packet Size",
    " Avg Fwd Segment Size", " Avg Bwd Segment Size",
    " Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", " Fwd Avg Packets/Bulk", " Fwd Avg Bulk Rate",
    " Bwd Avg Bytes/Bulk", " Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", " Subflow Fwd Bytes",
    "Subflow Bwd Packets", " Subflow Bwd Bytes",
    "Init_Win_bytes_forward", " Init_Win_bytes_backward",
    " act_data_pkt_fwd", " min_seg_size_forward",
    "Active Mean", " Active Std", " Active Max", " Active Min",
    "Idle Mean", " Idle Std", " Idle Max", " Idle Min",
]

LABEL_COL = " Label"


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _encode_label(raw: str) -> int:
    """Map raw string label to integer, handling encoding artefacts."""
    raw = raw.strip()
    if raw in LABEL_MAP:
        return LABEL_MAP[raw]
    # Fuzzy match for unicode artefacts (e.g. \ufffd → \x96, ï¿½ → -)
    cleaned = raw.replace("\ufffd", "\x96").replace("\u2013", "\x96")
    if cleaned in LABEL_MAP:
        return LABEL_MAP[cleaned]
    # latin-1 misread: ï¿½ (3 bytes \xef\xbf\xbd decoded as latin-1) → dash
    cleaned2 = raw.replace("ï¿½", "-")
    if cleaned2 in LABEL_MAP:
        return LABEL_MAP[cleaned2]
    # Check if it contains known keywords
    upper = raw.upper()
    for key, val in LABEL_MAP.items():
        if key.upper() in upper:
            return val
    logger.warning("Unknown label: %r — mapping to BENIGN(0)", raw)
    return 0


def load_single_csv(
    filepath: str,
    max_rows: Optional[int] = None,
    feature_cols: Optional[list] = None,
) -> pd.DataFrame:
    """Load one CICIDS2017 CSV, clean it, and return a tidy DataFrame."""
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    logger.info("Loading %s", os.path.basename(filepath))
    df = pd.read_csv(filepath, encoding="latin-1", low_memory=False)

    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    # Keep only the columns we need (features + label)
    available = [c for c in feature_cols if c in df.columns]
    if LABEL_COL in df.columns:
        df = df[available + [LABEL_COL]].copy()
    else:
        df = df[available].copy()

    # Encode label
    if LABEL_COL in df.columns:
        df["label_id"] = df[LABEL_COL].astype(str).apply(_encode_label)
        df.drop(columns=[LABEL_COL], inplace=True)

    # Convert all feature columns to numeric (coerce errors → NaN)
    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace inf with NaN then forward-fill / fill 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Clip extreme values to [-1e9, 1e9]
    for col in available:
        df[col] = df[col].clip(-1e9, 1e9)

    return df


def load_all_data(
    data_dir: str = "Kaggle Dataset",
    max_rows_per_file: Optional[int] = None,
    feature_cols: Optional[list] = None,
) -> pd.DataFrame:
    """Load and concatenate all CICIDS2017 CSV files."""
    pattern = os.path.join(data_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for f in files:
        try:
            df = load_single_csv(f, max_rows=max_rows_per_file, feature_cols=feature_cols)
            frames.append(df)
            logger.info("  -> %d rows, label distribution: %s",
                        len(df),
                        df["label_id"].value_counts().to_dict() if "label_id" in df.columns else "N/A")
        except Exception as e:
            logger.warning("Failed to load %s: %s", f, e)

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Total rows loaded: %d", len(combined))
    return combined


def get_feature_cols_present(df: pd.DataFrame) -> list:
    """Return which FEATURE_COLS are actually in this DataFrame."""
    return [c for c in FEATURE_COLS if c in df.columns]


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split (falls back to random if class counts are too low)."""
    from sklearn.model_selection import train_test_split

    test_ratio = 1.0 - train_ratio - val_ratio

    # Only stratify if every class has at least 2 samples
    labels = df["label_id"] if "label_id" in df.columns else None
    can_stratify = (labels is not None) and (labels.value_counts().min() >= 2)
    strat = labels if can_stratify else None
    if not can_stratify:
        logger.warning("Some classes have <2 samples — using random (non-stratified) split")

    try:
        train_df, temp_df = train_test_split(
            df, test_size=(val_ratio + test_ratio),
            stratify=strat, random_state=random_state
        )
        val_strat = temp_df["label_id"] if (can_stratify and "label_id" in temp_df.columns
                                            and temp_df["label_id"].value_counts().min() >= 2) else None
        val_size_rel = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df, test_size=(1 - val_size_rel),
            stratify=val_strat, random_state=random_state
        )
    except ValueError as e:
        logger.warning("Stratified split failed (%s) — falling back to random split", e)
        train_df, temp_df = train_test_split(
            df, test_size=(val_ratio + test_ratio), random_state=random_state
        )
        val_size_rel = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df, test_size=(1 - val_size_rel), random_state=random_state
        )

    logger.info("Split: train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
