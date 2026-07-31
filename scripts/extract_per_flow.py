"""
Extract Per-Flow Arrays from Raw CSV Files.

Reads the original CICIDS2017 CSVs, applies the same preprocessing as
training (feature selection, numeric coercion, scaler), and saves per-flow
(features, labels) arrays with CORRECT per-row labels — NOT session-level labels.

This feeds the per-flow XGBoost (primary detector) correctly.

Usage:
    python scripts/extract_per_flow.py
    python scripts/extract_per_flow.py --config configs/config.yaml
"""
import logging
import os
import pickle
import sys
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/extract_per_flow.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def main(config_path: str = "configs/config.yaml"):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from src.data.loader import (
        FEATURE_COLS, LABEL_COL, NUM_CLASSES, ID_TO_LABEL,
        _encode_label, load_single_csv,
    )
    from src.data.feature_engineer import FeatureScaler

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir      = cfg["paths"]["data_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    model_dir     = cfg["paths"]["models_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    csv_files = [
        os.path.join(data_dir, fn)
        for fn in cfg["data"]["csv_files"]
        if os.path.exists(os.path.join(data_dir, fn))
    ]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    logger.info("Loading %d CSV files for per-flow extraction…", len(csv_files))

    # ── Load all CSVs with per-row labels ─────────────────────────────────────
    frames = []
    for path in csv_files:
        logger.info("  Loading %s", os.path.basename(path))
        df = load_single_csv(path, feature_cols=FEATURE_COLS)
        frames.append(df)
        counts = df["label_id"].value_counts().to_dict()
        logger.info("    -> %d rows, labels: %s", len(df), {ID_TO_LABEL.get(k, str(k)): v for k, v in counts.items()})

    all_df = pd.concat(frames, ignore_index=True)
    logger.info("Total: %d rows", len(all_df))

    # ── Extract features and labels ───────────────────────────────────────────
    available_cols = [c for c in FEATURE_COLS if c in all_df.columns]
    X_raw = all_df[available_cols].values.astype(np.float32)

    # Pad any missing feature columns with zeros
    if len(available_cols) < len(FEATURE_COLS):
        logger.warning("Missing %d feature columns — padding with zeros",
                       len(FEATURE_COLS) - len(available_cols))
        X_full = np.zeros((len(X_raw), len(FEATURE_COLS)), dtype=np.float32)
        for i, col in enumerate(FEATURE_COLS):
            if col in available_cols:
                j = available_cols.index(col)
                X_full[:, i] = X_raw[:, j]
        X_raw = X_full

    y = all_df["label_id"].values.astype(np.int64)

    # Log distribution
    counts = np.bincount(y, minlength=NUM_CLASSES)
    logger.info("Label distribution (%d/%d classes present):", np.sum(counts > 0), NUM_CLASSES)
    for i, c in enumerate(counts):
        if c > 0:
            logger.info("  [%2d] %-25s  %10d  (%.2f%%)", i, ID_TO_LABEL.get(i, str(i)), c, 100*c/len(y))

    # ── Stratified split (same seed as main preprocess) ───────────────────────
    seed = cfg.get("seed", 42)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X_raw, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed
    )
    logger.info("Split: train=%d  val=%d  test=%d", len(X_train), len(X_val), len(X_test))

    # ── Scale features ────────────────────────────────────────────────────────
    # XGBoost is tree-based (no scaling needed), but we fit a scaler for the API
    # endpoint which may receive raw features and needs to normalise them.
    # Save as a separate file to avoid clobbering the session-based scaler (81 feats).
    flow_scaler_path = os.path.join(model_dir, "feature_scaler_flow.pkl")
    scaler = FeatureScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    scaler.save(flow_scaler_path)
    logger.info("Per-flow scaler (76 features) saved to %s", flow_scaler_path)
    logger.info("Note: XGBoost does not require scaling — scaler saved for API use only")

    # ── Save per-flow arrays ──────────────────────────────────────────────────
    out_path = os.path.join(processed_dir, "per_flow_data.pkl")
    payload = {
        # Raw features (no scaling) — preferred for tree-based models
        "train_features": X_train,
        "train_labels":   y_train,
        "val_features":   X_val,
        "val_labels":     y_val,
        "test_features":  X_test,
        "test_labels":    y_test,
        # Scaled features — for neural/distance-based models
        "train_features_scaled": X_train_s,
        "val_features_scaled":   X_val_s,
        "test_features_scaled":  X_test_s,
        "feature_cols":   FEATURE_COLS,
        "num_features":   len(FEATURE_COLS),
        "num_classes":    NUM_CLASSES,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=4)

    logger.info("Per-flow data saved to %s", out_path)
    logger.info("  train: %s  val: %s  test: %s",
                X_train_s.shape, X_val_s.shape, X_test_s.shape)
    logger.info("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
