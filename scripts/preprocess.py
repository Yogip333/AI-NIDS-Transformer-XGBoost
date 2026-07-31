"""
Preprocessing Script.

Steps:
  1. Load all CICIDS2017 CSV files
  2. Map flows to Zeek log types
  3. Build session windows
  4. Compute statistical features
  5. Fit and save feature scaler
  6. Save processed sessions to disk
"""
import logging
import os
import sys
import pickle
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/preprocess.log"),
    ],
)
logger = logging.getLogger(__name__)


def main(config_path: str = "configs/config.yaml", dev_mode: bool = False):
    import numpy as np
    from src.data.loader import load_all_data, split_data, get_feature_cols_present
    from src.data.zeek_mapper import map_to_zeek_records, get_zeek_extra_cols
    from src.data.sessionizer import Sessionizer
    from src.data.feature_engineer import compute_all_session_stats, scale_flow_features, FeatureScaler, STAT_FEATURE_NAMES

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir   = cfg["paths"]["data_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    model_dir  = cfg["paths"]["models_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    max_rows = 20_000 if dev_mode else cfg["data"].get("max_rows_per_file")
    logger.info("Loading data (dev_mode=%s, max_rows_per_file=%s)…", dev_mode, max_rows)
    df = load_all_data(data_dir, max_rows_per_file=max_rows)
    logger.info("Total flows loaded: %d", len(df))

    # ── 2. Map to Zeek log types ──────────────────────────────────────────────
    logger.info("Mapping to Zeek log types…")
    df = map_to_zeek_records(df)
    from src.data.zeek_mapper import describe_log_distribution
    logger.info("Log distribution: %s", describe_log_distribution(df))

    # ── 3. Split data ─────────────────────────────────────────────────────────
    logger.info("Splitting data…")
    train_df, val_df, test_df = split_data(
        df,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
    )

    # ── 4. Get feature columns ────────────────────────────────────────────────
    from src.data.loader import FEATURE_COLS
    zeek_extra = get_zeek_extra_cols()
    all_feat_cols = [c for c in (FEATURE_COLS + zeek_extra) if c in df.columns]
    logger.info("Feature columns: %d", len(all_feat_cols))

    # ── 5. Build sessions ─────────────────────────────────────────────────────
    session_size = cfg["data"]["session_size"]
    logger.info("Sessionizing (size=%d)…", session_size)
    sessionizer = Sessionizer(session_size=session_size, stride=session_size)

    train_sessions = sessionizer.build_sessions(train_df, all_feat_cols)
    val_sessions   = sessionizer.build_sessions(val_df,   all_feat_cols)
    test_sessions  = sessionizer.build_sessions(test_df,  all_feat_cols)

    benign_sessions = sessionizer.build_benign_sessions(train_df, all_feat_cols)
    logger.info("Sessions: train=%d val=%d test=%d benign=%d",
                len(train_sessions), len(val_sessions), len(test_sessions), len(benign_sessions))

    # ── 6. Compute statistical features (on RAW data, before scaling) ────────
    logger.info("Computing statistical features (on raw data)…")
    train_stats = compute_all_session_stats(train_sessions, all_feat_cols)
    val_stats   = compute_all_session_stats(val_sessions,   all_feat_cols)
    test_stats  = compute_all_session_stats(test_sessions,  all_feat_cols)

    # ── 7. Scale features (for Transformer input) ───────────────────────────
    logger.info("Fitting feature scaler on training sessions…")
    train_sessions, scaler = scale_flow_features(train_sessions, all_feat_cols, fit=True)
    val_sessions,   _      = scale_flow_features(val_sessions,   all_feat_cols, scaler=scaler, fit=False)
    test_sessions,  _      = scale_flow_features(test_sessions,  all_feat_cols, scaler=scaler, fit=False)
    benign_sessions, _     = scale_flow_features(benign_sessions, all_feat_cols, scaler=scaler, fit=False)

    scaler.save(os.path.join(model_dir, "feature_scaler.pkl"))
    logger.info("Feature scaler saved")
    logger.info("Stat features shape: train=%s val=%s test=%s",
                train_stats.shape, val_stats.shape, test_stats.shape)

    # ── 8. Save processed data ────────────────────────────────────────────────
    logger.info("Saving processed data…")
    data = {
        "train_sessions": train_sessions,
        "val_sessions":   val_sessions,
        "test_sessions":  test_sessions,
        "benign_sessions": benign_sessions,
        "train_stats":    train_stats,
        "val_stats":      val_stats,
        "test_stats":     test_stats,
        "feature_cols":   all_feat_cols,
        "stat_feature_names": STAT_FEATURE_NAMES,
        "num_features":   len(all_feat_cols),
        "session_size":   session_size,
    }

    with open(os.path.join(processed_dir, "processed_data.pkl"), "wb") as f:
        pickle.dump(data, f, protocol=4)

    # Also save metadata separately
    meta = {k: v for k, v in data.items()
            if not isinstance(v, (list, np.ndarray)) or k in ("feature_cols", "stat_feature_names")}
    meta["num_train_sessions"] = len(train_sessions)
    meta["num_val_sessions"]   = len(val_sessions)
    meta["num_test_sessions"]  = len(test_sessions)
    meta["num_benign_sessions"]= len(benign_sessions)

    import json
    with open(os.path.join(processed_dir, "metadata.json"), "w") as f:
        json.dump({k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in meta.items()
                   if isinstance(v, (int, float, str, list))}, f, indent=2)

    logger.info("Preprocessing complete!")
    logger.info("  Train sessions: %d", len(train_sessions))
    logger.info("  Val sessions:   %d", len(val_sessions))
    logger.info("  Test sessions:  %d", len(test_sessions))
    logger.info("  Benign sessions:%d", len(benign_sessions))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess CICIDS2017 data")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dev",    action="store_true", help="Use small subset for testing")
    args = parser.parse_args()
    main(config_path=args.config, dev_mode=args.dev)
