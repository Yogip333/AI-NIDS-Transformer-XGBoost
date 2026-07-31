"""
CSV Replay Script — simulates live traffic from CICIDS2017 CSV files.

Usage:
    # Stream all files, report stats only (no model):
    python scripts/stream_csv.py --dry-run

    # Stream with per-flow XGBoost detection:
    python scripts/stream_csv.py --model flow_xgb

    # Stream a specific file with rate limiting:
    python scripts/stream_csv.py --file "Kaggle Dataset/Monday-WorkingHours.pcap_ISCX.csv" --rate 1000

    # Stream and post alerts to the running API:
    python scripts/stream_csv.py --model flow_xgb --api-url http://localhost:8000
"""
import json
import logging
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/streaming.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def main(
    config_path: str = "configs/config.yaml",
    model_type: str = "flow_xgb",
    files: list[str] | None = None,
    rate_limit: float = 0.0,
    batch_size: int = 100,
    dry_run: bool = False,
):
    import numpy as np
    from src.streaming.csv_replay import CSVReplayStreamer, StreamStats
    from src.data.loader import NUM_CLASSES

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir  = cfg["paths"]["data_dir"]
    model_dir = cfg["paths"]["models_dir"]

    # Resolve CSV files to stream
    if files is None:
        files = [
            os.path.join(data_dir, fn)
            for fn in cfg["data"]["csv_files"]
            if os.path.exists(os.path.join(data_dir, fn))
        ]
    if not files:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    logger.info("Files to stream: %d", len(files))
    for f in files:
        logger.info("  %s", f)

    # Build streamer
    # Prefer the per-flow scaler (78 features) for flow_xgb; fall back to session scaler
    flow_scaler_path = os.path.join(model_dir, "feature_scaler_flow.pkl")
    scaler_path = flow_scaler_path if os.path.exists(flow_scaler_path) else os.path.join(model_dir, "feature_scaler.pkl")
    schema_path = os.path.join(model_dir, "feature_schema.json")
    stream_cfg  = cfg.get("streaming", {})

    streamer = CSVReplayStreamer.from_saved_artifacts(
        scaler_path=scaler_path,
        schema_path=schema_path,
        batch_size=batch_size or stream_cfg.get("batch_size", 100),
        rate_limit=rate_limit or stream_cfg.get("rate_limit", 0),
        has_labels=True,
    )

    # Build inference function
    model_fn = None
    if not dry_run:
        model_fn = _build_model_fn(model_type, model_dir, cfg, NUM_CLASSES)

    # Run streaming
    logger.info("=" * 60)
    logger.info("Starting CSV replay  model=%s  rate=%.0f/s  batch=%d",
                model_type if not dry_run else "none (dry-run)",
                rate_limit, streamer.batch_size)
    logger.info("=" * 60)

    stats = streamer.run_with_stats(files, model_fn=model_fn)

    # Final report
    logger.info("=" * 60)
    logger.info("Stream complete.")
    logger.info(stats.summary())
    if stats.total_flows > 0 and stats.true_positives + stats.false_negatives > 0:
        logger.info("Detection rate : %.4f", stats.detection_rate)
        logger.info("FP rate        : %.4f", stats.false_positive_rate)
    logger.info("=" * 60)

    # Save summary
    summary_path = os.path.join(model_dir, "stream_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "total_flows":       stats.total_flows,
            "total_attacks":     stats.total_attacks,
            "total_benign":      stats.total_benign,
            "true_positives":    stats.true_positives,
            "false_positives":   stats.false_positives,
            "false_negatives":   stats.false_negatives,
            "true_negatives":    stats.true_negatives,
            "detection_rate":    round(stats.detection_rate, 4),
            "false_positive_rate": round(stats.false_positive_rate, 4),
            "flows_per_second":  round(stats.flows_per_second, 1),
            "attack_counts":     stats.attack_counts,
        }, f, indent=2)
    logger.info("Summary saved to %s", summary_path)


def _build_model_fn(model_type: str, model_dir: str, cfg: dict, num_classes: int):
    """Return a callable(raw_features, scaled_features) -> (preds, probas)."""
    import numpy as np

    if model_type == "flow_xgb":
        from src.models.xgboost_model import ZeekXGBoost
        path = os.path.join(model_dir, "xgb_flow_primary.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Per-flow XGBoost not found: {path}. Run train_model.py first."
            )
        model = ZeekXGBoost.load(path)
        logger.info("Loaded per-flow XGBoost from %s", path)

        def model_fn(raw_features: np.ndarray, scaled_features: np.ndarray):
            # XGBoost was trained on RAW (unscaled) features — use raw
            preds  = model.predict(raw_features)
            probas = model.predict_proba(raw_features)
            return preds, probas

        return model_fn

    if model_type == "session_hybrid":
        import torch
        from src.models.transformer import build_classifier
        from src.models.hybrid import HybridIDS
        from src.data.sessionizer import SessionDataset, collate_sessions
        from torch.utils.data import DataLoader

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        classifier = build_classifier(cfg, input_dim=len(
            cfg["features"]["numerical_features"]), num_classes=num_classes)
        ckpt = os.path.join(model_dir, "classifier_final.pt")
        if os.path.exists(ckpt):
            state = torch.load(ckpt, map_location=device, weights_only=True)
            classifier.load_state_dict(state)
        classifier = classifier.to(device)

        hybrid = HybridIDS(cfg, classifier, num_classes=num_classes)
        hybrid.load_all(model_dir)
        logger.info("Loaded session hybrid model")

        def model_fn(raw_features: np.ndarray, scaled_features: np.ndarray):
            # Session hybrid uses scaled features (Transformer expects normalized input)
            from src.data.feature_engineer import compute_session_stats, STAT_FEATURE_NAMES
            from src.data.loader import FEATURE_COLS
            stats = compute_session_stats(scaled_features, FEATURE_COLS)
            lt    = np.zeros(len(scaled_features), dtype=np.int8)
            result = hybrid.predict_single_session(scaled_features, lt, stats, config="c")
            pred   = np.array([result["prediction"]])
            proba  = np.array([result["probabilities"]])
            return pred, proba

        return model_fn

    raise ValueError(f"Unknown model_type: {model_type!r}. Use 'flow_xgb' or 'session_hybrid'.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stream CICIDS2017 CSV as live traffic")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--model",      default="flow_xgb",
                        choices=["flow_xgb", "session_hybrid"],
                        help="Which model to use for detection")
    parser.add_argument("--file",       nargs="*", dest="files",
                        help="Specific CSV file(s) to stream (default: all)")
    parser.add_argument("--rate",       type=float, default=0.0,
                        help="Max flows/second (0 = unlimited)")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run",    action="store_true",
                        help="Load + validate only, no model inference")
    args = parser.parse_args()
    main(
        config_path=args.config,
        model_type=args.model,
        files=args.files,
        rate_limit=args.rate,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
