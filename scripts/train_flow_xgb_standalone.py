"""
Standalone Per-Flow XGBoost Training.

Extracts per-flow arrays from preprocessed sessions and trains the primary
detection model (xgb_flow_primary.joblib).

Usage:
    python scripts/train_flow_xgb_standalone.py
    python scripts/train_flow_xgb_standalone.py --config configs/config.yaml
"""
import logging
import os
import pickle
import sys
import json
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/train_flow_xgb.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def extract_per_flow(sessions: list, source: str = "") -> tuple[np.ndarray, np.ndarray]:
    """
    Flatten sessions into per-flow (features, labels) arrays.
    Uses session-level label as proxy for each flow in that session.
    This is valid because sessions are grouped by port/protocol similarity.
    """
    all_features = []
    all_labels   = []
    for s in sessions:
        feats = s["features"]  # (session_size, num_features)
        label = s["label"]     # int — majority label for this session
        all_features.append(feats)
        all_labels.extend([label] * len(feats))

    X = np.vstack(all_features).astype(np.float32)
    y = np.array(all_labels, dtype=np.int64)
    logger.info("  %s: %d flows, %d features", source, len(X), X.shape[1])
    return X, y


def main(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_dir = cfg["paths"]["processed_dir"]
    model_dir     = cfg["paths"]["models_dir"]
    os.makedirs(model_dir, exist_ok=True)

    # ── Load preprocessed sessions ────────────────────────────────────────────
    pkl_path = os.path.join(processed_dir, "processed_data.pkl")
    # Prefer the dedicated per-flow file (correct per-row labels) if available
    per_flow_path = os.path.join(processed_dir, "per_flow_data.pkl")
    if os.path.exists(per_flow_path):
        logger.info("Loading per-flow data from %s (correct per-row labels)", per_flow_path)
        with open(per_flow_path, "rb") as f:
            pf = pickle.load(f)
        X_train, y_train = pf["train_features"], pf["train_labels"]
        X_val,   y_val   = pf["val_features"],   pf["val_labels"]
        X_test,  y_test  = pf["test_features"],  pf["test_labels"]
        logger.info("Per-flow: train=%d  val=%d  test=%d",
                    len(X_train), len(X_val), len(X_test))
    else:
        logger.info("per_flow_data.pkl not found — falling back to session extraction")
        logger.info("Run scripts/extract_per_flow.py for correct per-row labels")
        logger.info("Loading preprocessed sessions from %s", pkl_path)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        train_sessions = data["train_sessions"]
        val_sessions   = data["val_sessions"]
        test_sessions  = data["test_sessions"]
        logger.info("Sessions: train=%d  val=%d  test=%d",
                    len(train_sessions), len(val_sessions), len(test_sessions))
        logger.info("Extracting per-flow arrays from sessions (session-level labels)…")
        X_train, y_train = extract_per_flow(train_sessions, "train")
        X_val,   y_val   = extract_per_flow(val_sessions,   "val")
        X_test,  y_test  = extract_per_flow(test_sessions,  "test")

    # ── Label distribution report ─────────────────────────────────────────────
    from src.data.loader import ID_TO_LABEL, NUM_CLASSES
    counts = np.bincount(y_train, minlength=NUM_CLASSES)
    logger.info("Training label distribution (%d/%d classes present):",
                np.sum(counts > 0), NUM_CLASSES)
    for i, c in enumerate(counts):
        if c > 0:
            logger.info("  [%2d] %-25s  %8d", i, ID_TO_LABEL.get(i, str(i)), c)

    # ── Train per-flow XGBoost ────────────────────────────────────────────────
    from src.training.train_flow_xgb import train_flow_xgb, evaluate_flow_xgb

    save_path = os.path.join(model_dir, "xgb_flow_primary.joblib")
    model = train_flow_xgb(
        train_features=X_train,
        train_labels=y_train,
        val_features=X_val,
        val_labels=y_val,
        cfg=cfg,
        save_path=save_path,
    )

    # ── Evaluate on test set ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Evaluating on held-out test set (%d flows)…", len(X_test))
    results = evaluate_flow_xgb(model, X_test, y_test)

    # Save evaluation results
    results_path = os.path.join(model_dir, "flow_xgb_eval.json")
    # Remove non-serialisable keys before saving
    save_results = {k: v for k, v in results.items()
                    if k not in ("confusion_matrix", "report")}
    save_results["confusion_matrix"] = results["confusion_matrix"]
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2)
    logger.info("Evaluation results saved to %s", results_path)

    logger.info("=" * 60)
    logger.info("Per-flow XGBoost training COMPLETE")
    logger.info("  Model saved   : %s", save_path)
    logger.info("  Accuracy      : %.4f", results["accuracy"])
    logger.info("  F1 weighted   : %.4f", results["f1_weighted"])
    logger.info("  Binary F1     : %.4f", results["binary_f1"])
    if results["binary_pr_auc"] is not None:
        logger.info("  Binary PR-AUC : %.4f", results["binary_pr_auc"])
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
