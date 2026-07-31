"""
Threat Modelling Evaluation for Mini AI-SOC.

Evaluates:
  1. Kill-chain stage coverage — detection rate per ATT&CK stage
  2. Detection latency — accuracy vs number of observed flows (truncation experiment)
  3. False positive profiling — statistical characterization of FP sessions
  4. Confidence calibration — predicted confidence vs actual correctness

Requires: processed_data.pkl, trained models, config.yaml

Output: models/checkpoints/threat_model_eval.json
"""
import json
import logging
import os
import pickle
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

from src.evaluation.config_names import CONFIG_C
from src.data.loader import ID_TO_LABEL


# ── Kill-chain stage mapping for CICIDS2017 attacks ──────────────────────────

ATTACK_STAGE_MAP = {
    0:  {"name": "BENIGN",                "stage": "N/A"},
    1:  {"name": "Bot",                   "stage": "Command and Control"},
    2:  {"name": "DDoS",                  "stage": "Impact"},
    3:  {"name": "DoS GoldenEye",         "stage": "Impact"},
    4:  {"name": "DoS Hulk",              "stage": "Impact"},
    5:  {"name": "DoS Slowhttptest",      "stage": "Impact"},
    6:  {"name": "DoS slowloris",         "stage": "Impact"},
    7:  {"name": "FTP-Patator",           "stage": "Credential Access"},
    8:  {"name": "Heartbleed",            "stage": "Exploitation"},
    9:  {"name": "Infiltration",          "stage": "Lateral Movement"},
    10: {"name": "PortScan",              "stage": "Reconnaissance"},
    11: {"name": "SSH-Patator",           "stage": "Credential Access"},
    12: {"name": "Web Attack Brute Force","stage": "Credential Access"},
    13: {"name": "Web Attack Sql Injection","stage": "Exploitation"},
    14: {"name": "Web Attack XSS",        "stage": "Exploitation"},
}


def evaluate_stage_coverage(y_true, y_pred):
    """Detection rate per kill-chain stage."""
    stage_stats = {}

    for attack_id in range(1, 15):
        info = ATTACK_STAGE_MAP.get(attack_id, {})
        stage = info.get("stage", "Unknown")
        name = info.get("name", str(attack_id))

        mask = y_true == attack_id
        if not mask.any():
            continue

        correct = np.sum((y_true == attack_id) & (y_pred == attack_id))
        total = np.sum(mask)
        dr = float(correct / total)

        if stage not in stage_stats:
            stage_stats[stage] = {
                "attacks": [],
                "total_samples": 0,
                "total_correct": 0,
            }
        stage_stats[stage]["attacks"].append({
            "name": name,
            "attack_id": attack_id,
            "n_samples": int(total),
            "n_correct": int(correct),
            "detection_rate": round(dr, 4),
        })
        stage_stats[stage]["total_samples"] += int(total)
        stage_stats[stage]["total_correct"] += int(correct)

    # Compute per-stage aggregate
    for stage, data in stage_stats.items():
        data["stage_detection_rate"] = round(
            data["total_correct"] / max(data["total_samples"], 1), 4
        )

    return stage_stats


def evaluate_detection_latency(cfg, classifier, hybrid, test_sessions, feat_cols, device):
    """
    Truncate sessions to different flow counts and measure accuracy per truncation.
    Uses Config C (primary serving detector).
    """
    import torch
    from torch.utils.data import DataLoader
    from src.data.sessionizer import SessionDataset, collate_sessions
    from src.data.feature_engineer import compute_all_session_stats
    from src.evaluation.metrics import compute_metrics

    max_seq_len = cfg["transformer"]["max_seq_len"]
    truncation_points = [3, 5, 8, 10, 15, 20]
    results = {}

    if hybrid.xgb_c is None:
        logger.warning("Config C not loaded — skipping latency analysis")
        return {}

    y_true = np.array([s["label"] for s in test_sessions])

    for n_flows in truncation_points:
        truncated = []
        for s in test_sessions:
            feat = s["features"][:n_flows]
            lt = s["log_types"][:n_flows]
            truncated.append({
                "features": feat, "log_types": lt, "label": s["label"],
                "has_attack": s["has_attack"], "attack_ids": s["attack_ids"],
                "session_idx": s["session_idx"],
            })

        trunc_stats = compute_all_session_stats(truncated, feat_cols)
        trunc_ds = SessionDataset(truncated, max_seq_len, return_labels=True)
        trunc_loader = DataLoader(trunc_ds, batch_size=64, shuffle=False,
                                  collate_fn=collate_sessions, num_workers=0)

        try:
            y_pred, y_proba = hybrid.predict(trunc_loader, trunc_stats, config="c")
            metrics = compute_metrics(y_true, y_pred, y_proba)
            results[f"{n_flows}_flows"] = {
                "n_flows": n_flows,
                "accuracy": round(metrics["accuracy"], 4),
                "f1_weighted": round(metrics["f1_weighted"], 4),
                "f1_macro": round(metrics["f1_macro"], 4),
            }
            logger.info("  %2d flows: acc=%.4f  f1_w=%.4f  f1_m=%.4f",
                        n_flows, metrics["accuracy"], metrics["f1_weighted"], metrics["f1_macro"])
        except Exception as e:
            logger.warning("  %d flows: FAILED — %s", n_flows, e)

    return results


def evaluate_false_positive_profile(y_true, y_pred, test_stats, stat_names):
    """Profile benign sessions that were wrongly classified as attacks."""
    from src.data.loader import ID_TO_LABEL

    # FP = benign (y_true==0) predicted as attack
    fp_mask = (y_true == 0) & (y_pred != 0)
    tp_mask = (y_true == 0) & (y_pred == 0)

    n_fp = int(fp_mask.sum())
    n_tp = int(tp_mask.sum())

    result = {
        "total_benign": int((y_true == 0).sum()),
        "true_negatives": n_tp,
        "false_positives": n_fp,
        "fp_rate": round(n_fp / max(int((y_true == 0).sum()), 1), 4),
    }

    if n_fp > 0 and test_stats is not None:
        fp_stats = test_stats[fp_mask]
        tn_stats = test_stats[tp_mask] if n_tp > 0 else np.zeros((1, test_stats.shape[1]))

        feature_comparison = {}
        for i, name in enumerate(stat_names):
            feature_comparison[name] = {
                "fp_mean": round(float(fp_stats[:, i].mean()), 4),
                "fp_std":  round(float(fp_stats[:, i].std()), 4),
                "tn_mean": round(float(tn_stats[:, i].mean()), 4) if n_tp > 0 else 0.0,
                "tn_std":  round(float(tn_stats[:, i].std()), 4) if n_tp > 0 else 0.0,
            }
        result["feature_comparison"] = feature_comparison

        # What attacks are FPs classified as?
        fp_pred_labels = y_pred[fp_mask]
        fp_distribution = {}
        for pred_id in np.unique(fp_pred_labels):
            name = ID_TO_LABEL.get(int(pred_id), str(pred_id))
            fp_distribution[name] = int((fp_pred_labels == pred_id).sum())
        result["fp_predicted_as"] = fp_distribution

    return result


def evaluate_confidence_calibration(y_true, y_pred, y_proba, n_bins=10):
    """Reliability diagram: binned confidence vs actual correctness."""
    confidences = y_proba[np.arange(len(y_pred)), y_pred]
    correct = (y_true == y_pred).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    calibration = []

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        calibration.append({
            "bin_lower": round(float(bins[i]), 2),
            "bin_upper": round(float(bins[i + 1]), 2),
            "mean_confidence": round(float(confidences[mask].mean()), 4),
            "mean_accuracy": round(float(correct[mask].mean()), 4),
            "n_samples": int(mask.sum()),
        })

    # Expected Calibration Error
    ece = 0.0
    for b in calibration:
        ece += (b["n_samples"] / len(y_pred)) * abs(b["mean_confidence"] - b["mean_accuracy"])

    return {
        "bins": calibration,
        "ece": round(ece, 4),
        "n_total": len(y_pred),
    }


def main(config_path: str = "configs/config.yaml"):
    import torch
    from torch.utils.data import DataLoader
    from src.data.sessionizer import SessionDataset, collate_sessions
    from src.data.feature_engineer import compute_all_session_stats, STAT_FEATURE_NAMES
    from src.models.transformer import build_classifier
    from src.models.hybrid import HybridIDS

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    processed_dir = cfg["paths"]["processed_dir"]
    model_dir     = cfg["paths"]["models_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load data ─────────────────────────────────────────────────────────────
    with open(os.path.join(processed_dir, "processed_data.pkl"), "rb") as f:
        data = pickle.load(f)

    test_sessions = data["test_sessions"]
    test_stats    = data["test_stats"]
    feat_cols     = data["feature_cols"]
    num_features  = data["num_features"]
    max_seq_len   = cfg["transformer"]["max_seq_len"]

    test_ds = SessionDataset(test_sessions, max_seq_len, return_labels=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             collate_fn=collate_sessions, num_workers=0)

    # ── Load models ───────────────────────────────────────────────────────────
    classifier = build_classifier(cfg, input_dim=num_features, num_classes=15)
    ckpt = os.path.join(model_dir, "classifier_final.pt")
    if os.path.exists(ckpt):
        classifier.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    classifier = classifier.to(device).eval()

    hybrid = HybridIDS(cfg, classifier, num_classes=15)
    hybrid.load_all(model_dir)

    # ── Get Config C predictions ──────────────────────────────────────────────
    y_true = np.array([s["label"] for s in test_sessions])

    if hybrid.xgb_c is not None:
        logger.info("Running Config C predictions for threat model eval…")
        y_pred, y_proba = hybrid.predict(test_loader, test_stats, config="c")
    else:
        logger.error("Config C not available — cannot run threat model eval")
        return

    output = {}

    # ══════════════════════════════════════════════════════════════════════════
    # 1. Kill-chain stage coverage
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Stage Coverage Analysis")
    logger.info("=" * 60)

    stage_results = evaluate_stage_coverage(y_true, y_pred)
    for stage, data in stage_results.items():
        logger.info("  %-25s  DR=%.4f  (%d/%d)",
                    stage, data["stage_detection_rate"],
                    data["total_correct"], data["total_samples"])
    output["stage_coverage"] = stage_results

    # ══════════════════════════════════════════════════════════════════════════
    # 2. Detection latency
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Detection Latency Analysis")
    logger.info("=" * 60)

    latency_results = evaluate_detection_latency(
        cfg, classifier, hybrid, test_sessions, feat_cols, device)
    output["detection_latency"] = latency_results

    # ══════════════════════════════════════════════════════════════════════════
    # 3. False positive profiling
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("False Positive Profiling")
    logger.info("=" * 60)

    fp_results = evaluate_false_positive_profile(
        y_true, y_pred, test_stats, list(STAT_FEATURE_NAMES))
    logger.info("  Benign sessions: %d  TN: %d  FP: %d  FP-rate: %.4f",
                fp_results["total_benign"], fp_results["true_negatives"],
                fp_results["false_positives"], fp_results["fp_rate"])
    if "fp_predicted_as" in fp_results:
        for name, count in fp_results["fp_predicted_as"].items():
            logger.info("    FP classified as %-25s: %d", name, count)
    output["false_positive_profile"] = fp_results

    # ══════════════════════════════════════════════════════════════════════════
    # 4. Confidence calibration
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Confidence Calibration")
    logger.info("=" * 60)

    cal_results = evaluate_confidence_calibration(y_true, y_pred, y_proba)
    logger.info("  ECE (Expected Calibration Error): %.4f", cal_results["ece"])
    for b in cal_results["bins"]:
        logger.info("    [%.2f-%.2f] conf=%.4f acc=%.4f n=%d",
                    b["bin_lower"], b["bin_upper"],
                    b["mean_confidence"], b["mean_accuracy"], b["n_samples"])
    output["confidence_calibration"] = cal_results

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(model_dir, "threat_model_eval.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x)
    logger.info("Saved -> %s", out_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Threat Model Evaluation")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
