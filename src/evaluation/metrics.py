"""
Evaluation metrics for the Mini AI SOC intrusion detection system.

Computes:
  • Per-class precision, recall, F1 (macro/weighted)
  • Accuracy and AUC-ROC
  • Confusion matrix
  • Comparative table for all three model configurations
"""
import json
import logging
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.loader import ID_TO_LABEL, NUM_CLASSES

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    label_names: Optional[list[str]] = None,
) -> dict:
    """
    Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true    : ground-truth labels (int)
    y_pred    : predicted labels (int)
    y_proba   : predicted probabilities (N, num_classes) for AUC
    label_names : list of class name strings

    Returns
    -------
    dict with accuracy, f1, precision, recall, auc, confusion_matrix, report
    """
    if label_names is None:
        present = sorted(set(y_true) | set(y_pred))
        label_names = [ID_TO_LABEL.get(i, str(i)) for i in present]

    acc  = float(accuracy_score(y_true, y_pred))
    f1_w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec  = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    auc = None
    if y_proba is not None:
        try:
            classes_present = np.unique(y_true)
            if len(classes_present) > 1:
                # Select only columns for classes in the test set and
                # re-normalise so rows sum to 1.0 (required by sklearn).
                proba_subset = y_proba[:, classes_present].copy()
                row_sums = proba_subset.sum(axis=1, keepdims=True)
                proba_subset /= np.maximum(row_sums, 1e-10)
                auc = float(roc_auc_score(
                    y_true, proba_subset,
                    multi_class="ovr",
                    labels=classes_present,
                    average="weighted",
                ))
        except Exception as e:
            logger.warning("AUC computation failed: %s", e)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist()

    # Full per-class report
    all_label_names = [ID_TO_LABEL.get(i, str(i)) for i in range(NUM_CLASSES)]
    report = classification_report(
        y_true, y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=all_label_names,
        zero_division=0,
        output_dict=True,
    )

    return {
        "accuracy":            acc,
        "f1_weighted":         f1_w,
        "f1_macro":            f1_m,
        "precision_weighted":  prec,
        "recall_weighted":     rec,
        "auc_weighted":        auc,
        "confusion_matrix":    cm,
        "classification_report": report,
        "n_samples":           int(len(y_true)),
        "n_attacks":           int(np.sum(y_true != 0)),
        "n_benign":            int(np.sum(y_true == 0)),
    }


def compare_configs(
    results: dict[str, dict],
    save_path: Optional[str] = None,
) -> str:
    """
    Build a comparison table for three model configurations.

    Parameters
    ----------
    results : {config_name: metrics_dict}
    save_path : if given, save JSON and formatted table

    Returns
    -------
    str : formatted comparison table
    """
    header = f"{'Config':<30} {'Accuracy':>10} {'F1(W)':>8} {'F1(M)':>8} {'Precision':>10} {'Recall':>8} {'AUC':>8}"
    sep = "-" * len(header)
    rows = [header, sep]

    for name, m in results.items():
        auc_str = f"{m.get('auc_weighted', 0) or 0:.4f}"
        row = (
            f"{name:<30} "
            f"{m['accuracy']:>10.4f} "
            f"{m['f1_weighted']:>8.4f} "
            f"{m['f1_macro']:>8.4f} "
            f"{m['precision_weighted']:>10.4f} "
            f"{m['recall_weighted']:>8.4f} "
            f"{auc_str:>8}"
        )
        rows.append(row)

    table = "\n".join(rows)
    logger.info("\n%s", table)

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", save_path)

    return table


def detection_rate_by_attack(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, dict]:
    """
    Per-attack-type detection statistics.

    Returns a dict keyed by attack name with:
      true_count, detected_count, detection_rate, false_positives
    """
    result = {}
    for attack_id in range(1, NUM_CLASSES):  # skip BENIGN
        name = ID_TO_LABEL.get(attack_id, str(attack_id))
        true_mask = y_true == attack_id
        pred_mask = y_pred == attack_id
        true_count = int(true_mask.sum())
        if true_count == 0:
            continue
        correct = int((true_mask & pred_mask).sum())
        fp = int((~true_mask & pred_mask).sum())
        result[name] = {
            "true_count":      true_count,
            "detected_count":  correct,
            "detection_rate":  round(correct / true_count, 4),
            "false_positives": fp,
            "fp_rate":         round(fp / max(int((~true_mask).sum()), 1), 6),
        }
    return result


def threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[list[float]] = None,
) -> dict:
    """
    Analyse binary detection quality (attack vs benign) across confidence thresholds.

    Returns dict with per-threshold TPR, FPR, precision, recall, F1 and PR-AUC.
    Useful for SOC operators choosing an operating point.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Binary: attack = 1, benign = 0
    y_binary   = (y_true != 0).astype(int)
    attack_proba = 1.0 - y_proba[:, 0]   # P(not benign)

    pr_auc = float(average_precision_score(y_binary, attack_proba))

    rows = []
    for thr in thresholds:
        y_binary_pred = (attack_proba >= thr).astype(int)
        tp = int(np.sum((y_binary == 1) & (y_binary_pred == 1)))
        fp = int(np.sum((y_binary == 0) & (y_binary_pred == 1)))
        fn = int(np.sum((y_binary == 1) & (y_binary_pred == 0)))
        tn = int(np.sum((y_binary == 0) & (y_binary_pred == 0)))
        tpr  = tp / max(tp + fn, 1)
        fpr  = fp / max(fp + tn, 1)
        prec = tp / max(tp + fp, 1)
        f1   = 2 * prec * tpr / max(prec + tpr, 1e-9)
        rows.append({
            "threshold": thr,
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "precision": round(prec, 4),
            "recall": round(tpr, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    logger.info("Threshold analysis (binary: attack vs benign)  PR-AUC=%.4f", pr_auc)
    header = f"{'Thr':>6} {'TPR':>7} {'FPR':>7} {'Prec':>7} {'F1':>7}"
    logger.info(header)
    for r in rows:
        logger.info("%6.2f %7.4f %7.4f %7.4f %7.4f",
                    r["threshold"], r["tpr"], r["fpr"], r["precision"], r["f1"])

    return {"pr_auc": pr_auc, "thresholds": rows}


def session_window_ablation(
    results_per_window: dict[str, dict],
) -> str:
    """
    Format ablation study results comparing session windows of 1/5/10 minutes
    (proxied by different session_size values).

    Parameters
    ----------
    results_per_window : {window_label: metrics_dict}
      e.g. {"10-flow": {...}, "20-flow": {...}, "50-flow": {...}}
    """
    header = f"{'Window':>12} {'F1(W)':>8} {'F1(M)':>8} {'Accuracy':>10} {'AUC':>8}"
    sep = "-" * 50
    rows = [header, sep]
    for label, m in results_per_window.items():
        auc_str = f"{m.get('auc_weighted', 0) or 0:.4f}"
        rows.append(
            f"{label:>12} "
            f"{m['f1_weighted']:>8.4f} "
            f"{m['f1_macro']:>8.4f} "
            f"{m['accuracy']:>10.4f} "
            f"{auc_str:>8}"
        )
    table = "\n".join(rows)
    logger.info("Session window ablation:\n%s", table)
    return table
