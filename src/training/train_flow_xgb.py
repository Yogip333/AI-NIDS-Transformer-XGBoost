"""
Per-flow XGBoost training — primary detection engine.

Trains directly on individual flow records (76 features → 15 classes).
This is the highest-accuracy component, targeting 95-98% on CICIDS2017.
No sessionization, no Transformer — pure tabular ML.
"""
import logging
import os

import numpy as np

from src.models.xgboost_model import ZeekXGBoost, compute_class_weights
from src.data.loader import ID_TO_LABEL, NUM_CLASSES

logger = logging.getLogger(__name__)


def train_flow_xgb(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    cfg: dict,
    save_path: str,
) -> ZeekXGBoost:
    """
    Train XGBoost on per-flow features.

    Parameters
    ----------
    train_features : (N_train, 76)  scaled flow features
    train_labels   : (N_train,)     integer class labels
    val_features   : (N_val, 76)
    val_labels     : (N_val,)
    cfg            : project config dict (reads cfg['xgboost_flow'])
    save_path      : where to save the trained model

    Returns
    -------
    Fitted ZeekXGBoost model
    """
    logger.info("=" * 60)
    logger.info("Training per-flow XGBoost (primary detector)")
    logger.info("  Train: %d flows  Val: %d flows  Features: %d",
                len(train_features), len(val_features), train_features.shape[1])

    # Log label distribution
    counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    present = np.sum(counts > 0)
    logger.info("  %d/%d attack classes present in training data", present - 1, NUM_CLASSES - 1)
    for i, c in enumerate(counts):
        if c > 0:
            logger.info("    [%2d] %-25s  %8d", i, ID_TO_LABEL.get(i, str(i)), c)

    # Build config subset for per-flow XGBoost
    flow_cfg = {"xgboost": cfg.get("xgboost_flow", cfg.get("xgboost", {}))}

    # Log-smoothed class weights to handle extreme imbalance
    sample_weights = compute_class_weights(train_labels, NUM_CLASSES)
    logger.info("  Class weights computed (log-smoothed inverse frequency)")

    model = ZeekXGBoost(cfg=flow_cfg, num_classes=NUM_CLASSES)

    # Use feature names from loader for SHAP interpretability
    try:
        from src.data.loader import FEATURE_COLS
        feature_names = list(FEATURE_COLS)
    except Exception:
        feature_names = [f"feat_{i}" for i in range(train_features.shape[1])]

    early_stopping = flow_cfg["xgboost"].get("early_stopping_rounds", 30)

    model.fit(
        train_features, train_labels,
        X_val=val_features,
        y_val=val_labels,
        feature_names=feature_names,
        sample_weight=sample_weights,
        early_stopping_rounds=early_stopping,
    )

    # Save model
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    model.save(save_path)
    logger.info("Per-flow XGBoost saved to %s", save_path)

    return model


def evaluate_flow_xgb(
    model: ZeekXGBoost,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    """
    Full evaluation of per-flow XGBoost on the test set.

    Returns comprehensive metrics dict.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        classification_report, confusion_matrix,
    )
    from sklearn.preprocessing import label_binarize

    logger.info("Evaluating per-flow XGBoost on %d test flows…", len(test_features))

    y_pred  = model.predict(test_features)
    y_proba = model.predict_proba(test_features)
    y_true  = test_labels

    acc  = float(accuracy_score(y_true, y_pred))
    f1w  = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1m  = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec  = float(recall_score(y_true, y_pred, average="weighted",  zero_division=0))

    label_names = [ID_TO_LABEL.get(i, str(i)) for i in range(NUM_CLASSES)]
    report_str  = classification_report(
        y_true, y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=label_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    # Binary metrics (attack vs benign)
    y_binary_true = (y_true != 0).astype(int)
    y_binary_pred = (y_pred != 0).astype(int)
    binary_acc    = float(accuracy_score(y_binary_true, y_binary_pred))
    binary_f1     = float(f1_score(y_binary_true, y_binary_pred, zero_division=0))

    # PR-AUC binary
    try:
        from sklearn.metrics import average_precision_score
        attack_proba = 1.0 - y_proba[:, 0]
        pr_auc = float(average_precision_score(y_binary_true, attack_proba))
    except Exception:
        pr_auc = float("nan")

    # Per-class detection rates
    per_class = {}
    for cls_id in range(NUM_CLASSES):
        mask = y_true == cls_id
        if mask.sum() == 0:
            continue
        dr = float(np.mean(y_pred[mask] == cls_id))
        fp = int(np.sum((y_pred == cls_id) & (y_true != cls_id)))
        per_class[ID_TO_LABEL.get(cls_id, str(cls_id))] = {
            "count": int(mask.sum()),
            "detection_rate": round(dr, 4),
            "false_positives": fp,
        }

    result = {
        "accuracy":       round(acc,        4),
        "f1_weighted":    round(f1w,        4),
        "f1_macro":       round(f1m,        4),
        "precision":      round(prec,       4),
        "recall":         round(rec,        4),
        "binary_accuracy":round(binary_acc, 4),
        "binary_f1":      round(binary_f1,  4),
        "binary_pr_auc":  round(pr_auc,     4) if not np.isnan(pr_auc) else None,
        "report":         report_str,
        "confusion_matrix": cm.tolist(),
        "per_class":      per_class,
    }

    logger.info("Per-flow XGBoost results:")
    logger.info("  Accuracy   : %.4f", acc)
    logger.info("  F1 weighted: %.4f", f1w)
    logger.info("  F1 macro   : %.4f", f1m)
    logger.info("  Binary F1  : %.4f", binary_f1)
    logger.info("  PR-AUC     : %.4f", pr_auc if not np.isnan(pr_auc) else -1)
    logger.info("\n%s", report_str)

    return result
