"""
Unified 4-Config Evaluation Script for AI-NIDS.

Evaluates all four ablation configurations on the full CICIDS2017 test split
(21,230 sessions).

Configs:
  A — XGBoost on 26 statistical features only          (baseline)
  B — XGBoost on 256-dim Transformer embeddings
  C — XGBoost on Transformer embeddings + stats         (full hybrid)
  D — Transformer classifier (CLS → MLP head)           (Transformer-only)

Outputs:
  evaluation_results_v4.json   — per-config metrics (confusion matrix, per-class, AUC)
  full_eval_results.json       — comparison summary table
  ablation_results.json        — session-window ablation (10/20/50 flows)
  shap_importance_config_a.json
  shap_importance_config_b.json
  shap_importance_config_c.json
  attention_analysis_config_d.json
  threshold_analysis.json
"""
import logging
import os
import pickle
import sys
import json

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/evaluation.log"),
    ],
)
logger = logging.getLogger(__name__)

from src.evaluation.config_names import (
    CONFIG_A, CONFIG_B, CONFIG_C, CONFIG_D,
    CONFIG_DESCRIPTIONS, SHAP_CONFIGS, ATTENTION_CONFIGS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_xgb_config(hybrid, test_loader, test_stats, test_sessions, internal_id, device):
    """Evaluate an XGBoost-based config (A/B/C) through HybridIDS."""
    from src.evaluation.metrics import compute_metrics, detection_rate_by_attack
    y_pred, y_proba = hybrid.predict(test_loader, test_stats, config=internal_id)
    y_true = np.array([s["label"] for s in test_sessions])
    metrics = compute_metrics(y_true, y_pred, y_proba)
    dr = detection_rate_by_attack(y_true, y_pred)
    return metrics, dr, y_pred, y_true, y_proba


def evaluate_transformer_standalone(classifier, test_loader, test_sessions, device):
    """Evaluate Config D: Transformer classifier independently of XGBoost."""
    import torch
    from src.evaluation.metrics import compute_metrics, detection_rate_by_attack

    classifier.eval()
    all_preds = []
    all_proba = []

    for batch in test_loader:
        features     = batch["features"].to(device)
        log_types    = batch["log_types"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        with torch.no_grad():
            logits = classifier(features, log_types, padding_mask)
            probs  = torch.softmax(logits, dim=-1)

        all_preds.append(probs.argmax(dim=-1).cpu().numpy())
        all_proba.append(probs.cpu().numpy())

    y_pred  = np.concatenate(all_preds)
    y_proba = np.vstack(all_proba)
    y_true  = np.array([s["label"] for s in test_sessions])

    metrics = compute_metrics(y_true, y_pred, y_proba)
    dr = detection_rate_by_attack(y_true, y_pred)
    return metrics, dr, y_pred, y_true, y_proba


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(config_path: str = "configs/config.yaml"):
    import torch
    from torch.utils.data import DataLoader

    from src.data.sessionizer import SessionDataset, collate_sessions
    from src.data.feature_engineer import (
        compute_all_session_stats, STAT_FEATURE_NAMES,
    )
    from src.models.transformer import build_classifier
    from src.models.hybrid import HybridIDS, extract_embeddings
    from src.evaluation.metrics import (
        compare_configs, session_window_ablation,
        threshold_analysis,
    )

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    processed_dir = cfg["paths"]["processed_dir"]
    model_dir     = cfg["paths"]["models_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load preprocessed data ────────────────────────────────────────────────
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

    logger.info("Test set: %d sessions, %d features", len(test_sessions), num_features)

    # ── Load models ───────────────────────────────────────────────────────────
    classifier_path = os.path.join(model_dir, "classifier_final.pt")
    classifier = build_classifier(cfg, input_dim=num_features, num_classes=15)
    if os.path.exists(classifier_path):
        classifier.load_state_dict(
            torch.load(classifier_path, map_location=device, weights_only=True)
        )
    classifier = classifier.to(device).eval()

    hybrid = HybridIDS(cfg, classifier, num_classes=15)
    hybrid.load_all(model_dir)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. Evaluate all 4 configs
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("PHASE 1 — Four-Config Ablation Evaluation")
    logger.info("=" * 70)

    all_results = {}
    all_dr = {}
    all_preds = {}
    all_probas = {}
    y_true = np.array([s["label"] for s in test_sessions])

    # Config A (stat-only XGBoost)
    if hybrid.xgb_a is not None:
        logger.info("Evaluating %s — %s", CONFIG_A, CONFIG_DESCRIPTIONS[CONFIG_A])
        m, dr, yp, yt, ypr = evaluate_xgb_config(
            hybrid, test_loader, test_stats, test_sessions, "a", device)
        all_results[CONFIG_A] = m
        all_dr[CONFIG_A] = dr
        all_preds[CONFIG_A] = yp
        all_probas[CONFIG_A] = ypr
        logger.info("  acc=%.4f  f1_w=%.4f  f1_m=%.4f", m["accuracy"], m["f1_weighted"], m["f1_macro"])
    else:
        logger.warning("Config A model not found — skipping")

    # Config B (embed-only XGBoost)
    if hybrid.xgb_b is not None:
        logger.info("Evaluating %s — %s", CONFIG_B, CONFIG_DESCRIPTIONS[CONFIG_B])
        m, dr, yp, yt, ypr = evaluate_xgb_config(
            hybrid, test_loader, test_stats, test_sessions, "b", device)
        all_results[CONFIG_B] = m
        all_dr[CONFIG_B] = dr
        all_preds[CONFIG_B] = yp
        all_probas[CONFIG_B] = ypr
        logger.info("  acc=%.4f  f1_w=%.4f  f1_m=%.4f", m["accuracy"], m["f1_weighted"], m["f1_macro"])
    else:
        logger.warning("Config B model not found — skipping")

    # Config C (hybrid XGBoost)
    if hybrid.xgb_c is not None:
        logger.info("Evaluating %s — %s", CONFIG_C, CONFIG_DESCRIPTIONS[CONFIG_C])
        m, dr, yp, yt, ypr = evaluate_xgb_config(
            hybrid, test_loader, test_stats, test_sessions, "c", device)
        all_results[CONFIG_C] = m
        all_dr[CONFIG_C] = dr
        all_preds[CONFIG_C] = yp
        all_probas[CONFIG_C] = ypr
        logger.info("  acc=%.4f  f1_w=%.4f  f1_m=%.4f", m["accuracy"], m["f1_weighted"], m["f1_macro"])
    else:
        logger.warning("Config C model not found — skipping")

    # Config D (Transformer-only classifier)
    logger.info("Evaluating %s — %s", CONFIG_D, CONFIG_DESCRIPTIONS[CONFIG_D])
    m, dr, yp, yt, ypr = evaluate_transformer_standalone(
        classifier, test_loader, test_sessions, device)
    all_results[CONFIG_D] = m
    all_dr[CONFIG_D] = dr
    all_preds[CONFIG_D] = yp
    all_probas[CONFIG_D] = ypr
    logger.info("  acc=%.4f  f1_w=%.4f  f1_m=%.4f", m["accuracy"], m["f1_weighted"], m["f1_macro"])

    # Save main evaluation artifact
    eval_path = os.path.join(model_dir, "evaluation_results_v4.json")
    _save_json(all_results, eval_path)
    logger.info("Saved -> %s", eval_path)

    # Save comparison table
    if len(all_results) > 1:
        table = compare_configs(
            all_results,
            save_path=os.path.join(model_dir, "full_eval_results.json"),
        )
        logger.info("\nComparison Table:\n%s", table)

    # Save per-config detection rates
    dr_path = os.path.join(model_dir, "detection_rates.json")
    _save_json(all_dr, dr_path)
    logger.info("Saved detection rates -> %s", dr_path)

    # ══════════════════════════════════════════════════════════════════════════
    # 2. Session Window Ablation
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("PHASE 2 — Session Window Ablation Study")
    logger.info("=" * 70)

    ablation_sizes = cfg["data"].get("session_sizes_ablation", [10, 20, 50])
    ablation_results = {}

    if hybrid.xgb_c is not None:
        for size in ablation_sizes:
            label = f"{size}-flow"
            new_sessions = []
            for s in test_sessions:
                feat = s["features"][:size]
                lt   = s["log_types"][:size]
                new_sessions.append({
                    "features": feat, "log_types": lt, "label": s["label"],
                    "has_attack": s["has_attack"], "attack_ids": s["attack_ids"],
                    "session_idx": s["session_idx"],
                })

            new_stats = compute_all_session_stats(new_sessions, feat_cols)
            new_ds = SessionDataset(new_sessions, max_seq_len, return_labels=True)
            new_loader = DataLoader(new_ds, batch_size=64, shuffle=False,
                                    collate_fn=collate_sessions, num_workers=0)

            m, _, _, _, _ = evaluate_xgb_config(
                hybrid, new_loader, new_stats, new_sessions, "c", device)
            ablation_results[label] = {k: v for k, v in m.items()
                                       if not isinstance(v, (list, dict))}
            logger.info("  Window=%s: acc=%.4f f1_w=%.4f f1_m=%.4f",
                        label, m["accuracy"], m["f1_weighted"], m["f1_macro"])

        if ablation_results:
            ab_table = session_window_ablation(ablation_results)
            ab_path = os.path.join(model_dir, "ablation_results.json")
            _save_json(ablation_results, ab_path)
            logger.info("Saved -> %s", ab_path)

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Threshold Analysis (Config C — primary serving detector)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("PHASE 3 — Threshold Analysis (Config C)")
    logger.info("=" * 70)

    if CONFIG_C in all_probas:
        thr_result = threshold_analysis(y_true, all_probas[CONFIG_C])
        thr_path = os.path.join(model_dir, "threshold_analysis.json")
        _save_json(thr_result, thr_path)
        logger.info("Saved -> %s", thr_path)

    # ══════════════════════════════════════════════════════════════════════════
    # 4. SHAP Feature Attribution (Configs A, B, C)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("PHASE 4 — SHAP Feature Attribution")
    logger.info("=" * 70)

    from src.evaluation.explainability import SHAPExplainer

    d_model = cfg["transformer"]["d_model"]
    embed_names = [f"emb_{i}" for i in range(d_model)]
    stat_names  = list(STAT_FEATURE_NAMES)

    # Extract embeddings once (reused across configs B/C)
    logger.info("Extracting embeddings for SHAP analysis...")
    test_emb, test_labels = extract_embeddings(classifier, test_loader, device)

    n_shap = min(500, len(test_emb))
    rng = np.random.RandomState(42)
    shap_idx = rng.choice(len(test_emb), n_shap, replace=False)

    # Config A — SHAP on 26 statistical features
    if hybrid.xgb_a is not None and hybrid.xgb_a._fitted:
        logger.info("SHAP for %s (statistical features)", CONFIG_A)
        shap_a = SHAPExplainer(hybrid.xgb_a, feature_names=stat_names)
        imp_a = shap_a.global_importance(test_stats[shap_idx], n_top=26)
        _save_json(imp_a, os.path.join(model_dir, "shap_importance_config_a.json"))
        for item in imp_a[:10]:
            logger.info("  %-40s %.6f", item["feature"], item["importance"])

    # Config B — SHAP on 256-dim embeddings
    if hybrid.xgb_b is not None and hybrid.xgb_b._fitted:
        logger.info("SHAP for %s (embedding features)", CONFIG_B)
        shap_b = SHAPExplainer(hybrid.xgb_b, feature_names=embed_names)
        imp_b = shap_b.global_importance(test_emb[shap_idx], n_top=20)
        _save_json(imp_b, os.path.join(model_dir, "shap_importance_config_b.json"))
        for item in imp_b[:10]:
            logger.info("  %-40s %.6f", item["feature"], item["importance"])

    # Config C — SHAP on embeddings + stats
    if hybrid.xgb_c is not None and hybrid.xgb_c._fitted:
        logger.info("SHAP for %s (hybrid features)", CONFIG_C)
        hybrid_names = embed_names + stat_names
        X_hybrid = np.hstack([test_emb, test_stats])
        shap_c = SHAPExplainer(hybrid.xgb_c, feature_names=hybrid_names)
        imp_c = shap_c.global_importance(X_hybrid[shap_idx], n_top=20)
        _save_json(imp_c, os.path.join(model_dir, "shap_importance_config_c.json"))
        for item in imp_c[:10]:
            logger.info("  %-40s %.6f", item["feature"], item["importance"])

    # ══════════════════════════════════════════════════════════════════════════
    # 5. Attention Analysis (Config D — Transformer-only)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("PHASE 5 — Attention Analysis (Config D)")
    logger.info("=" * 70)

    from src.evaluation.explainability import AttentionAnalyser
    from src.data.loader import ID_TO_LABEL

    analyser = AttentionAnalyser(classifier, device)

    # Pick one high-confidence correctly-classified session per attack type
    if CONFIG_D in all_preds:
        y_pred_d = all_preds[CONFIG_D]
        y_proba_d = all_probas[CONFIG_D]
        confidences = y_proba_d[np.arange(len(y_pred_d)), y_pred_d]

        attention_results = {}
        for attack_id in range(15):
            name = ID_TO_LABEL.get(attack_id, str(attack_id))
            mask = (y_true == attack_id) & (y_pred_d == attack_id)
            if not mask.any():
                continue

            # Best correctly-classified session for this attack
            candidates = np.where(mask)[0]
            best_idx = candidates[np.argmax(confidences[candidates])]
            session = test_sessions[best_idx]

            features = np.array(session["features"], dtype=np.float32)
            log_types = np.array(session["log_types"], dtype=np.int64)
            real_len = len(features)

            # Pad to max_seq_len
            pad_len = max(0, max_seq_len - real_len)
            if pad_len > 0:
                features = np.vstack([features, np.zeros((pad_len, features.shape[1]), dtype=np.float32)])
                log_types = np.concatenate([log_types, np.zeros(pad_len, dtype=np.int64)])
            features = features[:max_seq_len]
            log_types = log_types[:max_seq_len]
            padding_mask = np.zeros(max_seq_len, dtype=bool)
            if pad_len > 0:
                padding_mask[real_len:] = True

            try:
                attn_maps = analyser.get_attention_maps(features, log_types, padding_mask)
                cls_attn = analyser.cls_attention_over_events(attn_maps, layer=-1)
                top_events = analyser.identify_anomalous_events(cls_attn[:real_len], top_n=5)
                rollout = analyser.layer_wise_relevance(attn_maps)

                attention_results[name] = {
                    "session_idx": int(best_idx),
                    "confidence": float(confidences[best_idx]),
                    "n_flows": real_len,
                    "cls_attention_last_layer": cls_attn[:real_len].tolist(),
                    "top_attended_events": top_events,
                    "rollout_cls_row": rollout[0, 1:real_len + 1].tolist(),
                    "attention_entropy": float(-np.sum(
                        cls_attn[:real_len] * np.log(cls_attn[:real_len] + 1e-10)
                    )),
                }
                logger.info("  %-25s  conf=%.4f  top_event=%d  entropy=%.4f",
                            name, confidences[best_idx],
                            top_events[0]["event_idx"] if top_events else -1,
                            attention_results[name]["attention_entropy"])
            except Exception as e:
                logger.warning("  %-25s  FAILED: %s", name, e)

        attn_path = os.path.join(model_dir, "attention_analysis_config_d.json")
        _save_json(attention_results, attn_path)
        logger.info("Saved -> %s", attn_path)

    # ══════════════════════════════════════════════════════════════════════════
    # Done
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("Evaluation complete — all artifacts saved to %s", model_dir)
    logger.info("=" * 70)


def _save_json(obj, path):
    """Save dict/list to JSON, handling numpy types."""
    class NpEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=NpEncoder)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Mini AI SOC — 4-config ablation")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
