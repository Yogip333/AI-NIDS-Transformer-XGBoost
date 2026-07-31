"""
Model Training Script.

This runs the full training pipeline:
  1. Load preprocessed sessions
  2. Self-supervised pre-training (MFP + NEP on benign sessions)
  3. Supervised fine-tuning (Transformer classifier)
  4. XGBoost training for all three configurations
  5. Save all models and training history
"""
import logging
import os
import pickle
import sys
import json
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/training.log"),
    ],
)
logger = logging.getLogger(__name__)


def main(config_path: str = "configs/config.yaml", skip_pretrain: bool = False,
         skip_finetune: bool = False, skip_xgb: bool = False,
         resume: bool = False):
    import torch
    import numpy as np
    from torch.utils.data import DataLoader

    from src.data.sessionizer import SessionDataset, collate_sessions
    from src.data.feature_engineer import STAT_FEATURE_NAMES
    from src.models.transformer import build_pretrain_model, build_classifier
    from src.models.hybrid import HybridIDS
    from src.training.pretrain import PreTrainer
    from src.training.finetune import FineTuner, build_weighted_sampler

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    processed_dir = cfg["paths"]["processed_dir"]
    model_dir     = cfg["paths"]["models_dir"]
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info("CUDA device: %s (%.2f GB VRAM)", gpu_name, total_mem_gb)
        # Enables kernel autotuning for fixed-size inputs and improves throughput.
        torch.backends.cudnn.benchmark = True

    # ── Load preprocessed data ────────────────────────────────────────────────
    data_path = os.path.join(processed_dir, "processed_data.pkl")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Run preprocess.py first: {data_path}")

    logger.info("Loading preprocessed data…")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    train_sessions  = data["train_sessions"]
    val_sessions    = data["val_sessions"]
    test_sessions   = data["test_sessions"]
    benign_sessions = data["benign_sessions"]
    train_stats     = data["train_stats"]
    val_stats       = data["val_stats"]
    test_stats      = data["test_stats"]
    feat_cols       = data["feature_cols"]
    num_features    = data["num_features"]

    max_seq_len = cfg["transformer"]["max_seq_len"]
    pcfg = cfg.get("pretraining", {})
    fcfg = cfg.get("finetuning", {})
    NUM_CLASSES = 15

    # ── Build datasets ────────────────────────────────────────────────────────
    benign_ds  = SessionDataset(benign_sessions, max_seq_len, return_labels=False)
    train_ds   = SessionDataset(train_sessions,  max_seq_len, return_labels=True)
    val_ds     = SessionDataset(val_sessions,    max_seq_len, return_labels=True)
    test_ds    = SessionDataset(test_sessions,   max_seq_len, return_labels=True)

    if len(benign_ds) == 0:
        raise ValueError("No benign sessions found for pre-training. Re-run preprocess and verify label mapping.")
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise ValueError("Empty train/val/test session split detected. Re-run preprocess and verify data split.")

    logger.info("Session counts | benign=%d train=%d val=%d test=%d",
                len(benign_ds), len(train_ds), len(val_ds), len(test_ds))

    # Weighted sampler for imbalanced training
    train_labels = np.array([s["label"] for s in train_sessions])
    sampler = build_weighted_sampler(train_labels, NUM_CLASSES)

    loader_workers = int(cfg.get("training", {}).get(
        "num_workers", 2 if device.type == "cuda" else 0
    ))
    loader_kwargs = {
        "collate_fn": collate_sessions,
        "num_workers": loader_workers,
        "pin_memory": device.type == "cuda",
    }
    if loader_workers > 0:
        loader_kwargs["persistent_workers"] = True

    logger.info("DataLoader settings | workers=%d pin_memory=%s",
                loader_workers, loader_kwargs["pin_memory"])

    benign_loader = DataLoader(benign_ds, batch_size=pcfg.get("batch_size", 64),
                               shuffle=True, **loader_kwargs)
    train_loader  = DataLoader(train_ds,  batch_size=fcfg.get("batch_size", 32),
                               sampler=sampler, **loader_kwargs)
    val_loader    = DataLoader(val_ds,    batch_size=64, shuffle=False,
                               **loader_kwargs)
    test_loader   = DataLoader(test_ds,   batch_size=64, shuffle=False,
                               **loader_kwargs)

    history = {}

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1: Self-supervised Pre-training
    # ══════════════════════════════════════════════════════════════════════════
    pretrain_ckpt = os.path.join(model_dir, "pretrain_best.pt")
    pretrain_final_ckpt = os.path.join(model_dir, "pretrain_final.pt")
    finetune_best_path = os.path.join(model_dir, "finetune_best.pt")
    classifier_path = os.path.join(model_dir, "classifier_final.pt")
    xgb_stat_path = os.path.join(model_dir, "xgb_stat_only.joblib")
    xgb_embed_path = os.path.join(model_dir, "xgb_embed_only.joblib")
    xgb_hybrid_path = os.path.join(model_dir, "xgb_hybrid.joblib")

    if resume:
        has_pretrain = os.path.exists(pretrain_ckpt) or os.path.exists(pretrain_final_ckpt)
        has_finetune = os.path.exists(classifier_path) or os.path.exists(finetune_best_path)
        has_xgb = all(os.path.exists(p) for p in [xgb_stat_path, xgb_embed_path, xgb_hybrid_path])

        if has_pretrain and not skip_pretrain:
            skip_pretrain = True
            logger.info("Resume mode: found pretrain checkpoints, skipping Stage 1")
        if has_finetune and not skip_finetune:
            skip_finetune = True
            logger.info("Resume mode: found fine-tune checkpoint, skipping Stage 2")
        if has_xgb and not skip_xgb:
            skip_xgb = True
            logger.info("Resume mode: found XGBoost models, skipping Stage 3")

        logger.info("Resume effective flags | skip_pretrain=%s skip_finetune=%s skip_xgb=%s",
                    skip_pretrain, skip_finetune, skip_xgb)

    if not skip_pretrain:
        logger.info("=" * 60)
        logger.info("Stage 1: Self-Supervised Pre-Training")
        logger.info("=" * 60)
        pretrain_model = build_pretrain_model(cfg, input_dim=num_features)
        logger.info("Pretrain model params: %d",
                    sum(p.numel() for p in pretrain_model.parameters()))

        pre_trainer = PreTrainer(pretrain_model, cfg, device, save_dir=model_dir)
        history["pretrain"] = pre_trainer.train(benign_loader, val_loader=None)
        logger.info("Pre-training complete")
    else:
        logger.info("Skipping pre-training (skip_pretrain=True)")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 2: Supervised Fine-Tuning
    # ══════════════════════════════════════════════════════════════════════════
    if not skip_finetune:
        logger.info("=" * 60)
        logger.info("Stage 2: Supervised Fine-Tuning")
        logger.info("=" * 60)

        # Build classifier (load pretrained encoder if available)
        classifier = build_classifier(cfg, input_dim=num_features, num_classes=NUM_CLASSES)

        if os.path.exists(pretrain_ckpt):
            logger.info("Loading pre-trained encoder weights from %s", pretrain_ckpt)
            from src.training.pretrain import PreTrainer as PT
            PT.load_encoder_weights(pretrain_ckpt, classifier)
        elif os.path.exists(pretrain_final_ckpt):
            logger.info("Loading pre-trained encoder weights from %s", pretrain_final_ckpt)
            from src.training.pretrain import PreTrainer as PT
            PT.load_encoder_weights(pretrain_final_ckpt, classifier)
        else:
            logger.warning("No pre-training checkpoint found — training from scratch")

        ft_trainer = FineTuner(classifier, cfg, device, save_dir=model_dir,
                               num_classes=NUM_CLASSES)
        history["finetune"] = ft_trainer.train(train_loader, val_loader)

        # Persist final classifier before evaluation so later steps can resume
        # even if reporting/evaluation fails.
        ft_trainer.save_model(classifier_path)
        logger.info("Classifier saved to %s", classifier_path)

        # Evaluate transformer
        logger.info("Evaluating Transformer classifier on test set…")
        ft_result = ft_trainer.evaluate(test_loader)
        history["transformer_test"] = {k: v for k, v in ft_result.items()
                                        if not isinstance(v, (list, type(None), np.ndarray))}
    else:
        logger.info("Skipping fine-tuning (skip_finetune=True)")
        if not os.path.exists(classifier_path) and not os.path.exists(finetune_best_path):
            logger.warning("No classifier checkpoint — building untrained model for XGBoost step")
        classifier = build_classifier(cfg, input_dim=num_features, num_classes=NUM_CLASSES)
        if os.path.exists(classifier_path):
            import torch
            classifier.load_state_dict(
                torch.load(classifier_path, map_location=device, weights_only=True)
            )
        elif os.path.exists(finetune_best_path):
            import torch
            best_state = torch.load(finetune_best_path, map_location=device, weights_only=True)
            classifier.load_state_dict(best_state["model_state_dict"])
            logger.info("Loaded classifier weights from %s", finetune_best_path)
        classifier = classifier.to(device)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 3: XGBoost Training (3 configurations)
    # ══════════════════════════════════════════════════════════════════════════
    if not skip_xgb:
        logger.info("=" * 60)
        logger.info("Stage 3: XGBoost Training (3 configs)")
        logger.info("=" * 60)

        from src.data.loader import ID_TO_LABEL
        stat_names = STAT_FEATURE_NAMES
        embed_names = [f"emb_{i}" for i in range(cfg["transformer"]["d_model"])]

        # Sequential loader for XGBoost — no sampler, no shuffle.
        # This ensures embeddings/labels stay aligned with train_stats.
        xgb_train_loader = DataLoader(train_ds, batch_size=64, shuffle=False,
                                       **loader_kwargs)

        hybrid = HybridIDS(cfg, classifier, num_classes=NUM_CLASSES)
        hybrid.train_all(
            train_loader=xgb_train_loader,
            val_loader=val_loader,
            train_stats=train_stats,
            val_stats=val_stats,
            stat_feature_names=stat_names,
            embed_feature_names=embed_names,
        )
        hybrid.save_all(model_dir)
        logger.info("XGBoost models saved to %s", model_dir)

        # Evaluate all configs
        from src.evaluation.metrics import compute_metrics, compare_configs, detection_rate_by_attack

        all_results = {}
        for config_id in ["a", "b", "c"]:
            label_map = {
                "a": "Statistical Only",
                "b": "Transformer Embeddings",
                "c": "Hybrid (Embeddings + Stats)",
            }
            logger.info("Evaluating Config-%s (%s)…", config_id.upper(), label_map[config_id])
            y_pred, y_proba = hybrid.predict(test_loader, test_stats, config=config_id)
            y_true = np.array([s["label"] for s in test_sessions])

            metrics = compute_metrics(y_true, y_pred, y_proba)
            all_results[label_map[config_id]] = metrics
            history[f"eval_config_{config_id}"] = {
                k: v for k, v in metrics.items()
                if not isinstance(v, (list, dict))
            }

            # Per-attack detection rates
            dr = detection_rate_by_attack(y_true, y_pred)
            logger.info("Detection rates for Config-%s:", config_id.upper())
            for atk, stats in dr.items():
                logger.info("  %-30s  DR=%.3f  FP=%d", atk, stats["detection_rate"], stats["false_positives"])

        # Print comparison table
        from src.evaluation.metrics import compare_configs
        table = compare_configs(all_results,
                                save_path=os.path.join(model_dir, "evaluation_results.json"))
        logger.info("\n%s", table)
    else:
        logger.info("Skipping XGBoost training (skip_xgb=True)")

    # Save training history
    def _json_safe(obj):
        """Recursively convert numpy types so json.dump doesn't choke."""
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    history_path = os.path.join(model_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(_json_safe(history), f, indent=2)
    logger.info("Training history saved to %s", history_path)
    logger.info("=" * 60)
    logger.info("Training pipeline complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Mini AI SOC models")
    parser.add_argument("--config",        default="configs/config.yaml")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--skip-xgb",      action="store_true")
    parser.add_argument("--resume",        action="store_true")
    args = parser.parse_args()
    main(args.config,
         skip_pretrain=args.skip_pretrain,
         skip_finetune=args.skip_finetune,
         skip_xgb=args.skip_xgb,
         resume=args.resume)
