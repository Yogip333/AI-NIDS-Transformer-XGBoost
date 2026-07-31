#!/usr/bin/env python
"""
AI-NIDS — Main Entry Point.

Usage:
  python main.py preprocess           # Step 1: load & sessionize data
  python main.py preprocess --dev     # Step 1 with small sample (fast)
  pyrain + finetune + XGBoost
  python main.py train --skip-pretrainthon main.py train                # Step 2: pret
  python main.py evaluate             # Step 3: full evaluation + ablation + SHAP
  python main.py serve                # Step 4: start API server + open dashboard
  python main.py rag-build            # Build RAG knowledge base only
  python main.py all --dev            # Run all steps (dev mode)
"""
import os
import sys
import logging
import argparse

# Let Python find modules in this project folder
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger("main")


def cmd_preprocess(args):
    from scripts.preprocess import main as preprocess_main
    preprocess_main(config_path=args.config, dev_mode=args.dev)


def cmd_train(args):
    from scripts.train_model import main as train_main
    train_main(
        config_path=args.config,
        skip_pretrain=args.skip_pretrain,
        skip_finetune=args.skip_finetune,
        skip_xgb=args.skip_xgb,
        resume=args.resume,
    )


def cmd_evaluate(args):
    from scripts.evaluate_model import main as eval_main
    eval_main(config_path=args.config)


def cmd_serve(args):
    import webbrowser
    import threading
    import time
    import yaml

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    host = cfg.get("api", {}).get("host", "0.0.0.0")
    port = int(cfg.get("api", {}).get("port", 8000))

    def open_browser():
        time.sleep(2)
        url = f"http://localhost:{port}"
        logger.info("Opening browser at %s", url)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    import uvicorn
    from src.api.app import app

    logger.info("Starting Mini AI SOC server at http://%s:%d", host, port)
    logger.info("Dashboard: http://localhost:%d", port)
    logger.info("API docs:  http://localhost:%d/docs", port)
    uvicorn.run(app, host=host, port=port, log_level="info")


def cmd_rag_build(args):
    import yaml
    from src.rag.threat_rag import ThreatRAG

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    logger.info("Building RAG knowledge base…")
    rag = ThreatRAG(
        db_path=cfg["paths"].get("rag_db_dir", "data/rag_db"),
        collection=cfg["rag"]["collection_name"],
        top_k=cfg["rag"]["top_k"],
    )
    rag.build_knowledge_base(force_rebuild=True)
    logger.info("RAG knowledge base built successfully")


def cmd_all(args):
    """Run the full pipeline end-to-end."""
    logger.info("Running full pipeline (dev_mode=%s)…", args.dev)
    cmd_preprocess(args)
    cmd_train(args)
    cmd_evaluate(args)
    logger.info("Pipeline complete! Now run: python main.py serve")


def main():
    os.makedirs("logs", exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Mini AI SOC — Zeek NIDS with Transformer + XGBoost + RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to config YAML")

    sub = parser.add_subparsers(dest="command")

    # preprocess
    p_pre = sub.add_parser("preprocess", help="Load & sessionize CICIDS2017 data")
    p_pre.add_argument("--dev", action="store_true",
                       help="Use small subset (20k rows/file) for fast testing")

    # train
    p_train = sub.add_parser("train", help="Train all models")
    p_train.add_argument("--skip-pretrain", action="store_true")
    p_train.add_argument("--skip-finetune", action="store_true")
    p_train.add_argument("--skip-xgb",      action="store_true")
    p_train.add_argument("--resume",        action="store_true",
                         help="Auto-skip completed stages based on checkpoints")

    # evaluate
    sub.add_parser("evaluate", help="Evaluate models + ablation + SHAP")

    # serve
    sub.add_parser("serve", help="Start FastAPI server + open dashboard")

    # rag-build
    sub.add_parser("rag-build", help="Build RAG threat knowledge base")

    # all
    p_all = sub.add_parser("all", help="Run full pipeline")
    p_all.add_argument("--dev", action="store_true")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Add missing attributes for subparsers that don't define them
    if not hasattr(args, "dev"):           args.dev = False
    if not hasattr(args, "skip_pretrain"): args.skip_pretrain = False
    if not hasattr(args, "skip_finetune"): args.skip_finetune = False
    if not hasattr(args, "skip_xgb"):      args.skip_xgb = False
    if not hasattr(args, "resume"):        args.resume = False

    cmd_map = {
        "preprocess": cmd_preprocess,
        "train":      cmd_train,
        "evaluate":   cmd_evaluate,
        "serve":      cmd_serve,
        "rag-build":  cmd_rag_build,
        "all":        cmd_all,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
