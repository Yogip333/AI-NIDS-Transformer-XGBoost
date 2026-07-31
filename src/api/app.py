"""
FastAPI Application for AI-NIDS.

Startup sequence:
  1. Load config
  2. Load trained models (Transformer classifier + session XGBoost configs A/B/C/D)
  3. Load / build RAG knowledge base
  4. Expose REST API endpoints

Ablation configs (session-level, 21,230 test sessions on full CICIDS2017):
  A — XGBoost on 26 statistical features only          (61.93% acc)
  B — XGBoost on 256-dim Transformer embeddings        (90.94% acc)
  C — XGBoost on embeddings + statistical features     (90.94% acc)  [primary]
  D — Transformer classifier (CLS → MLP head)          (89.86% acc)
"""
import logging
import os
import sys
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logger = logging.getLogger(__name__)

# Global state accessible to all routes.
# Alert history and detection counters live in src/api/alert_store.py (SQLite)
# so the streaming engine and the REST API see a unified view.
APP_STATE: dict = {
    "config":        None,
    # Session-level detector — Transformer classifier (Config B)
    "classifier":    None,
    # Session-level XGBoost configs (A=stat-only, C=embed-only, D=hybrid)
    "hybrid_ids":    None,
    # Shared
    "rag":           None,
    "scaler":        None,
    "feature_cols":  None,
    "ready":         False,   # True when session hybrid is loaded
}


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_models(cfg: dict, state: dict) -> None:
    """
    Load session-level models only (per-flow XGBoost removed from architecture).

    Loading order:
      1. RAG (always — works without ML models)
      2. Feature scaler (session scaler, 81 features)
      3. Transformer classifier (Config B)
      4. Session XGBoost configs A / C / D via HybridIDS
         Ready = True when hybrid XGBoost (Config D) is loaded.
    """
    import torch
    from src.api.alert_store import get_alert_store
    from src.data.feature_engineer import FeatureScaler
    from src.data.loader import FEATURE_COLS
    from src.models.transformer import build_classifier
    from src.models.hybrid import HybridIDS
    from src.rag.threat_rag import ThreatRAG
    from src.rag.groq_analyst import GroqAnalyst

    # Initialise the unified alert store eagerly so /api/alerts works
    # even before the first detection.
    try:
        get_alert_store()
    except Exception as e:
        logger.warning("AlertStore init failed (non-fatal): %s", e)

    device    = torch.device("cpu")
    model_dir = cfg["paths"]["models_dir"]
    state["feature_cols"] = FEATURE_COLS

    # ── Groq analyst (optional LLM enrichment) ───────────────────────────────
    groq_analyst = None
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        try:
            groq_analyst = GroqAnalyst(api_key=groq_key)
            if groq_analyst.available:
                logger.info("Groq LLM analyst ready (model=%s)", "llama-3.3-70b-versatile")
        except Exception as e:
            logger.warning("Groq analyst init failed (non-fatal): %s", e)
    else:
        logger.info("GROQ_API_KEY not set — LLM enrichment disabled, RAG-only mode")

    # ── RAG (always loaded first — works without ML models) ───────────────────
    logger.info("Initialising RAG pipeline…")
    try:
        rag = ThreatRAG(
            db_path=cfg["paths"].get("rag_db_dir", "data/rag_db"),
            collection=cfg["rag"]["collection_name"],
            top_k=cfg["rag"]["top_k"],
            groq_analyst=groq_analyst,
        )
        rag.build_knowledge_base(force_rebuild=False)
        state["rag"] = rag
        logger.info("RAG ready (Groq enrichment: %s)", "enabled" if groq_analyst else "disabled")
    except Exception as e:
        logger.error("RAG initialisation failed: %s", e)

    # ── Feature scaler (session scaler — 81 features) ────────────────────────
    session_scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
    if os.path.exists(session_scaler_path):
        state["scaler"] = FeatureScaler.load(session_scaler_path)
        logger.info("Loaded session scaler (81 features)")
    else:
        logger.warning("Session scaler not found — inference will use raw features")

    # ── Transformer classifier (Config B) ────────────────────────────────────
    # Trained on sessions with 81 features (76 CICIDS + 5 Zeek log-type cols).
    processed_pkl = os.path.join(cfg["paths"]["processed_dir"], "processed_data.pkl")
    if os.path.exists(processed_pkl):
        import pickle
        with open(processed_pkl, "rb") as _f:
            _d = pickle.load(_f)
        num_features = int(_d.get("num_features", len(FEATURE_COLS)))
    else:
        num_features = len(FEATURE_COLS)

    classifier_path = os.path.join(model_dir, "classifier_final.pt")
    try:
        classifier = build_classifier(cfg, input_dim=num_features)
        if os.path.exists(classifier_path):
            classifier.load_state_dict(
                torch.load(classifier_path, map_location=device, weights_only=True)
            )
            logger.info("Loaded Transformer classifier (Config B)")
        else:
            logger.warning("Transformer classifier checkpoint not found — using untrained weights")
        classifier.eval().to(device)
        state["classifier"] = classifier

        # ── Session XGBoost — Configs A / C / D via HybridIDS ────────────────
        hybrid = HybridIDS(cfg, classifier, num_classes=15)
        hybrid.load_all(model_dir)
        state["hybrid_ids"] = hybrid

        loaded = [k for k, v in [("A", hybrid.xgb_a), ("C", hybrid.xgb_b), ("D", hybrid.xgb_c)] if v is not None]
        if loaded:
            logger.info("Session XGBoost configs loaded: %s", loaded)
        if hybrid.xgb_c is not None:
            state["ready"] = True
            logger.info("System READY — primary detector: Config D (embeddings + stats)")
        elif state["classifier"] is not None:
            state["ready"] = True
            logger.info("System READY — primary detector: Config B (Transformer classifier)")

    except Exception as e:
        logger.error("Failed to load Transformer/hybrid models: %s", e)

    if not state["ready"]:
        logger.warning("No trained models found — system in RAG-ONLY / DEMO mode")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    )
    # Load .env so GROQ_API_KEY and HUGGINGFACE_TOKEN are available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    logger.info("Starting Mini AI SOC backend…")

    try:
        cfg = load_config()
        APP_STATE["config"] = cfg
        load_models(cfg, APP_STATE)
    except Exception as e:
        logger.error("Model loading failed: %s", e, exc_info=True)
        # Keep running in degraded mode so RAG still works

    yield

    logger.info("Shutting down Mini AI SOC backend")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Mini AI SOC — Zeek NIDS",
        description=(
            "Hybrid Self-Supervised Transformer + XGBoost Intrusion Detection System. "
            "Uses CICIDS2017-derived Zeek telemetry with RAG-powered threat intelligence."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    from src.api.routes import router
    app.include_router(router, prefix="/api")

    # Serve static dashboard files
    static_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static"))
    if os.path.isdir(static_dir):
        from starlette.staticfiles import StaticFiles
        from fastapi.responses import FileResponse
        
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/")
        async def read_index():
            index_path = os.path.join(static_dir, "dashboard.html")
            if os.path.exists(index_path):
                return FileResponse(index_path)
            return {"message": "Mini AI SOC API Ready. Dashboard not found at static/dashboard.html"}

    return app


app = create_app()
