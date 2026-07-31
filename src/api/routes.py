"""
API Routes for AI-NIDS.

Endpoints:
  GET  /api/health           — system health and readiness
  GET  /api/model-info       — model configuration info
  POST /api/analyze          — analyse a batch of network flows
  GET  /api/alerts           — recent alert history
  GET  /api/stats            — detection statistics
  GET  /api/threat-intel/{attack_name} — RAG threat intelligence
  POST /api/threat-intel/search       — free-text RAG search
  GET  /api/label-map        — integer → attack name mapping
  POST /api/demo             — demo analysis with synthetic data
"""
import logging
import os
import time
import uuid
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.alert_store import get_alert_store
from src.api.app import APP_STATE
from src.data.loader import ID_TO_LABEL, FEATURE_COLS, NUM_CLASSES
from src.data.session_pipeline import realign_to_trained_schema, trained_feature_names
from src.data.zeek_mapper import LOG_TYPE_NAMES
from src.detection.exceptions import InferenceError, ModelNotLoadedError

logger = logging.getLogger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class FlowRecord(BaseModel):
    """Single network flow record (matches CIC-IDS2017 feature names)."""
    features: list[float] = Field(..., description="78 numerical flow features in CICIDS2017 order")
    log_type: int = Field(0, description="Zeek log type: 0=conn 1=dns 2=http 3=ssl 4=files")


class AnalyzeRequest(BaseModel):
    flows: list[FlowRecord] = Field(..., description="List of flow records forming a session")
    model_config_: str = Field("c", alias="model_config",
                                description="'b'=embed-only 'c'=hybrid (default). Config 'a' (stat-only) is not a production path and is ignored.")


class ThreatSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filter_severity: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Health / Info
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    return {
        "status":   "ready" if APP_STATE["ready"] else "degraded",
        "message":  "All models loaded" if APP_STATE["ready"] else "RAG-only mode (models not trained yet)",
        "rag_ready": APP_STATE["rag"] is not None,
    }


@router.get("/model-info")
async def model_info():
    cfg = APP_STATE.get("config", {})
    t   = cfg.get("transformer", {})
    hybrid = APP_STATE.get("hybrid_ids")
    return {
        "ablation_configs": {
            "A": {
                "description": "XGBoost on 26 statistical features only (baseline)",
                "loaded":      hybrid is not None and hybrid.xgb_a is not None,
                "accuracy":    "61.93%",
                "f1_weighted": "61.01%",
                "f1_macro":    "17.47%",
            },
            "B": {
                "description": "XGBoost on 256-dim Transformer embeddings",
                "loaded":      hybrid is not None and hybrid.xgb_b is not None,
                "accuracy":    "90.94%",
                "f1_weighted": "90.92%",
                "f1_macro":    "60.42%",
            },
            "C": {
                "description": "XGBoost on Transformer embeddings + statistical features (full hybrid)",
                "loaded":      hybrid is not None and hybrid.xgb_c is not None,
                "accuracy":    "90.94%",
                "f1_weighted": "90.91%",
                "f1_macro":    "60.44%",
            },
            "D": {
                "description": "Transformer classifier (CLS token → MLP head)",
                "loaded":      APP_STATE.get("classifier") is not None,
                "accuracy":    "89.86%",
                "f1_weighted": "90.18%",
                "f1_macro":    "48.00%",
            },
        },
        "transformer": {
            "d_model":    t.get("d_model", 256),
            "nhead":      t.get("nhead", 8),
            "num_layers": t.get("num_encoder_layers", 4),
            "max_seq_len":t.get("max_seq_len", 60),
            "use_amp":    t.get("use_amp", True),
            "pretraining": "Self-supervised (MFP + NEP on benign sessions)",
        },
        "num_classes":     NUM_CLASSES,
        "num_features":    len(FEATURE_COLS),
        "session_size":    cfg.get("data", {}).get("session_size", 20),
        "embedding_model": cfg.get("rag", {}).get("embedding_model", "nomic-embed-text-v1.5"),
        "system_ready":    APP_STATE.get("ready", False),
    }


@router.get("/label-map")
async def label_map():
    return {"labels": {str(k): v for k, v in ID_TO_LABEL.items()}}


# ─────────────────────────────────────────────────────────────────────────────
# Core Analysis Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """
    Analyse a session (list of flows) and return a threat detection result.

    Inference path: Transformer encoder → Config C (hybrid XGBoost) → [Config B fallback].
    Returns HTTP 503 if neither Config-C nor Config-B checkpoint is loaded.
    No heuristic or rule-based fallback — if the model cannot run, the error is explicit.
    """
    start_ts = time.time()
    flows = request.flows

    if len(flows) == 0:
        raise HTTPException(status_code=400, detail="At least one flow required")

    max_flows = APP_STATE.get("config", {}).get("api", {}).get("max_flows_per_request", 1000)
    if len(flows) > max_flows:
        raise HTTPException(status_code=400, detail=f"Max {max_flows} flows per request")

    # ── Validate models are loaded ────────────────────────────────────────────
    if not APP_STATE.get("ready") or APP_STATE.get("hybrid_ids") is None:
        raise HTTPException(
            status_code=503,
            detail=str(ModelNotLoadedError(
                "HybridIDS (Transformer + XGBoost)",
                "models/checkpoints/",
            )),
        )
    if APP_STATE.get("scaler") is None:
        raise HTTPException(
            status_code=503,
            detail=str(ModelNotLoadedError(
                "FeatureScaler",
                "models/checkpoints/feature_scaler.pkl",
            )),
        )

    # Build feature arrays
    feat_array = np.array([f.features for f in flows], dtype=np.float32)
    lt_array   = np.array([f.log_type for f in flows], dtype=np.int64)

    # Clip and clean boundary values before schema validation
    feat_array = np.nan_to_num(feat_array, nan=0.0, posinf=1e6, neginf=-1e6)
    feat_array = np.clip(feat_array, -1e9, 1e9)

    # ── Realign to 81-col trained schema ─────────────────────────────────────
    # Single source of truth shared with scripts/live_demo.py.
    # Raises FeatureDimensionError / NaNInfError on contract violations.
    feat_array_81 = realign_to_trained_schema(feat_array, lt_array)

    # ── Statistical features — computed on the unscaled 81-col array ─────────
    # Training order: stats(raw_81) → scale(raw_81). Must match here.
    from src.data.feature_engineer import compute_session_stats, STAT_FEATURE_NAMES
    session_stats_arr = compute_session_stats(feat_array_81, trained_feature_names())
    stats_dict = dict(zip(STAT_FEATURE_NAMES, session_stats_arr.tolist()))

    # ── Scale features for Transformer / XGBoost ─────────────────────────────
    scaled_feat_array = APP_STATE["scaler"].transform(feat_array_81)

    # ── Inference — Transformer → XGBoost hybrid (Config C, fallback B) ───────
    # predict_with_fallback tries Config C first, then Config B.
    # Config A (stat-only, bypasses Transformer) is not a production path.
    # Raises ModelNotLoadedError if neither C nor B checkpoint is present.
    # Raises InferenceError if the loaded checkpoint has a schema mismatch.
    try:
        result = APP_STATE["hybrid_ids"].predict_with_fallback(
            scaled_feat_array, lt_array.astype(np.int8), session_stats_arr
        )
    except (ModelNotLoadedError, InferenceError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    attack_id    = result["prediction"]
    probabilities = result["probabilities"]
    config_used   = result.get("config_used", "C")

    # ── Build alert ────────────────────────────────────────────────────────────
    alert_id = str(uuid.uuid4())[:8]
    rag = APP_STATE["rag"]
    alert = {
        "alert_id":    alert_id,
        "timestamp":   time.time(),
        "elapsed_ms":  round((time.time() - start_ts) * 1000, 2),
        "attack_id":   attack_id,
        "attack_name": ID_TO_LABEL.get(attack_id, "Unknown"),
        "confidence":  round(float(probabilities[attack_id]), 4),
        "is_attack":   attack_id != 0,
        "config_used": config_used,
        "n_flows":     len(flows),
        "session_stats": {k: round(float(v), 4) for k, v in stats_dict.items()},
        "log_type_distribution": _log_type_dist(lt_array),
        "top_predictions": _top_n(probabilities, 3),
    }

    if rag is not None and attack_id != 0:
        try:
            full_alert = rag.format_alert(
                attack_id=attack_id,
                probabilities=probabilities,
                session_stats=stats_dict,
                config=config_used.lower(),
            )
            alert["threat_intelligence"] = {
                "severity":        full_alert["severity"],
                "description":     full_alert["description"][:400],
                "mitre_tactics":   full_alert["mitre_tactics"],
                "techniques":      full_alert["techniques"][:3],
                "zeek_indicators": full_alert["zeek_indicators"][:3],
                "response_actions": full_alert["response_actions"][:3],
                "tags":            full_alert["tags"][:8],
            }
        except Exception as e:
            logger.warning("RAG alert generation failed: %s", e)

    # Persist to the unified alert store (shared with the live streaming engine)
    try:
        get_alert_store().record(alert, source="rest")
    except Exception as e:
        logger.warning("AlertStore.record failed (non-fatal): %s", e)

    return alert


# ─────────────────────────────────────────────────────────────────────────────
# Alerts & Stats
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/alerts")
async def get_alerts(limit: int = 50, source: Optional[str] = None):
    """
    Return recent alerts from the unified alert store. The optional `source`
    filter accepts 'rest' or 'stream' to scope to one detection path.
    """
    store = get_alert_store()
    alerts = store.recent(limit=limit, source=source)
    return {
        "alerts": alerts,
        "total":  len(alerts),
        "source": source or "all",
    }


@router.get("/stats")
async def get_stats():
    """
    Aggregate detection counters from the unified alert store. Includes
    breakdown by source so REST vs streaming activity is comparable.
    """
    return get_alert_store().stats()


# ─────────────────────────────────────────────────────────────────────────────
# Threat Intelligence (RAG)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/threat-intel/{attack_name}")
async def get_threat_intel(attack_name: str):
    rag = APP_STATE.get("rag")
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG not available")
    result = rag.get_threat_intelligence(attack_name)
    return result


@router.post("/threat-intel/search")
async def search_threat_intel(request: ThreatSearchRequest):
    rag = APP_STATE.get("rag")
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG not available")
    results = rag.query(request.query, top_k=request.top_k,
                        filter_severity=request.filter_severity)
    return {"query": request.query, "results": results}


# ─────────────────────────────────────────────────────────────────────────────
# Demo endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/demo")
async def demo():
    """Run demo analysis using real CICIDS2017 attack samples for accurate detection."""
    import random as _rng

    # Use cached samples if available; retry if cache is empty (e.g. CSVs
    # were absent on first call but have since been restored).
    if not APP_STATE.get("_demo_cache"):
        APP_STATE["_demo_cache"] = _load_demo_samples()

    cache = APP_STATE["_demo_cache"]
    if not cache:
        raise HTTPException(status_code=503, detail="No CSV data available for demo")

    # Pick a random attack type from cached samples
    attack_type = _rng.choice(list(cache.keys()))
    sample_rows = cache[attack_type]

    flows = [FlowRecord(features=row, log_type=0) for row in sample_rows]
    req = AnalyzeRequest(flows=flows, **{"model_config": "c"})
    return await analyze(req)


def _load_demo_samples() -> dict[str, list[list[float]]]:
    """
    Load real attack samples from CICIDS2017 CSVs for demo purposes.
    Caches 20-flow sessions for several attack types.
    """
    import glob as _glob
    import pandas as pd

    cfg = APP_STATE.get("config", {})
    data_dir = cfg.get("paths", {}).get("data_dir", "Kaggle Dataset")
    csv_files = sorted(_glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        logger.warning("No CSV files found in %s for demo samples", data_dir)
        return {}

    # Map CSV files to expected attack types
    attack_file_hints = {
        "PortScan": "PortScan",
        "DDos": "DDoS",
        "Morning": None,  # skip — mixed / benign
    }
    demo_cache: dict[str, list[list[float]]] = {}
    target_attacks = {"PortScan", "DDoS", "DoS Hulk", "DoS GoldenEye",
                      "Bot", "FTP-Patator", "SSH-Patator"}

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
            if " Label" not in df.columns:
                continue
            for attack_name in target_attacks:
                if attack_name in demo_cache:
                    continue
                mask = df[" Label"].astype(str).str.strip() == attack_name
                attack_rows = df[mask]
                if len(attack_rows) >= 20:
                    sample = attack_rows.head(20)
                    feat_rows = []
                    for _, row in sample.iterrows():
                        feats = []
                        for c in FEATURE_COLS:
                            v = row.get(c, 0)
                            try:
                                v = float(v)
                            except (ValueError, TypeError):
                                v = 0.0
                            if not np.isfinite(v):
                                v = 0.0
                            feats.append(round(v, 6))
                        feat_rows.append(feats)
                    demo_cache[attack_name] = feat_rows
                    logger.info("Cached %d %s flows for demo", len(feat_rows), attack_name)
        except Exception as e:
            logger.warning("Failed to load demo samples from %s: %s", csv_path, e)

    logger.info("Demo cache ready: %d attack types cached", len(demo_cache))
    return demo_cache


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log_type_dist(lt_array: np.ndarray) -> dict:
    from collections import Counter
    cnt = Counter(lt_array.tolist())
    return {LOG_TYPE_NAMES.get(k, str(k)): v for k, v in cnt.items()}


def _top_n(probs: list[float], n: int = 3) -> list[dict]:
    indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:n]
    return [
        {"attack_id": i, "attack_name": ID_TO_LABEL.get(i, str(i)),
         "probability": round(p, 4)}
        for i, p in indexed
    ]
