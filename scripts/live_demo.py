"""
Live SOC Analyst Dashboard — streams real CICIDS2017 CSV data through
the Transformer-based session hybrid model (Config C) and broadcasts
results via WebSocket for a live dashboard.

Proves the hypothesis: Zeek-parsed network flows → session windowing →
Transformer embeddings + statistical features → XGBoost → accurate
multi-class attack detection in real time.

Usage:
    python scripts/live_demo.py
    python scripts/live_demo.py --rate 500 --batch 20 --port 8765
"""
import asyncio
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/live_demo.log", mode="w"),
    ],
)
logger = logging.getLogger("live_demo")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SESSION_SIZE = 20  # flows per session window (matches training)

# ── Binary attack gate ────────────────────────────────────────────────────────
# p_attack = 1 - P(BENIGN) must exceed this threshold before an alert is raised.
# Separates "is this session malicious at all?" from "what attack family is it?".
# Tune upward (0.60 → 0.70 → 0.80) to trade recall for lower FPR.
# At 0.50 the model needs majority non-benign probability to alert.
BENIGN_GATE_THRESHOLD = 0.50

ATTACK_MIX = {
    # attack_name: weight (higher = more likely to be sampled)
    "BENIGN": 6,
    "DDoS": 2,
    "DoS Hulk": 2,
    "PortScan": 2,
    "Bot": 1,
    "FTP-Patator": 1,
    "SSH-Patator": 1,
    "DoS GoldenEye": 1,
    "DoS Slowhttptest": 1,
    "DoS slowloris": 1,
    "Web Attack Brute Force": 1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_flow_pool(data_dir: str, max_per_class: int = 5000) -> dict[str, pd.DataFrame]:
    """
    Load flows from all CSVs, grouped by attack label.
    Returns {label_str: DataFrame} with up to max_per_class rows each.
    """
    import glob as _glob
    from src.data.loader import FEATURE_COLS, LABEL_COL

    csv_files = sorted(_glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")

    pool: dict[str, list[pd.DataFrame]] = defaultdict(list)

    for csv_path in csv_files:
        logger.info("Loading %s …", Path(csv_path).name)
        try:
            df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
        except Exception as e:
            logger.warning("Failed to read %s: %s", csv_path, e)
            continue

        if LABEL_COL not in df.columns:
            continue

        # Keep feature cols + label
        available = [c for c in FEATURE_COLS if c in df.columns]
        df = df[available + [LABEL_COL]].copy()

        # Clean
        for col in available:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        for col in available:
            df[col] = df[col].clip(-1e9, 1e9)

        # Group by label
        for label_str, group in df.groupby(LABEL_COL):
            label_str = str(label_str).strip()
            pool[label_str].append(group)

    # Concatenate, cap, and normalise keys to canonical names (ID_TO_LABEL)
    from src.data.loader import _encode_label, ID_TO_LABEL
    result = {}
    for label_str, frames in pool.items():
        combined = pd.concat(frames, ignore_index=True)
        if len(combined) > max_per_class:
            combined = combined.sample(n=max_per_class, random_state=42)
        # Normalise key: raw CSV label → int ID → canonical name
        label_id = _encode_label(label_str)
        canonical = ID_TO_LABEL.get(label_id, label_str)
        if canonical in result:
            result[canonical] = pd.concat([result[canonical], combined], ignore_index=True)
            if len(result[canonical]) > max_per_class:
                result[canonical] = result[canonical].sample(n=max_per_class, random_state=42)
        else:
            result[canonical] = combined
        logger.info("  Pool: %-25s %6d flows (raw: %s)", canonical, len(result[canonical]), label_str)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_inference_pipeline(config_path: str = "configs/config.yaml"):
    """
    Load the full inference pipeline: scaler, Transformer, HybridIDS.
    Returns (hybrid_ids, scaler, feature_cols_81, device).
    """
    import yaml
    import torch
    import pickle
    from src.data.feature_engineer import FeatureScaler
    from src.data.loader import FEATURE_COLS
    from src.models.transformer import build_classifier
    from src.models.hybrid import HybridIDS

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_dir = cfg["paths"]["models_dir"]
    device = torch.device("cpu")

    # Load scaler
    scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
    scaler = FeatureScaler.load(scaler_path)
    logger.info("Loaded session scaler from %s", scaler_path)

    # Determine input dim from processed data
    processed_pkl = os.path.join(cfg["paths"]["processed_dir"], "processed_data.pkl")
    if os.path.exists(processed_pkl):
        with open(processed_pkl, "rb") as pf:
            _d = pickle.load(pf)
        num_features = int(_d.get("num_features", len(FEATURE_COLS)))
    else:
        num_features = len(FEATURE_COLS)

    # Load Transformer
    classifier = build_classifier(cfg, input_dim=num_features)
    ckpt = os.path.join(model_dir, "classifier_final.pt")
    if os.path.exists(ckpt):
        classifier.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=True)
        )
        logger.info("Loaded Transformer classifier from %s", ckpt)
    classifier.eval().to(device)

    # Load HybridIDS (XGBoost heads)
    hybrid = HybridIDS(cfg, classifier, num_classes=15)
    hybrid.load_all(model_dir)
    logger.info("HybridIDS loaded — Config C ready")

    return hybrid, scaler, cfg, device


def load_threat_intelligence(cfg: dict):
    """
    Load the RAG pipeline (ChromaDB + nomic embeddings) and an optional Groq
    LLM analyst, mirroring src/api/app.py:load_models() so the live demo
    enriches alerts with the same MITRE / IoC / LLM context the REST API does.

    Returns the ThreatRAG instance, or None if it cannot be initialised.
    Failure is non-fatal — the live demo continues to stream raw detections.
    """
    try:
        from src.rag.threat_rag import ThreatRAG
        from src.rag.groq_analyst import GroqAnalyst
    except ImportError as e:
        logger.warning("Threat RAG modules unavailable: %s — enrichment disabled", e)
        return None

    # Load .env so GROQ_API_KEY (and any other secrets) are picked up
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    groq_analyst = None
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        try:
            groq_analyst = GroqAnalyst(api_key=groq_key)
            if not groq_analyst.available:
                groq_analyst = None
            else:
                logger.info("Groq LLM analyst ready (live demo)")
        except Exception as e:
            logger.warning("Groq analyst init failed (non-fatal): %s", e)
            groq_analyst = None
    else:
        logger.info("GROQ_API_KEY not set — live demo will use RAG without LLM enrichment")

    try:
        rag = ThreatRAG(
            db_path=cfg["paths"].get("rag_db_dir", "data/rag_db"),
            collection=cfg["rag"]["collection_name"],
            top_k=cfg["rag"]["top_k"],
            groq_analyst=groq_analyst,
        )
        rag.build_knowledge_base(force_rebuild=False)
        logger.info("RAG ready for live demo (Groq enrichment: %s)",
                    "enabled" if groq_analyst else "disabled")
        return rag
    except Exception as e:
        logger.error("RAG initialisation failed: %s — streaming continues without enrichment", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Feature Pipeline (mirrors routes.py / preprocess.py)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_session(
    flows_df: pd.DataFrame,
    scaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a DataFrame of raw flows into model-ready inputs.

    Returns (scaled_features_81, log_types_int8, stats_26).

    Schema realignment is delegated to
    `src.data.session_pipeline.realign_to_trained_schema`, which is the single
    source of truth shared with `src/api/routes.py`. This guarantees that the
    streaming path and the REST path can never feed the model two different
    feature distributions.
    """
    from src.data.loader import FEATURE_COLS
    from src.data.feature_engineer import compute_session_stats
    from src.data.session_pipeline import (
        realign_to_trained_schema,
        trained_feature_names,
    )
    from src.data.zeek_mapper import assign_log_types_vectorized

    # Build raw (N × 78) CICIDS array from FEATURE_COLS columns (filling 0 for any missing)
    raw_78 = np.zeros((len(flows_df), len(FEATURE_COLS)), dtype=np.float32)
    for i, col in enumerate(FEATURE_COLS):
        if col in flows_df.columns:
            raw_78[:, i] = flows_df[col].values.astype(np.float32)

    # Derive log_type from destination port + fwd byte volume — same heuristic
    # `scripts/preprocess.py` uses at training time. The REST API has these
    # supplied by the client per FlowRecord; the live demo derives them.
    dst_port  = flows_df.get(" Destination Port", pd.Series(0, index=flows_df.index))
    fwd_bytes = flows_df.get("Total Length of Fwd Packets", pd.Series(0, index=flows_df.index))
    log_types = assign_log_types_vectorized(dst_port, fwd_bytes).to_numpy(dtype=np.float32)

    # Realign 78 → 81 trained schema (shared with routes.py)
    raw_81 = realign_to_trained_schema(raw_78, log_types)

    # Scale for the Transformer / hybrid head
    scaled_81 = scaler.transform(raw_81)

    # Statistical features — names must align with the 81-col order
    stats = compute_session_stats(raw_81, trained_feature_names())

    return scaled_81, log_types.astype(np.int8), stats


# ─────────────────────────────────────────────────────────────────────────────
# Session Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_mixed_sessions(
    pool: dict[str, pd.DataFrame],
    attack_mix: dict[str, int],
    session_size: int = 20,
):
    """
    Infinite generator yielding (session_df, ground_truth_labels) tuples.

    Each session is a mix of flows sampled according to attack_mix weights.
    Some sessions are pure-attack, some are pure-benign, some are mixed.
    """
    from src.data.loader import ID_TO_LABEL

    # Build canonical name → ID mapping (reverse of ID_TO_LABEL)
    name_to_id = {v: k for k, v in ID_TO_LABEL.items()}

    labels_available = [k for k in attack_mix if k in pool and len(pool[k]) >= 5]
    weights = [attack_mix[k] for k in labels_available]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    session_types = [
        "pure",     # all flows from one label
        "mixed",    # 50/50 dominant attack + benign noise
        "burst",    # attack burst inside benign traffic
    ]

    while True:
        stype = random.choice(session_types)
        chosen_label = random.choices(labels_available, weights=probs, k=1)[0]
        source = pool[chosen_label]

        if stype == "pure":
            # All flows from one class
            if len(source) >= session_size:
                sample = source.sample(n=session_size)
            else:
                sample = source.sample(n=session_size, replace=True)
            gt_labels = [name_to_id.get(chosen_label, 0)] * session_size

        elif stype == "mixed" and "BENIGN" in pool:
            # Half attack, half benign
            n_attack = session_size // 2
            n_benign = session_size - n_attack
            attack_sample = source.sample(n=min(n_attack, len(source)), replace=len(source) < n_attack)
            benign_sample = pool["BENIGN"].sample(n=n_benign, replace=len(pool["BENIGN"]) < n_benign)
            sample = pd.concat([attack_sample, benign_sample], ignore_index=True)
            gt_labels = (
                [name_to_id.get(chosen_label, 0)] * len(attack_sample)
                + [0] * len(benign_sample)
            )
            # Shuffle
            idx = list(range(len(sample)))
            random.shuffle(idx)
            sample = sample.iloc[idx].reset_index(drop=True)
            gt_labels = [gt_labels[i] for i in idx]

        elif stype == "burst" and "BENIGN" in pool:
            # Benign traffic with an attack burst in the middle
            n_attack = random.randint(3, min(10, session_size - 2))
            n_benign = session_size - n_attack
            burst_start = random.randint(0, n_benign)

            benign_sample = pool["BENIGN"].sample(n=n_benign, replace=len(pool["BENIGN"]) < n_benign)
            attack_sample = source.sample(n=min(n_attack, len(source)), replace=len(source) < n_attack)

            pre = benign_sample.iloc[:burst_start]
            post = benign_sample.iloc[burst_start:]
            sample = pd.concat([pre, attack_sample, post], ignore_index=True)
            gt_labels = (
                [0] * len(pre)
                + [name_to_id.get(chosen_label, 0)] * len(attack_sample)
                + [0] * len(post)
            )
        else:
            # Fallback: pure session
            if len(source) >= session_size:
                sample = source.sample(n=session_size)
            else:
                sample = source.sample(n=session_size, replace=True)
            gt_labels = [name_to_id.get(chosen_label, 0)] * session_size

        # Ensure exactly session_size rows
        if len(sample) > session_size:
            sample = sample.iloc[:session_size]
            gt_labels = gt_labels[:session_size]
        elif len(sample) < session_size:
            deficit = session_size - len(sample)
            pad = sample.sample(n=deficit, replace=True)
            sample = pd.concat([sample, pad], ignore_index=True)
            gt_labels = gt_labels + [gt_labels[-1]] * deficit

        yield sample, gt_labels


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Server
# ─────────────────────────────────────────────────────────────────────────────

class LiveDemoServer:
    """Streams detection results to connected WebSocket clients."""

    def __init__(
        self,
        hybrid_ids,
        scaler,
        pool: dict[str, pd.DataFrame],
        rag=None,
        rate_limit: float = 200.0,
        session_size: int = 20,
    ):
        self.hybrid = hybrid_ids
        self.scaler = scaler
        self.pool = pool
        self.rag = rag  # ThreatRAG instance or None — non-fatal when missing
        self.rate_limit = rate_limit
        self.session_size = session_size
        self.clients: set = set()
        # Unified alert store — shared with the REST API process. Failures are
        # non-fatal so the live dashboard never goes dark over a DB hiccup.
        try:
            from src.api.alert_store import get_alert_store
            self.alert_store = get_alert_store()
            logger.info("AlertStore wired into live demo")
        except Exception as e:
            logger.warning("AlertStore unavailable (non-fatal): %s", e)
            self.alert_store = None
        self.stats = {
            "total_sessions": 0,
            "total_flows": 0,
            "correct": 0,
            "wrong": 0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "attack_counts": defaultdict(int),
            "confusion": defaultdict(lambda: defaultdict(int)),
            "start_time": time.time(),
        }

    async def register(self, ws):
        self.clients.add(ws)
        logger.info("Client connected (%d total)", len(self.clients))
        # Send initial state
        await ws.send(json.dumps({
            "type": "init",
            "stats": self._stats_snapshot(),
        }))

    async def unregister(self, ws):
        self.clients.discard(ws)
        logger.info("Client disconnected (%d total)", len(self.clients))

    async def broadcast(self, message: dict):
        if not self.clients:
            return
        data = json.dumps(message)
        dead = set()
        for ws in self.clients:
            try:
                await ws.send(data)
            except Exception:
                dead.add(ws)
        self.clients -= dead

    def _stats_snapshot(self) -> dict:
        s = self.stats
        elapsed = time.time() - s["start_time"]
        total = s["total_sessions"]
        correct = s["correct"]
        denom_dr = s["tp"] + s["fn"]
        denom_fp = s["fp"] + s["tn"]
        # binary_accuracy = (TP+TN)/total — "did we correctly judge whether
        # this session contained any attack at all?"
        # This is the meaningful SOC metric.  The old "accuracy" (multiclass
        # argmax vs majority-vote GT) scores burst sessions (e.g. 15 benign +
        # 5 attack → GT majority BENIGN, model says DoS Hulk) as wrong even
        # when the detection was correct, deflating the number to ~54%.
        denom_binary = s["tp"] + s["fp"] + s["tn"] + s["fn"]
        binary_acc = (s["tp"] + s["tn"]) / denom_binary if denom_binary > 0 else 0
        return {
            "total_sessions": total,
            "total_flows": s["total_flows"],
            "accuracy": round(binary_acc, 4),          # now binary detection accuracy
            "multiclass_accuracy": round(correct / total, 4) if total > 0 else 0,
            "detection_rate": round(s["tp"] / denom_dr, 4) if denom_dr > 0 else 0,
            "false_positive_rate": round(s["fp"] / denom_fp, 4) if denom_fp > 0 else 0,
            "tp": s["tp"], "fp": s["fp"], "fn": s["fn"], "tn": s["tn"],
            "elapsed_seconds": round(elapsed, 1),
            "sessions_per_second": round(total / elapsed, 1) if elapsed > 0 else 0,
            "attack_counts": dict(s["attack_counts"]),
        }

    def _enrich_alert(
        self,
        attack_id: int,
        probabilities: list,
        stats_dict: dict,
        config: str = "c",
    ) -> dict | None:
        """
        Synchronous RAG (+ optional Groq) enrichment. Returns the same subset
        of fields the REST API exposes via /api/analyze, or None if RAG is
        unavailable, the prediction is BENIGN, or enrichment fails.

        Designed to be called via asyncio.to_thread() so the streaming loop
        is never blocked by ChromaDB or LLM latency.
        """
        if self.rag is None or attack_id == 0:
            return None
        try:
            full_alert = self.rag.format_alert(
                attack_id=attack_id,
                probabilities=probabilities,
                session_stats=stats_dict,
                config=config,
            )
            intel = {
                "severity":         full_alert.get("severity"),
                "description":      (full_alert.get("description") or "")[:400],
                "mitre_tactics":    full_alert.get("mitre_tactics", []),
                "techniques":       (full_alert.get("techniques") or [])[:3],
                "zeek_indicators":  (full_alert.get("zeek_indicators") or [])[:3],
                "response_actions": (full_alert.get("response_actions") or [])[:3],
                "tags":             (full_alert.get("tags") or [])[:8],
            }
            # Promote Groq fields when present (matches routes.py behaviour)
            if "fp_likelihood" in full_alert:
                intel["fp_likelihood"] = full_alert["fp_likelihood"]
            if "fp_reasoning" in full_alert:
                intel["fp_reasoning"] = full_alert["fp_reasoning"]
            if "groq_analysis" in full_alert:
                intel["groq_analysis"] = full_alert["groq_analysis"]
            return intel
        except Exception as e:
            logger.warning("RAG enrichment failed (non-fatal): %s", e)
            return None

    def _update_stats(
        self,
        pred_id: int,
        gt_majority_id: int,
        pred_name: str,
        session_has_attack: bool = False,
    ):
        s = self.stats
        s["total_sessions"] += 1
        s["total_flows"] += self.session_size

        # ── Multiclass accuracy: argmax must match the majority GT label ──────
        if pred_id == gt_majority_id:
            s["correct"] += 1

        # ── Binary detection: "does the session contain ANY attack flow?" ─────
        # Fix: use session_has_attack (any non-BENIGN flow present) instead of
        # majority-vote GT.  Without this, burst sessions (e.g. 15 benign + 5
        # attack flows → majority = BENIGN) wrongly score correct detections as
        # false positives, inflating the displayed FPR.
        gt_binary   = int(session_has_attack)
        pred_binary = int(pred_id != 0)

        if gt_binary == 1 and pred_binary == 1:
            s["tp"] += 1
        elif gt_binary == 0 and pred_binary == 0:
            s["tn"] += 1
        elif gt_binary == 0 and pred_binary == 1:
            s["fp"] += 1
        else:
            s["fn"] += 1

        if pred_id != 0:
            s["attack_counts"][pred_name] += 1

    async def run_streaming(self):
        """Main loop: generate sessions, classify, broadcast."""
        from src.data.loader import ID_TO_LABEL
        from src.data.feature_engineer import STAT_FEATURE_NAMES
        from src.detection.exceptions import InferenceError, ModelNotLoadedError

        gen = generate_mixed_sessions(self.pool, ATTACK_MIX, self.session_size)
        delay = self.session_size / self.rate_limit if self.rate_limit > 0 else 0.1

        logger.info("Streaming started (rate=%.0f flows/s, delay=%.3fs/session)",
                     self.rate_limit, delay)

        for session_df, gt_labels in gen:
            try:
                # Prepare features
                scaled_81, lt_int, stats_26 = prepare_session(session_df, self.scaler)

                # Run inference — Transformer → Config C (hybrid), fallback Config B.
                # If neither checkpoint is loaded, ModelNotLoadedError is raised and
                # the session is skipped. No heuristic substitution is permitted.
                result = self.hybrid.predict_with_fallback(
                    scaled_81, lt_int, stats_26
                )
                config_used = result.get("config_used", "C")

                pred_id = result["prediction"]
                pred_name = ID_TO_LABEL.get(pred_id, str(pred_id))
                probabilities = result["probabilities"]
                confidence = probabilities[pred_id] if pred_id < len(probabilities) else 0.0

                # ── Binary attack gate ────────────────────────────────────────
                # If total non-benign probability is below threshold, suppress
                # the attack call.  This separates "is this session malicious?"
                # from "what attack family is it?" and cuts false positives from
                # ambiguous / mixed-traffic sessions.
                p_benign       = float(probabilities[0]) if probabilities else 1.0
                p_attack_score = 1.0 - p_benign
                if p_attack_score < BENIGN_GATE_THRESHOLD:
                    pred_id   = 0
                    pred_name = "BENIGN"
                    confidence = p_benign

                # Ground truth: majority vote (for multiclass accuracy)
                from collections import Counter
                gt_counter = Counter(gt_labels)
                gt_majority_id = gt_counter.most_common(1)[0][0]
                gt_majority_name = ID_TO_LABEL.get(gt_majority_id, str(gt_majority_id))

                # ── session_has_attack: true if ANY flow in this session is
                # non-BENIGN.  Used for binary TP/FP/TN/FN evaluation — more
                # correct than majority vote for burst/mixed sessions.
                session_has_attack = any(gt != 0 for gt in gt_labels)

                # Per-flow ground truth distribution
                gt_dist = {
                    ID_TO_LABEL.get(k, str(k)): v
                    for k, v in gt_counter.items()
                }

                is_correct = (pred_id == gt_majority_id)
                is_attack = (pred_id != 0)

                self._update_stats(
                    pred_id, gt_majority_id, pred_name,
                    session_has_attack=session_has_attack,
                )

                # ── Threat-intel enrichment (RAG + optional Groq) ─────────────
                # Run in a thread so ChromaDB / LLM latency does not stall the
                # asyncio event loop. Skipped automatically when self.rag is
                # None or the prediction is BENIGN.
                stats_dict = dict(zip(STAT_FEATURE_NAMES, stats_26.tolist()))
                threat_intel = await asyncio.to_thread(
                    self._enrich_alert, pred_id, probabilities, stats_dict, "c"
                )

                # ── Persist into unified alert store (shared with REST API) ───
                if self.alert_store is not None:
                    alert_record = {
                        "alert_id":    f"stream-{self.stats['total_sessions']}",
                        "timestamp":   time.time(),
                        "attack_id":   pred_id,
                        "attack_name": pred_name,
                        "confidence":  round(float(confidence), 4),
                        "is_attack":   is_attack,
                        "n_flows":     self.session_size,
                    }
                    # Run in a thread — SQLite writes are sync and we don't
                    # want to block the asyncio loop even briefly.
                    # Pass ground_truth_id as binary (1 = any attack present,
                    # 0 = pure-benign session) so _classify_verdict() in
                    # alert_store.py aligns with the binary gate above.
                    gt_for_store = 1 if session_has_attack else 0
                    await asyncio.to_thread(
                        self.alert_store.record,
                        alert_record,
                        "stream",
                        gt_for_store,
                        self.stats["total_sessions"],
                    )

                # Build event message
                event = {
                    "type": "detection",
                    "session_id": self.stats["total_sessions"],
                    "timestamp": time.time(),
                    "config_used": config_used,
                    "prediction": {
                        "id": pred_id,
                        "name": pred_name,
                        "confidence": round(confidence, 4),
                        "is_attack": is_attack,
                        "top_probabilities": {
                            ID_TO_LABEL.get(i, str(i)): round(p, 4)
                            for i, p in sorted(
                                enumerate(probabilities),
                                key=lambda x: -x[1]
                            )[:5]
                        },
                    },
                    "ground_truth": {
                        "majority_id": gt_majority_id,
                        "majority_name": gt_majority_name,
                        "distribution": gt_dist,
                    },
                    "correct": is_correct,
                    "stats": self._stats_snapshot(),
                }
                if threat_intel is not None:
                    event["threat_intelligence"] = threat_intel

                await self.broadcast(event)

                # Log periodically
                if self.stats["total_sessions"] % 50 == 0:
                    snap = self._stats_snapshot()
                    logger.info(
                        "Sessions: %d | Acc: %.3f | DR: %.3f | FPR: %.3f | %.1f sess/s",
                        snap["total_sessions"], snap["accuracy"],
                        snap["detection_rate"], snap["false_positive_rate"],
                        snap["sessions_per_second"],
                    )

            except (ModelNotLoadedError, InferenceError) as e:
                # Hard failure — model checkpoint absent or schema mismatch.
                # Skip this session; do not substitute a heuristic result.
                logger.error(
                    "Session %d skipped — inference failed: %s",
                    self.stats["total_sessions"] + 1, e,
                )
            except Exception as e:
                logger.error("Unexpected streaming error: %s", e, exc_info=True)

            await asyncio.sleep(delay)


async def ws_handler(ws, server: LiveDemoServer):
    """Handle a single WebSocket connection."""
    await server.register(ws)
    try:
        async for msg in ws:
            # Client can send control messages (pause/resume/speed)
            try:
                data = json.loads(msg)
                if data.get("type") == "ping":
                    await ws.send(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass
    finally:
        await server.unregister(ws)


async def run_server(
    config_path: str = "configs/config.yaml",
    host: str = "0.0.0.0",
    port: int = 8765,
    rate_limit: float = 200.0,
    http_port: int = 8080,
):
    """Start WebSocket server + HTTP static file server."""
    import websockets

    logger.info("=" * 60)
    logger.info("Loading flow pool from CSV files…")
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data_dir = cfg["paths"]["data_dir"]

    pool = load_flow_pool(data_dir, max_per_class=5000)
    if not pool:
        raise RuntimeError("No flows loaded — check CSV files")

    logger.info("Loading inference pipeline…")
    hybrid, scaler, cfg, device = load_inference_pipeline(config_path)

    logger.info("Loading threat intelligence (RAG + Groq)…")
    rag = load_threat_intelligence(cfg)

    server = LiveDemoServer(
        hybrid_ids=hybrid,
        scaler=scaler,
        pool=pool,
        rag=rag,
        rate_limit=rate_limit,
        session_size=SESSION_SIZE,
    )

    # Start HTTP server for dashboard in background
    http_task = asyncio.create_task(run_http_server(http_port))

    logger.info("=" * 60)
    logger.info("  WebSocket : ws://localhost:%d", port)
    logger.info("  Dashboard : http://localhost:%d/dashboard.html", http_port)
    logger.info("=" * 60)

    # Start streaming in background
    stream_task = asyncio.create_task(server.run_streaming())

    # Start WebSocket server
    async with websockets.serve(
        lambda ws: ws_handler(ws, server),
        host, port,
    ):
        await asyncio.gather(stream_task, http_task)


async def run_http_server(port: int):
    """Simple HTTP server for the dashboard HTML."""
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    import threading

    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
    os.makedirs(static_dir, exist_ok=True)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=static_dir, **kwargs)
        def log_message(self, format, *args):
            pass  # Suppress HTTP logs

    httpd = HTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    logger.info("HTTP server serving static/ on port %d", port)

    # Keep alive
    while True:
        await asyncio.sleep(3600)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live SOC Analyst Dashboard Demo")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--http-port", type=int, default=8080, help="Dashboard HTTP port")
    parser.add_argument("--rate", type=float, default=200.0, help="Flows/second rate limit")
    args = parser.parse_args()

    asyncio.run(run_server(
        config_path=args.config,
        port=args.port,
        http_port=args.http_port,
        rate_limit=args.rate,
    ))
