"""
Unified Alert Store — SQLite-backed persistence layer shared by both
detection paths:

  - src/api/routes.py        (REST `/api/analyze` endpoint)
  - scripts/live_demo.py     (WebSocket streaming engine)

Before this module existed, the REST path stored alerts in
``APP_STATE["alerts_history"]`` (in-memory list, lost on restart, only visible
to the FastAPI process) and the live streaming engine kept its own counters
inside ``LiveDemoServer.stats`` (only visible to WebSocket clients in that
process). The two paths could not see each other's alerts, and the dashboard
served by uvicorn would show zero detections even while live_demo.py was
streaming hundreds.

With the AlertStore, every detection — regardless of source — lands in
``data/alerts.db``. The REST `/api/alerts` and `/api/stats` endpoints query
the same DB and therefore reflect a unified view across processes.

Schema is intentionally minimal so it can run as the project moves into the
real-time SIEM phase without further migration.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join("data", "alerts.db")
DEFAULT_MAX_ALERTS = 10_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_uuid    TEXT UNIQUE,
    timestamp     REAL    NOT NULL,
    attack_id     INTEGER NOT NULL,
    attack_name   TEXT    NOT NULL,
    confidence    REAL    NOT NULL,
    is_attack     INTEGER NOT NULL,
    source        TEXT    NOT NULL,         -- 'rest' or 'stream'
    session_id    INTEGER,
    ground_truth  INTEGER,                  -- attack_id from labelled data, when known
    verdict       TEXT,                     -- 'TP' / 'FP' / 'TN' / 'FN' / NULL
    n_flows       INTEGER DEFAULT 0,
    raw           TEXT                      -- JSON blob with the full alert minus large RAG context
);
CREATE INDEX IF NOT EXISTS idx_alerts_ts     ON alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_source ON alerts(source);
CREATE INDEX IF NOT EXISTS idx_alerts_attack ON alerts(attack_name);
"""


def _classify_verdict(pred_attack: bool, gt_attack: Optional[bool]) -> Optional[str]:
    if gt_attack is None:
        return None
    if pred_attack and gt_attack:
        return "TP"
    if pred_attack and not gt_attack:
        return "FP"
    if not pred_attack and gt_attack:
        return "FN"
    return "TN"


class AlertStore:
    """
    SQLite-backed alert store. Thread-safe via a single ``threading.Lock``.

    SQLite is plenty fast for the project's expected rates (live demo runs
    ~10 sessions/second; SQLite handles 1000s of inserts/sec on local disk).
    Using stdlib only — no extra dependency.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        max_alerts: int = DEFAULT_MAX_ALERTS,
    ):
        self.db_path = db_path
        self.max_alerts = max_alerts

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        # check_same_thread=False because we serialise via self._lock; this lets
        # both the asyncio loop and any worker threads share one connection.
        # isolation_level=None → autocommit, simpler at our write rate.
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        logger.info("AlertStore ready at %s (cap=%d alerts)", db_path, max_alerts)

    # ── Write path ──────────────────────────────────────────────────────────

    def record(
        self,
        alert: dict,
        source: str = "rest",
        ground_truth_id: Optional[int] = None,
        session_id: Optional[int] = None,
    ) -> None:
        """
        Persist a single alert. Both the REST path (``source='rest'``) and the
        streaming path (``source='stream'``) call this. Failures are logged
        and swallowed — alert persistence must never break detection.
        """
        is_attack = bool(alert.get("is_attack", False))
        gt_attack = None if ground_truth_id is None else (ground_truth_id != 0)
        verdict = _classify_verdict(is_attack, gt_attack)

        # Drop the threat_intelligence blob from the JSON copy. Keeping it
        # would balloon the row size and the data is fully reconstructable
        # from `attack_id` via the RAG knowledge base.
        raw_copy = {k: v for k, v in alert.items() if k != "threat_intelligence"}

        with self._lock:
            try:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO alerts (
                        alert_uuid, timestamp, attack_id, attack_name,
                        confidence, is_attack, source, session_id,
                        ground_truth, verdict, n_flows, raw
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        alert.get("alert_id") or alert.get("alert_uuid"),
                        float(alert.get("timestamp", time.time())),
                        int(alert.get("attack_id", 0)),
                        str(alert.get("attack_name", "Unknown")),
                        float(alert.get("confidence", 0.0)),
                        1 if is_attack else 0,
                        source,
                        session_id,
                        ground_truth_id,
                        verdict,
                        int(alert.get("n_flows", 0)),
                        json.dumps(raw_copy, default=str),
                    ),
                )
                # Trim oldest rows beyond the cap. Cheap because the index keeps
                # the ORDER BY id DESC scan to a single B-tree walk.
                self._conn.execute(
                    """
                    DELETE FROM alerts
                    WHERE id IN (
                        SELECT id FROM alerts
                        ORDER BY id DESC
                        LIMIT -1 OFFSET ?
                    )
                    """,
                    (self.max_alerts,),
                )
            except Exception as e:
                logger.warning("AlertStore.record failed (non-fatal): %s", e)

    # ── Read path ───────────────────────────────────────────────────────────

    def recent(
        self,
        limit: int = 50,
        source: Optional[str] = None,
    ) -> list[dict]:
        """Return the most recent alerts (newest first)."""
        with self._lock:
            if source:
                rows = self._conn.execute(
                    """
                    SELECT alert_uuid, timestamp, attack_id, attack_name,
                           confidence, is_attack, source, session_id, verdict
                    FROM alerts
                    WHERE source = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (source, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT alert_uuid, timestamp, attack_id, attack_name,
                           confidence, is_attack, source, session_id, verdict
                    FROM alerts
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        """Aggregate counters across the entire store."""
        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*) FROM alerts"
            ).fetchone()[0]
            attacks = self._conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE is_attack = 1"
            ).fetchone()[0]
            attack_counts = self._conn.execute(
                """
                SELECT attack_name, COUNT(*) AS n
                FROM alerts
                WHERE is_attack = 1
                GROUP BY attack_name
                ORDER BY n DESC
                """
            ).fetchall()
            by_source = self._conn.execute(
                "SELECT source, COUNT(*) FROM alerts GROUP BY source"
            ).fetchall()
            verdicts = self._conn.execute(
                """
                SELECT verdict, COUNT(*)
                FROM alerts
                WHERE verdict IS NOT NULL
                GROUP BY verdict
                """
            ).fetchall()

        v = {row[0]: row[1] for row in verdicts}
        return {
            "total_sessions_analyzed": total,
            "total_attacks_detected":  attacks,
            "detection_rate":          round(attacks / max(total, 1), 4),
            "attack_type_counts":      {row[0]: row[1] for row in attack_counts},
            "top_attack_types":        [(row[0], row[1]) for row in attack_counts[:10]],
            "by_source":               {row[0]: row[1] for row in by_source},
            "tp": v.get("TP", 0),
            "fp": v.get("FP", 0),
            "tn": v.get("TN", 0),
            "fn": v.get("FN", 0),
        }

    def clear(self) -> None:
        """Wipe the alert table — used by tests."""
        with self._lock:
            self._conn.execute("DELETE FROM alerts")

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


# ── Module-level singleton ──────────────────────────────────────────────────
_instance: Optional[AlertStore] = None
_instance_lock = threading.Lock()


def get_alert_store(db_path: str = DEFAULT_DB_PATH) -> AlertStore:
    """
    Return the process-wide singleton AlertStore. The first call decides
    the DB path; subsequent calls ignore the argument.
    """
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = AlertStore(db_path=db_path)
    return _instance


def reset_alert_store() -> None:
    """Drop the singleton — primarily for tests / reconfiguration."""
    global _instance
    with _instance_lock:
        if _instance is not None:
            _instance.close()
        _instance = None
