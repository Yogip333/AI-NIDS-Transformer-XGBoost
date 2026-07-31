"""
Statistical Feature Engineering.

Computes per-session aggregate features that capture threat-indicative
statistical signatures (entropy, ratios, distribution patterns, etc.).
These features are combined with Transformer embeddings in the hybrid model.
"""
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy


# ── Column shortcuts ─────────────────────────────────────────────────────────

_DST_PORT   = " Destination Port"
_FLOW_DUR   = " Flow Duration"
_FWD_PKTS   = " Total Fwd Packets"
_BWD_PKTS   = " Total Backward Packets"
_FWD_BYTES  = "Total Length of Fwd Packets"
_BWD_BYTES  = " Total Length of Bwd Packets"
_FLOW_BPS   = "Flow Bytes/s"
_FLOW_PPS   = " Flow Packets/s"
_FLOW_IAT_M = " Flow IAT Mean"
_FIN_FLAG   = "FIN Flag Count"
_SYN_FLAG   = " SYN Flag Count"
_RST_FLAG   = " RST Flag Count"
_PSH_FLAG   = " PSH Flag Count"
_ACK_FLAG   = " ACK Flag Count"
_URG_FLAG   = " URG Flag Count"
_DOWN_UP    = " Down/Up Ratio"
_WIN_FWD    = "Init_Win_bytes_forward"
_WIN_BWD    = " Init_Win_bytes_backward"
_ACT_MEAN   = "Active Mean"
_IDLE_MEAN  = "Idle Mean"
_LOG_TYPE   = "log_type"

STAT_FEATURE_NAMES = [
    "stat_flow_count",
    "stat_total_fwd_bytes",
    "stat_total_bwd_bytes",
    "stat_avg_bytes_per_flow",
    "stat_std_bytes_per_flow",
    "stat_avg_duration",
    "stat_std_duration",
    "stat_avg_pkts_per_flow",
    "stat_std_pkts_per_flow",
    "stat_bytes_ratio",            # fwd / (fwd + bwd + eps)
    "stat_pkt_ratio",              # fwd_pkts / (total_pkts + eps)
    "stat_port_entropy",           # entropy of dst port distribution
    "stat_log_type_entropy",       # entropy of Zeek log type distribution
    "stat_syn_ratio",
    "stat_fin_ratio",
    "stat_rst_ratio",
    "stat_psh_ratio",
    "stat_urg_ratio",
    "stat_avg_flow_bps",
    "stat_max_flow_bps",
    "stat_avg_flow_iat",
    "stat_std_flow_iat",
    "stat_unique_dst_ports",
    "stat_short_flow_ratio",       # flows with duration < 0.1 s
    "stat_avg_win_size",           # mean TCP window
    "stat_idle_active_ratio",
]

NUM_STAT_FEATURES = len(STAT_FEATURE_NAMES)


def _safe_entropy(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    if counts.sum() == 0:
        return 0.0
    probs = counts / counts.sum()
    return float(scipy_entropy(probs, base=2))


def compute_session_stats(
    session_features: np.ndarray,
    feature_col_names: list[str],
) -> np.ndarray:
    """
    Given a (session_size, num_features) float32 array, compute the
    26 statistical threat-indicative features for that session.
    """
    def _col(name: str) -> np.ndarray:
        if name in feature_col_names:
            return session_features[:, feature_col_names.index(name)]
        return np.zeros(len(session_features), dtype=np.float32)

    n = len(session_features)

    fwd_bytes  = _col(_FWD_BYTES)
    bwd_bytes  = _col(_BWD_BYTES)
    fwd_pkts   = _col(_FWD_PKTS)
    bwd_pkts   = _col(_BWD_PKTS)
    duration   = _col(_FLOW_DUR)
    bps        = _col(_FLOW_BPS)
    iat_mean   = _col(_FLOW_IAT_M)
    dst_ports  = _col(_DST_PORT)
    log_types  = _col(_LOG_TYPE)
    fin_flags  = _col(_FIN_FLAG)
    syn_flags  = _col(_SYN_FLAG)
    rst_flags  = _col(_RST_FLAG)
    psh_flags  = _col(_PSH_FLAG)
    urg_flags  = _col(_URG_FLAG)
    win_fwd    = _col(_WIN_FWD)
    win_bwd    = _col(_WIN_BWD)
    active_m   = _col(_ACT_MEAN)
    idle_m     = _col(_IDLE_MEAN)

    total_bytes = fwd_bytes + bwd_bytes
    total_pkts  = fwd_pkts + bwd_pkts
    total_flags = fin_flags + syn_flags + rst_flags + psh_flags + urg_flags + 1e-9

    stats = np.zeros(NUM_STAT_FEATURES, dtype=np.float32)
    stats[0]  = float(n)
    stats[1]  = float(np.sum(fwd_bytes))
    stats[2]  = float(np.sum(bwd_bytes))
    stats[3]  = float(np.mean(total_bytes))
    stats[4]  = float(np.std(total_bytes) + 1e-9)
    stats[5]  = float(np.mean(duration))
    stats[6]  = float(np.std(duration) + 1e-9)
    stats[7]  = float(np.mean(total_pkts))
    stats[8]  = float(np.std(total_pkts) + 1e-9)
    stats[9]  = float(np.sum(fwd_bytes) / (np.sum(total_bytes) + 1e-9))
    stats[10] = float(np.sum(fwd_pkts) / (np.sum(total_pkts) + 1e-9))
    stats[11] = _safe_entropy(dst_ports.astype(int))
    stats[12] = _safe_entropy(log_types.astype(int))
    stats[13] = float(np.sum(syn_flags) / total_flags.sum())
    stats[14] = float(np.sum(fin_flags) / total_flags.sum())
    stats[15] = float(np.sum(rst_flags) / total_flags.sum())
    stats[16] = float(np.sum(psh_flags) / total_flags.sum())
    stats[17] = float(np.sum(urg_flags) / total_flags.sum())
    stats[18] = float(np.mean(np.clip(bps, 0, 1e8)))
    stats[19] = float(np.max(np.clip(bps, 0, 1e8)))
    stats[20] = float(np.mean(np.clip(iat_mean, 0, 1e9)))
    stats[21] = float(np.std(np.clip(iat_mean, 0, 1e9)) + 1e-9)
    stats[22] = float(len(np.unique(dst_ports.astype(int))))
    stats[23] = float(np.mean(duration < 100_000))  # < 0.1 second (μs units)
    avg_win = (win_fwd + win_bwd) / 2.0
    stats[24] = float(np.mean(avg_win))
    total_act  = np.sum(active_m) + 1e-9
    total_idle = np.sum(idle_m)
    stats[25] = float(total_idle / (total_act + total_idle))

    # Clip and replace any residual inf/nan
    stats = np.nan_to_num(stats, nan=0.0, posinf=1e6, neginf=-1e6)
    return stats


def compute_all_session_stats(
    sessions: list[dict],
    feature_col_names: list[str],
) -> np.ndarray:
    """
    Compute statistical features for all sessions.

    Returns
    -------
    np.ndarray : shape (num_sessions, NUM_STAT_FEATURES)
    """
    all_stats = []
    for s in sessions:
        stat = compute_session_stats(s["features"], feature_col_names)
        all_stats.append(stat)
    return np.vstack(all_stats).astype(np.float32)


class FeatureScaler:
    """Fit-transform / transform wrapper around sklearn StandardScaler."""

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        self._scaler.fit(X)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._scaler.transform(X).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self._scaler.fit_transform(X).astype(np.float32)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self._scaler, path)

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        import joblib
        obj = cls()
        obj._scaler = joblib.load(path)
        obj._fitted = True
        return obj


def scale_flow_features(
    sessions: list[dict],
    feature_col_names: list[str],
    scaler: FeatureScaler | None = None,
    fit: bool = True,
) -> tuple[list[dict], FeatureScaler]:
    """
    Normalise the per-flow feature arrays inside every session dict in-place.

    Returns the (modified) sessions list and the fitted FeatureScaler.
    """
    # Collect all flow arrays to fit scaler
    all_arrays = np.vstack([s["features"] for s in sessions])

    if scaler is None:
        scaler = FeatureScaler()
    if fit:
        scaler.fit(all_arrays)

    for s in sessions:
        s["features"] = scaler.transform(s["features"])

    return sessions, scaler
