"""
Zeek Log Simulator.

Maps CICIDS2017 flow-level CSV features into structured records that mirror
the five Zeek telemetry log types:
  0 → conn.log   (default connection record)
  1 → dns.log    (port 53)
  2 → http.log   (ports 80/8080/8000)
  3 → ssl.log    (ports 443/8443)
  4 → files.log  (FTP/SSH/SMTP and large-data flows)

Each row keeps ALL original numerical features plus a synthetic 'log_type'
integer so the Transformer can learn cross-protocol context.
"""
import numpy as np
import pandas as pd

# ── Port → Log-type mapping ──────────────────────────────────────────────────
_DNS_PORTS   = {53}
_HTTP_PORTS  = {80, 8080, 8000, 3000}
_SSL_PORTS   = {443, 8443, 4443}
_FILES_PORTS = {21, 22, 25, 143, 993, 995, 3306, 5432}

LOG_TYPE_NAMES = {0: "conn", 1: "dns", 2: "http", 3: "ssl", 4: "files"}

# Byte threshold for files.log classification (flows with > 1 MB)
_FILES_BYTE_THRESHOLD = 1_000_000


def assign_log_type(dst_port: float, fwd_bytes: float = 0.0) -> int:
    """Determine which Zeek log type a flow belongs to."""
    port = int(dst_port) if not np.isnan(dst_port) else 0
    if port in _DNS_PORTS:
        return 1
    if port in _HTTP_PORTS:
        return 2
    if port in _SSL_PORTS:
        return 3
    if port in _FILES_PORTS or fwd_bytes > _FILES_BYTE_THRESHOLD:
        return 4
    return 0  # conn.log (default)


def assign_log_types_vectorized(
    dst_ports: "pd.Series",
    fwd_bytes: "pd.Series",
) -> "pd.Series":
    """Vectorised version for an entire DataFrame column."""
    ports = dst_ports.fillna(0).astype(int)
    is_dns   = ports.isin(_DNS_PORTS)
    is_http  = ports.isin(_HTTP_PORTS)
    is_ssl   = ports.isin(_SSL_PORTS)
    is_files = ports.isin(_FILES_PORTS) | (fwd_bytes > _FILES_BYTE_THRESHOLD)

    log_types = pd.Series(0, index=dst_ports.index, dtype=np.int8)
    log_types[is_files] = 4
    log_types[is_ssl]   = 3
    log_types[is_http]  = 2
    log_types[is_dns]   = 1  # highest priority wins (dns overwrites ssl if dual)
    return log_types


# ── Field-level synthetic Zeek attributes ───────────────────────────────────

def _infer_protocol(fin_cnt: float, syn_cnt: float) -> str:
    """Heuristic: flows with SYN or FIN flags are TCP, others UDP."""
    return "tcp" if (fin_cnt + syn_cnt) > 0 else "udp"


def _infer_conn_state(
    syn: float, fin: float, rst: float, fwd_pkts: float, bwd_pkts: float
) -> str:
    """Map flag counts to a Zeek conn_state token (simplified)."""
    if rst > 0:
        return "RSTO"
    if syn > 0 and fin > 0:
        return "SF"
    if syn > 0 and bwd_pkts == 0:
        return "S0"
    if fwd_pkts > 0 and bwd_pkts > 0:
        return "SF"
    return "OTH"


def map_to_zeek_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the DataFrame with synthetic Zeek-like metadata columns:
      - log_type          (int 0-4)
      - zeek_proto        (str: tcp/udp)
      - zeek_conn_state   (str: SF/S0/RSTO/OTH)
      - zeek_service      (str: dns/http/ssl/ftp/ssh/-)

    The original 76 numerical features are preserved unchanged.
    """
    out = df.copy()

    # Log type
    fwd_bytes = out.get("Total Length of Fwd Packets", pd.Series(0, index=out.index))
    dst_port  = out.get(" Destination Port",            pd.Series(0, index=out.index))
    out["log_type"] = assign_log_types_vectorized(dst_port, fwd_bytes).values

    # Protocol
    fin = out.get("FIN Flag Count", pd.Series(0, index=out.index)).fillna(0)
    syn = out.get(" SYN Flag Count", pd.Series(0, index=out.index)).fillna(0)
    rst = out.get(" RST Flag Count", pd.Series(0, index=out.index)).fillna(0)
    fwd_pkts = out.get(" Total Fwd Packets", pd.Series(0, index=out.index)).fillna(0)
    bwd_pkts = out.get(" Total Backward Packets", pd.Series(0, index=out.index)).fillna(0)

    out["zeek_proto"] = np.where((fin + syn) > 0, 1, 0).astype(np.int8)  # 1=tcp, 0=udp

    # Connection state (encoded as int for the model)
    # SF=0, S0=1, RSTO=2, OTH=3
    state = np.full(len(out), 3, dtype=np.int8)
    state[(rst > 0)] = 2
    state[(syn > 0) & (fin > 0) & (rst == 0)] = 0
    state[(syn > 0) & (bwd_pkts == 0) & (rst == 0)] = 1
    out["zeek_conn_state"] = state

    # Service (port-based integer mapping)
    ports = dst_port.fillna(0).astype(int)
    service = pd.Series(0, index=out.index, dtype=np.int8)  # 0 = unknown
    service[ports == 53]   = 1   # dns
    service[ports == 80]   = 2   # http
    service[ports == 443]  = 3   # ssl
    service[ports == 21]   = 4   # ftp
    service[ports == 22]   = 5   # ssh
    service[ports == 25]   = 6   # smtp
    service[ports == 143]  = 7   # imap
    out["zeek_service"] = service

    return out


def get_zeek_extra_cols() -> list[str]:
    """Return the names of synthetic Zeek metadata columns added by this module."""
    return ["log_type", "zeek_proto", "zeek_conn_state", "zeek_service"]


def describe_log_distribution(df: pd.DataFrame) -> dict:
    """Return per-log-type row counts."""
    if "log_type" not in df.columns:
        return {}
    counts = df["log_type"].value_counts().to_dict()
    return {LOG_TYPE_NAMES.get(k, k): v for k, v in counts.items()}
