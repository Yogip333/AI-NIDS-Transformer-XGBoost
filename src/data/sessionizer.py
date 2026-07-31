"""
Sessionizer: groups individual network flows into fixed-size session windows.

Because CICIDS2017 CSV files lack source-IP and timestamp columns,
hence using a sliding-window approach over the ordered rows as a proxy for
temporal session grouping.  In a real Zeek deployment the window would
be keyed on (src_ip, 5-min bucket).

Each session is a sequence of `session_size` consecutive flows.
The session label is the majority label among its constituent flows.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class Sessionizer:
    """
    Converts a flat DataFrame of flows into a list of session dicts.

    Parameters
    ----------
    session_size : int
        Number of flows per session window.
    stride : int
        Step between successive windows.  stride == session_size gives
        non-overlapping windows; stride < session_size gives overlapping ones.
    """

    def __init__(self, session_size: int = 20, stride: Optional[int] = None):
        self.session_size = session_size
        self.stride = stride if stride is not None else session_size

    def build_sessions(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = "label_id",
        log_type_col: str = "log_type",
    ) -> list[dict]:
        """
        Returns a list of session dicts, each containing:
          - 'features'    : np.ndarray  (session_size, num_features)
          - 'log_types'   : np.ndarray  (session_size,)  int8
          - 'label'       : int  (majority label)
          - 'has_attack'  : bool
          - 'attack_ids'  : set of attack label IDs present in session
          - 'session_idx' : int
        """
        feats = df[feature_cols].values.astype(np.float32)
        log_types = df[log_type_col].values.astype(np.int8) if log_type_col in df.columns \
            else np.zeros(len(df), dtype=np.int8)
        labels = df[label_col].values.astype(np.int64) if label_col in df.columns \
            else np.zeros(len(df), dtype=np.int64)

        sessions = []
        n = len(df)
        idx = 0
        session_counter = 0

        while idx + self.session_size <= n:
            end = idx + self.session_size
            f_slice = feats[idx:end]
            lt_slice = log_types[idx:end]
            lbl_slice = labels[idx:end]

            # Majority label
            counts = np.bincount(lbl_slice, minlength=15)
            maj_label = int(np.argmax(counts))

            # For imbalanced sessions: if any attack present → flag it
            attack_ids = set(lbl_slice[lbl_slice != 0].tolist())
            has_attack = len(attack_ids) > 0

            # If session has mixed labels, use majority non-benign label
            if has_attack:
                attack_counts = counts.copy()
                attack_counts[0] = 0  # zero out BENIGN
                maj_label = int(np.argmax(attack_counts))

            sessions.append({
                "features":    f_slice,
                "log_types":   lt_slice,
                "label":       maj_label,
                "has_attack":  has_attack,
                "attack_ids":  attack_ids,
                "session_idx": session_counter,
            })
            idx += self.stride
            session_counter += 1

        logger.info(
            "Created %d sessions (size=%d, stride=%d) — %d with attacks",
            len(sessions), self.session_size, self.stride,
            sum(1 for s in sessions if s["has_attack"]),
        )
        return sessions

    def build_benign_sessions(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        log_type_col: str = "log_type",
    ) -> list[dict]:
        """Build sessions from BENIGN-only rows for self-supervised pre-training."""
        benign_df = df[df["label_id"] == 0].reset_index(drop=True) \
            if "label_id" in df.columns else df.reset_index(drop=True)
        logger.info("Benign rows for pre-training: %d", len(benign_df))
        return self.build_sessions(benign_df, feature_cols, label_col="label_id",
                                   log_type_col=log_type_col)


class SessionDataset(Dataset):
    """
    PyTorch Dataset wrapping a list of session dicts.

    Parameters
    ----------
    sessions : list of dicts from Sessionizer
    max_seq_len : int
        Pad / truncate to this length.
    return_labels : bool
        Whether to return label tensors (False during pre-training).
    """

    def __init__(
        self,
        sessions: list[dict],
        max_seq_len: int = 60,
        return_labels: bool = True,
    ):
        self.sessions = sessions
        self.max_seq_len = max_seq_len
        self.return_labels = return_labels

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int) -> dict:
        s = self.sessions[idx]
        feat = s["features"]     # (L, F)
        lt   = s["log_types"]    # (L,)
        L, F = feat.shape

        # Pad or truncate to max_seq_len
        if L >= self.max_seq_len:
            feat = feat[:self.max_seq_len]
            lt   = lt[:self.max_seq_len]
            length = self.max_seq_len
        else:
            pad_len = self.max_seq_len - L
            feat = np.vstack([feat, np.zeros((pad_len, F), dtype=np.float32)])
            lt   = np.concatenate([lt, np.zeros(pad_len, dtype=np.int8)])
            length = L

        # Padding mask: True = padded position (ignored by attention)
        padding_mask = np.zeros(self.max_seq_len, dtype=bool)
        padding_mask[length:] = True

        item = {
            "features":     torch.from_numpy(feat),          # (max_seq_len, F)
            "log_types":    torch.from_numpy(lt.astype(np.int64)),  # (max_seq_len,)
            "padding_mask": torch.from_numpy(padding_mask),  # (max_seq_len,)
            "length":       torch.tensor(length, dtype=torch.long),
        }
        if self.return_labels:
            item["label"] = torch.tensor(s["label"], dtype=torch.long)
        return item


def collate_sessions(batch: list[dict]) -> dict:
    """Custom collate that stacks session dicts into batched tensors."""
    out = {}
    for key in batch[0]:
        tensors = [item[key] for item in batch]
        out[key] = torch.stack(tensors)
    return out
