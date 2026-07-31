"""
CSV Replay / Streaming Simulator.

Streams CICIDS2017 CSV rows through the EXACT same preprocessing path
used at training time: same feature order, same scaler, same validation.

This simulates "live" traffic for demo and testing purposes. Because
training and inference use identical pipelines, accuracy on replayed
CSV is representative of model quality on this dataset.

Usage:
    from src.streaming.csv_replay import CSVReplayStreamer

    streamer = CSVReplayStreamer.from_saved_artifacts(
        scaler_path="models/checkpoints/feature_scaler.pkl",
        schema_path="models/checkpoints/feature_schema.json",
    )
    for batch_result in streamer.stream("Kaggle Dataset/Monday-WorkingHours.pcap_ISCX.csv"):
        print(batch_result)
"""
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from src.data.loader import (
    FEATURE_COLS, ID_TO_LABEL, LABEL_COL, NUM_CLASSES,
    _encode_label, load_config,
)
from src.data.validator import load_and_verify_schema, validate_dataframe
from src.data.feature_engineer import FeatureScaler, STAT_FEATURE_NAMES

logger = logging.getLogger(__name__)


@dataclass
class FlowBatch:
    """A batch of processed flows ready for inference."""
    features: np.ndarray          # (batch_size, num_features) scaled
    raw_features: np.ndarray      # (batch_size, num_features) unscaled — for stats
    labels: Optional[np.ndarray]  # (batch_size,) None if no ground truth
    feature_names: list[str]
    batch_idx: int
    source_file: str
    rows_processed: int


@dataclass
class DetectionEvent:
    """Enriched detection result for a single flow."""
    flow_idx: int
    attack_id: int
    attack_name: str
    confidence: float
    probabilities: list[float]
    is_attack: bool
    ground_truth_id: Optional[int] = None
    ground_truth_name: Optional[str] = None
    correct: Optional[bool] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class StreamStats:
    """Running statistics for a streaming session."""
    total_flows: int = 0
    total_attacks: int = 0
    total_benign: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    attack_counts: dict = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def flows_per_second(self) -> float:
        elapsed = self.elapsed_seconds
        return self.total_flows / elapsed if elapsed > 0 else 0.0

    @property
    def detection_rate(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def false_positive_rate(self) -> float:
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom > 0 else 0.0

    def update(self, event: DetectionEvent) -> None:
        self.total_flows += 1
        gt = event.ground_truth_id
        pred = event.attack_id

        if gt is not None:
            gt_binary   = int(gt != 0)
            pred_binary = int(pred != 0)
            if gt_binary == 1:
                self.total_attacks += 1
                if pred_binary == 1:
                    self.true_positives += 1
                else:
                    self.false_negatives += 1
            else:
                self.total_benign += 1
                if pred_binary == 0:
                    self.true_negatives += 1
                else:
                    self.false_positives += 1
        else:
            if pred != 0:
                self.total_attacks += 1
            else:
                self.total_benign += 1

        if pred != 0:
            name = ID_TO_LABEL.get(pred, str(pred))
            self.attack_counts[name] = self.attack_counts.get(name, 0) + 1

    def summary(self) -> str:
        lines = [
            f"Flows: {self.total_flows:,}  ({self.flows_per_second:.0f}/s)",
            f"  Attacks detected: {self.total_attacks:,}",
            f"  Benign: {self.total_benign:,}",
        ]
        if self.true_positives + self.false_negatives > 0:
            lines += [
                f"  Detection rate : {self.detection_rate:.4f}",
                f"  FP rate        : {self.false_positive_rate:.4f}",
            ]
        if self.attack_counts:
            top = sorted(self.attack_counts.items(), key=lambda x: -x[1])[:5]
            lines.append("  Top attacks: " + ", ".join(f"{k}={v}" for k, v in top))
        return "\n".join(lines)


class CSVReplayStreamer:
    """
    Streams CSV rows through the training-identical preprocessing pipeline
    and invokes a detection model for each batch.

    Parameters
    ----------
    scaler       : fitted FeatureScaler from training
    feature_cols : exact feature columns in training order
    batch_size   : flows per inference batch
    rate_limit   : max flows/second (0 = no limit)
    has_labels   : whether to decode the Label column for ground-truth metrics
    """

    def __init__(
        self,
        scaler: FeatureScaler,
        feature_cols: list[str],
        batch_size: int = 100,
        rate_limit: float = 0.0,
        has_labels: bool = True,
    ):
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.batch_size = batch_size
        self.rate_limit = rate_limit
        self.has_labels = has_labels
        self._sleep_per_batch = (batch_size / rate_limit) if rate_limit > 0 else 0.0

    @classmethod
    def from_saved_artifacts(
        cls,
        scaler_path: str = "models/checkpoints/feature_scaler.pkl",
        schema_path: str = "models/checkpoints/feature_schema.json",
        batch_size: int = 100,
        rate_limit: float = 0.0,
        has_labels: bool = True,
    ) -> "CSVReplayStreamer":
        """Load from persisted training artifacts."""
        if not Path(scaler_path).exists():
            raise FileNotFoundError(
                f"Scaler not found: {scaler_path}. Run preprocess.py first."
            )
        scaler = FeatureScaler.load(scaler_path)
        logger.info("Loaded scaler from %s", scaler_path)

        # Verify schema parity
        load_and_verify_schema(FEATURE_COLS, schema_path)

        return cls(
            scaler=scaler,
            feature_cols=FEATURE_COLS,
            batch_size=batch_size,
            rate_limit=rate_limit,
            has_labels=has_labels,
        )

    def _load_csv(self, filepath: str) -> pd.DataFrame:
        """Load and clean one CSV file, same logic as loader.py."""
        logger.info("Loading %s …", Path(filepath).name)
        df = pd.read_csv(filepath, encoding="latin-1", low_memory=False)

        # Keep only feature columns (+ label if present)
        available = [c for c in self.feature_cols if c in df.columns]
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            logger.warning("Missing %d features in CSV: %s", len(missing), missing[:5])

        cols_to_keep = available
        if self.has_labels and LABEL_COL in df.columns:
            cols_to_keep = available + [LABEL_COL]

        df = df[cols_to_keep].copy()

        # Decode labels
        if self.has_labels and LABEL_COL in df.columns:
            df["label_id"] = df[LABEL_COL].astype(str).apply(_encode_label)
            df.drop(columns=[LABEL_COL], inplace=True)

        # Numeric coercion + clean
        for col in available:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        for col in available:
            df[col] = df[col].clip(-1e9, 1e9)

        logger.info("Loaded %d rows from %s", len(df), Path(filepath).name)
        return df

    def _pad_missing_features(self, df: pd.DataFrame) -> np.ndarray:
        """Return (N, len(feature_cols)) array, padding missing cols with 0."""
        out = np.zeros((len(df), len(self.feature_cols)), dtype=np.float32)
        for i, col in enumerate(self.feature_cols):
            if col in df.columns:
                out[:, i] = df[col].values.astype(np.float32)
        return out

    def stream(
        self,
        csv_path: str,
        model_fn=None,
    ) -> Iterator[list[DetectionEvent]]:
        """
        Stream CSV rows through the pipeline, yielding batches of DetectionEvents.

        Parameters
        ----------
        csv_path : path to CICIDS2017 CSV file
        model_fn : callable(raw: np.ndarray, scaled: np.ndarray) -> (preds, probas)
                   Receives BOTH raw and scaled features; each model chooses which to use.
                   If None, yields batches without predictions (features only).

        Yields
        ------
        list[DetectionEvent]  — one event per flow in the batch
        """
        df = self._load_csv(csv_path)
        n = len(df)
        rows_done = 0
        batch_idx = 0

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            chunk = df.iloc[start:end]

            raw_features = self._pad_missing_features(chunk)
            scaled_features = self.scaler.transform(raw_features)

            labels = None
            if self.has_labels and "label_id" in chunk.columns:
                labels = chunk["label_id"].values.astype(np.int64)

            events = []
            if model_fn is not None:
                try:
                    preds, probas = model_fn(raw_features, scaled_features)
                except Exception as e:
                    logger.error("Model inference failed on batch %d: %s", batch_idx, e)
                    preds  = np.zeros(len(chunk), dtype=np.int64)
                    probas = np.zeros((len(chunk), NUM_CLASSES), dtype=np.float32)
                    probas[:, 0] = 1.0

                for i in range(len(chunk)):
                    att_id = int(preds[i])
                    gt_id  = int(labels[i]) if labels is not None else None
                    events.append(DetectionEvent(
                        flow_idx=rows_done + i,
                        attack_id=att_id,
                        attack_name=ID_TO_LABEL.get(att_id, str(att_id)),
                        confidence=float(probas[i, att_id]),
                        probabilities=probas[i].tolist(),
                        is_attack=(att_id != 0),
                        ground_truth_id=gt_id,
                        ground_truth_name=ID_TO_LABEL.get(gt_id, str(gt_id)) if gt_id is not None else None,
                        correct=(att_id == gt_id) if gt_id is not None else None,
                    ))
            else:
                for i in range(len(chunk)):
                    gt_id = int(labels[i]) if labels is not None else None
                    events.append(DetectionEvent(
                        flow_idx=rows_done + i,
                        attack_id=-1,
                        attack_name="",
                        confidence=0.0,
                        probabilities=[],
                        is_attack=False,
                        ground_truth_id=gt_id,
                    ))

            rows_done += len(chunk)
            batch_idx += 1

            if self._sleep_per_batch > 0:
                time.sleep(self._sleep_per_batch)

            yield events

            if rows_done % 10_000 == 0:
                logger.info("  Streamed %d/%d flows (%.1f%%)",
                            rows_done, n, 100 * rows_done / n)

        logger.info("Stream complete: %d flows from %s", rows_done, Path(csv_path).name)

    def stream_all_files(
        self,
        csv_paths: list[str],
        model_fn=None,
    ) -> Iterator[list[DetectionEvent]]:
        """Stream multiple CSV files sequentially."""
        for path in csv_paths:
            yield from self.stream(path, model_fn=model_fn)

    def run_with_stats(
        self,
        csv_paths: list[str],
        model_fn,
        log_interval: int = 50_000,
    ) -> StreamStats:
        """
        Stream all files, collect stats, and return a final StreamStats object.
        Logs progress every `log_interval` flows.
        """
        stats = StreamStats()
        last_log = 0

        for events in self.stream_all_files(csv_paths, model_fn=model_fn):
            for ev in events:
                stats.update(ev)

            if stats.total_flows - last_log >= log_interval:
                logger.info("Progress | %s", stats.summary())
                last_log = stats.total_flows

        logger.info("Final stream stats:\n%s", stats.summary())
        return stats
