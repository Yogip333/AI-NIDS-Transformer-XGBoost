# Adapted from concepts by:
# Lin, T.-Y. et al. (2018) 'Focal Loss for Dense Object Detection',
#   IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(2), pp. 318-327.
# Loshchilov, I. and Hutter, F. (2019) 'Decoupled Weight Decay Regularization',
#   Proceedings of ICLR 2019.
# Loshchilov, I. and Hutter, F. (2017) 'SGDR: Stochastic Gradient Descent with
#   Warm Restarts', Proceedings of ICLR 2017.

"""Supervised fine-tuning for the Zeek Transformer classifier.

The encoder is initialised from the self-supervised checkpoint, a 2-layer
MLP head is attached, and the whole network is trained end-to-end with
Focal Loss (Lin et al., 2018) on a 40 per cent slice of the labelled attack
data plus all benign sessions. AdamW (Loshchilov and Hutter, 2019) is used
with a linear warm-up followed by cosine annealing (Loshchilov and Hutter,
2017). Class imbalance is addressed both by the focal modulation factor
and by per-class alpha weights computed from the training distribution.
"""
import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from src.models.transformer import TransformerClassifier
from src.data.loader import ID_TO_LABEL, NUM_CLASSES

logger = logging.getLogger(__name__)


def compute_class_weights_tensor(
    labels: np.ndarray, num_classes: int, device: torch.device
) -> torch.Tensor:
    """Return inverse-frequency class weights as a CUDA/CPU tensor.

    Classes with zero samples get weight=0 so they don't dominate the loss.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    present = counts > 0
    weights = np.zeros(num_classes, dtype=float)
    weights[present] = 1.0 / counts[present]
    weights[present] /= weights[present].sum()
    weights[present] *= present.sum()  # scale so present classes average ~1.0
    return torch.tensor(weights, dtype=torch.float, device=device)


class FocalLoss(nn.Module):
    """Multi-class Focal Loss as introduced by Lin et al. (2018).

    The modulation factor ``(1 - p_t) ** gamma`` scales down the loss
    contribution of examples the model already classifies confidently, so
    the gradient is dominated by the minority-class samples the network
    currently gets wrong — the exact regime this project sits in, given
    the CICIDS2017 long tail. Label smoothing is passed through to the
    underlying cross-entropy call and left at a conservative 0.05.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha          # per-class weight tensor
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Per-sample cross-entropy (no class weights — applied separately)
        ce = F.cross_entropy(
            inputs, targets, reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)                        # probability of correct class
        focal = ((1 - pt) ** self.gamma) * ce       # focal modulation

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal = alpha_t * focal

        return focal.mean()


def build_weighted_sampler(labels: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler for class-imbalanced data."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    class_weights = 1.0 / counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(labels),
        replacement=True,
    )


class FineTuner:
    """
    Supervised fine-tuning for multi-class intrusion detection.

    Parameters
    ----------
    model   : TransformerClassifier (encoder pre-loaded from pre-training)
    cfg     : top-level config dict
    device  : torch.device
    """

    def __init__(
        self,
        model: TransformerClassifier,
        cfg: dict,
        device: torch.device,
        save_dir: str = "models/checkpoints",
        num_classes: int = NUM_CLASSES,
    ):
        self.model      = model.to(device)
        self.device     = device
        self.cfg        = cfg
        self.save_dir   = save_dir
        self.num_classes = num_classes
        os.makedirs(save_dir, exist_ok=True)

        ftcfg = cfg.get("finetuning", {})
        self.lr           = float(ftcfg.get("learning_rate", 5e-5))
        self.num_epochs   = int(ftcfg.get("num_epochs", 25))
        self.grad_clip    = float(ftcfg.get("grad_clip", 1.0))
        self.patience     = int(ftcfg.get("early_stopping_patience", 5))
        self.warmup_steps = int(ftcfg.get("warmup_steps", 200))
        self.use_weights  = bool(ftcfg.get("class_weights", True))

        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self._class_weights: Optional[torch.Tensor] = None

    def _build_scheduler(self, total_steps: int) -> SequentialLR:
        warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0,
                          total_iters=self.warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer,
                                   T_max=max(1, total_steps - self.warmup_steps))
        return SequentialLR(self.optimizer, [warmup, cosine],
                            milestones=[self.warmup_steps])

    def _setup_loss(self, train_labels: np.ndarray) -> nn.Module:
        ftcfg = self.cfg.get("finetuning", {})
        gamma = float(ftcfg.get("focal_gamma", 2.0))
        if self.use_weights:
            self._class_weights = compute_class_weights_tensor(
                train_labels, self.num_classes, self.device
            )
            logger.info("Using FocalLoss (gamma=%.1f) with class weights", gamma)
            return FocalLoss(alpha=self._class_weights, gamma=gamma,
                             label_smoothing=0.05)
        return FocalLoss(gamma=gamma, label_smoothing=0.05)

    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        training: bool,
        epoch: int | None = None,
    ) -> dict:
        self.model.train(training)
        total_loss = 0.0
        correct = 0
        total = 0

        phase = "train" if training else "val"
        desc = f"Finetune {phase}"
        if epoch is not None:
            desc = f"Finetune {phase} e{epoch:02d}"

        iterator = tqdm(loader, desc=desc, leave=False, disable=(len(loader) == 0))
        for batch_idx, batch in enumerate(iterator, start=1):
            features     = batch["features"].to(self.device)
            log_types    = batch["log_types"].to(self.device)
            padding_mask = batch["padding_mask"].to(self.device)
            labels       = batch["label"].to(self.device)

            if training:
                self.optimizer.zero_grad()

            logits = self.model(features, log_types, padding_mask)
            loss   = criterion(logits, labels)

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                if hasattr(self, "_scheduler"):
                    self._scheduler.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            if training and self.device.type == "cuda" and (batch_idx == 1 or batch_idx % 50 == 0):
                mem_gb = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
                iterator.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / max(total, 1):.4f}", mem_gb=f"{mem_gb:.2f}")
            else:
                iterator.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / max(total, 1):.4f}")

        return {
            "loss": total_loss / max(len(loader), 1),
            "accuracy": correct / max(total, 1),
        }

    def _collect_labels(self, loader: DataLoader) -> np.ndarray:
        all_labels = []
        for batch in loader:
            if "label" in batch:
                all_labels.append(batch["label"].numpy())
        return np.concatenate(all_labels) if all_labels else np.array([])

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict:
        """Run full fine-tuning loop."""
        if len(train_loader) == 0 or len(val_loader) == 0:
            raise ValueError("Fine-tuning DataLoader is empty. Verify train/val sessions in processed data.")

        # Collect train labels for class-weight setup
        train_labels = self._collect_labels(train_loader)
        if train_labels.size == 0:
            raise ValueError("No training labels were collected for fine-tuning.")

        criterion    = self._setup_loss(train_labels)

        total_steps = self.num_epochs * len(train_loader)
        self._scheduler = self._build_scheduler(total_steps)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_path = os.path.join(self.save_dir, "finetune_best.pt")

        for epoch in range(1, self.num_epochs + 1):
            tr = self._run_epoch(train_loader, criterion, training=True, epoch=epoch)
            history["train_loss"].append(tr["loss"])
            history["train_acc"].append(tr["accuracy"])

            with torch.no_grad():
                va = self._run_epoch(val_loader, criterion, training=False, epoch=epoch)
            history["val_loss"].append(va["loss"])
            history["val_acc"].append(va["accuracy"])

            logger.info(
                "FT epoch %02d/%d | train loss=%.4f acc=%.4f | val loss=%.4f acc=%.4f",
                epoch, self.num_epochs,
                tr["loss"], tr["accuracy"],
                va["loss"], va["accuracy"],
            )

            if va["loss"] < best_val_loss:
                best_val_loss = va["loss"]
                patience_counter = 0
                self.save_checkpoint(best_path, epoch, va["loss"])
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        # Load best weights back
        ckpt = torch.load(best_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Fine-tuning complete. Best val loss: %.4f", best_val_loss)
        return history

    def evaluate(self, test_loader: DataLoader) -> dict:
        """Return per-class classification report on the test set."""
        from sklearn.metrics import classification_report, f1_score, accuracy_score

        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                features     = batch["features"].to(self.device)
                log_types    = batch["log_types"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)
                labels       = batch["label"]

                logits = self.model(features, log_types, padding_mask)
                preds  = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        y_pred = np.array(all_preds)
        y_true = np.array(all_labels)
        label_names = [ID_TO_LABEL.get(i, str(i)) for i in range(self.num_classes)]

        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(self.num_classes)),
            target_names=label_names,
            zero_division=0,
        )
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        logger.info("Transformer eval  acc=%.4f  f1=%.4f", acc, f1)
        logger.info("\n%s", report)
        return {"accuracy": acc, "f1": f1, "report": report,
                "y_pred": y_pred, "y_true": y_true}

    def save_checkpoint(self, path: str, epoch: int, loss: float) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "loss":  loss,
            "model_state_dict": self.model.state_dict(),
        }, path)
        logger.debug("Checkpoint saved: %s", path)

    def save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info("Model state dict saved to %s", path)

    @classmethod
    def load_model(cls, model: TransformerClassifier, path: str, device: torch.device) -> None:
        state = torch.load(path, map_location=device, weights_only=True)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.to(device)
        logger.info("Loaded model weights from %s", path)
