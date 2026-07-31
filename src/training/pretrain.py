# Adapted from concepts by:
# Devlin, J. et al. (2019) 'BERT: Pre-training of Deep Bidirectional Transformers
#   for Language Understanding', Proceedings of NAACL-HLT 2019, pp. 4171-4186.
# Loshchilov, I. and Hutter, F. (2019) 'Decoupled Weight Decay Regularization',
#   Proceedings of ICLR 2019.
# Loshchilov, I. and Hutter, F. (2017) 'SGDR: Stochastic Gradient Descent with
#   Warm Restarts', Proceedings of ICLR 2017.

"""Self-supervised pre-training for the Zeek Transformer.

Two pretext objectives are optimised jointly on benign-only sessions so
that the encoder learns a usable representation of normal network
behaviour without ever seeing an attack label. Masked Field Prediction is
the BERT-style masked-autoencoding objective of Devlin et al. (2019),
adapted to operate on continuous numerical fields rather than sub-word
tokens; Next-Event Prediction is a simple one-step-ahead auto-regressive
target over the session. The optimiser is AdamW (Loshchilov and Hutter,
2019) and the schedule is a linear warm-up followed by cosine annealing.
"""
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.transformer import MultiTaskTransformer

logger = logging.getLogger(__name__)


class PreTrainer:
    """
    Runs self-supervised pre-training on benign sessions.

    Parameters
    ----------
    model    : MultiTaskTransformer
    cfg      : top-level config dict
    device   : torch.device
    """

    def __init__(
        self,
        model: MultiTaskTransformer,
        cfg: dict,
        device: torch.device,
        save_dir: str = "models/checkpoints",
    ):
        self.model   = model.to(device)
        self.device  = device
        self.cfg     = cfg
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        pcfg = cfg.get("pretraining", {})
        self.lr           = float(pcfg.get("learning_rate", 1e-4))
        self.num_epochs   = int(pcfg.get("num_epochs", 15))
        self.mask_prob    = float(pcfg.get("mask_prob", 0.15))
        self.grad_clip    = float(pcfg.get("grad_clip", 1.0))
        self.patience     = int(pcfg.get("early_stopping_patience", 3))
        self.warmup_steps = int(pcfg.get("warmup_steps", 500))

        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

    def _build_scheduler(self, total_steps: int) -> SequentialLR:
        warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0,
                          total_iters=self.warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer,
                                   T_max=max(1, total_steps - self.warmup_steps))
        return SequentialLR(self.optimizer, [warmup, cosine],
                            milestones=[self.warmup_steps])

    def _run_epoch(self, loader: DataLoader, training: bool, epoch: int | None = None) -> dict:
        self.model.train(training)
        totals = {"total_loss": 0.0, "mfp_loss": 0.0, "nep_loss": 0.0}
        n_batches = 0

        phase = "train" if training else "val"
        desc = f"Pretrain {phase}"
        if epoch is not None:
            desc = f"Pretrain {phase} e{epoch:02d}"

        iterator = tqdm(loader, desc=desc, leave=False, disable=(len(loader) == 0))
        for batch_idx, batch in enumerate(iterator, start=1):
            features     = batch["features"].to(self.device)
            log_types    = batch["log_types"].to(self.device)
            padding_mask = batch["padding_mask"].to(self.device)

            if training:
                self.optimizer.zero_grad()

            losses = self.model(features, log_types, padding_mask, mask_prob=self.mask_prob)

            if training:
                losses["total_loss"].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                if hasattr(self, "_scheduler"):
                    self._scheduler.step()

            for k in totals:
                totals[k] += losses[k].item()
            n_batches += 1

            if training and self.device.type == "cuda" and (batch_idx == 1 or batch_idx % 50 == 0):
                mem_gb = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
                iterator.set_postfix(loss=f"{losses['total_loss'].item():.4f}", mem_gb=f"{mem_gb:.2f}")
            else:
                iterator.set_postfix(loss=f"{losses['total_loss'].item():.4f}")

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> dict:
        """
        Run full pre-training loop.

        Returns
        -------
        history : dict with per-epoch loss lists
        """
        if len(train_loader) == 0:
            raise ValueError("Pre-training DataLoader is empty. Verify benign sessions in processed data.")

        total_steps = self.num_epochs * len(train_loader)
        self._scheduler = self._build_scheduler(total_steps)

        history = {"train_loss": [], "val_loss": []}
        best_metric = float("inf")
        patience_counter = 0
        best_path = os.path.join(self.save_dir, "pretrain_best.pt")

        for epoch in range(1, self.num_epochs + 1):
            train_metrics = self._run_epoch(train_loader, training=True, epoch=epoch)
            history["train_loss"].append(train_metrics["total_loss"])

            log_msg = (
                f"Pretrain epoch {epoch:02d}/{self.num_epochs} | "
                f"loss={train_metrics['total_loss']:.4f} "
                f"(mfp={train_metrics['mfp_loss']:.4f}, "
                f"nep={train_metrics['nep_loss']:.4f})"
            )

            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = self._run_epoch(val_loader, training=False, epoch=epoch)
                history["val_loss"].append(val_metrics["total_loss"])
                log_msg += f" | val_loss={val_metrics['total_loss']:.4f}"

                if val_metrics["total_loss"] < best_metric:
                    best_metric = val_metrics["total_loss"]
                    patience_counter = 0
                    self.save_checkpoint(best_path, epoch, best_metric)
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info("Early stopping at epoch %d", epoch)
                        break
            else:
                # When no validation loader is provided, persist the best train-loss checkpoint.
                if train_metrics["total_loss"] < best_metric:
                    best_metric = train_metrics["total_loss"]
                    self.save_checkpoint(best_path, epoch, best_metric)

            logger.info(log_msg)

        # Save final checkpoint
        final_path = os.path.join(self.save_dir, "pretrain_final.pt")
        self.save_checkpoint(final_path, epoch, train_metrics["total_loss"])
        logger.info("Pre-training complete. Best monitored loss: %.4f", best_metric)
        return history

    def save_checkpoint(self, path: str, epoch: int, loss: float) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "epoch":      epoch,
            "loss":       loss,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        logger.debug("Checkpoint saved to %s", path)

    @classmethod
    def load_encoder_weights(
        cls,
        pretrain_ckpt: str,
        target_model: nn.Module,
    ) -> nn.Module:
        """
        Load encoder weights from a pre-training checkpoint into any model
        that contains a `.encoder` attribute.
        """
        ckpt = torch.load(pretrain_ckpt, map_location="cpu", weights_only=True)
        state = ckpt["model_state_dict"]

        # Filter only encoder keys
        encoder_state = {
            k.replace("encoder.", "", 1): v
            for k, v in state.items()
            if k.startswith("encoder.")
        }
        target_model.encoder.load_state_dict(encoder_state, strict=True)
        logger.info("Loaded pre-trained encoder weights from %s", pretrain_ckpt)
        return target_model
