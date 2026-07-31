# Adapted from concepts by:
# Vaswani, A. et al. (2017) 'Attention Is All You Need',
#   Advances in Neural Information Processing Systems 30.
# Devlin, J. et al. (2019) 'BERT: Pre-training of Deep Bidirectional Transformers
#   for Language Understanding', Proceedings of NAACL-HLT 2019, pp. 4171-4186.
# Xiong, R. et al. (2020) 'On Layer Normalization in the Transformer Architecture',
#   Proceedings of the 37th International Conference on Machine Learning.
# Hendrycks, D. and Gimpel, K. (2016) 'Gaussian Error Linear Units (GELUs)',
#   arXiv:1606.08415.

"""Zeek-aware Transformer used as the shared encoder backbone.

The design follows the standard encoder-only Transformer of Vaswani et al.
(2017) with two small adaptations that matter for network traffic:

- each event token is the concatenation of the raw numerical flow vector
  and a learned embedding of its Zeek log family (conn, dns, http, ssl,
  files), so the encoder always knows which protocol view a flow came from;
- every encoder block uses pre-layer-normalisation (``norm_first=True``)
  following Xiong et al. (2020), which is noticeably easier to train on a
  single-GPU workstation than the original post-norm variant.

Two self-supervised pretext heads sit on top of the encoder for pre-training
on benign sessions — Masked Field Prediction in the spirit of BERT's masked
language modelling (Devlin et al., 2019), and a simple Next-Event Prediction
objective. After pre-training, the encoder is wrapped by a 2-layer MLP for
supervised classification over the CLS token.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding, identical to Vaswani et al. (2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)           # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, L, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# Core Encoder
# ─────────────────────────────────────────────────────────────────────────────

class ZeekTransformerEncoder(nn.Module):
    """
    Shared encoder backbone used by all three model variants.

    Parameters
    ----------
    input_dim        : number of raw flow features (after cleaning)
    d_model          : transformer hidden dimension
    nhead            : number of attention heads
    num_layers       : number of encoder layers
    dim_feedforward  : FFN hidden size
    dropout          : dropout rate
    max_seq_len      : maximum sequence length (for positional encoding)
    num_log_types    : number of distinct Zeek log types (0-4)
    log_embed_dim    : dimension of log-type embedding (concatenated before projection)
    """

    def __init__(
        self,
        input_dim: int = 76,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 60,
        num_log_types: int = 5,
        log_embed_dim: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim

        # Log-type embedding
        self.log_embedding = nn.Embedding(num_log_types + 1, log_embed_dim, padding_idx=num_log_types)

        # Project combined input → d_model
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + log_embed_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding (+1 for CLS)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len + 1, dropout=dropout)

        # Standard PyTorch encoder layer with pre-norm (Xiong et al., 2020)
        # and GELU activation (Hendrycks and Gimpel, 2016).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

    def forward(
        self,
        features: Tensor,       # (B, L, input_dim)
        log_types: Tensor,      # (B, L)
        padding_mask: Tensor,   # (B, L)  True = pad position
    ) -> tuple[Tensor, Tensor]:
        """
        Returns
        -------
        cls_repr  : (B, d_model)   – CLS token representation
        all_repr  : (B, L+1, d_model) – full sequence output (including CLS)
        """
        B, L, _ = features.shape

        # Embed log types
        lt_embed = self.log_embedding(log_types)           # (B, L, log_embed_dim)

        # Concatenate and project
        x = torch.cat([features, lt_embed], dim=-1)        # (B, L, input_dim + log_embed_dim)
        x = self.input_proj(x)                             # (B, L, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)             # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                    # (B, L+1, d_model)

        # Positional encoding
        x = self.pos_enc(x)                               # (B, L+1, d_model)

        # Extend padding mask for CLS (never masked)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)  # (B, L+1)

        # Transformer encoding
        out = self.transformer(x, src_key_padding_mask=full_mask)  # (B, L+1, d_model)

        cls_repr = out[:, 0, :]   # (B, d_model)
        return cls_repr, out

    def get_attention_weights(
        self,
        features: Tensor,
        log_types: Tensor,
        padding_mask: Tensor,
    ) -> list[Tensor]:
        """
        Extract per-layer attention weights for interpretability.
        Returns list of (B, nhead, L+1, L+1) tensors.
        """
        B, L, _ = features.shape
        lt_embed = self.log_embedding(log_types)
        x = torch.cat([features, lt_embed], dim=-1)
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_enc(x)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)

        attn_weights = []
        current = x
        for layer in self.transformer.layers:
            # Manually run the sub-layers to capture attention weights
            # Pre-norm
            normed = layer.norm1(current)
            attn_out, weights = layer.self_attn(
                normed, normed, normed,
                key_padding_mask=full_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            attn_weights.append(weights.detach())
            current = current + layer.dropout1(attn_out)
            normed2 = layer.norm2(current)
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(normed2))))
            current = current + layer.dropout2(ff_out)

        return attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Task Pre-Training Model
# ─────────────────────────────────────────────────────────────────────────────

class MultiTaskTransformer(nn.Module):
    """Encoder plus two self-supervised pretext heads used during pre-training.

    The Masked Field Prediction objective zero-masks 15 per cent of the
    numerical field values and asks the model to reconstruct them, mirroring
    the masked-language-model pretext of Devlin et al. (2019) but operating
    on continuous fields rather than sub-word tokens. Next-Event Prediction
    is a much simpler auto-regressive target: given the first ``L - 1``
    events of a session, predict the feature vector of event ``L``. The two
    losses are summed with equal weight.
    """

    def __init__(self, encoder: ZeekTransformerEncoder):
        super().__init__()
        self.encoder = encoder
        d = encoder.d_model
        inp = encoder.input_dim

        # Masked field prediction head: event repr → feature values
        self.mfp_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, inp),
        )

        # Next-event prediction head: CLS of truncated seq → feature values
        self.nep_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, inp),
        )

    def forward(
        self,
        features: Tensor,           # (B, L, F)
        log_types: Tensor,          # (B, L)
        padding_mask: Tensor,       # (B, L)
        mask_prob: float = 0.15,
    ) -> dict[str, Tensor]:
        """
        Returns a dict with:
          'mfp_loss' : scalar
          'nep_loss' : scalar
          'total_loss' : scalar
        """
        B, L, n_feat = features.shape

        # ── Task 1: Masked Field Prediction ──────────────────────────────────
        mask = (torch.rand(B, L, n_feat, device=features.device) < mask_prob)
        # Don't mask padded positions
        pad_expanded = padding_mask.unsqueeze(-1).expand_as(features)
        mask = mask & ~pad_expanded

        masked_features = features.clone()
        masked_features[mask] = 0.0   # replace with zero (ablation study can try noise for future)

        _, seq_out = self.encoder(masked_features, log_types, padding_mask)
        event_repr = seq_out[:, 1:, :]      # (B, L, d)  skip CLS
        pred_fields = self.mfp_head(event_repr)  # (B, L, F)

        if mask.any():
            mfp_loss = F.mse_loss(pred_fields[mask], features[mask])
        else:
            mfp_loss = torch.tensor(0.0, device=features.device)

        # ── Task 2: Next-Event Prediction ────────────────────────────────────
        # Use first L-1 events as context, predict last event's features
        # Only valid where length > 1
        lengths = (~padding_mask).sum(dim=1)         # (B,) real lengths
        valid = lengths > 1
        nep_loss = torch.tensor(0.0, device=features.device)

        if valid.any():
            nep_losses = []
            for b in range(B):
                if not valid[b]:
                    continue
                real_len = lengths[b].item()
                # Context: first real_len - 1 events
                ctx_feat = features[b:b+1, :real_len-1, :]
                ctx_lt   = log_types[b:b+1, :real_len-1]
                ctx_pad  = torch.zeros(1, real_len-1, dtype=torch.bool, device=features.device)

                cls_repr, _ = self.encoder(ctx_feat, ctx_lt, ctx_pad)
                pred_next = self.nep_head(cls_repr)        # (1, F)
                target_next = features[b, real_len-1, :]  # (F,)
                nep_losses.append(F.mse_loss(pred_next.squeeze(0), target_next))

            if nep_losses:
                nep_loss = torch.stack(nep_losses).mean()

        total_loss = mfp_loss + nep_loss
        return {"mfp_loss": mfp_loss, "nep_loss": nep_loss, "total_loss": total_loss}


# ─────────────────────────────────────────────────────────────────────────────
# Classification Model
# ─────────────────────────────────────────────────────────────────────────────

class TransformerClassifier(nn.Module):
    """
    Fine-tuned classifier built on top of a (pre-trained) encoder.

    Uses the CLS token representation for classification.
    """

    def __init__(
        self,
        encoder: ZeekTransformerEncoder,
        num_classes: int = 15,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = encoder
        d = encoder.d_model

        self.classifier = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        features: Tensor,       # (B, L, F)
        log_types: Tensor,      # (B, L)
        padding_mask: Tensor,   # (B, L)
    ) -> Tensor:
        """Returns logits (B, num_classes)."""
        cls_repr, _ = self.encoder(features, log_types, padding_mask)
        return self.classifier(cls_repr)

    def get_embedding(       #used for Xgboost
        self,
        features: Tensor,
        log_types: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        """Returns CLS embedding (B, d_model) for use in hybrid model."""
        cls_repr, _ = self.encoder(features, log_types, padding_mask)
        return cls_repr.detach()


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_encoder(cfg: dict, input_dim: int) -> ZeekTransformerEncoder:
    t = cfg.get("transformer", {})
    return ZeekTransformerEncoder(
        input_dim=input_dim,
        d_model=t.get("d_model", 256),
        nhead=t.get("nhead", 8),
        num_layers=t.get("num_encoder_layers", 4),
        dim_feedforward=t.get("dim_feedforward", 512),
        dropout=t.get("dropout", 0.1),
        max_seq_len=t.get("max_seq_len", 60),
        num_log_types=cfg.get("features", {}).get("num_log_types", 5),
        log_embed_dim=t.get("log_type_embed_dim", 32),
    )


def build_pretrain_model(cfg: dict, input_dim: int) -> MultiTaskTransformer:
    encoder = build_encoder(cfg, input_dim)
    return MultiTaskTransformer(encoder)


def build_classifier(
    cfg: dict,
    input_dim: int,
    num_classes: int = 15,
    pretrained_encoder: ZeekTransformerEncoder | None = None,
) -> TransformerClassifier:
    encoder = pretrained_encoder if pretrained_encoder is not None else build_encoder(cfg, input_dim)
    return TransformerClassifier(
        encoder=encoder,
        num_classes=num_classes,
        hidden_dim=128,
        dropout=cfg.get("finetuning", {}).get("dropout", 0.2),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
