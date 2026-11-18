"""
SmolGPT Architecture and Utilities for Weight Trajectory Forecasting.
Provides a clean, self-contained implementation of the modern causal decoder architecture:
- GQA (Grouped Query Attention) with QK-Norm
- RoPE (Rotary Position Embeddings)
- SwiGLU Feed-Forward Networks
- RMSNorm pre-normalization
- Weight-tied Token Embeddings and Output LM Head
- Autoregressive Cross-Entropy Loss & Perplexity (PPL)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SmolGPTConfig:
    block_size: int = 128
    vocab_size: int = 4096
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 2
    n_embed: int = 128
    dropout: float = 0.0
    bias: bool = False
    use_rotary: bool = True
    use_qk_norm: bool = True


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[1]
        if seq_len != self.seq_len_cached or self.inv_freq is None or self.inv_freq.device != q.device:
            self.inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, device=q.device).float() / self.dim)
            )
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=q.device).float()
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().type_as(q)
            self.sin_cached = freqs.sin().type_as(q)

        cos = self.cos_cached[None, :, None, :]
        sin = self.sin_cached[None, :, None, :]
        q_rot = self._apply_rotary(q, cos, sin)
        k_rot = self._apply_rotary(k, cos, sin)
        return q_rot, k_rot

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        d = x.shape[3] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], dim=3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: SmolGPTConfig):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        self.config = config
        self.head_dim = config.n_embed // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep_kv = config.n_head // config.n_kv_head

        self.c_q = nn.Linear(config.n_embed, config.n_head * self.head_dim, bias=config.bias)
        self.c_k = nn.Linear(config.n_embed, config.n_kv_head * self.head_dim, bias=config.bias)
        self.c_v = nn.Linear(config.n_embed, config.n_kv_head * self.head_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)

        if config.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        if config.use_rotary:
            self.rotary = RotaryEmbedding(self.head_dim)
        else:
            self.rotary = None

        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.n_rep_kv > 1:
            k = k.repeat_interleave(self.n_rep_kv, dim=1)
            v = v.repeat_interleave(self.n_rep_kv, dim=1)

        if self.rotary is not None:
            # Transpose to (B, T, n_head, head_dim) for rotary then back
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            q_rot, k_rot = self.rotary(q_t, k_t)
            q = q_rot.transpose(1, 2)
            k = k_rot.transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, config: SmolGPTConfig):
        super().__init__()
        hidden_dim = int(2 * (4 * config.n_embed) / 3)
        self.w1 = nn.Linear(config.n_embed, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embed, bias=False)
        self.w3 = nn.Linear(config.n_embed, hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: SmolGPTConfig):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embed)
        self.mlp = SwiGLUFeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SmolGPT(nn.Module):
    def __init__(self, config: SmolGPTConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embed)
        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.RMSNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # Weight tying: share parameters between embedding and output projection
        self.wte.weight = self.lm_head.weight

        # Weight initialization
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("w2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.wte(idx)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss


def get_flat_params(model: nn.Module) -> torch.Tensor:
    """Returns a 1D tensor concatenating all unique model parameters."""
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def set_flat_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    """Loads a 1D tensor of parameters back into the model."""
    offset = 0
    device = next(model.parameters()).device
    flat_params = flat_params.to(device)
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset:offset + numel].view_as(p))
        offset += numel


def get_mlp_params(model: nn.Module) -> torch.Tensor:
    """Returns a 1D tensor concatenating only MLP parameters (w1, w2, w3 across all layers)."""
    return torch.cat([p.detach().flatten() for name, p in model.named_parameters() if "mlp" in name])


def set_mlp_params(model: nn.Module, flat_mlp: torch.Tensor) -> None:
    """Loads a 1D tensor strictly into MLP parameters, leaving Embeddings, Attention, and Norms untouched."""
    offset = 0
    device = next(model.parameters()).device
    flat_mlp = flat_mlp.to(device)
    for name, p in model.named_parameters():
        if "mlp" in name:
            numel = p.numel()
            p.data.copy_(flat_mlp[offset:offset + numel].view_as(p))
            offset += numel


def evaluate_transformer_loss_and_ppl(
    model: nn.Module,
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[float, float]:
    """
    Evaluates cross-entropy loss and perplexity (PPL) over pre-materialized validation batches.
    Returns: (val_loss, perplexity)
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in val_batches:
            _, loss = model(x, y)
            losses.append(loss.item())
    mean_loss = float(np.mean(losses))
    perplexity = float(math.exp(min(mean_loss, 20.0)))
    return mean_loss, perplexity
