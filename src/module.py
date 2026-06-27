"""Conditional-injection Transformer building blocks.

Adapted from the `leworldmodel` reference (``le-wm/module.py``) so that the
predictor can be used as a world-model backbone inside this repo:

* AdaLN-zero conditioning (``ConditionalBlock``) injects a per-step action
  vector into every Transformer block.
* The conditioning stream can have a different dimensionality than the token
  stream (``cond_dim`` != ``input_dim``), which lets the action embedding
  (``embed_dim_act``) differ from the state embedding (``embed_dim``).
* Attention is causal; a finite *memory window* is realised by the caller
  simply slicing the context to the last ``mem_window`` tokens before each
  autoregressive step (full causal attention inside the slice is equivalent to
  a sliding window over the full sequence).

No ``einops`` dependency: tensor reshapes are done with ``reshape``/``permute``.
"""

import torch
from torch import nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------------------------------
def modulate(x, shift, scale):
    """AdaLN-zero modulation."""
    return x * (1 + scale) + shift


# ======================================================================================================================
class FeedForward(nn.Module):
    """FeedForward network used in Transformers."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ======================================================================================================================
class Attention(nn.Module):
    """Scaled dot-product attention with causal masking."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """x : (B, T, D)."""
        B, T, _ = x.shape
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # (B, T, inner) -> (B, heads, T, dim_head)
        q, k, v = (
            t.reshape(B, T, self.heads, self.dim_head).permute(0, 2, 1, 3)
            for t in qkv
        )
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        # (B, heads, T, dim_head) -> (B, T, heads*dim_head)
        out = out.permute(0, 2, 1, 3).reshape(B, T, self.heads * self.dim_head)
        return self.to_out(out)


# ======================================================================================================================
class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ======================================================================================================================
class Block(nn.Module):
    """Standard Transformer block (no conditioning)."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ======================================================================================================================
class Transformer(nn.Module):
    """Transformer supporting AdaLN-zero (conditional) blocks.

    Unlike the reference implementation, the conditioning stream is projected
    from its own ``cond_dim`` so the action embedding can differ in size from
    the token embedding.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        cond_dim=None,
        block_class=ConditionalBlock,
    ):
        super().__init__()
        cond_dim = cond_dim if cond_dim is not None else input_dim

        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(cond_dim, hidden_dim)
            if cond_dim != hidden_dim
            else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):
        x = self.input_proj(x)

        if c is not None:
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        return self.output_proj(x)


# ======================================================================================================================
class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step latent prediction.

    Tokens ``x`` (B, T, input_dim) are predicted one step ahead under causal
    attention, with a per-step conditioning stream ``c`` (B, T, cond_dim)
    injected through AdaLN-zero. A finite memory window is obtained by the
    caller slicing ``x``/``c`` to the last ``mem_window`` steps before each
    rollout step.
    """

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        cond_dim=None,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            cond_dim=cond_dim,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """x: (B, T, input_dim), c: (B, T, cond_dim)."""
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        return self.transformer(x, c)
