"""
Manifold Encoder for MSEF Framework

This module implements the manifold encoder phi_theta that maps multimodal character
features to era-conditioned manifold coordinates as described in Section 3.1.

The encoder uses:
- Input projection (352 -> 256)
- 4 residual blocks with hidden dim 512 (Appendix E)
- 12-layer Transformer with Adaptive Layer Normalization (AdaLN)
  to condition on temporal information
- Output projection (256 -> 256)
"""

import math
import torch
import torch.nn as nn
from typing import Tuple


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding as commonly used in diffusion models.

    Maps scalar time t in [0,1] to a high-dimensional representation.
    """

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        half_dim = self.embedding_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t * embeddings.unsqueeze(0)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings


class ResidualBlock(nn.Module):
    """
    Residual block with hidden dim for feature processing (Appendix E).

    Architecture: x -> Linear -> GELU -> Dropout -> Linear -> Dropout -> + x
    With a skip connection and layer normalization.
    """

    def __init__(self, dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class AdaLNTransformerBlock(nn.Module):
    """
    Transformer block with Adaptive Layer Normalization (AdaLN) for time conditioning.

    AdaLN modulates the normalization parameters based on the time embedding,
    allowing the model to adapt its processing based on the era.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        time_emb_dim: int = 64,
        attention_inner_dim=None,
    ):
        super().__init__()

        # Multi-head self-attention
        inner = attention_inner_dim or d_model
        if inner % nhead:
            raise ValueError("Attention width must be divisible by number of heads")
        self.attn_input = (
            nn.Identity() if inner == d_model else nn.Linear(d_model, inner)
        )
        self.attn_output = (
            nn.Identity() if inner == d_model else nn.Linear(inner, d_model)
        )
        self.self_attn = nn.MultiheadAttention(
            inner, nhead, dropout=dropout, batch_first=True
        )
        self.capture_attention = False
        self.last_attention = None
        self.head_mask = None

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # AdaLN modulation parameters
        # Each layer norm has scale and shift parameters conditioned on time
        self.adln_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, 4 * d_model))

        # Layer norms (without affine parameters, since AdaLN provides them)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        # Get AdaLN modulation parameters
        adln_params = self.adln_mlp(time_emb)
        scale_attn, shift_attn, scale_ffn, shift_ffn = adln_params.chunk(4, dim=-1)

        # Reshape for broadcasting over sequence length
        scale_attn = scale_attn.unsqueeze(1)
        shift_attn = shift_attn.unsqueeze(1)
        scale_ffn = scale_ffn.unsqueeze(1)
        shift_ffn = shift_ffn.unsqueeze(1)

        # Self-attention with AdaLN
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_attn) + shift_attn
        attn_input = self.attn_input(x_norm)
        if self.head_mask is None:
            attn_out, weights = self.self_attn(
                attn_input,
                attn_input,
                attn_input,
                need_weights=self.capture_attention,
                average_attn_weights=False,
            )
        else:
            import torch.nn.functional as F

            B, L, D = attn_input.shape
            H = self.self_attn.num_heads
            d = D // H
            q, k, v = F.linear(
                attn_input, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias
            ).chunk(3, -1)
            q, k, v = [a.reshape(B, L, H, d).transpose(1, 2) for a in (q, k, v)]
            weights = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(d), -1)
            weights = weights * self.head_mask.to(weights).reshape(1, H, 1, 1)
            weights = F.dropout(
                weights, p=self.self_attn.dropout, training=self.training
            )
            attn_out = (weights @ v).transpose(1, 2).reshape(B, L, D)
            attn_out = F.linear(
                attn_out, self.self_attn.out_proj.weight, self.self_attn.out_proj.bias
            )
        if self.capture_attention:
            self.last_attention = weights.detach()
        x = x + self.attn_output(attn_out)

        # Feedforward with AdaLN
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_ffn) + shift_ffn
        x = x + self.ffn(x_norm)

        return x


class ManifoldEncoder(nn.Module):
    """
    Manifold Encoder phi_theta: R^352 x [0,1] -> R^256

    Maps multimodal character features x_c^t in R^352 and era time t in [0,1]
    to manifold coordinates z_c^t in R^256.

    Architecture (Section 3.1, Appendix E):
    - Input projection: 352 -> 256
    - 4 residual blocks with hidden dim 512
    - Time embedding: Sinusoidal encoding (64-dim)
    - Transformer: 12 layers with AdaLN for time conditioning
    - Output: 256-dim manifold coordinates
    """

    def __init__(
        self,
        input_dim: int = 352,
        manifold_dim: int = 256,
        num_layers: int = 12,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        time_emb_dim: int = 64,
        num_resblocks: int = 4,
        resblock_hidden_dim: int = 512,
        attention_inner_dim=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.manifold_dim = manifold_dim
        self.num_layers = num_layers

        # Time embedding
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)

        # Input projection
        self.input_proj = nn.Linear(input_dim, manifold_dim)

        # Five modality tokens, with block dimensions from Appendix M.
        self.modality_dims = (128, 64, 64, 64, 32)
        self.modality_projections = (
            nn.ModuleList([nn.Linear(dim, manifold_dim) for dim in self.modality_dims])
            if input_dim == 352
            else None
        )

        # Residual blocks (Appendix E: 4 residual blocks with hidden dim 512)
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(manifold_dim, resblock_hidden_dim, dropout)
                for _ in range(num_resblocks)
            ]
        )

        # Modality sequence positional buffer
        self.register_buffer(
            "pos_embedding", torch.zeros(1, 5 if input_dim == 352 else 1, manifold_dim)
        )

        # Transformer layers with AdaLN
        self.transformer_blocks = nn.ModuleList(
            [
                AdaLNTransformerBlock(
                    d_model=manifold_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    time_emb_dim=time_emb_dim,
                    attention_inner_dim=attention_inner_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(manifold_dim)

        # Output projection
        self.output_proj = nn.Linear(manifold_dim, manifold_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Encode multimodal features to manifold coordinates conditioned on era time.

        Implements: z_c^t = phi_theta(x_c^t, t)

        Args:
            x: Input features of shape (batch_size, input_dim=352)
            t: Era time of shape (batch_size,) with values in [0,1]

        Returns:
            Manifold coordinates of shape (batch_size, manifold_dim=256)
        """
        batch_size = x.shape[0]

        # Embed time
        time_emb = self.time_embedding(t)  # (batch_size, time_emb_dim)

        if self.modality_projections is not None:
            parts = torch.split(x, self.modality_dims, dim=-1)
            tokens = torch.stack(
                [proj(part) for proj, part in zip(self.modality_projections, parts)],
                dim=1,
            )
            # The common feature projection supplies a global residual context.
            x = tokens + self.input_proj(x).unsqueeze(1)
        else:
            x = self.input_proj(x).unsqueeze(1)
        for resblock in self.residual_blocks:
            x = resblock(x)

        # Add positional embedding
        x = x + self.pos_embedding

        # Apply transformer blocks with time conditioning
        for block in self.transformer_blocks:
            x = block(x, time_emb)

        # Remove sequence dimension
        x = x.mean(dim=1)  # pool modality tokens to (batch_size, manifold_dim)

        # Final normalization and projection
        x = self.final_norm(x)
        x = self.output_proj(x)

        return x

    def get_manifold_dim(self) -> int:
        """Return the dimension of the manifold coordinates."""
        return self.manifold_dim
