"""
Velocity Field for MSEF Framework

This module implements the velocity field v_θ(z, t) that governs the Neural ODE
dynamics on the manifold, as described in Section 3.2.

The velocity field defines how manifold coordinates evolve over time according to
the ODE: dz/dt = v_θ(z, t)
"""

import math
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from typing import Tuple


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for continuous time representation.

    Maps scalar time t to a high-dimensional representation using sinusoidal
    functions at different frequencies.
    """

    def __init__(self, embedding_dim: int = 64):
        """
        Args:
            embedding_dim: Dimension of the time embedding (default: 64)
        """
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time tensor of shape (batch_size,) or (batch_size, 1)

        Returns:
            Time embeddings of shape (batch_size, embedding_dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        half_dim = self.embedding_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t * embeddings.unsqueeze(0)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings


class VelocityField(nn.Module):
    """
    Velocity Field v_θ: R^256 × [0,1] → R^256

    Defines the temporal evolution of manifold coordinates through the ODE:
        dz/dt = v_θ(z, t)

    Architecture (Section 3.2):
    - Input: z ∈ R^256 (manifold coordinates) and t ∈ [0,1] (time)
    - Time encoding: 64-dim sinusoidal embedding
    - Network: 3-layer MLP with hidden dimension 512
    - Activation: Tanh (bounded, smooth)
    - Regularization: Spectral normalization for Lipschitz continuity
    - Output: dz/dt ∈ R^256

    Key properties:
    - Lipschitz continuity ensures ODE stability and unique solutions
    - Smooth activations (Tanh) ensure differentiability
    - Time conditioning allows the velocity field to adapt across eras

    The forward and backward flows defined by this velocity field satisfy:
        Φ_θ^{t_1→t_2}(z) ∘ Φ_θ^{t_2→t_1}(z) = I  (Equation 2, 3)
    """

    def __init__(
        self,
        manifold_dim: int = 256,
        hidden_dim: int = 512,
        time_emb_dim: int = 64,
        num_layers: int = 3,
        use_spectral_norm: bool = True,
    ):
        """
        Args:
            manifold_dim: Dimension of manifold coordinates (default: 256)
            hidden_dim: Hidden dimension of MLP layers (default: 512)
            time_emb_dim: Dimension of time embedding (default: 64)
            num_layers: Number of MLP layers (default: 3)
            use_spectral_norm: Whether to use spectral normalization (default: True)
        """
        super().__init__()

        self.manifold_dim = manifold_dim
        self.hidden_dim = hidden_dim
        self.time_emb_dim = time_emb_dim
        self.num_layers = num_layers

        # Time embedding
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)

        # Build MLP layers
        layers = []

        # Input layer: concatenate z and time embedding
        input_dim = manifold_dim + time_emb_dim
        if use_spectral_norm:
            layers.append(
                spectral_norm(nn.Linear(input_dim, hidden_dim), n_power_iterations=5)
            )
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(num_layers - 2):
            if use_spectral_norm:
                layers.append(
                    spectral_norm(
                        nn.Linear(hidden_dim, hidden_dim), n_power_iterations=5
                    )
                )
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # Output layer
        if use_spectral_norm:
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, manifold_dim), n_power_iterations=5)
            )
        else:
            layers.append(nn.Linear(hidden_dim, manifold_dim))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the velocity field at manifold coordinates z and time t.

        Implements: dz/dt = v_θ(z, t)

        Args:
            z: Manifold coordinates of shape (batch_size, manifold_dim=256)
            t: Time of shape (batch_size,) with values in [0,1]

        Returns:
            Velocity (time derivative) of shape (batch_size, manifold_dim=256)
        """
        # Embed time
        time_emb = self.time_embedding(t)  # (batch_size, time_emb_dim)

        # Concatenate z and time embedding
        zt = torch.cat(
            [z, time_emb], dim=-1
        )  # (batch_size, manifold_dim + time_emb_dim)

        # Compute velocity
        velocity = self.mlp(zt)  # (batch_size, manifold_dim)

        return velocity

    def ode_func(self, t: float, z: torch.Tensor) -> torch.Tensor:
        """
        ODE function compatible with torchdiffeq.odeint.

        This function signature is required by the ODE solver.

        Args:
            t: Scalar time (float)
            z: Manifold coordinates of shape (batch_size, manifold_dim)

        Returns:
            Velocity of shape (batch_size, manifold_dim)
        """
        # Convert scalar time to tensor matching z's batch size
        batch_size = z.shape[0]
        t_tensor = torch.as_tensor(t, dtype=z.dtype, device=z.device).expand(batch_size)

        return self.forward(z, t_tensor)

    def get_lipschitz_constant(self) -> float:
        """
        Estimate the Lipschitz constant of the velocity field.

        With spectral normalization, the Lipschitz constant is bounded by
        the product of spectral norms across layers.

        Returns:
            Estimated Lipschitz constant
        """
        lipschitz = 1.0
        for module in self.modules():
            if isinstance(module, nn.Linear) and hasattr(module, "weight_sigma"):
                # Spectral normalized layer
                lipschitz *= module.weight_sigma.item()
            elif isinstance(module, nn.Linear):
                # Regular layer - use spectral norm of weight matrix
                lipschitz *= torch.linalg.matrix_norm(module.weight, ord=2).item()

        return lipschitz
