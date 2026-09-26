"""
MSEF (Manifold-based Script Evolution Framework)

This module implements the unified MSEF framework that combines the manifold
encoder, velocity field, and survival network to model script evolution.

The framework uses Neural ODEs to model continuous evolution on a learned
manifold and predicts character survival/extinction.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable
from torchdiffeq import odeint

from .manifold_encoder import ManifoldEncoder
from .velocity_field import VelocityField
from .survival_network import SurvivalNetwork


class MSEF(nn.Module):
    """
    Manifold-based Script Evolution Framework

    The MSEF framework consists of three main components:
    1. Manifold Encoder phi_theta: Maps multimodal features to manifold coordinates
    2. Velocity Field v_theta: Defines ODE dynamics on the manifold
    3. Survival Network S_psi: Predicts character survival probability

    Key Operations:
    - encode(x, t): Maps features to manifold space at time t
    - flow_forward(z, t1, t2): Evolves z from time t1 to t2 using ODE
    - flow_backward(z, t2, t1): Reverse evolution from t2 to t1
    - predict_survival(z_obi, t): Predicts survival probability

    Mathematical Formulation:
    - Forward flow (Eq. 2): Phi^{t1->t2}(z) = z + int_{t1}^{t2} v(z(tau), tau) dtau
    - Backward flow (Eq. 3): Phi^{t2->t1}(z) = z - int_{t1}^{t2} v(z(tau), tau) dtau
    - Invertibility: Phi^{t1->t2} o Phi^{t2->t1} = I
    """

    def __init__(
        self,
        input_dim: int = 352,
        manifold_dim: int = 256,
        encoder_layers: int = 12,
        encoder_heads: int = 8,
        encoder_dim_feedforward: int = 1024,
        velocity_hidden_dim: int = 512,
        velocity_layers: int = 3,
        survival_hidden_dim: int = 128,
        survival_layers: int = 3,
        time_emb_dim: int = 64,
        dropout: float = 0.1,
        num_resblocks: int = 4,
        resblock_hidden_dim: int = 512,
        ode_solver: str = "dopri5",
        ode_rtol: float = 1e-5,
        ode_atol: float = 1e-5,
        disabled_modalities=(),
        coarse_time=False,
        use_ode=True,
        attention_inner_dim=None,
    ):
        super().__init__()

        self.disabled_modalities = tuple(disabled_modalities)
        self.coarse_time, self.use_ode = coarse_time, use_ode
        self.input_dim = input_dim
        self.manifold_dim = manifold_dim
        self.ode_solver = ode_solver
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol

        # Initialize components
        self.encoder = ManifoldEncoder(
            input_dim=input_dim,
            manifold_dim=manifold_dim,
            num_layers=encoder_layers,
            nhead=encoder_heads,
            dim_feedforward=encoder_dim_feedforward,
            dropout=dropout,
            time_emb_dim=time_emb_dim,
            num_resblocks=num_resblocks,
            resblock_hidden_dim=resblock_hidden_dim,
            attention_inner_dim=attention_inner_dim,
        )

        self.velocity_field = VelocityField(
            manifold_dim=manifold_dim,
            hidden_dim=velocity_hidden_dim,
            time_emb_dim=time_emb_dim,
            num_layers=velocity_layers,
            use_spectral_norm=True,
        )

        self.survival_network = SurvivalNetwork(
            manifold_dim=manifold_dim,
            hidden_dim=survival_hidden_dim,
            num_layers=survival_layers,
            dropout=dropout,
            time_emb_dim=time_emb_dim,
        )

    @classmethod
    def from_config(cls, config) -> "MSEF":
        """
        Create MSEF model from a configuration object (e.g., MSEFConfig).

        Args:
            config: Configuration object with model hyperparameters.

        Returns:
            Initialized MSEF model.
        """
        return cls(
            input_dim=config.feature_dim,
            manifold_dim=config.manifold_dim,
            encoder_layers=config.manifold_num_layers,
            encoder_heads=config.manifold_num_heads,
            encoder_dim_feedforward=getattr(config, "encoder_dim_feedforward", 1024),
            velocity_hidden_dim=config.velocity_hidden_dim,
            velocity_layers=config.velocity_num_layers,
            survival_hidden_dim=config.survival_hidden_dim,
            survival_layers=getattr(config, "survival_num_layers", 3),
            time_emb_dim=config.time_embed_dim,
            dropout=getattr(config, "dropout", 0.1),
            num_resblocks=getattr(config, "resblocks", 4),
            resblock_hidden_dim=getattr(config, "resblock_hidden_dim", 512),
            ode_solver=config.method,
            ode_rtol=config.rtol,
            ode_atol=config.atol,
            disabled_modalities=config.disabled_modalities,
            coarse_time=config.coarse_time,
            use_ode=config.use_ode,
            attention_inner_dim=config.attention_inner_dim,
        )

    def encode(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Encode multimodal features to manifold coordinates at time t.

        Implements: z_c^t = phi_theta(x_c^t, t)

        Args:
            x: Input features of shape (batch_size, input_dim=352)
            t: Time of shape (batch_size,) with values in [0,1]

        Returns:
            Manifold coordinates of shape (batch_size, manifold_dim=256)
        """
        x = x.clone()
        blocks = {
            "visual": (0, 128),
            "structural": (128, 192),
            "semantic": (192, 256),
            "context": (256, 320),
            "spatiotemporal": (320, 352),
        }
        for name in self.disabled_modalities:
            if name not in blocks:
                raise ValueError("Unknown modality " + name)
            a, b = blocks[name]
            x[:, a:b] = 0
        t = self.condition_time(t)
        if self.coarse_time and "spatiotemporal" not in self.disabled_modalities:
            freq = torch.exp(
                -torch.log(x.new_tensor(10000.0))
                * torch.arange(0, 16, 2, device=x.device)
                / 16
            )
            phase = t[:, None] * 10000 * freq
            x[:, 320:336:2] = torch.sin(phase)
            x[:, 321:336:2] = torch.cos(phase)
        return self.encoder(x, t)

    def condition_time(self, t):
        if self.coarse_time:
            centers = t.new_tensor([0.05, 0.35, 0.70, 0.85, 1.0])
            t = centers[(t[:, None] - centers).abs().argmin(-1)]
        return t

    def flow_forward(
        self, z: torch.Tensor, t1: float, t2: float, **ode_kwargs
    ) -> torch.Tensor:
        """
        Flow manifold coordinates forward in time from t1 to t2.

        Implements forward flow (Eq. 2):
            Phi^{t1->t2}(z) = z + int_{t1}^{t2} v(z(tau), tau) dtau

        Args:
            z: Initial manifold coordinates of shape (batch_size, manifold_dim)
            t1: Start time (scalar)
            t2: End time (scalar)
            **ode_kwargs: Additional arguments for ODE solver

        Returns:
            Final manifold coordinates of shape (batch_size, manifold_dim)
        """
        if isinstance(t1, torch.Tensor):
            t1 = t1.item()
        if isinstance(t2, torch.Tensor):
            t2 = t2.item()

        if not self.use_ode or t1 == t2:
            return z
        t_span = torch.tensor([t1, t2], dtype=z.dtype, device=z.device)

        rtol = ode_kwargs.pop("rtol", self.ode_rtol)
        atol = ode_kwargs.pop("atol", self.ode_atol)
        method = ode_kwargs.pop("method", self.ode_solver)

        trajectory = odeint(
            func=self.velocity_field.ode_func,
            y0=z,
            t=t_span,
            method=method,
            rtol=rtol,
            atol=atol,
            **ode_kwargs
        )

        z_final = trajectory[-1]
        return z_final

    def flow_batch(self, z, source_times, target_times):
        """Group identical time spans; preserve per-occurrence times and gradients."""
        source_times = self.condition_time(source_times.to(z.device).reshape(-1))
        target_times = self.condition_time(target_times.to(z.device).reshape(-1))
        if len(source_times) != len(z) or len(target_times) != len(z):
            raise ValueError("Times must match batch size")
        spans, inverse = torch.unique(
            torch.stack([source_times, target_times], -1), dim=0, return_inverse=True
        )
        output = torch.zeros_like(z)
        for group, span in enumerate(spans):
            indices = (inverse == group).nonzero(as_tuple=True)[0]
            initial = z.index_select(0, indices)
            a, b = span.tolist()
            final = initial if a == b else self.flow_forward(initial, a, b)
            output = output.index_copy(0, indices, final)
        return output

    def flow_backward(
        self, z: torch.Tensor, t2: float, t1: float, **ode_kwargs
    ) -> torch.Tensor:
        """
        Flow manifold coordinates backward in time from t2 to t1.

        Implements backward flow (Eq. 3):
            Phi^{t2->t1}(z) = z - int_{t1}^{t2} v(z(tau), tau) dtau

        Args:
            z: Initial manifold coordinates of shape (batch_size, manifold_dim)
            t2: Start time (scalar)
            t1: End time (scalar, t1 < t2)
            **ode_kwargs: Additional arguments for ODE solver

        Returns:
            Final manifold coordinates of shape (batch_size, manifold_dim)
        """
        return self.flow_forward(z, t2, t1, **ode_kwargs)

    def predict_survival(self, z_obi: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict character survival probability.

        Args:
            z_obi: OBI era (t=0) manifold coordinates of shape (batch_size, manifold_dim)
            t: Target time of shape (batch_size,) with values in [0,1]

        Returns:
            Survival probability of shape (batch_size, 1) with values in [0, 1]
        """
        return self.survival_network(z_obi, t)

    def reconstruct(
        self, z: torch.Tensor, t1: float, t2: float, **ode_kwargs
    ) -> torch.Tensor:
        """
        Test invertibility by flowing forward then backward.

        Implements: z_recon = Phi^{t2->t1}(Phi^{t1->t2}(z))

        Args:
            z: Initial manifold coordinates of shape (batch_size, manifold_dim)
            t1: Start time (scalar)
            t2: Intermediate time (scalar)
            **ode_kwargs: Additional arguments for ODE solver

        Returns:
            Reconstructed coordinates of shape (batch_size, manifold_dim)
        """
        z_forward = self.flow_forward(z, t1, t2, **ode_kwargs)
        z_recon = self.flow_backward(z_forward, t2, t1, **ode_kwargs)
        return z_recon

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, target_t: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Full forward pass through the MSEF framework.

        Args:
            x: Input features of shape (batch_size, input_dim)
            t: Source time of shape (batch_size,)
            target_t: Optional target time for flow (batch_size,)

        Returns:
            Dictionary containing:
                - 'z': Manifold coordinates at time t
                - 'z_flow': Manifold coordinates at target_t (if provided)
                - 'survival_prob': Survival probability (if target_t provided)
        """
        z = self.encode(x, t)
        output = {"z": z}

        if target_t is not None:
            output["z_flow"] = self.flow_batch(z, t, target_t)

            z_obi = self.flow_batch(z, t, torch.full_like(t, 0.05))
            output["survival_prob"] = self.predict_survival(z_obi, target_t)

        return output

    def get_manifold_dim(self) -> int:
        """Return the dimension of the manifold."""
        return self.manifold_dim

    def get_velocity_lipschitz(self) -> float:
        """Return estimated Lipschitz constant of the velocity field."""
        return self.velocity_field.get_lipschitz_constant()
