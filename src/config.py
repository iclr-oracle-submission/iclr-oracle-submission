"""
Configuration module for MSEF (Manifold-based Script Evolution Framework).

Model settings and configurable defaults; see IMPLEMENTATION.md.
Era time values follow Table 1 and Appendix C.
"""

import math
from dataclasses import dataclass
from typing import Literal, Dict


@dataclass
class MSEFConfig:
    """
    Configuration for MSEF model and training.

    Architecture, optimization and inference settings for the model variants.

    Architecture Parameters:
        feature_dim: Input feature dimension (Section 3.1)
        manifold_dim: Latent manifold dimension (Section 3.2)
        manifold_num_layers: Number of transformer layers in manifold encoder
        manifold_num_heads: Number of attention heads in manifold encoder
        encoder_dim_feedforward: Feedforward dimension in encoder transformer
        velocity_hidden_dim: Hidden dimension for velocity field network (Section 3.3)
        velocity_num_layers: Number of layers in velocity field MLP
        survival_hidden_dim: Hidden dimension for survival predictor (Section 3.4)
        survival_num_layers: Number of layers in survival network MLP
        time_embed_dim: Dimension for time embeddings
        resblocks: Number of residual blocks in encoder (Appendix E)
        resblock_hidden_dim: Hidden dimension for residual blocks (Appendix E)
        dropout: Dropout rate

    ODE Parameters:
        method: ODE solver method (Section 3.3)
        rtol: Relative tolerance for ODE solver
        atol: Absolute tolerance for ODE solver

    Training Parameters:
        optimizer: Optimization algorithm
        lr: Learning rate
        weight_decay: Weight decay for regularization
        batch_size: Training batch size
        epochs: Number of training epochs
        survival_epochs: Number of survival-training epochs
        grad_clip: Gradient clipping max norm

    Loss Weights (Equation 10):
        lambda_adj: Weight for adjacent-era evolution loss (lambda_1)
        lambda_skip: Weight for skip-era evolution loss (lambda_2)
        lambda_full: Weight for full-trajectory evolution loss (lambda_3)
        lambda_cyc: Weight for cycle-consistency loss (lambda_4)
        lambda_surv: Weight of the independently optimized survival BCE (Table 28)

    Inference Parameters (Algorithm 1):
        retrieval_depth_K: Number of nearest neighbors to retrieve per era
        survival_threshold_tau: Survival probability threshold
    """

    # Architecture parameters
    feature_dim: int = 352
    manifold_dim: int = 256
    manifold_num_layers: int = 12
    manifold_num_heads: int = 8
    encoder_dim_feedforward: int = 1024
    velocity_hidden_dim: int = 512
    velocity_num_layers: int = 3
    survival_hidden_dim: int = 128
    survival_num_layers: int = 3
    time_embed_dim: int = 64
    resblocks: int = 4
    resblock_hidden_dim: int = 512
    dropout: float = 0.1

    # ODE parameters
    method: Literal["dopri5", "rk4", "euler"] = "dopri5"
    rtol: float = 1e-5
    atol: float = 1e-5

    # Training parameters
    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 100
    survival_epochs: int = 20
    grad_clip: float = 1.0

    # Loss weights (Equation 10)
    lambda_adj: float = 1.0  # lambda_1: adjacent-era evolution
    lambda_skip: float = 0.5  # lambda_2: skip-era evolution
    lambda_full: float = 0.3  # lambda_3: full-trajectory evolution
    lambda_cyc: float = 0.5  # lambda_4: cycle-consistency
    lambda_surv: float = 0.2  # scale of the separate survival objective

    # Inference parameters (Algorithm 1)
    retrieval_depth_K: int = 5
    survival_threshold_tau: float = 0.1

    # Experiment variants
    disabled_modalities: tuple = ()
    coarse_time: bool = False
    use_ode: bool = True
    train_survival: bool = True
    pair_types: tuple = ("adjacent", "skip", "complete")
    training_eras: tuple = ("OBI", "Bronze", "Seal", "Clerical", "Regular")
    max_training_pairs: int = None
    attention_inner_dim: int = None

    # Era definitions (Section 2.1)
    eras: tuple = ("OBI", "Bronze", "Seal", "Clerical", "Regular")
    era_times: Dict[str, float] = None

    def __post_init__(self):
        """Initialize era time mappings using historically-grounded values (Table 1)."""
        for name in (
            "manifold_dim",
            "manifold_num_layers",
            "manifold_num_heads",
            "encoder_dim_feedforward",
            "velocity_hidden_dim",
            "velocity_num_layers",
            "survival_hidden_dim",
            "survival_num_layers",
            "time_embed_dim",
            "resblock_hidden_dim",
            "batch_size",
            "epochs",
            "retrieval_depth_K",
        ):
            if (
                not isinstance(getattr(self, name), int)
                or isinstance(getattr(self, name), bool)
                or getattr(self, name) < 1
            ):
                raise ValueError(name + " must be a positive integer")
        if self.train_survival and self.survival_epochs < 1:
            raise ValueError(
                "survival_epochs must be positive when survival training is enabled"
            )
        if self.optimizer.lower() != "adamw":
            raise ValueError("Supported optimizer: adamw")
        if self.method not in ("dopri5", "rk4", "euler"):
            raise ValueError("Unsupported ODE method")
        for name in ("lr", "rtol", "atol", "grad_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive")
        for name in (
            "weight_decay",
            "lambda_adj",
            "lambda_skip",
            "lambda_full",
            "lambda_cyc",
            "lambda_surv",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name + " must be finite and nonnegative")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if not 0 <= self.survival_threshold_tau <= 1:
            raise ValueError("Survival threshold must be in [0,1]")
        if self.resblocks < 0:
            raise ValueError("resblocks must be nonnegative")
        if self.feature_dim != 352:
            raise ValueError("Five-modality input must have 352 dimensions")
        if self.time_embed_dim < 4 or self.time_embed_dim % 2:
            raise ValueError("time_embed_dim must be even and >=4")
        if (self.attention_inner_dim or self.manifold_dim) % self.manifold_num_heads:
            raise ValueError("Attention width must divide into heads")
        if self.max_training_pairs is not None and self.max_training_pairs <= 0:
            raise ValueError("max_training_pairs must be positive")
        if not self.pair_types:
            raise ValueError("At least one pair type is required")
        if not self.training_eras:
            raise ValueError("At least one training era is required")
        if not set(self.disabled_modalities) <= {
            "visual",
            "structural",
            "semantic",
            "context",
            "spatiotemporal",
        }:
            raise ValueError("Unknown disabled modality")
        if self.attention_inner_dim is not None and self.attention_inner_dim < 1:
            raise ValueError("attention_inner_dim must be positive")
        if not set(self.pair_types) <= {"adjacent", "skip", "complete"}:
            raise ValueError("Unknown pair type")
        if not set(self.training_eras) <= set(self.eras):
            raise ValueError("Unknown training era")
        if self.era_times is None:
            # Historically-grounded normalized time values (Table 1, Appendix C)
            # OBI: center of [0.00, 0.10]
            # Bronze: center of [0.15, 0.55]
            # Seal: 0.70
            # Clerical: 0.85
            # Regular: 1.00
            self.era_times = {
                "OBI": 0.05,
                "Bronze": 0.35,
                "Seal": 0.70,
                "Clerical": 0.85,
                "Regular": 1.00,
            }

    def get_era_time(self, era: str) -> float:
        """Get normalized time value for an era."""
        return self.era_times[era]

    def get_era_index(self, era: str) -> int:
        """Get index of an era in the evolution sequence."""
        return self.eras.index(era)

    def get_adjacent_era_pairs(self):
        """Get list of adjacent era pairs with their times."""
        pairs = []
        for i in range(len(self.eras) - 1):
            era_src = self.eras[i]
            era_tgt = self.eras[i + 1]
            pairs.append(
                (era_src, era_tgt, self.era_times[era_src], self.era_times[era_tgt])
            )
        return pairs
