"""
Survival Network for MSEF Framework

This module implements the survival network S_psi that predicts character extinction
probability based on manifold coordinates, as described in Section 3.3.

The network takes OBI (Oracle Bone Inscription) era manifold representations
and predicts the probability that a character becomes extinct by a target time.

Uses sinusoidal time embedding consistent with the velocity field.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for continuous time representation.

    Maps scalar time t to a high-dimensional representation using sinusoidal
    functions at different frequencies. Consistent with velocity field embedding.
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


class SurvivalNetwork(nn.Module):
    """
    Survival Network S_psi: R^256 x [0,1] -> [0,1]

    Predicts the probability that a character becomes extinct by time t, given
    its manifold representation at the OBI era (t=0).

    Architecture (Section 3.3):
    - Input: z_c^0 in R^256 (OBI manifold coordinates) and t in [0,1] (target time)
    - Time encoding: 64-dim sinusoidal embedding (consistent with velocity field)
    - Network: MLP with hidden dimension 128
    - Output: s in [0,1] (survival probability)
    - Activation: Sigmoid for probability output

    The survival network learns to predict:
        P(character c extinct by time t | z_c^0)

    This is used for the survival loss (Equation 9):
        L_survival = -[y_c log(s_c) + (1-y_c)log(1-s_c)]
    where y_c in {0,1} indicates extinction status.
    """

    def __init__(
        self,
        manifold_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        time_emb_dim: int = 64,
    ):
        super().__init__()

        self.manifold_dim = manifold_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_emb_dim = time_emb_dim

        # Sinusoidal time embedding (consistent with velocity field)
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)

        # Build MLP
        layers = []

        # Input layer: manifold coordinates + time embedding
        input_dim = manifold_dim + time_emb_dim  # 256 + 64 = 320
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer: single probability value
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())

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

    def forward(self, z_obi: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict survival probability.

        Implements: s_c = S_psi(z_c^0, t)

        Args:
            z_obi: OBI era manifold coordinates of shape (batch_size, manifold_dim=256)
            t: Target time of shape (batch_size,) or (batch_size, 1) with values in [0,1]

        Returns:
            Survival probability of shape (batch_size, 1) with values in [0, 1]
        """
        # Ensure t has the right shape for time embedding
        if t.dim() == 2:
            t = t.squeeze(-1)  # (batch_size,)

        # Compute sinusoidal time embedding
        time_emb = self.time_embedding(t)  # (batch_size, time_emb_dim)

        # Concatenate z_obi and time embedding
        zt = torch.cat(
            [z_obi, time_emb], dim=-1
        )  # (batch_size, manifold_dim + time_emb_dim)

        # Compute survival probability
        survival_prob = self.mlp(zt)  # (batch_size, 1)

        return survival_prob

    def predict(
        self, z_obi: torch.Tensor, t: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Predict binary survival labels based on threshold.

        Args:
            z_obi: OBI era manifold coordinates of shape (batch_size, manifold_dim)
            t: Target time of shape (batch_size,) or (batch_size, 1)
            threshold: Probability threshold for classification (default: 0.5)

        Returns:
            Binary predictions of shape (batch_size, 1)
            1 = survived, 0 = extinct
        """
        with torch.no_grad():
            probs = self.forward(z_obi, t)
            predictions = (probs >= threshold).float()

        return predictions

    def compute_survival_loss(
        self,
        z_obi: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute binary cross-entropy survival loss (Equation 9).

        L_survival = -[y_c log(s_c) + (1-y_c)log(1-s_c)]

        Args:
            z_obi: OBI era manifold coordinates of shape (batch_size, manifold_dim)
            t: Target time of shape (batch_size,) or (batch_size, 1)
            labels: Ground truth survival labels of shape (batch_size,) or (batch_size, 1)
                   1 = survived, 0 = extinct
            reduction: Loss reduction method ('mean', 'sum', or 'none')

        Returns:
            Survival loss (scalar if reduction is 'mean' or 'sum')
        """
        pred_probs = self.forward(z_obi, t)  # (batch_size, 1)

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)

        loss = nn.functional.binary_cross_entropy(
            pred_probs, labels.float(), reduction=reduction
        )

        return loss
