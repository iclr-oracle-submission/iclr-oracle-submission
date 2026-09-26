"""Conditional VP diffusion and probability-flow transport in frozen manifold coordinates.

Noise time u is separate from historical era h. Transport encodes a source-era
coordinate into the common noise space, then decodes it under the target-era
condition. VP/PF definitions: https://arxiv.org/abs/2011.13456 .
"""

import torch
from torch import nn
from torchdiffeq import odeint
from .manifold_encoder import SinusoidalTimeEmbedding
from .msef import MSEF


class ConditionalNoisePredictor(nn.Module):
    def __init__(self, dim, hidden=512, time_dim=64):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(dim + 2 * time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z, u, h):
        return self.net(torch.cat([z, self.time(u), self.time(h)], -1))


class ProbabilityFlowTransport(nn.Module):
    def __init__(
        self, encoder_config, hidden=512, beta_min=0.1, beta_max=20.0, noise_min=1e-3
    ):
        if not 0 < beta_min <= beta_max or not 0 < noise_min < 1:
            raise ValueError("Invalid VP schedule")
        super().__init__()
        self.encoder_model = MSEF.from_config(encoder_config)
        self.manifold_dim = encoder_config.manifold_dim
        self.score = ConditionalNoisePredictor(
            self.manifold_dim, hidden, encoder_config.time_embed_dim
        )
        self.beta_min, self.beta_max, self.noise_min = beta_min, beta_max, noise_min
        self.ode_rtol, self.ode_atol = encoder_config.rtol, encoder_config.atol
        for p in self.encoder_model.parameters():
            p.requires_grad = False

    def coefficients(self, u):
        log_alpha = (
            -0.25 * (self.beta_max - self.beta_min) * u.square()
            - 0.5 * self.beta_min * u
        )
        alpha = log_alpha.exp()
        sigma = (1 - alpha.square()).clamp_min(1e-12).sqrt()
        return alpha, sigma

    def drift(self, z, u, h):
        alpha, sigma = self.coefficients(u)
        score = -self.score(z, u, h) / sigma[:, None]
        beta = self.beta_min + u * (self.beta_max - self.beta_min)
        return -0.5 * beta[:, None] * (z + score)

    def dsm_loss(self, clean, h, generator=None):
        u = (
            torch.rand(len(clean), device=clean.device, generator=generator)
            * (1 - self.noise_min)
            + self.noise_min
        )
        noise = torch.randn(
            clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
        )
        alpha, sigma = self.coefficients(u)
        noisy = alpha[:, None] * clean + sigma[:, None] * noise
        return (self.score(noisy, u, h) - noise).square().sum(-1).mean()

    def encode(self, x, t):
        return self.encoder_model.encode(x, t)

    def predict_survival(self, z, t):
        return self.encoder_model.predict_survival(z, t)

    def noise_flow(self, z, h, a, b):
        if a == b:
            return z

        def field(u, state):
            noise_time = u.to(state).expand(len(state))
            history = state.new_full((len(state),), float(h))
            return self.drift(state, noise_time, history)

        return odeint(
            field,
            z,
            z.new_tensor([a, b]),
            rtol=self.ode_rtol,
            atol=self.ode_atol,
            method="dopri5",
        )[-1]

    def flow_forward(self, z, t1, t2, **kwargs):
        if float(t1) == float(t2):
            return z
        noise = self.noise_flow(z, float(t1), self.noise_min, 1.0)
        return self.noise_flow(noise, float(t2), 1.0, self.noise_min)

    def flow_backward(self, z, t2, t1, **kwargs):
        return self.flow_forward(z, t2, t1, **kwargs)

    def flow_batch(self, z, a, b):
        pairs, inverse = torch.unique(
            torch.stack([a, b], -1), dim=0, return_inverse=True
        )
        out = torch.zeros_like(z)
        for i, pair in enumerate(pairs):
            idx = (inverse == i).nonzero(as_tuple=True)[0]
            out = out.index_copy(
                0, idx, self.flow_forward(z.index_select(0, idx), *pair.tolist())
            )
        return out
