"""Conventional CNN adapter from natural-object images to frozen glyph coordinates."""

import torch
from torch import nn
import torch.nn.functional as F


class PhotoAssociation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.photos = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, dim),
        )
        self.log_scale = nn.Parameter(torch.tensor(2.302585))

    def forward(self, glyph, photos):
        b, k = photos.shape[:2]
        vectors = self.photos(photos.flatten(0, 1)).reshape(b, k, -1)
        return (F.normalize(glyph, dim=-1)[:, None] * F.normalize(vectors, dim=-1)).sum(
            -1
        ) * self.log_scale.clamp(0, 4.60517).exp()
