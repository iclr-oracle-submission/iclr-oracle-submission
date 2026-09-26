"""
MSEF: Manifold-based Script Evolution Framework

This package implements the MSEF framework for deciphering Oracle Bone Inscriptions
through evolutionary character analysis across multiple historical eras.

Main components:
    - MSEF: Core model with manifold encoder, velocity field, and survival predictor
    - CascadedBidirectionalDecipherment: CBED algorithm for decipherment
    - FGCCESDataset: Dataset loader for evolution pairs
    - MSEFLoss: Multi-objective loss function

Reference: Section 3, 4, 5 of the paper
"""

from .config import MSEFConfig
from .models.msef import MSEF
from .losses import MSEFLoss
from .data.dataset import FGCCESDataset
from .algorithms.cbed import (
    CascadedBidirectionalDecipherment,
    DeciphermentResult,
    DeciphermentOutput,
)

__version__ = "1.0.0"

__all__ = [
    "MSEFConfig",
    "MSEF",
    "MSEFLoss",
    "FGCCESDataset",
    "CascadedBidirectionalDecipherment",
    "DeciphermentResult",
    "DeciphermentOutput",
]
