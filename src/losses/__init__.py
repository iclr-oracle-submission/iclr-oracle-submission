"""
Loss functions for MSEF paper implementation.

Exports:
    - MSEFLoss: Multi-Scale Evolution Flow loss (combination of all loss terms)
"""

from .evolution_losses import MSEFLoss

__all__ = ["MSEFLoss"]
