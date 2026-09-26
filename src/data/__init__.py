"""
Data module for MSEF paper implementation.

Exports:
    - FGCCESDataset: Fine-Grained Chinese Character Evolution Shapes dataset
    - EvolutionPairCollator: Custom collator for batching evolution pairs
"""

from .dataset import FGCCESDataset, EvolutionPairCollator

__all__ = ["FGCCESDataset", "EvolutionPairCollator"]
