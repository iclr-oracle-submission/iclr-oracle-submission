"""Ordered script coordinates, final manuscript Table 1 and Appendix K.

Script category, scribal group and archaeological date are separate fields.
The default checkpoints are modeling choices, not calendar-year estimates.
"""
from typing import Optional, Dict, Tuple
import numpy as np

ERA_TIMES: Dict[str, float] = {
    "OBI": 0.05, "Bronze": 0.35, "Seal": 0.70, "Clerical": 0.85, "Regular": 1.0,
}
ERA_RANGES: Dict[str, Tuple[float, float]] = {
    "OBI": (0.0, 0.30), "Bronze": (0.30, 0.70), "Seal": (0.70, 0.85),
    "Clerical": (0.85, 1.0), "Regular": (1.0, 1.0),
}
MAIN_PERIODS = list(ERA_TIMES)

def get_time_for_era(era: str, subperiod: Optional[str] = None) -> float:
    if subperiod is not None:
        raise ValueError("Subperiod/group labels require an explicitly justified assigned time")
    return ERA_TIMES[era]

def get_era_for_time(t: float, include_subperiods: bool = False) -> str:
    if not np.isfinite(t) or not 0 <= t <= 1:
        raise ValueError("Time must be finite and in [0, 1]")
    for era, (lo, hi) in ERA_RANGES.items():
        if lo <= t < hi or era == "Regular" and t == 1:
            return era
    raise ValueError("No script interval for time")

def is_adjacent_period(era1: str, era2: str) -> bool:
    return abs(MAIN_PERIODS.index(era1) - MAIN_PERIODS.index(era2)) == 1

def get_adjacent_pairs() -> list:
    return [(a, b, ERA_TIMES[a], ERA_TIMES[b]) for a,b in zip(MAIN_PERIODS, MAIN_PERIODS[1:])]

def encode_time_vector(t: float, encoding_dim: int = 16) -> np.ndarray:
    """
    Encode time value into a sinusoidal positional encoding vector.

    This follows the positional encoding scheme used in transformers,
    adapted for temporal encoding in the MSEF model (Appendix K).

    Args:
        t: Normalized time value in [0, 1]
        encoding_dim: Dimension of the encoding (must be even)

    Returns:
        Time encoding vector of shape (encoding_dim,)

    Raises:
        ValueError: If encoding_dim is not even
    """
    if encoding_dim % 2 != 0:
        raise ValueError(f"Encoding dimension must be even, got {encoding_dim}")

    # Create positional encoding
    position = t * 10000  # Scale time to larger range
    div_term = np.exp(np.arange(0, encoding_dim, 2) * -(np.log(10000.0) / encoding_dim))

    encoding = np.zeros(encoding_dim)
    encoding[0::2] = np.sin(position * div_term)
    encoding[1::2] = np.cos(position * div_term)

    return encoding.astype(np.float32)


def get_time_difference(era1: str, era2: str) -> float:
    """
    Calculate the temporal distance between two eras.

    Args:
        era1: First era name
        era2: Second era name

    Returns:
        Absolute time difference

    Examples:
        >>> get_time_difference('OBI', 'Bronze')
        0.30
        >>> get_time_difference('Seal', 'Regular')
        0.30
    """
    time1 = ERA_TIMES[era1]
    time2 = ERA_TIMES[era2]
    return abs(time2 - time1)
