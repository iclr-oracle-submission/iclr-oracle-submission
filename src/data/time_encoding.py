"""
Time encoding for historical periods in Chinese character evolution.

This module implements the temporal encoding scheme described in Table 1 of the MSEF paper.
Time values are normalized to [0, 1] range representing the chronological progression of scripts.

Historical Periods:
    - OBI (Oracle Bone Inscriptions): t=0.00-0.10
        - Bin: 0.00-0.02
        - Chu: 0.02-0.04
        - Li: 0.04-0.06
        - Zi: 0.06-0.08
        - Huang: 0.08-0.10
    - Bronze: t=0.15-0.55
        - Early Western Zhou: 0.15-0.25
        - Late Western Zhou: 0.25-0.35
        - Spring & Autumn: 0.35-0.45
        - Warring States: 0.45-0.55
    - Seal: t=0.70
    - Clerical: t=0.85
    - Regular: t=1.00
"""

from typing import Optional, Dict, Tuple
import numpy as np


# Time encoding dictionary mapping eras to their temporal values
# Based on Table 1 from the MSEF paper
ERA_TIMES: Dict[str, float] = {
    # OBI sub-periods (t=0.00-0.10)
    "OBI": 0.05,  # Center of OBI period
    "Bin": 0.01,
    "Chu": 0.03,
    "Li": 0.05,
    "Zi": 0.07,
    "Huang": 0.09,
    # Bronze sub-periods (t=0.15-0.55)
    "Bronze": 0.35,  # Center of Bronze period
    "Early Western Zhou": 0.20,
    "Late Western Zhou": 0.30,
    "Spring&Autumn": 0.40,
    "Warring States": 0.50,
    # Later scripts
    "Seal": 0.70,
    "Clerical": 0.85,
    "Regular": 1.00,
}

# Period ranges for validation and inverse lookup
ERA_RANGES: Dict[str, Tuple[float, float]] = {
    "OBI": (0.00, 0.10),
    "Bin": (0.00, 0.02),
    "Chu": (0.02, 0.04),
    "Li": (0.04, 0.06),
    "Zi": (0.06, 0.08),
    "Huang": (0.08, 0.10),
    "Bronze": (0.15, 0.55),
    "Early Western Zhou": (0.15, 0.25),
    "Late Western Zhou": (0.25, 0.35),
    "Spring&Autumn": (0.35, 0.45),
    "Warring States": (0.45, 0.55),
    "Seal": (0.68, 0.72),
    "Clerical": (0.83, 0.87),
    "Regular": (0.98, 1.00),
}

# Main period identifiers
MAIN_PERIODS = ["OBI", "Bronze", "Seal", "Clerical", "Regular"]

# Sub-period groupings
OBI_SUBPERIODS = ["Bin", "Chu", "Li", "Zi", "Huang"]
BRONZE_SUBPERIODS = [
    "Early Western Zhou",
    "Late Western Zhou",
    "Spring&Autumn",
    "Warring States",
]


def get_time_for_era(era: str, subperiod: Optional[str] = None) -> float:
    """
    Get the normalized time value for a given era and optional sub-period.

    Args:
        era: Main historical period (e.g., 'OBI', 'Bronze', 'Seal', 'Clerical', 'Regular')
        subperiod: Optional sub-period within the era (e.g., 'Bin', 'Early Western Zhou')

    Returns:
        Normalized time value in [0, 1] range

    Raises:
        ValueError: If era or subperiod is not recognized

    Examples:
        >>> get_time_for_era('OBI')
        0.05
        >>> get_time_for_era('OBI', 'Bin')
        0.01
        >>> get_time_for_era('Bronze', 'Early Western Zhou')
        0.20
        >>> get_time_for_era('Regular')
        1.00
    """
    if subperiod is not None:
        if subperiod not in ERA_TIMES:
            raise ValueError(f"Unknown sub-period: {subperiod}")
        return ERA_TIMES[subperiod]

    if era not in ERA_TIMES:
        raise ValueError(f"Unknown era: {era}")

    return ERA_TIMES[era]


def get_era_for_time(t: float, include_subperiods: bool = True) -> str:
    """
    Get the era name corresponding to a given time value.

    Args:
        t: Normalized time value in [0, 1] range
        include_subperiods: If True, return specific sub-period names;
                           if False, return main period only

    Returns:
        Era or sub-period name

    Raises:
        ValueError: If time value is out of valid range

    Examples:
        >>> get_era_for_time(0.01)
        'Bin'
        >>> get_era_for_time(0.01, include_subperiods=False)
        'OBI'
        >>> get_era_for_time(0.70)
        'Seal'
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"Time value must be in [0, 1] range, got {t}")

    # Find the closest matching era
    if include_subperiods:
        # Check all eras including sub-periods
        min_dist = float("inf")
        closest_era = None

        for era, era_time in ERA_TIMES.items():
            dist = abs(t - era_time)
            if dist < min_dist:
                min_dist = dist
                closest_era = era

        return closest_era
    else:
        # Return only main periods
        if t <= 0.10:
            return "OBI"
        elif t <= 0.55:
            return "Bronze"
        elif t <= 0.72:
            return "Seal"
        elif t <= 0.87:
            return "Clerical"
        else:
            return "Regular"


def is_adjacent_period(era1: str, era2: str) -> bool:
    """
    Check if two eras are adjacent in chronological order.

    Args:
        era1: First era name
        era2: Second era name

    Returns:
        True if eras are adjacent (consecutive) periods

    Examples:
        >>> is_adjacent_period('OBI', 'Bronze')
        True
        >>> is_adjacent_period('OBI', 'Seal')
        False
        >>> is_adjacent_period('Bronze', 'Seal')
        True
    """
    # Get main periods
    main1 = (
        era1
        if era1 in MAIN_PERIODS
        else get_era_for_time(ERA_TIMES[era1], include_subperiods=False)
    )
    main2 = (
        era2
        if era2 in MAIN_PERIODS
        else get_era_for_time(ERA_TIMES[era2], include_subperiods=False)
    )

    # Check if they are consecutive in the main periods list
    try:
        idx1 = MAIN_PERIODS.index(main1)
        idx2 = MAIN_PERIODS.index(main2)
        return abs(idx1 - idx2) == 1
    except ValueError:
        return False


def get_adjacent_pairs() -> list:
    """
    Get all adjacent period pairs for training.

    Returns:
        List of tuples (era1, era2, time1, time2) for adjacent periods

    Examples:
        >>> pairs = get_adjacent_pairs()
        >>> ('OBI', 'Bronze', 0.05, 0.35) in pairs
        True
    """
    adjacent_pairs = []
    for i in range(len(MAIN_PERIODS) - 1):
        era1 = MAIN_PERIODS[i]
        era2 = MAIN_PERIODS[i + 1]
        time1 = ERA_TIMES[era1]
        time2 = ERA_TIMES[era2]
        adjacent_pairs.append((era1, era2, time1, time2))

    return adjacent_pairs


def encode_time_vector(t: float, encoding_dim: int = 16) -> np.ndarray:
    """
    Encode time value into a sinusoidal positional encoding vector.

    This follows the positional encoding scheme used in transformers,
    adapted for temporal encoding in the MSEF model (Appendix D).

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
