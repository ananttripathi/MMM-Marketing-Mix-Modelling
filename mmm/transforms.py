"""Adstock and saturation transforms for MMM.

All saturation transforms follow the law of diminishing marginal returns:
- Negative exponential: 1 - exp(-x/k)
- Log: log(1 + x/k), scaled
- Linear (reciprocal): x/(k+x)
- Power (Hill): x^α/(k^α+x^α), α in (0,1)
"""

from typing import Optional

import numpy as np

# Transform types that satisfy diminishing marginal returns
TRANSFORM_TYPES = ["hill", "negative_exponential", "log", "linear", "power"]


def adstock_transform(
    x: np.ndarray,
    decay: float,
    max_lag: int = 4,
) -> np.ndarray:
    """
    Apply geometric adstock (carryover) effect.
    Past spend continues to influence current period with exponential decay.
    """
    n = len(x)
    result = np.zeros(n)
    for i in range(n):
        for lag in range(min(max_lag + 1, i + 1)):
            result[i] += x[i - lag] * (decay ** lag)
    return result


def adstock_weight_sum(decay: float, max_lag: int) -> float:
    """Sum of adstock weights: sum(decay^i) for i=0..max_lag."""
    return (1 - decay ** (max_lag + 1)) / (1 - decay) if decay < 1.0 else max_lag + 1


def decay_from_weight_sum(target_sum: float, max_lag: int) -> float:
    """
    Solve for decay such that adstock_weight_sum(decay, max_lag) = target_sum.
    Uses binary search.
    """
    if target_sum >= max_lag + 1:
        return 1.0
    if target_sum <= 1.0:
        return 0.0

    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        s = adstock_weight_sum(mid, max_lag)
        if abs(s - target_sum) < 1e-6:
            return mid
        if s > target_sum:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def saturation_transform(
    x: np.ndarray,
    alpha: float = 1.0,
    half_saturation: float = 50.0,
    transform_type: str = "negative_exponential",
) -> np.ndarray:
    """
    Apply saturation transform. All types satisfy diminishing marginal returns.

    Parameters
    ----------
    x : np.ndarray
        Input (adstocked spend).
    alpha : float
        Shape parameter (for hill/power).
    half_saturation : float
        Scale parameter (spend at ~50% saturation).
    transform_type : str
        "hill", "negative_exponential", "log", "linear", "power"
    """
    x_safe = np.maximum(x, 1e-10)
    k = max(half_saturation, 1e-6)

    if transform_type == "hill":
        # Hill: x^α / (k^α + x^α)
        return (x_safe ** alpha) / ((k ** alpha) + (x_safe ** alpha))

    elif transform_type == "negative_exponential":
        # 1 - exp(-x/k): saturates at 1, concave
        return 1.0 - np.exp(-x_safe / k)

    elif transform_type == "log":
        # log(1 + x/k) / log(1 + 50): scaled to ~[0,1], concave
        log_scale = np.log(1 + 50)
        return np.log(1 + x_safe / k) / log_scale

    elif transform_type == "linear":
        # x/(k+x): reciprocal, saturates at 1, concave
        return x_safe / (k + x_safe)

    elif transform_type == "power":
        # x^α/(k^α+x^α) with α in (0,1): stronger concavity
        a = max(0.1, min(alpha, 0.99))
        return (x_safe ** a) / ((k ** a) + (x_safe ** a))

    else:
        raise ValueError(f"Unknown transform_type: {transform_type}. Use {TRANSFORM_TYPES}")


def apply_transforms(
    df,
    channel_cols: list,
    decay: float,
    max_lag: int,
    alpha: float,
    half_saturation: float,
    transform_type: str = "negative_exponential",
    channel_transform_types: Optional[dict] = None,
) -> np.ndarray:
    """
    Apply adstock then saturation to channel columns.
    channel_transform_types: {channel_name: transform_type} for per-channel override.
    """
    channel_transform_types = channel_transform_types or {}
    X_channel = []
    for col in channel_cols:
        x = df[col].values.astype(float)
        x_adstock = adstock_transform(x, decay, max_lag)
        t = channel_transform_types.get(col, transform_type)
        x_sat = saturation_transform(x_adstock, alpha, half_saturation, t)
        X_channel.append(x_sat)
    return np.column_stack(X_channel)
