"""Adstock and saturation transforms for MMM."""

import numpy as np


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
) -> np.ndarray:
    """
    Hill saturation: x^alpha / (half_sat^alpha + x^alpha).
    alpha (curvature): >1 = S-curve, 1 = concave, <1 = convex.
    """
    x_safe = np.maximum(x, 1e-10)
    return (x_safe ** alpha) / ((half_saturation ** alpha) + (x_safe ** alpha))


def apply_transforms(
    df,
    channel_cols: list,
    decay: float,
    max_lag: int,
    alpha: float,
    half_saturation: float,
) -> np.ndarray:
    """
    Apply adstock then saturation to channel columns.
    Returns (n_samples, n_channels) array.
    """
    X_channel = []
    for col in channel_cols:
        x = df[col].values.astype(float)
        x_adstock = adstock_transform(x, decay, max_lag)
        x_sat = saturation_transform(x_adstock, alpha, half_saturation)
        X_channel.append(x_sat)
    return np.column_stack(X_channel)
