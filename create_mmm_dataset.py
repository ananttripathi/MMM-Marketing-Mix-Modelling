"""
Marketing Mix Modelling (MMM) Dataset Generator

Creates synthetic time-series datasets with user-configurable:
- Frequency: daily, weekly, monthly, yearly
- Channel names: custom list (e.g. TV, Digital, Radio) or default
"""

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd

FREQ_MAP = {
    "daily": "D",
    "weekly": "W",
    "monthly": "M",   # month-end
    "yearly": "Y",    # year-end
}

DEFAULT_CHANNELS = ["tv_spend", "digital_spend", "radio_spend", "print_spend", "social_spend"]


def adstock_transform(x: np.ndarray, decay: float, max_lag: int = 4) -> np.ndarray:
    """Apply adstock (carryover) effect - past spend continues to influence current period."""
    n = len(x)
    result = np.zeros(n)
    for i in range(n):
        for lag in range(min(max_lag + 1, i + 1)):
            result[i] += x[i - lag] * (decay ** lag)
    return result


def saturation_curve(
    x: np.ndarray,
    half_saturation: float,
    shape: float = 1.0,
    transform_type: str = "negative_exponential",
) -> np.ndarray:
    """
    Saturation transforms - all follow diminishing marginal returns.
    Types: hill, negative_exponential, log, linear, power
    """
    x_safe = np.maximum(x, 1e-10)
    k = max(half_saturation, 1e-6)

    if transform_type == "hill":
        return (x_safe ** shape) / ((k ** shape) + (x_safe ** shape))
    elif transform_type == "negative_exponential":
        return 1.0 - np.exp(-x_safe / k)
    elif transform_type == "log":
        log_scale = np.log(1 + 50)
        return np.log(1 + x_safe / k) / log_scale
    elif transform_type == "linear":
        return x_safe / (k + x_safe)
    elif transform_type == "power":
        a = max(0.1, min(shape, 0.99))
        return (x_safe ** a) / ((k ** a) + (x_safe ** a))
    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")


def generate_mmm_dataset(
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    freq: str = "W",
    channel_names: Optional[List[str]] = None,
    target_col: str = "sales",
    transform_type: str = "negative_exponential",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic marketing mix modelling dataset.

    Parameters
    ----------
    start_date : str
        Start of time series.
    end_date : str
        End of time series.
    freq : str
        'D' daily, 'W' weekly, 'ME' monthly, 'YE' yearly.
    channel_names : list of str, optional
        Custom channel names (e.g. ['TV', 'Digital', 'Radio']).
        Default: tv_spend, digital_spend, radio_spend, print_spend, social_spend
    target_col : str
        Name of target variable.
    transform_type : str
        hill, negative_exponential, log, linear, power (all diminishing returns).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Dataset with date, target, channel spends, and control variables.
    """
    np.random.seed(seed)

    channel_names = channel_names or DEFAULT_CHANNELS
    n_channels = len(channel_names)

    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(date_range)

    if n < 2:
        raise ValueError(f"Date range too short: {n} periods. Use a broader range.")

    # ----- Marketing channel spends (per channel) -----
    channel_spends = {}
    channel_adstocks = {}
    channel_effects = {}

    for i, name in enumerate(channel_names):
        # Vary base spend and pattern per channel
        base = 30 + 20 * (i % 3)
        amp = 15 + 10 * (i % 2)
        phase = i * 0.5
        spend = base + amp * np.sin(np.linspace(phase, 4 * np.pi + phase, n)) + np.random.normal(0, 5, n)
        spend = np.maximum(spend, 1)
        channel_spends[name] = spend

        decay = 0.3 + 0.2 * (i % 4)  # vary decay per channel
        adstock = adstock_transform(spend, decay=decay)
        channel_adstocks[name] = adstock

        half_sat = 30 + 20 * i
        effect_coef = 40 + 30 * (n_channels - i) / max(n_channels, 1)
        effect = effect_coef * saturation_curve(adstock, half_saturation=half_sat, shape=1.0, transform_type=transform_type)
        channel_effects[name] = effect

    # ----- Base sales + trend -----
    trend = np.linspace(0, 50, n)
    base_sales = 200 + trend

    # ----- Seasonality -----
    week_of_year = np.array([d.isocalendar()[1] for d in date_range])
    month = np.array([d.month for d in date_range])
    seasonality = 30 * np.sin(2 * np.pi * week_of_year / 52) + 15 * np.sin(2 * np.pi * month / 12)

    # ----- Holiday dummy (Q4 boost) -----
    holiday = (month >= 10) & (month <= 12)
    holiday_effect = np.where(holiday, 25 + np.random.normal(0, 5, n), 0)

    # ----- Promotion (random spikes) -----
    promotion = np.zeros(n)
    n_promo = min(max(5, n // 15), n - 1)
    promo_idx = np.random.choice(n, size=n_promo, replace=False)
    promotion[promo_idx] = 1
    promotion_effect = promotion * (20 + np.random.normal(0, 5, n))

    # ----- Compose sales -----
    sales = base_sales + sum(channel_effects.values()) + seasonality + holiday_effect + promotion_effect
    sales = sales + np.random.normal(0, 15, n)
    sales = np.maximum(sales, 50)

    # ----- Build DataFrame -----
    data = {"date": date_range, target_col: np.round(sales, 2)}
    for name in channel_names:
        data[name] = np.round(channel_spends[name], 2)
    data["week_of_year"] = week_of_year
    data["month"] = month
    data["holiday_period"] = holiday.astype(int)
    data["promotion"] = promotion.astype(int)

    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic MMM dataset")
    parser.add_argument(
        "--start",
        default="2020-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default="2023-12-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--freq",
        choices=["daily", "weekly", "monthly", "yearly"],
        default="weekly",
        help="Data frequency",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default=None,
        help="Comma-separated channel names (e.g. TV,Digital,Radio,Print,Social)",
    )
    parser.add_argument(
        "--target",
        default="sales",
        help="Target column name",
    )
    parser.add_argument(
        "--transform",
        choices=["hill", "negative_exponential", "log", "linear", "power"],
        default="negative_exponential",
        help="Saturation transform (all follow diminishing returns)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output path (default: data/marketing_mix_<freq>.csv)",
    )

    args = parser.parse_args()

    channel_names = None
    if args.channels:
        channel_names = [c.strip() for c in args.channels.split(",") if c.strip()]

    df = generate_mmm_dataset(
        start_date=args.start,
        end_date=args.end,
        freq=FREQ_MAP[args.freq],
        channel_names=channel_names,
        target_col=args.target,
        transform_type=args.transform,
        seed=args.seed,
    )

    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)

    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(out_dir, f"marketing_mix_{args.freq}.csv")

    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"Shape: {df.shape}")
    print(f"Channels: {list(df.columns)[2:-4]}")  # exclude date, target, controls
    print(df.head())


if __name__ == "__main__":
    main()
