"""
Marketing Mix Modelling (MMM) Dataset Generator

Creates a synthetic time-series dataset suitable for MMM analysis, including:
- Target variable: sales/revenue
- Marketing channel spends (TV, Digital, Radio, Print, Social)
- Control variables: seasonality, holidays, promotions
- Realistic saturation curves (diminishing returns) for channel effects
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def adstock_transform(x: np.ndarray, decay: float, max_lag: int = 4) -> np.ndarray:
    """Apply adstock (carryover) effect - past spend continues to influence current period."""
    n = len(x)
    result = np.zeros(n)
    for i in range(n):
        for lag in range(min(max_lag + 1, i + 1)):
            result[i] += x[i - lag] * (decay ** lag)
    return result


def saturation_curve(x: np.ndarray, half_saturation: float, shape: float = 1.0) -> np.ndarray:
    """Hill transformation: diminishing returns as spend increases."""
    return x ** shape / (half_saturation ** shape + x ** shape)


def generate_mmm_dataset(
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    freq: str = "W",  # "W" weekly, "D" daily
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
        'W' for weekly, 'D' for daily.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Dataset with date, sales, channel spends, and control variables.
    """
    np.random.seed(seed)

    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(date_range)

    # ----- Marketing channel spends (in thousands) -----
    # Simulate varying spend levels with some trend and randomness
    tv_spend = 50 + 30 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 8, n)
    tv_spend = np.maximum(tv_spend, 5)

    digital_spend = 40 + 25 * np.sin(np.linspace(0.5, 4 * np.pi + 0.5, n)) + np.random.normal(0, 6, n)
    digital_spend = np.maximum(digital_spend, 5)

    radio_spend = 20 + 10 * np.sin(np.linspace(1, 4 * np.pi + 1, n)) + np.random.normal(0, 4, n)
    radio_spend = np.maximum(radio_spend, 2)

    print_spend = 15 + 5 * np.sin(np.linspace(1.5, 4 * np.pi + 1.5, n)) + np.random.normal(0, 3, n)
    print_spend = np.maximum(print_spend, 0)

    social_spend = 35 + 20 * np.sin(np.linspace(0.8, 4 * np.pi + 0.8, n)) + np.random.normal(0, 5, n)
    social_spend = np.maximum(social_spend, 5)

    # ----- Adstock (carryover) -----
    tv_adstock = adstock_transform(tv_spend, decay=0.5)
    digital_adstock = adstock_transform(digital_spend, decay=0.4)
    radio_adstock = adstock_transform(radio_spend, decay=0.3)
    print_adstock = adstock_transform(print_spend, decay=0.4)
    social_adstock = adstock_transform(social_spend, decay=0.35)

    # ----- Saturation (diminishing returns) -----
    tv_effect = 80 * saturation_curve(tv_adstock, half_saturation=60, shape=1.2)
    digital_effect = 70 * saturation_curve(digital_adstock, half_saturation=50, shape=1.0)
    radio_effect = 25 * saturation_curve(radio_adstock, half_saturation=25, shape=1.1)
    print_effect = 15 * saturation_curve(print_adstock, half_saturation=20, shape=0.9)
    social_effect = 45 * saturation_curve(social_adstock, half_saturation=40, shape=1.0)

    # ----- Base sales + trend -----
    trend = np.linspace(0, 50, n)
    base_sales = 200 + trend

    # ----- Seasonality (quarterly + yearly) -----
    week_of_year = np.array([d.isocalendar()[1] for d in date_range])
    month = np.array([d.month for d in date_range])
    seasonality = 30 * np.sin(2 * np.pi * week_of_year / 52) + 15 * np.sin(2 * np.pi * month / 12)

    # ----- Holiday dummy (e.g. Q4 boost) -----
    holiday = (month >= 10) & (month <= 12)
    holiday_effect = np.where(holiday, 25 + np.random.normal(0, 5, n), 0)

    # ----- Promotion weeks (random spikes) -----
    promotion = np.zeros(n)
    promo_weeks = np.random.choice(n, size=min(15, n // 10), replace=False)
    promotion[promo_weeks] = 1
    promotion_effect = promotion * (20 + np.random.normal(0, 5, n))

    # ----- Compose sales -----
    sales = (
        base_sales
        + tv_effect
        + digital_effect
        + radio_effect
        + print_effect
        + social_effect
        + seasonality
        + holiday_effect
        + promotion_effect
        + np.random.normal(0, 15, n)
    )
    sales = np.maximum(sales, 50)

    # ----- Control variables for regression -----
    df = pd.DataFrame(
        {
            "date": date_range,
            "sales": np.round(sales, 2),
            "tv_spend": np.round(tv_spend, 2),
            "digital_spend": np.round(digital_spend, 2),
            "radio_spend": np.round(radio_spend, 2),
            "print_spend": np.round(print_spend, 2),
            "social_spend": np.round(social_spend, 2),
            "week_of_year": week_of_year,
            "month": month,
            "holiday_period": holiday.astype(int),
            "promotion": promotion.astype(int),
        }
    )

    return df


if __name__ == "__main__":
    import os

    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)

    # Weekly data (typical for MMM)
    df_weekly = generate_mmm_dataset(
        start_date="2020-01-01",
        end_date="2023-12-31",
        freq="W",
        seed=42,
    )
    path_weekly = os.path.join(out_dir, "marketing_mix_weekly.csv")
    df_weekly.to_csv(path_weekly, index=False)
    print(f"Saved weekly MMM dataset: {path_weekly}")
    print(f"Shape: {df_weekly.shape}")
    print(df_weekly.head(10))

    # Optional: daily data (smaller date range to keep size reasonable)
    df_daily = generate_mmm_dataset(
        start_date="2022-01-01",
        end_date="2023-12-31",
        freq="D",
        seed=42,
    )
    path_daily = os.path.join(out_dir, "marketing_mix_daily.csv")
    df_daily.to_csv(path_daily, index=False)
    print(f"\nSaved daily MMM dataset: {path_daily}")
    print(f"Shape: {df_daily.shape}")
