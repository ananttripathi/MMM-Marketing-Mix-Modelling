"""Configuration for MMM pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd


@dataclass
class MMMConfig:
    """MMM pipeline configuration. Column names are user-mapped, not hardcoded."""

    # Target and features (set by user mapping - no defaults)
    target_col: str = ""
    date_col: str = "date"
    channel_cols: List[str] = field(default_factory=list)
    control_cols: List[str] = field(default_factory=list)
    segment_cols: List[str] = field(default_factory=list)
    channel_transform_types: Optional[dict] = None  # {channel: transform_type}

    # Adstock
    adstock_decay: float = 0.5
    adstock_max_lag: int = 4

    # Saturation (diminishing returns)
    saturation_alpha: float = 1.0  # shape for hill/power
    saturation_half_sat: Optional[float] = None  # auto from data if None
    saturation_transform_type: str = "negative_exponential"  # hill, negative_exponential, log, linear, power

    # Constraints
    positive_constraints: bool = True
    lag_sum_lower: Optional[float] = None  # e.g. 0.8
    lag_sum_upper: Optional[float] = None  # e.g. 1.0

    # Model
    model_type: str = "linear"  # linear, ridge, lasso, bayesian, hierarchical
    ridge_alpha: float = 1.0
    lasso_alpha: float = 0.1

    # Bayesian
    bayesian_samples: int = 1000
    bayesian_tune: int = 500
    bayesian_chains: int = 2

    def get_channel_cols_from_df(self, df_columns: List[str]) -> List[str]:
        """Get channel columns that exist in the dataframe."""
        return [c for c in self.channel_cols if c in df_columns]

    def get_control_cols_from_df(self, df_columns: List[str]) -> List[str]:
        """Get control columns that exist in the dataframe."""
        return [c for c in self.control_cols if c in df_columns]


def infer_target_column(df_columns: List[str]) -> str:
    """Infer likely target column from common names."""
    candidates = ["sales", "revenue", "conversions", "conversion", "target", "y", "outcome"]
    for c in candidates:
        for col in df_columns:
            if c.lower() in col.lower():
                return col
    return df_columns[0] if df_columns else ""


def infer_channel_columns(df, target_col: str, exclude: List[str]) -> List[str]:
    """Infer likely channel columns (numeric, excluding target/date)."""
    numeric = [
        c for c in df.columns
        if c != target_col and c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return numeric[:5] if len(numeric) > 5 else numeric
