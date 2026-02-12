"""MMM pipeline orchestrating transforms and models."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import MMMConfig
from .transforms import apply_transforms


def get_model(config: MMMConfig, channel_names: List[str], control_names: List[str]):
    """Instantiate model based on config."""
    from .models.linear import LinearMMM
    from .models.ridge_lasso import RidgeLassoMMM

    n_channels = len(channel_names)
    n_controls = len(control_names)

    common = dict(
        n_channels=n_channels,
        n_controls=n_controls,
        positive_constraints=config.positive_constraints,
        lag_sum_lower=config.lag_sum_lower,
        lag_sum_upper=config.lag_sum_upper,
        channel_names=channel_names,
    )

    if config.model_type == "linear":
        return LinearMMM(**common)
    elif config.model_type == "ridge":
        return RidgeLassoMMM(**common, mode="ridge", alpha=config.ridge_alpha)
    elif config.model_type == "lasso":
        return RidgeLassoMMM(**common, mode="lasso", alpha=config.lasso_alpha)
    elif config.model_type == "bayesian":
        from .models.bayesian import BayesianMMM

        return BayesianMMM(
            **common,
            samples=config.bayesian_samples,
            tune=config.bayesian_tune,
            chains=config.bayesian_chains,
        )
    elif config.model_type == "hierarchical":
        from .models.hierarchical import HierarchicalMMM

        return HierarchicalMMM(
            **common,
            samples=config.bayesian_samples,
            tune=config.bayesian_tune,
            chains=config.bayesian_chains,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


class MMMPipeline:
    """End-to-end MMM pipeline."""

    def __init__(self, config: MMMConfig):
        self.config = config
        self.model = None
        self.channel_names_: List[str] = []
        self.control_names_: List[str] = []
        self.half_saturation_: Optional[float] = None

    def fit(self, df: pd.DataFrame) -> "MMMPipeline":
        """Fit pipeline on dataframe."""
        channel_cols = self.config.get_channel_cols_from_df(df.columns.tolist())
        control_cols = self.config.get_control_cols_from_df(df.columns.tolist())

        if not channel_cols:
            raise ValueError(
                "No channel columns found. Expected one of: "
                + ", ".join(self.config.channel_cols)
            )
        if self.config.target_col not in df.columns:
            raise ValueError(f"Target column '{self.config.target_col}' not found.")

        self.channel_names_ = channel_cols
        self.control_names_ = control_cols

        # Half-saturation: median of each channel's max (or use config)
        if self.config.saturation_half_sat is not None:
            self.half_saturation_ = self.config.saturation_half_sat
        else:
            max_per_channel = df[channel_cols].max().values
            self.half_saturation_ = float(np.median(max_per_channel) * 0.5)

        X_channel = apply_transforms(
            df,
            channel_cols,
            decay=self.config.adstock_decay,
            max_lag=self.config.adstock_max_lag,
            alpha=self.config.saturation_alpha,
            half_saturation=self.half_saturation_,
            transform_type=self.config.saturation_transform_type,
            channel_transform_types=self.config.channel_transform_types,
        )

        X_control = None
        if control_cols:
            X_control = df[control_cols].values.astype(float)

        y = df[self.config.target_col].values.astype(float)

        self.model = get_model(
            self.config,
            self.channel_names_,
            self.control_names_,
        )
        self.model.fit(X_channel, y, X_control)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict on dataframe."""
        X_channel = apply_transforms(
            df,
            self.channel_names_,
            decay=self.config.adstock_decay,
            max_lag=self.config.adstock_max_lag,
            alpha=self.config.saturation_alpha,
            half_saturation=self.half_saturation_,
            transform_type=self.config.saturation_transform_type,
            channel_transform_types=self.config.channel_transform_types,
        )
        X_control = None
        if self.control_names_:
            X_control = df[self.control_names_].values.astype(float)
        return self.model.predict(X_channel, X_control)

    def score(self, df: pd.DataFrame) -> float:
        """R² score on dataframe."""
        y = df[self.config.target_col].values.astype(float)
        pred = self.predict(df)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def get_coefficients(self) -> Dict[str, float]:
        """Channel and control coefficients with proper names."""
        coef = self.model.get_coefficients()
        out = {}
        for k, v in coef.items():
            if k.startswith("control_"):
                idx = int(k.split("_")[1])
                if idx < len(self.control_names_):
                    out[self.control_names_[idx]] = v
                else:
                    out[k] = v
            else:
                out[k] = v
        return out

    def get_channel_contributions(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get per-channel contribution to the target for each row.
        Returns dict of channel_name -> array of contributions (length n_rows).
        """
        X_channel = apply_transforms(
            df,
            self.channel_names_,
            decay=self.config.adstock_decay,
            max_lag=self.config.adstock_max_lag,
            alpha=self.config.saturation_alpha,
            half_saturation=self.half_saturation_,
            transform_type=self.config.saturation_transform_type,
            channel_transform_types=self.config.channel_transform_types,
        )
        coef = self.model.get_coefficients()
        out = {}
        for i, ch in enumerate(self.channel_names_):
            if ch in coef:
                out[ch] = coef[ch] * X_channel[:, i]
        return out

    def get_marginal_roi(self, df: pd.DataFrame, channel: str, delta_pct: float = 0.01) -> float:
        """
        Approximate marginal ROI for a channel: (dRevenue/dSpend) at current spend.
        Uses numerical differentiation: add delta_pct of spend, measure revenue change.
        Returns mROI as a multiplier (e.g. 2.0 = $2 revenue per $1 extra spend).
        """
        contributions = self.get_channel_contributions(df)
        base_contrib = float(np.sum(contributions.get(channel, np.zeros(len(df)))))
        df_perturb = df.copy()
        df_perturb[channel] = df_perturb[channel] * (1 + delta_pct)
        contrib_perturb = self.get_channel_contributions(df_perturb)
        new_contrib = float(np.sum(contrib_perturb.get(channel, np.zeros(len(df)))))
        delta_revenue_units = new_contrib - base_contrib
        delta_spend = float(df[channel].sum() * delta_pct)
        if delta_spend <= 0:
            return 0.0
        # mROI = delta_revenue / delta_spend (in target units per $ spend)
        return delta_revenue_units / delta_spend
