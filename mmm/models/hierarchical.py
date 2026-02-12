"""Hierarchical (multi-level) Bayesian MMM."""

from typing import Dict, List, Optional

import numpy as np

from .base import BaseMMM

try:
    import pymc as pm

    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False


class HierarchicalMMM(BaseMMM):
    """
    Hierarchical MMM: channel coefficients drawn from a population-level prior.
    Allows partial pooling across channels.
    """

    def __init__(
        self,
        n_channels: int,
        n_controls: int,
        positive_constraints: bool = True,
        lag_sum_lower: Optional[float] = None,
        lag_sum_upper: Optional[float] = None,
        channel_names: Optional[List[str]] = None,
        samples: int = 1000,
        tune: int = 500,
        chains: int = 2,
    ):
        if not PYMC_AVAILABLE:
            raise ImportError(
                "PyMC required for Hierarchical model. Install with: pip install pymc arviz"
            )
        self.n_channels = n_channels
        self.n_controls = n_controls
        self.positive_constraints = positive_constraints
        self.lag_sum_lower = lag_sum_lower
        self.lag_sum_upper = lag_sum_upper
        self.channel_names = channel_names or [f"channel_{i}" for i in range(n_channels)]
        self.samples = samples
        self.tune = tune
        self.chains = chains
        self.idata_ = None
        self.coef_ = None
        self.intercept_ = None

    def _design_matrix(self, X: np.ndarray, X_control: Optional[np.ndarray]) -> np.ndarray:
        if X_control is not None and X_control.size > 0:
            return np.hstack([X, X_control])
        return X

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_control: Optional[np.ndarray] = None,
    ) -> "HierarchicalMMM":
        X_design = self._design_matrix(X, X_control)

        with pm.Model() as model:
            intercept = pm.Normal("intercept", mu=np.mean(y), sigma=50)

            # Hierarchical: population mean and std for channel effects
            mu_channel = pm.HalfNormal("mu_channel", sigma=30) if self.positive_constraints else pm.Normal("mu_channel", mu=0, sigma=30)
            sigma_channel = pm.HalfNormal("sigma_channel", sigma=20)

            # Channel-specific coefficients (partially pooled)
            if self.positive_constraints:
                channel_coef = pm.HalfNormal(
                    "channel_coef",
                    sigma=pm.math.abs(sigma_channel) + 0.1,
                    shape=self.n_channels,
                )
            else:
                channel_coef = pm.Normal(
                    "channel_coef",
                    mu=mu_channel,
                    sigma=pm.math.abs(sigma_channel) + 0.1,
                    shape=self.n_channels,
                )

            if self.n_controls > 0:
                control_coef = pm.Normal(
                    "control_coef",
                    mu=0,
                    sigma=20,
                    shape=self.n_controls,
                )
                mu = (
                    intercept
                    + pm.math.dot(X[:, : self.n_channels], channel_coef)
                    + pm.math.dot(X[:, self.n_channels :], control_coef)
                )
            else:
                mu = intercept + pm.math.dot(X, channel_coef)

            if self.lag_sum_lower is not None and self.lag_sum_upper is not None:
                lag_sum = pm.math.sum(channel_coef)
                mid = (self.lag_sum_lower + self.lag_sum_upper) / 2
                half_range = (self.lag_sum_upper - self.lag_sum_lower) / 4
                pm.Potential(
                    "lag_sum_prior",
                    pm.logp(pm.Normal.dist(mu=mid, sigma=half_range + 1), lag_sum),
                )

            sigma = pm.HalfNormal("sigma", sigma=20)
            pm.Normal("y", mu=mu, sigma=sigma, observed=y)

            self.idata_ = pm.sample(
                draws=self.samples,
                tune=self.tune,
                chains=self.chains,
                return_inferencedata=True,
                progressbar=False,
                target_accept=0.9,
            )

        post = self.idata_.posterior
        self.intercept_ = float(post["intercept"].mean())
        self.coef_ = np.concatenate([
            [self.intercept_],
            post["channel_coef"].mean(dim=["chain", "draw"]).values,
            post["control_coef"].mean(dim=["chain", "draw"]).values
            if self.n_controls > 0
            else [],
        ])
        return self

    def predict(
        self,
        X: np.ndarray,
        X_control: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        X_design = self._design_matrix(X, X_control)
        ones = np.ones((X.shape[0], 1))
        X_full = np.hstack([ones, X_design])
        return X_full @ self.coef_

    def get_coefficients(self) -> Dict[str, float]:
        d = {"intercept": float(self.intercept_)}
        post = self.idata_.posterior
        ch = post["channel_coef"]
        for i, name in enumerate(self.channel_names):
            d[name] = float(ch.isel({ch.dims[0]: i}).mean())
        if self.n_controls > 0:
            ctrl = post["control_coef"]
            for i in range(self.n_controls):
                d[f"control_{i}"] = float(ctrl.isel({ctrl.dims[0]: i}).mean())
        return d
