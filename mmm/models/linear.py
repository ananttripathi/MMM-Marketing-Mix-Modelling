"""Linear MMM with optional constraints."""

from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize

from .base import BaseMMM


class LinearMMM(BaseMMM):
    """Linear regression MMM with optional positive and lag-sum constraints."""

    def __init__(
        self,
        n_channels: int,
        n_controls: int,
        positive_constraints: bool = True,
        lag_sum_lower: Optional[float] = None,
        lag_sum_upper: Optional[float] = None,
        channel_names: Optional[List[str]] = None,
    ):
        self.n_channels = n_channels
        self.n_controls = n_controls
        self.positive_constraints = positive_constraints
        self.lag_sum_lower = lag_sum_lower
        self.lag_sum_upper = lag_sum_upper
        self.channel_names = channel_names or [f"channel_{i}" for i in range(n_channels)]
        self.n_features = n_channels + n_controls
        self.coef_ = None
        self.intercept_ = None

    def _design_matrix(self, X: np.ndarray, X_control: Optional[np.ndarray]) -> np.ndarray:
        ones = np.ones((X.shape[0], 1))
        if X_control is not None and X_control.size > 0:
            return np.hstack([ones, X, X_control])
        return np.hstack([ones, X])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_control: Optional[np.ndarray] = None,
    ) -> "LinearMMM":
        """Fit linear model with optional constraints."""
        X_design = self._design_matrix(X, X_control)
        n_params = X_design.shape[1]

        def loss(beta):
            return np.sum((y - X_design @ beta) ** 2)

        # Bounds
        bounds = []
        for i in range(n_params):
            if i == 0:
                bounds.append((None, None))  # intercept
            elif i <= self.n_channels:
                if self.positive_constraints:
                    bounds.append((0, None))
                else:
                    bounds.append((None, None))
            else:
                bounds.append((None, None))

        # Constraints: sum of channel coefficients in [lag_sum_lower, lag_sum_upper]
        constraints = []
        if self.lag_sum_lower is not None or self.lag_sum_upper is not None:
            from scipy.optimize import LinearConstraint

            # Channel coeff indices: 1 to n_channels (0 is intercept)
            A = np.zeros(n_params)
            A[1 : 1 + self.n_channels] = 1

            lb = self.lag_sum_lower if self.lag_sum_lower is not None else -np.inf
            ub = self.lag_sum_upper if self.lag_sum_upper is not None else np.inf
            constraints.append(LinearConstraint(A, lb, ub))

        x0 = np.linalg.lstsq(X_design, y, rcond=None)[0]
        if self.positive_constraints:
            x0 = np.maximum(x0, 0.01)

        method = "SLSQP" if constraints else "L-BFGS-B"
        res = minimize(
            loss,
            x0,
            method=method,
            bounds=bounds,
            constraints=constraints if constraints else (),
        )
        self.coef_ = res.x
        self.intercept_ = res.x[0]
        return self

    def predict(
        self,
        X: np.ndarray,
        X_control: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        X_design = self._design_matrix(X, X_control)
        return X_design @ self.coef_

    def get_coefficients(self) -> Dict[str, float]:
        d = {"intercept": float(self.intercept_)}
        for i, name in enumerate(self.channel_names):
            d[name] = float(self.coef_[1 + i])
        for i in range(self.n_controls):
            d[f"control_{i}"] = float(self.coef_[1 + self.n_channels + i])
        return d
