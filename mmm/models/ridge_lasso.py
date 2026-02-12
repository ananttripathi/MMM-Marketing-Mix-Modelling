"""Ridge and Lasso MMM with optional constraints."""

from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize

from .base import BaseMMM


class RidgeLassoMMM(BaseMMM):
    """Ridge or Lasso MMM with optional positive and lag-sum constraints."""

    def __init__(
        self,
        n_channels: int,
        n_controls: int,
        mode: str = "ridge",  # "ridge" or "lasso"
        alpha: float = 1.0,
        positive_constraints: bool = True,
        lag_sum_lower: Optional[float] = None,
        lag_sum_upper: Optional[float] = None,
        channel_names: Optional[List[str]] = None,
    ):
        self.n_channels = n_channels
        self.n_controls = n_controls
        self.mode = mode
        self.alpha = alpha
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

    def _loss(self, beta: np.ndarray, X_design: np.ndarray, y: np.ndarray) -> float:
        mse = np.sum((y - X_design @ beta) ** 2) / len(y)
        if self.mode == "ridge":
            reg = self.alpha * np.sum(beta[1:] ** 2)
        else:  # lasso
            reg = self.alpha * np.sum(np.abs(beta[1:]))
        return mse + reg

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_control: Optional[np.ndarray] = None,
    ) -> "RidgeLassoMMM":
        X_design = self._design_matrix(X, X_control)
        n_params = X_design.shape[1]

        def loss(beta):
            return self._loss(beta, X_design, y)

        bounds = []
        for i in range(n_params):
            if i == 0:
                bounds.append((None, None))
            elif i <= self.n_channels:
                if self.positive_constraints:
                    bounds.append((0, None))
                else:
                    bounds.append((None, None))
            else:
                bounds.append((None, None))

        constraints = []
        if self.lag_sum_lower is not None or self.lag_sum_upper is not None:
            from scipy.optimize import LinearConstraint

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
