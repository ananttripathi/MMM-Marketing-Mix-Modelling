"""Base MMM model interface."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np


class BaseMMM(ABC):
    """Base class for MMM models."""

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_control: Optional[np.ndarray] = None,
    ) -> "BaseMMM":
        """Fit the model."""
        pass

    @abstractmethod
    def predict(
        self,
        X: np.ndarray,
        X_control: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict."""
        pass

    @abstractmethod
    def get_coefficients(self) -> Dict[str, float]:
        """Return channel coefficients."""
        pass

    def score(self, X: np.ndarray, y: np.ndarray, X_control: Optional[np.ndarray] = None) -> float:
        """R² score."""
        pred = self.predict(X, X_control)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
