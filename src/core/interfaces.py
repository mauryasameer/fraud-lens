from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class FraudClassifier(ABC):
    """Abstract interface for all fraud-classification backends.

    Concrete implementations live in src/providers/. Swap providers by
    changing one constructor argument — no service code changes.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model on preprocessed, training-fold-only-oversampled data."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Hard class predictions (0 = legitimate, 1 = fraud)."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Fraud-class probability estimates, shape (n_samples,)."""
        ...
