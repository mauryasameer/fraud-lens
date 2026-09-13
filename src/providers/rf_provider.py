from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.core.interfaces import FraudClassifier


class RandomForestFraudClassifier(FraudClassifier):
    """Random forest. n_estimators=860/criterion=entropy/min_samples_leaf=30 is the
    notebook's own CV-found winner on the non-oversampled data."""

    def __init__(
        self,
        n_estimators: int = 860,
        criterion: str = "entropy",
        min_samples_leaf: int = 30,
        random_state: int = 42,
    ) -> None:
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("RandomForestFraudClassifier.predict() called before fit()")
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("RandomForestFraudClassifier.predict_proba() called before fit()")
        return self._model.predict_proba(X)[:, 1]
