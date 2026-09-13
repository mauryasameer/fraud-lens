from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.core.interfaces import FraudClassifier


class LogRegFraudClassifier(FraudClassifier):
    """Logistic regression baseline. C=0.01 is the notebook's own CV-found winner."""

    def __init__(self, C: float = 0.01, random_state: int = 42) -> None:
        self._model = LogisticRegression(C=C, random_state=random_state, max_iter=1000)
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LogRegFraudClassifier.predict() called before fit()")
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LogRegFraudClassifier.predict_proba() called before fit()")
        return self._model.predict_proba(X)[:, 1]
