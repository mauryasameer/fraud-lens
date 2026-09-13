from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.core.interfaces import FraudClassifier


class XGBoostFraudClassifier(FraudClassifier):
    """XGBoost. learning_rate=0.11/max_depth=4/min_child_weight=30/n_estimators=285 is
    the notebook's own CV-found winner on the SMOTE-oversampled data."""

    def __init__(
        self,
        learning_rate: float = 0.11,
        max_depth: int = 4,
        min_child_weight: int = 30,
        n_estimators: int = 285,
        random_state: int = 42,
    ) -> None:
        self._model = XGBClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            n_estimators=n_estimators,
            random_state=random_state,
            eval_metric="logloss",
        )
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("XGBoostFraudClassifier.predict() called before fit()")
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("XGBoostFraudClassifier.predict_proba() called before fit()")
        return self._model.predict_proba(X)[:, 1]
