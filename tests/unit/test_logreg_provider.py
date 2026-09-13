import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.providers.logreg_provider import LogRegFraudClassifier


def _synthetic_imbalanced_data():
    X, y = make_classification(
        n_samples=300,
        n_features=6,
        n_informative=4,
        weights=[0.9, 0.1],
        random_state=42,
    )
    import pandas as pd

    return pd.DataFrame(X, columns=[f"V{i}" for i in range(6)]), pd.Series(y)


def test_fit_predict_predict_proba_shapes_and_ranges():
    X, y = _synthetic_imbalanced_data()
    clf = LogRegFraudClassifier()
    clf.fit(X, y)

    preds = clf.predict(X)
    probs = clf.predict_proba(X)

    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1})
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_predict_before_fit_raises():
    X, _y = _synthetic_imbalanced_data()
    clf = LogRegFraudClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(X)


def test_predict_proba_before_fit_raises():
    X, _y = _synthetic_imbalanced_data()
    clf = LogRegFraudClassifier()
    with pytest.raises(RuntimeError):
        clf.predict_proba(X)
