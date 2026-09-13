import pytest

from src.core.interfaces import FraudClassifier


def test_fraud_classifier_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        FraudClassifier()


def test_fraud_classifier_subclass_must_implement_all_methods():
    class Incomplete(FraudClassifier):
        def fit(self, X, y):
            pass

    with pytest.raises(TypeError):
        Incomplete()
