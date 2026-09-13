import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.core.interfaces import FraudClassifier
from src.services.fraud_service import run_fraud_pipeline


class StubFraudClassifier(FraudClassifier):
    """Deterministic stub: flags anything with V0 > 0."""

    def __init__(self) -> None:
        self._fitted = False

    def fit(self, X, y):
        self._fitted = True

    def predict(self, X):
        return (X["V0"] > 0).astype(int).to_numpy()

    def predict_proba(self, X):
        return (X["V0"] > 0).astype(float).to_numpy()


class SmoteStubFraudClassifier(StubFraudClassifier):
    uses_smote = True


def _synthetic_dataset(n=400, positive_rate=0.1):
    X, y = make_classification(
        n_samples=n,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        weights=[1 - positive_rate, positive_rate],
        random_state=42,
    )
    df = pd.DataFrame(X, columns=["V0", "V1", "V2", "Amount"])
    df["Time"] = np.arange(n)
    df["Amount"] = np.abs(df["Amount"]) * 100
    df["Class"] = y
    return df


def test_run_fraud_pipeline_produces_expected_shape():
    data = _synthetic_dataset()
    result = run_fraud_pipeline(data, StubFraudClassifier())

    assert result.metrics.support == len(result.y_test)
    assert result.confusion_matrix_fig is not None
    assert result.roc_curve_fig is not None
    assert len(result.amount_test) == len(result.X_test) == len(result.y_test)
    assert result.n_train_real == len(result.y_train_res)


def test_run_fraud_pipeline_applies_smote_only_when_provider_requests_it():
    data = _synthetic_dataset()
    result = run_fraud_pipeline(data, SmoteStubFraudClassifier())

    train_counts = result.y_train_res.value_counts()
    assert result.n_train_real < len(result.y_train_res)
    assert train_counts[0] == train_counts[1]


def test_run_fraud_pipeline_excludes_time_from_features():
    data = _synthetic_dataset()
    result = run_fraud_pipeline(data, StubFraudClassifier())

    assert "Time" not in result.X_test.columns
    assert "Class" not in result.X_test.columns
    assert "Amount" in result.X_test.columns


def test_run_fraud_pipeline_excludes_renamed_target_column():
    data = _synthetic_dataset().rename(columns={"Class": "fraud"})
    result = run_fraud_pipeline(data, StubFraudClassifier(), target_col="fraud")

    assert "fraud" not in result.X_test.columns
    assert "Time" not in result.X_test.columns


def test_run_fraud_pipeline_reports_pre_smote_train_count():
    data = _synthetic_dataset()
    result = run_fraud_pipeline(data, SmoteStubFraudClassifier())

    assert result.n_train_real < len(result.y_train_res)


def test_run_fraud_pipeline_handles_single_class_test_fold(mocker):
    data = _synthetic_dataset()
    train = data.iloc[:280].reset_index(drop=True)
    # Force every row in the test fold to be the legitimate class, but keep
    # the train fold balanced enough for SMOTE to run without error.
    test = data.iloc[280:].copy()
    test["Class"] = 0
    test = test.reset_index(drop=True)
    mocker.patch("src.services.fraud_service.split_fraud_data", return_value=(train, test))

    result = run_fraud_pipeline(data, StubFraudClassifier())

    assert result.metrics.auc_roc is None
    assert result.confusion_matrix_fig is not None
