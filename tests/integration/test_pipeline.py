import numpy as np
import pandas as pd

from src.core.interfaces import FraudClassifier
from src.services.fairness_check import amount_quartile_breakdown
from src.services.fraud_service import run_fraud_pipeline
from src.services.narrative_service import compute_normal_stats, generate_top_n_narratives
from src.services.report_service import GOVERNANCE_BANNER, build_report


class StubFraudClassifier(FraudClassifier):
    def __init__(self):
        self._fitted = False

    def fit(self, X, y):
        self._fitted = True

    def predict(self, X):
        return (X["V0"] > 0).astype(int).to_numpy()

    def predict_proba(self, X):
        return (X["V0"] > 0).astype(float).to_numpy()


class StubLLM:
    def generate(self, prompt, system=None, images=None, **kwargs):
        class _Resp:
            content = "stub statistical explanation"

        return _Resp()


def _synthetic_dataset(n=500, positive_rate=0.08):
    from sklearn.datasets import make_classification

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


def test_end_to_end_pipeline_writes_governed_report(tmp_path):
    data = _synthetic_dataset()
    result = run_fraud_pipeline(data, StubFraudClassifier())

    fairness_df = amount_quartile_breakdown(result.amount_test, result.y_test.to_numpy(), result.y_pred)
    mean, std = compute_normal_stats(result.X_train_res, result.y_train_res)
    narratives = generate_top_n_narratives(result.X_test, result.y_prob, mean, std, StubLLM(), top_n=3)

    report = build_report(
        title="FraudLens — Fraud Detection Report",
        metrics=result.metrics,
        confusion_matrix_fig=result.confusion_matrix_fig,
        roc_curve_fig=result.roc_curve_fig,
        fairness_df=fairness_df,
        narratives=narratives,
        provider_name="stub",
        hyperparameters={},
        row_counts={"train": len(result.X_train_res), "test": len(result.X_test)},
    )

    output_path = tmp_path / "report.html"
    report.save(output_path)

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert GOVERNANCE_BANNER in content
    assert "Run Info" in content
    assert "stub statistical explanation" in content
