import matplotlib.pyplot as plt
import pandas as pd
from meerax.eval.classification import ClassificationMetrics

from src.services.report_service import GOVERNANCE_BANNER, build_report


def _dummy_fig():
    fig, _ax = plt.subplots()
    return fig


def test_build_report_includes_all_required_sections():
    metrics = ClassificationMetrics(
        accuracy=0.99, precision=0.85, recall=0.80, f1=0.82, auc_roc=0.97, support=100
    )
    fairness_df = pd.DataFrame(
        [
            {"quartile": "Q1", "n": 25, "flag_rate": 0.1, "precision": 0.5},
            {"quartile": "Q4", "n": 25, "flag_rate": 0.3, "precision": 0.9},
        ]
    )
    narratives = ["explanation one text", "explanation two text"]

    report = build_report(
        title="FraudLens — Fraud Detection Report",
        metrics=metrics,
        confusion_matrix_fig=_dummy_fig(),
        roc_curve_fig=_dummy_fig(),
        fairness_df=fairness_df,
        narratives=narratives,
        provider_name="xgboost",
        hyperparameters={"learning_rate": 0.11, "max_depth": 4},
        row_counts={"train": 800, "test": 200},
    )
    html = report.to_html()

    assert GOVERNANCE_BANNER in html
    assert "0.99" in html or "0.8500" in html or "accuracy" in html.lower()
    assert "explanation one text" in html
    assert "explanation two text" in html
    assert "Q1" in html and "Q4" in html
    assert "xgboost" in html
    assert "learning_rate" in html
    assert "800" in html and "200" in html
