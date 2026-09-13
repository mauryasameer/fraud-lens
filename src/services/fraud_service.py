from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from meerax.data.imbalance import smote_oversample
from meerax.eval.classification import ClassificationMetrics, evaluate_classifier
from meerax.viz.classification import plot_confusion_matrix, plot_roc_curve
from sklearn.metrics import confusion_matrix, roc_curve

from src.core.interfaces import FraudClassifier
from src.services.preprocessing import power_transform, split_fraud_data

EXCLUDED_TIME_COLUMN = "Time"


@dataclass
class FraudPipelineResult:
    provider: FraudClassifier
    metrics: ClassificationMetrics
    confusion_matrix_fig: Figure
    roc_curve_fig: Figure
    X_train_res: pd.DataFrame
    y_train_res: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    y_prob: np.ndarray
    amount_test: pd.Series
    n_train_real: int


def run_fraud_pipeline(
    data: pd.DataFrame,
    provider: FraudClassifier,
    target_col: str = "Class",
    random_state: int = 42,
) -> FraudPipelineResult:
    train, test = split_fraud_data(data, target_col=target_col, random_state=random_state)
    feature_cols = [c for c in data.columns if c not in {target_col, EXCLUDED_TIME_COLUMN}]
    amount_test = test["Amount"].reset_index(drop=True)

    X_train, X_test = power_transform(train[feature_cols], test[feature_cols])
    y_train = train[target_col].reset_index(drop=True)
    y_test = test[target_col].reset_index(drop=True)
    n_train_real = len(X_train)

    if provider.uses_smote:
        X_train_res_arr, y_train_res_arr = smote_oversample(
            X_train.to_numpy(), y_train.to_numpy(), random_state=random_state
        )
        X_train_res = pd.DataFrame(X_train_res_arr, columns=feature_cols)
        y_train_res = pd.Series(y_train_res_arr)
    else:
        X_train_res = X_train.reset_index(drop=True)
        y_train_res = y_train.reset_index(drop=True)

    provider.fit(X_train_res, y_train_res)
    y_pred = provider.predict(X_test)
    y_prob = provider.predict_proba(X_test)

    metrics = evaluate_classifier(y_test.to_numpy(), y_pred, y_prob)
    if metrics.auc_roc is not None and np.isnan(metrics.auc_roc):
        # Newer scikit-learn returns NaN (with an UndefinedMetricWarning) for a
        # single-class y_true instead of raising ValueError, which is the only
        # case evaluate_classifier catches to produce auc_roc=None. Normalize
        # here so the None-based "undefined" contract meerax's own
        # ClassificationMetrics.__str__/to_dict rely on still holds.
        metrics = replace(metrics, auc_roc=None)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    cm_fig = plot_confusion_matrix(cm, labels=["Legit", "Fraud"], title="Confusion Matrix")
    roc_fig = plot_roc_curve(fpr, tpr, metrics.auc_roc or 0.0, title="ROC Curve")

    return FraudPipelineResult(
        provider=provider,
        metrics=metrics,
        confusion_matrix_fig=cm_fig,
        roc_curve_fig=roc_fig,
        X_train_res=X_train_res,
        y_train_res=y_train_res,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        amount_test=amount_test,
        n_train_real=n_train_real,
    )
