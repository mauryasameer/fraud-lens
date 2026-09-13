from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from meerax.llm.base import LLMProvider
from meerax.llm.prompt import PromptTemplate

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This explanation describes statistical deviation from normal-transaction patterns in "
    "anonymized model features. It is not a causal or business explanation of fraud."
)

NARRATIVE_PROMPT = PromptTemplate(
    "A transaction has a predicted fraud probability of {probability:.2%}. Its most unusual "
    "features versus normal transactions are: {feature_summary}. These features are anonymized "
    "PCA components with no recoverable real-world meaning — do not invent what they represent "
    "or assert a causal reason for the flag. In 2-3 sentences, describe the statistical pattern "
    "that drove this flag, referencing the feature names and their deviation."
)


def compute_normal_stats(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Mean/std of the normal (Class == 0) rows only, for z-score computation."""
    normal = X_train[y_train.to_numpy() == 0]
    std = normal.std().replace(0, 1.0)
    return normal.mean(), std


def top_deviant_features(
    row: pd.Series,
    mean: pd.Series,
    std: pd.Series,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    z_scores = (row - mean) / std
    ranked = z_scores.abs().sort_values(ascending=False)
    top_features = ranked.head(top_k).index
    return [(feat, float(z_scores[feat])) for feat in top_features]


def generate_transaction_narrative(
    row: pd.Series,
    probability: float,
    mean: pd.Series,
    std: pd.Series,
    llm: LLMProvider,
    top_k: int = 3,
) -> str:
    top_features = top_deviant_features(row, mean, std, top_k=top_k)
    feature_summary = ", ".join(f"{feat} (z={z:+.2f})" for feat, z in top_features)
    try:
        prompt = NARRATIVE_PROMPT.render(probability=probability, feature_summary=feature_summary)
        explanation = llm.generate(prompt, temperature=0.0).content
    except Exception:
        logger.exception("Narrative generation failed for a flagged transaction")
        return f"explanation unavailable. {DISCLAIMER}"
    return f"{explanation} {DISCLAIMER}"


def generate_top_n_narratives(
    X_test: pd.DataFrame,
    y_prob: np.ndarray,
    mean: pd.Series,
    std: pd.Series,
    llm: LLMProvider,
    top_n: int = 5,
    top_k: int = 3,
) -> list[str]:
    if top_n < 0:
        raise ValueError("top_n must be non-negative")

    order = np.argsort(-np.asarray(y_prob))[:top_n]
    return [
        generate_transaction_narrative(X_test.iloc[i], float(y_prob[i]), mean, std, llm, top_k=top_k)
        for i in order
    ]
