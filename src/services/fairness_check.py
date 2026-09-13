from __future__ import annotations

import numpy as np
import pandas as pd


def amount_quartile_breakdown(
    amount: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Flag rate and precision within each Amount quartile of the test set.

    This is an explicitly partial fairness proxy: the dataset carries no
    protected-attribute data, so Amount (the one non-anonymized numeric
    feature) is the only real-world-meaningful lens available.

    Note: when duplicate Amount values collapse the number of actual bins
    below 4 (via duplicates="drop"), returns however many quartile rows
    actually resulted (1 to 4), not a forced 4-row output.
    """
    quartile_raw = pd.qcut(amount, q=4, duplicates="drop")
    categories = list(quartile_raw.cat.categories)
    label_map = {cat: f"Q{i + 1}" for i, cat in enumerate(categories)}
    quartile = quartile_raw.map(label_map)

    df = pd.DataFrame(
        {"quartile": quartile.to_numpy(), "y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred)}
    )

    rows = []
    for q in [f"Q{i + 1}" for i in range(len(categories))]:
        subset = df[df["quartile"] == q]
        n = len(subset)
        flagged = int((subset["y_pred"] == 1).sum())
        flag_rate = flagged / n if n else 0.0
        true_positives = int(((subset["y_pred"] == 1) & (subset["y_true"] == 1)).sum())
        precision = true_positives / flagged if flagged else 0.0
        rows.append({"quartile": q, "n": n, "flag_rate": flag_rate, "precision": precision})

    return pd.DataFrame(rows)
