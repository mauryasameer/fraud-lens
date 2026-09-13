import numpy as np
import pandas as pd
from scipy.stats import skew

from src.services.preprocessing import power_transform, split_fraud_data


def test_power_transform_reduces_skew():
    rng = np.random.default_rng(42)
    skewed = pd.DataFrame({"Amount": rng.exponential(scale=50.0, size=500)})
    train, test = skewed.iloc[:400].reset_index(drop=True), skewed.iloc[400:].reset_index(drop=True)

    train_t, test_t = power_transform(train, test)

    assert abs(skew(train_t["Amount"])) < abs(skew(train["Amount"]))
    assert list(train_t.columns) == list(train.columns)
    assert list(test_t.columns) == list(test.columns)
    assert len(test_t) == len(test)


def test_split_fraud_data_ratios_and_class_balance():
    rng = np.random.default_rng(42)
    n = 1000
    df = pd.DataFrame(
        {
            "Amount": rng.exponential(scale=50.0, size=n),
            "Class": [0] * 950 + [1] * 50,
        }
    )

    train, held_out = split_fraud_data(df, target_col="Class", random_state=42)

    assert len(train) + len(held_out) == n
    assert 0.65 <= len(train) / n <= 0.75

    train_positive_rate = train["Class"].mean()
    held_out_positive_rate = held_out["Class"].mean()
    assert abs(train_positive_rate - 0.05) < 0.02
    assert abs(held_out_positive_rate - 0.05) < 0.02


def test_split_fraud_data_no_overlap():
    rng = np.random.default_rng(7)
    n = 400
    df = pd.DataFrame(
        {
            "id": range(n),
            "Amount": rng.exponential(scale=50.0, size=n),
            "Class": [0] * 380 + [1] * 20,
        }
    )

    train, held_out = split_fraud_data(df, target_col="Class", random_state=42)

    assert set(train["id"]).isdisjoint(set(held_out["id"]))
