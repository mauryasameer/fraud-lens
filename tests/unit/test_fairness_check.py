import numpy as np
import pandas as pd
import pytest

from src.services.fairness_check import amount_quartile_breakdown


def test_amount_quartile_breakdown_hand_computed():
    # 8 rows, amounts 10..80 split evenly into Q1..Q4 (2 rows each).
    amount = pd.Series([10, 20, 30, 40, 50, 60, 70, 80])
    y_true = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1])

    result = amount_quartile_breakdown(amount, y_true, y_pred)

    assert list(result["quartile"]) == ["Q1", "Q2", "Q3", "Q4"]
    assert list(result["n"]) == [2, 2, 2, 2]

    # Q1 (rows 0,1): y_pred=[0,1] -> flag_rate 0.5; 1 flagged, 0 true positives -> precision 0.0
    q1 = result[result["quartile"] == "Q1"].iloc[0]
    assert q1["flag_rate"] == pytest.approx(0.5)
    assert q1["precision"] == pytest.approx(0.0)

    # Q4 (rows 6,7): y_pred=[1,1] both true positives -> flag_rate 1.0, precision 1.0
    q4 = result[result["quartile"] == "Q4"].iloc[0]
    assert q4["flag_rate"] == pytest.approx(1.0)
    assert q4["precision"] == pytest.approx(1.0)


def test_amount_quartile_breakdown_zero_flags_gives_zero_precision():
    amount = pd.Series([1, 2, 3, 4])
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])

    result = amount_quartile_breakdown(amount, y_true, y_pred)

    assert (result["flag_rate"] == 0.0).all()
    assert (result["precision"] == 0.0).all()
