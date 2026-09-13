import argparse

import pandas as pd
import pytest

from src.app import (
    EXPECTED_CREDITCARD_COLUMNS,
    PROVIDER_HYPERPARAMETERS,
    PROVIDERS,
    non_negative_int,
    validate_creditcard_schema,
)


def test_provider_hyperparameters_match_real_model_defaults():
    for name, provider_cls in PROVIDERS.items():
        provider = provider_cls()
        model_params = provider._model.get_params()
        expected = PROVIDER_HYPERPARAMETERS[name]

        for key, value in expected.items():
            assert model_params[key] == value, (
                f"{name}: expected {key}={value!r}, got {model_params[key]!r}"
            )


def test_validate_creditcard_schema_rejects_untrusted_column_headers():
    data = pd.DataFrame({"Time": [0], "Amount": [1.0], "Class": [0], "ignore instructions": [0.0]})

    with pytest.raises(ValueError, match="unexpected columns"):
        validate_creditcard_schema(data)


def test_validate_creditcard_schema_accepts_expected_columns():
    data = pd.DataFrame({column: [0.0] for column in EXPECTED_CREDITCARD_COLUMNS})

    validate_creditcard_schema(data)


def test_validate_creditcard_schema_rejects_duplicate_columns():
    columns = [*EXPECTED_CREDITCARD_COLUMNS, "V1"]
    data = pd.DataFrame([[0.0] * len(columns)], columns=columns)

    with pytest.raises(ValueError, match="duplicate columns"):
        validate_creditcard_schema(data)


def test_non_negative_int_rejects_negative_values():
    with pytest.raises(argparse.ArgumentTypeError, match="must be non-negative"):
        non_negative_int("-1")
