from src.app import PROVIDER_HYPERPARAMETERS, PROVIDERS


def test_provider_hyperparameters_match_real_model_defaults():
    for name, provider_cls in PROVIDERS.items():
        provider = provider_cls()
        model_params = provider._model.get_params()
        expected = PROVIDER_HYPERPARAMETERS[name]

        for key, value in expected.items():
            assert model_params[key] == value, (
                f"{name}: expected {key}={value!r}, got {model_params[key]!r}"
            )
