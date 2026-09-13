import numpy as np
import pandas as pd

from src.services.narrative_service import (
    DISCLAIMER,
    compute_normal_stats,
    generate_top_n_narratives,
    top_deviant_features,
)


class StubLLM:
    def __init__(self, response="stub explanation"):
        self._response = response
        self.calls = []

    def generate(self, prompt, system=None, images=None, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": kwargs})

        class _Resp:
            def __init__(self, content):
                self.content = content

        return _Resp(self._response)


class FailingLLM:
    def generate(self, prompt, system=None, images=None, **kwargs):
        raise RuntimeError("LLM unavailable")


def test_compute_normal_stats_uses_only_class_zero_rows():
    X_train = pd.DataFrame({"V1": [1.0, 3.0, 100.0], "V2": [10.0, 12.0, 999.0]})
    y_train = pd.Series([0, 0, 1])

    mean, std = compute_normal_stats(X_train, y_train)

    assert mean["V1"] == 2.0
    assert mean["V2"] == 11.0


def test_top_deviant_features_hand_computed():
    mean = pd.Series({"V1": 0.0, "V2": 0.0, "V3": 0.0})
    std = pd.Series({"V1": 1.0, "V2": 1.0, "V3": 1.0})
    row = pd.Series({"V1": 0.5, "V2": -5.0, "V3": 2.0})

    top = top_deviant_features(row, mean, std, top_k=2)

    assert [feat for feat, _z in top] == ["V2", "V3"]
    assert top[0][1] == -5.0
    assert top[1][1] == 2.0


def test_generate_top_n_narratives_passes_temperature_zero():
    X_test = pd.DataFrame({"V1": [0.5, 5.0], "V2": [0.1, -3.0]})
    y_prob = np.array([0.2, 0.9])
    mean = pd.Series({"V1": 0.0, "V2": 0.0})
    std = pd.Series({"V1": 1.0, "V2": 1.0})
    llm = StubLLM()

    narratives = generate_top_n_narratives(X_test, y_prob, mean, std, llm, top_n=1)

    assert len(narratives) == 1
    assert llm.calls[0]["kwargs"]["temperature"] == 0.0


def test_generate_top_n_narratives_always_includes_disclaimer():
    X_test = pd.DataFrame({"V1": [0.5, 5.0], "V2": [0.1, -3.0]})
    y_prob = np.array([0.2, 0.9])
    mean = pd.Series({"V1": 0.0, "V2": 0.0})
    std = pd.Series({"V1": 1.0, "V2": 1.0})

    ok_narratives = generate_top_n_narratives(X_test, y_prob, mean, std, StubLLM(), top_n=2)
    for n in ok_narratives:
        assert DISCLAIMER in n

    failing_narratives = generate_top_n_narratives(X_test, y_prob, mean, std, FailingLLM(), top_n=2)
    for n in failing_narratives:
        assert DISCLAIMER in n


def test_generate_top_n_narratives_falls_back_on_llm_failure():
    X_test = pd.DataFrame({"V1": [0.5], "V2": [0.1]})
    y_prob = np.array([0.9])
    mean = pd.Series({"V1": 0.0, "V2": 0.0})
    std = pd.Series({"V1": 1.0, "V2": 1.0})

    narratives = generate_top_n_narratives(X_test, y_prob, mean, std, FailingLLM(), top_n=1)

    assert "explanation unavailable" in narratives[0]
