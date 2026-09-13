from __future__ import annotations

import argparse
import sys

import pandas as pd
from meerax.llm.claude import ClaudeProvider
from meerax.llm.ollama import OllamaProvider
from meerax.llm.openai_provider import OpenAIProvider

from src.providers.logreg_provider import LogRegFraudClassifier
from src.providers.rf_provider import RandomForestFraudClassifier
from src.providers.xgboost_provider import XGBoostFraudClassifier
from src.services.fairness_check import amount_quartile_breakdown
from src.services.fraud_service import run_fraud_pipeline
from src.services.narrative_service import compute_normal_stats, generate_top_n_narratives
from src.services.report_service import build_report

PROVIDERS: dict[str, type] = {
    "logreg": LogRegFraudClassifier,
    "rf": RandomForestFraudClassifier,
    "xgboost": XGBoostFraudClassifier,
}

LLM_PROVIDERS: dict[str, type] = {
    "ollama": OllamaProvider,
    "claude": ClaudeProvider,
    "openai": OpenAIProvider,
}

PROVIDER_HYPERPARAMETERS: dict[str, dict[str, object]] = {
    "logreg": {"C": 0.01, "random_state": 42},
    "rf": {"n_estimators": 860, "criterion": "entropy", "min_samples_leaf": 30, "random_state": 42},
    "xgboost": {
        "learning_rate": 0.11,
        "max_depth": 4,
        "min_child_weight": 30,
        "n_estimators": 285,
        "random_state": 42,
    },
}

EXPECTED_CREDITCARD_COLUMNS = frozenset(
    {"Time", "Amount", "Class", *(f"V{i}" for i in range(1, 29))}
)


def non_negative_int(value: str) -> int:
    """Parse a non-negative command-line integer."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def validate_creditcard_schema(data: pd.DataFrame) -> None:
    """Ensure input matches the fixed Kaggle credit-card fraud schema."""
    if data.columns.has_duplicates:
        raise ValueError("duplicate columns are not allowed")

    columns = set(data.columns)
    missing = EXPECTED_CREDITCARD_COLUMNS - columns
    unexpected = columns - EXPECTED_CREDITCARD_COLUMNS

    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing columns: {', '.join(sorted(missing))}")
        if unexpected:
            details.append(f"unexpected columns: {', '.join(sorted(unexpected))}")
        raise ValueError("; ".join(details))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fraud-lens")
    parser.add_argument("--data", default="src/data/creditcard.csv")
    parser.add_argument("--provider", choices=list(PROVIDERS.keys()), default="xgboost")
    parser.add_argument("--llm-provider", choices=list(LLM_PROVIDERS.keys()), default="ollama")
    parser.add_argument("--top-n", type=non_negative_int, default=5)
    parser.add_argument("--output", default="reports/fraud_report.html")
    args = parser.parse_args(argv)

    try:
        data = pd.read_csv(args.data)
    except (FileNotFoundError, pd.errors.ParserError) as exc:
        print(f"error: could not read data file {args.data}: {exc}", file=sys.stderr)
        return 1

    try:
        validate_creditcard_schema(data)
    except ValueError as exc:
        print(f"error: invalid credit-card data schema: {exc}", file=sys.stderr)
        return 1

    provider = PROVIDERS[args.provider]()
    llm = LLM_PROVIDERS[args.llm_provider]()
    result = run_fraud_pipeline(data, provider)

    fairness_df = amount_quartile_breakdown(result.amount_test, result.y_test.to_numpy(), result.y_pred)
    mean, std = compute_normal_stats(result.X_train_res, result.y_train_res)

    narratives = generate_top_n_narratives(result.X_test, result.y_prob, mean, std, llm, top_n=args.top_n)

    run_metadata = {
        **PROVIDER_HYPERPARAMETERS[args.provider],
        "llm_provider": args.llm_provider,
        "llm_narratives_requested": len(narratives),
        "llm_narrative_fallbacks": sum(1 for n in narratives if "explanation unavailable" in n),
    }

    report = build_report(
        title="FraudLens — Fraud Detection Report",
        metrics=result.metrics,
        confusion_matrix_fig=result.confusion_matrix_fig,
        roc_curve_fig=result.roc_curve_fig,
        fairness_df=fairness_df,
        narratives=narratives,
        provider_name=args.provider,
        hyperparameters=run_metadata,
        row_counts={
            "train_real": result.n_train_real,
            "train_resampled": len(result.X_train_res),
            "test": len(result.X_test),
        },
    )
    report.save(args.output)
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
