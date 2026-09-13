from __future__ import annotations

import html
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from matplotlib.figure import Figure
from meerax.eval.classification import ClassificationMetrics
from meerax.report.builder import ReportBuilder, ReportSection

GOVERNANCE_BANNER = (
    "GOVERNANCE NOTICE: This report supports human fraud-analyst decisions; it does not "
    "autonomously block or act on transactions. V1-V28 are anonymized PCA components with no "
    "recoverable real-world meaning — every explanation below describes a statistical pattern, "
    "never a causal or business claim."
)


def _fairness_table_html(fairness_df: pd.DataFrame) -> str:
    header = "".join(f"<th>{html.escape(str(c))}</th>" for c in fairness_df.columns)
    body_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
        for row in fairness_df.itertuples(index=False)
    )
    return f"<table class='metrics'><thead><tr>{header}</tr></thead><tbody>{body_rows}</tbody></table>"


def _run_info_html(
    provider_name: str,
    hyperparameters: dict[str, Any],
    row_counts: dict[str, int],
    timestamp: str,
) -> str:
    params = ", ".join(f"{k}={v}" for k, v in hyperparameters.items())
    counts = ", ".join(f"{k}={v}" for k, v in row_counts.items())
    return (
        f"<p class='meta'>Run Info &mdash; provider: {html.escape(provider_name)} | "
        f"hyperparameters: {html.escape(params)} | dataset rows: {html.escape(counts)} | "
        f"generated: {html.escape(timestamp)}</p>"
    )


def build_report(
    title: str,
    metrics: ClassificationMetrics,
    confusion_matrix_fig: Figure,
    roc_curve_fig: Figure,
    fairness_df: pd.DataFrame,
    narratives: list[str],
    provider_name: str,
    hyperparameters: dict[str, Any],
    row_counts: dict[str, int],
) -> ReportBuilder:
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    report = ReportBuilder(title, subtitle=GOVERNANCE_BANNER)

    report.add_section(
        ReportSection(
            title="Model Performance",
            metrics=metrics.to_dict(),
            figures=[confusion_matrix_fig, roc_curve_fig],
        )
    )
    report.add_section(
        ReportSection(
            title="Fairness — Flag Rate & Precision by Amount Quartile",
            content=(
                "Partial fairness proxy: no protected-attribute data exists in this dataset. "
                "This table reports outcome rates by the one available continuous, "
                "non-anonymized feature (Amount)." + _fairness_table_html(fairness_df)
            ),
        )
    )
    report.add_section(
        ReportSection(
            title="Highest Fraud-Probability Transactions — Explanations",
            content=" ".join(f"<p>{html.escape(n)}</p>" for n in narratives),
        )
    )
    report.add_section(
        ReportSection(
            title="Run Info",
            content=_run_info_html(provider_name, hyperparameters, row_counts, timestamp),
        )
    )
    return report
