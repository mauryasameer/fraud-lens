# fraud-lens

![Version](https://img.shields.io/badge/version-0.1.0-blue)

Fraud detection (logistic regression / random forest / SMOTE+XGBoost) with GenAI
per-transaction explanations, built on [meerax](https://github.com/mauryasameer/the-forge).

## Setup

```bash
pip install -r requirements.txt
```

## Fetch data

Requires a Kaggle API token at `~/.kaggle/kaggle.json`:

```bash
python scripts/fetch_data.py
```

## Run

```bash
python -m src.app --provider xgboost --llm-provider ollama --output reports/fraud_report.html
```

## Docker

```bash
docker compose up --build
```

Set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` in `.env` (copy from `.env.example`) if using
`--llm-provider claude` / `--llm-provider openai`; the default `ollama` provider expects Ollama
running on the host.

## Governance

See [GOVERNANCE.md](./GOVERNANCE.md) for intended use, explainability boundaries, fairness
scope, LLM controls, and audit-trail details.

## Testing

```bash
pytest tests/ -v
```
