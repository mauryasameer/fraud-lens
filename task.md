# fraud-lens — Task Tracker

Live progress tracker. Keep this in sync with actual state.

## Backlog

- [x] `FraudClassifier` interface
- [x] Logistic regression / random forest / XGBoost providers
- [x] Preprocessing (power transform + stratified split)
- [x] Fraud pipeline orchestration service
- [x] Amount-quartile fairness proxy
- [x] Per-transaction GenAI narrative service
- [x] HTML report service with governance banner and Run Info
- [x] Data fetch script + CLI driver + end-to-end integration test
- [x] GOVERNANCE.md, Docker, README
- [ ] Real end-to-end run against the actual Kaggle dataset (requires Kaggle credentials, not exercised in CI)
- [ ] Real LLM narrative review against actual Ollama output (stubbed in tests)
