# GOVERNANCE.md

## Intended Use

FraudLens supports a human fraud analyst's decision — it does not autonomously block, hold, or
act on any transaction. Every score and explanation it produces is decision support, to be
reviewed by a person before any action is taken.

## Explainability Boundary

`V1`-`V28` are PCA-anonymized components of the original transaction data; they carry no
recoverable real-world meaning. Every generated explanation (see "Top Flagged Transactions —
Explanations" in the report) describes a *statistical* pattern — which anonymized features
deviated most from normal-class behavior, and by how much — never a causal or business claim
about why a transaction is fraudulent. This boundary is stated in the report itself (a permanent
banner), not only here.

## Fairness

This dataset carries no protected-attribute data (no age, gender, location, or similar fields).
A true disparate-impact analysis is not possible on the features available. `Amount` is the one
non-anonymized, real-world-meaningful numeric feature, so every report includes a flag-rate and
precision breakdown by `Amount` quartile — an explicitly partial proxy, not a fairness clearance.

## LLM Controls

The narrative-generation step calls the configured LLM provider at `temperature=0` for every
request, so a given transaction's explanation is reproducible rather than randomly sampled. The
only inputs reaching the prompt are numeric feature names, computed z-scores, and a probability
value — no retrieved documents or user-supplied free text ever reach it, so the prompt-injection
surface is assessed as low.

## Audit Trail

Every generated report embeds a "Run Info" block recording the provider used, its key
hyperparameters, the train/test row counts, and a timestamp — so any given report is traceable
to exactly what produced it.

## Regulatory Framing

FraudLens is a financial-services fraud-scoring model. The applicable model-risk governance
framework is SR 11-7 (model risk management) and, where personal data is in scope, the EU AI
Act and GDPR — the same lens `llm_eval` applies to its own evaluation framework. Naming these
frameworks is not a compliance claim; it states which regulatory lens governs this domain.
