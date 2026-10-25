# Product Brief — Credit Risk Portfolio Analytics

**Audience:** Credit-risk function (CRO, portfolio risk managers, credit analysts). **Status:** working prototype on the supplied synthetic dataset.

## Problem
The bank's loan book (14,685 accounts, 10,000 customers, $601m originated) defaults at 16.1% loan-level and 21.1% by exposure, and the losses are concentrated: mortgages are 52% of exposure and default at 24.7%; borrowers under $40k income default at 32.4%; customers recorded as unemployed at 35.0%. Today there is no single view of where that risk sits, no scored watch-list of accounts drifting towards default, and no auditable way to explain why an account looks risky.

## Target user
Portfolio risk managers who set policy and monitor concentrations, and credit/collections analysts who review individual accounts. Secondary: model-risk and compliance teams who need to validate and audit the outputs.

## User needs and business value
- See exposure and default rates by income, employment, region, age and loan type, with correct denominators, in seconds rather than through ad-hoc SQL.
- Rank accounts by estimated default risk with a per-account explanation that an analyst can defend.
- Surface unusual repayment or bureau patterns early enough to act, with a recorded human decision.
- Value (hypothesis, not measured): earlier engagement with the 10% of accounts in the High segment, which held 23% of defaulters in the test set, and policy rules on the segments with 2-3× the book's default rate.

## Proposed solution
A reproducible Python pipeline that joins and cleans the five source tables into one loan-level dataset, profiles the portfolio, trains and validates two default-prediction models (logistic regression baseline, LightGBM; both test AUC ≈ 0.68, Gini ≈ 0.37, KS ≈ 0.27), and serves the results in a Streamlit dashboard with a portfolio view, an account view (PD estimate, risk segment, SHAP drivers, repayment and bureau detail) and working filters. An embedded anomaly-detection feature flags accounts whose delinquency departs from their own history or their peer segment and routes every flag through an analyst review queue.

## Scope — included
Data ingestion, validation and documented cleaning; feature engineering from bureau and repayment history; portfolio analytics with five evidence-based insights and cross-dimensional cuts; two validated models with AUC/KS/Gini, SHAP, calibration, out-of-time and sensitivity checks; interactive dashboard with five filters and account search; rule-based anomaly detection with human-in-the-loop review; automated tests; architecture diagram, assumptions log and CRO deck.

## Scope — explicitly excluded
Production deployment, authentication, access control and audit logging; live data feeds or scheduled scoring; loss-given-default / expected-loss modelling; automated decisions of any kind; LLM or external API integration; monitoring jobs (PSI/CSI are designed, not operated); regulatory compliance assessment; fraud detection (no transaction data exists).

## Key outputs and success measures
| Output | Success measure |
|---|---|
| Clean loan-level dataset + data quality report | Row counts reconcile at every join (14,685 → 14,685); every issue has a documented treatment |
| Portfolio findings | 5 insights each with numerator, denominator, comparison group and action |
| Scoring model | Held-out AUC ≥ 0.65 and out-of-time result within 0.02 of it (achieved: 0.684 / 0.691); High segment default rate ≥ 3× Low (achieved: 36% vs 11%) |
| Dashboard | Starts, loads, filters recompute every metric, account lookup handles invalid IDs — all covered by automated tests (12/12 passing) |
| Anomaly feature | Every flag shows evidence and threshold; no action without a recorded analyst decision |

## Important limitations
The default definition, prediction timing and bureau as-of date are assumptions (A1-A3); collections dates are unreliable and that table is excluded from the model; age is a model driver and a protected characteristic requiring a compliance ruling; the dataset is synthetic, so no number here transfers to a real book; the model ranks risk usefully but at the High cut-off two in three flagged accounts would not default, so outputs are decision support, not decisions.
