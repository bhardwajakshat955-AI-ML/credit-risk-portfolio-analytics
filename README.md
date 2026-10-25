# Credit Risk Portfolio Analytics

End-to-end credit-risk analytics on the supplied five-table synthetic portfolio (10,000 customers, 14,685 loans): data ingestion and cleaning, portfolio profiling, default prediction (logistic regression + LightGBM), an interactive Streamlit dashboard, and an embedded anomaly-detection feature with human review.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
python run_pipeline.py                                  # ~1-2 minutes: dataset → analytics → models
streamlit run app/dashboard.py                          # opens http://localhost:8501
pytest -q                                               # 12 tests
# optional: open notebooks/credit_risk_walkthrough.ipynb for an executed walkthrough with outputs
```

Python 3.10+ is required. No API keys, external services or external datasets are used.

## Project structure

```
data/raw/                 the five source CSVs + DATA_DICTIONARY.md (read-only, never modified)
data/processed/           generated: loan_level_dataset.parquet/.csv, scored_loans.parquet, shap_values.parquet,
                          cleaned repayment / bureau / collections tables, build_summary.json
src/config.py             paths, seed, banding rules, timing assumptions, segment definitions
src/ingestion.py          load five tables, validate primary/foreign keys and coverage
src/cleaning.py           per-table treatments with a structured issue log
src/features.py           bureau (customer grain), repayment (account grain), collections (analytics only), ratios
src/build_dataset.py      orchestrates the above, asserts row counts, writes the dataset and the data quality report
src/analytics.py          profiling across income/employment/region/age/loan type, cross-cuts, trends, insights, figures
src/modeling.py           customer-grouped split, LR + LightGBM, AUC/KS/Gini, SHAP, calibration, OOT and sensitivity runs
src/anomaly.py            the embedded AI feature: four documented anomaly rules with evidence per flag
app/dashboard.py          Streamlit dashboard (Portfolio · Account · Anomalies · Method & data notes)
models/                   generated: lightgbm.joblib, logistic_regression.joblib, model_metadata.json
reports/                  generated reports + hand-written notes (see below) and figures/, tables/
deliverables/             product brief, architecture diagram (+ sources), CRO deck (+ generator), assignment PDF
notebooks/                credit_risk_walkthrough.ipynb (+ .html): executed walkthrough of the whole pipeline with outputs
tests/test_pipeline.py    unit tests on transforms/features/metrics/anomaly rules + integration tests on outputs and dashboard
run_pipeline.py           runs build_dataset → analytics → modeling
```

## Reports and deliverables

| File | Content |
|---|---|
| `reports/data_quality_report.md` | keys, referential integrity, join plan, record reconciliation, every issue with found/handled/why/impact |
| `reports/feature_dictionary.md` | every engineered feature with grain, definition and rationale (generated from code) |
| `reports/portfolio_findings.md` | profiles by dimension, vintage trend, three cross-cuts, five insights with "so what" and actions |
| `reports/model_evaluation.md` | set-up, test/train/out-of-time metrics, thresholds, calibration, sensitivity, importance/SHAP, fairness check, monitoring paragraph |
| `reports/model_evaluation_notes.md` | interpretation of the results and the model-selection recommendation |
| `reports/assumptions_log.md` | 14 assumptions, 7 open questions, what a real bank would need to provide |
| `deliverables/product_brief.md` | one-page product brief |
| `deliverables/architecture.png` / `.svg` | architecture diagram (sources: `architecture.mmd` Mermaid, `architecture.dot` Graphviz) |
| `deliverables/cro_pitch_deck.pptx` | 10-slide CRO deck, generated from the actual results by `build_cro_deck.js` |

## Key definitions (used consistently everywhere)

- **Grain:** one row per loan account; customer-level rates de-duplicate on `customer_id`.
- **Default rate:** loan-level = defaulted loans ÷ loans; customer-level = customers with ≥1 defaulted loan ÷ customers; exposure-weighted = exposure of defaulted loans ÷ exposure.
- **Exposure:** originated `loan_amount` (assumption A6).
- **Prediction point:** end of the 12-month repayment window; `target_default` taken as the outcome after it (A1-A2).
- **Risk segments:** Low / Medium / High at the 70th / 90th percentile of training-set LightGBM scores (cut-offs stored in `models/model_metadata.json`).
- **Excluded from the model:** `gender`, collections events, `customer_since`.

## Results summary (seeded run, `RANDOM_SEED = 42`)

| Model | Test ROC-AUC | Gini | KS | OOT (2023) AUC |
|---|---|---|---|---|
| Logistic regression | 0.684 | 0.368 | 0.276 | 0.696 |
| LightGBM | 0.684 | 0.368 | 0.271 | 0.691 |

Observed default rate by model segment on the test set: Low 11.2%, Medium 22.2%, High 36.4%. Full detail and limitations in `reports/model_evaluation.md`.

## Dashboard

- **Portfolio:** KPIs with explicit denominators, risk-segment mix, exposure and default rate by any of the five dimensions, vintage trend (year/quarter), cross-dimensional heatmap with n<30 masking, full-book insights.
- **Filters (sidebar):** region, income band, employment type, loan type, risk segment — every metric, chart, the account list and the anomaly list recompute on change.
- **Account:** select from the filtered list or type an ID; shows identifiers only, PD estimate and segment (labelled as estimates, with an in-/out-of-sample note), top-8 SHAP drivers with values, repayment chart, bureau lines, collections events (with the before-origination warning) and any anomaly flags. Invalid IDs return a clear error; thin-file customers are handled.
- **Anomalies (AI feature):** rules R1-R4, flagged-account list, per-flag evidence, analyst review queue (session-scoped, downloadable), and the false-positive / false-negative / safeguards documentation.
- **Method & data notes:** model metrics, definitions, and the reports rendered in-app.

## Verification status

Everything below was actually run in the development environment:

- `python run_pipeline.py` — completes; row counts reconcile (14,685 at every join; 2,364 defaults in raw and final); re-running reproduces identical metrics.
- `pytest -q` — 12 passed (ingestion validation, cleaning rules, feature aggregation, KS/Gini definitions, anomaly rules on constructed cases, dataset reconciliation, no customer overlap between folds, model artefacts, dashboard start-up / filter / account lookup / anomaly page via Streamlit's `AppTest`).
- Dashboard: additionally exercised headlessly for dimension and cross-cut selectors, quarter toggle, invalid ID, thin-file account, empty selections on all pages, rule filter and review-queue save/download.
- CRO deck: passes the OOXML validator and renders to PDF; slides visually inspected.

Not run here: the dashboard in a real browser (only headless), and any monitoring job (none exists — PSI/CSI is a proposed design).
