"""
Build the analysis-ready, loan-level dataset.

Grain: one row per loan account (14,685 rows). The customer dimension is
joined one-to-one onto each loan (so customer attributes repeat for
multi-loan customers — customer-level metrics must therefore de-duplicate
on customer_id, which analytics.py does). Bureau, repayment and collections
tables are aggregated to their join grain first.

Run:  python -m src.build_dataset
"""
from __future__ import annotations

import json

import pandas as pd

from src import cleaning, features
from src.config import AGE_BANDS, INCOME_BANDS, PROCESSED_DIR, REPORTS_DIR
from src.ingestion import load_raw_tables, validate_tables

JOIN_PLAN = [
    # (left, right, key, how, cardinality, rationale)
    ("loan_accounts", "customers", "customer_id", "left", "many-to-one",
     "Every loan has exactly one customer (0 orphans). Left join keeps all loans; row count cannot change."),
    ("loan_accounts", "bureau features", "customer_id", "left", "one-to-one after aggregation",
     "Bureau is many-to-one with customers, so it is aggregated to one row per customer first. "
     "Left join because 1,008 customers (1,462 loans) have no bureau file; they get NaN features and has_bureau_file=0."),
    ("loan_accounts", "repayment features", "account_id", "left", "one-to-one after aggregation",
     "12 monthly rows per account are aggregated to one row per account. Every loan has repayment rows, so this is effectively inner."),
    ("loan_accounts", "collections features", "account_id", "left", "one-to-one after aggregation",
     "Only 4,278 loans have collections events; others get 0 events. Analytics only — not a model input."),
    ("loan_accounts", "time-filtered bureau features", "account_id", "left", "one-to-one after aggregation",
     "Sensitivity variant keyed on the loan because the cut-off date differs per loan."),
]


def add_bands(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["income_band"] = pd.cut(df["annual_income"], bins=INCOME_BANDS["bins"], labels=INCOME_BANDS["labels"], right=False)
    df["age_band"] = pd.cut(df["age"], bins=AGE_BANDS["bins"], labels=AGE_BANDS["labels"], right=True)
    return df


def build(raw_dir=None, write: bool = True) -> tuple[pd.DataFrame, dict]:
    tables = load_raw_tables(raw_dir) if raw_dir else load_raw_tables()
    validation = validate_tables(tables)
    log = cleaning.IssueLog()

    customers = cleaning.clean_customers(tables["customers"], log)
    loans = cleaning.clean_loan_accounts(tables["loan_accounts"], customers, log)
    bureau = cleaning.clean_bureau(tables["bureau_credit"], log)
    rep = cleaning.clean_repayment(tables["repayment_history"], log)
    col = cleaning.clean_collections(tables["collections_events"], loans, log)

    features.FEATURE_DOCS.clear()
    bureau_f = features.build_bureau_features(bureau)
    bureau_tf = features.build_bureau_features_time_filtered(bureau, loans)
    rep_f = features.build_repayment_features(rep)
    col_f = features.build_collections_features(col)

    # ---- Joins, with a row-count assertion after every step ---------------
    n0 = len(loans)
    reconciliation = {"loan_accounts_start": n0}
    df = loans.merge(customers, on="customer_id", how="left", validate="many_to_one")
    reconciliation["after_customers"] = len(df)
    df = df.merge(bureau_f, on="customer_id", how="left", validate="many_to_one")
    reconciliation["after_bureau"] = len(df)
    df = df.merge(rep_f, on="account_id", how="left", validate="one_to_one")
    reconciliation["after_repayment"] = len(df)
    df = df.merge(col_f, on="account_id", how="left", validate="one_to_one")
    reconciliation["after_collections"] = len(df)
    df = df.merge(bureau_tf, on="account_id", how="left", validate="one_to_one")
    reconciliation["after_bureau_time_filtered"] = len(df)
    for k, v in reconciliation.items():
        if v != n0:
            raise AssertionError(f"Row multiplication at {k}: {v} != {n0}")

    for c in ["col_n_events", "col_n_legal_or_writeoff", "col_total_recovered", "col_n_events_before_orig"]:
        df[c] = df[c].fillna(0)
    df["has_collections"] = (df["col_n_events"] > 0).astype(int)

    df = features.add_loan_context_features(df)
    df = add_bands(df)

    # Sanity: target preserved and every loan still has a repayment profile.
    assert df["target_default"].isin([0, 1]).all()
    assert df["rep_months_observed"].eq(12).all()
    reconciliation["customers_in_final"] = int(df["customer_id"].nunique())
    reconciliation["defaults_in_final"] = int(df["target_default"].sum())
    reconciliation["defaults_in_raw"] = int(tables["loan_accounts"]["target_default"].sum())

    summary = {
        "validation": validation,
        "reconciliation": reconciliation,
        "issue_log": log.records,
        "final_shape": list(df.shape),
    }
    if write:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PROCESSED_DIR / "loan_level_dataset.parquet", index=False)
        df.to_csv(PROCESSED_DIR / "loan_level_dataset.csv", index=False)
        # Cleaned repayment and collections tables are kept for the dashboard's
        # account view (monthly history) and the anomaly feature.
        rep.to_parquet(PROCESSED_DIR / "repayment_history_clean.parquet", index=False)
        col.to_parquet(PROCESSED_DIR / "collections_events_clean.parquet", index=False)
        bureau.to_parquet(PROCESSED_DIR / "bureau_credit_clean.parquet", index=False)
        with open(PROCESSED_DIR / "build_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        write_data_quality_report(summary, df)
        write_feature_dictionary()
    return df, summary


def write_data_quality_report(summary: dict, df: pd.DataFrame) -> None:
    v, r = summary["validation"], summary["reconciliation"]
    lines = ["# Data Quality Report", "",
             "Generated by `python -m src.build_dataset`. All counts are computed from the supplied files.", "",
             "## 1. Source tables and keys", "",
             "| Table | Rows | Primary key | Duplicate key rows |", "|---|---|---|---|"]
    for t, rc in v["row_counts"].items():
        pk = v["primary_keys"][t]
        lines.append(f"| {t} | {rc:,} | {', '.join(pk['keys'])} | {pk['duplicate_key_rows']} |")
    lines += ["", "No table has full-row or key duplicates, so no de-duplication was required.", "",
              "## 2. Referential integrity", "", "| Relationship | Orphan child rows |", "|---|---|"]
    for k, fk in v["foreign_keys"].items():
        lines.append(f"| {k} | {fk['orphan_rows']} |")
    cov = v["coverage"]
    lines += ["", f"- Customers without any loan: {cov['customers_without_loans']}",
              f"- Customers without a bureau file: {cov['customers_without_bureau']:,} (their loans get NaN bureau features and `has_bureau_file = 0`)",
              f"- Loans without repayment history: {cov['loans_without_repayment']}",
              f"- Loans with at least one collections event: {cov['loans_with_collections']:,}",
              f"- Repayment months per account: {v['repayment_months_per_account']} (every account has exactly 12 months)",
              f"- Loans per customer: {v['loans_per_customer']}", "",
              "## 3. Join plan", "",
              "| Left | Right | Key | Join | Cardinality | Rationale |", "|---|---|---|---|---|---|"]
    for l, rt, k, how, card, why in JOIN_PLAN:
        lines.append(f"| {l} | {rt} | {k} | {how} | {card} | {why} |")
    lines += ["", "## 4. Record reconciliation", "", "| Step | Rows |", "|---|---|"]
    for k, val in r.items():
        lines.append(f"| {k} | {val:,} |")
    lines += ["", "The loan count is unchanged at every join (asserted in code), and the number of defaults in the "
              "final dataset equals the number in the raw loan table.", "",
              "## 5. Issues found and treatments", ""]
    for i, rec in enumerate(summary["issue_log"], 1):
        lines += [f"### {i}. `{rec['table']}.{rec['field']}` — {rec['rows_affected']:,} rows", "",
                  f"- **Found:** {rec['found']}", f"- **Handled:** {rec['handled']}",
                  f"- **Why:** {rec['why']}", f"- **Impact:** {rec['impact']}", ""]
    lines += ["## 6. Outliers that are legitimate signal", "",
              "- Mortgage loan amounts (up to 444,803) are 20-50× the median personal loan. They are real product differences, "
              "not errors, and are kept; exposure-weighted metrics are reported separately from count-based ones for this reason.",
              "- Repayment DPD of 60-89 days and bureau write-offs are extreme values that carry most of the default signal; "
              "they are kept untouched and no winsorisation is applied before modelling (tree models are scale-insensitive; "
              "the logistic model uses a standard scaler fitted on the training fold).", "",
              "## 7. Final dataset", "",
              f"- File: `data/processed/loan_level_dataset.parquet` (and `.csv`), shape {summary['final_shape'][0]:,} × {summary['final_shape'][1]}",
              "- Grain: one row per loan account; customer attributes repeat for multi-loan customers.",
              "- Target: `target_default` (1 = defaulted, 0 = performing), loan-level, taken as supplied. "
              f"Default rate {df['target_default'].mean():.2%} loan-level; "
              f"{df.groupby('customer_id')['target_default'].max().mean():.2%} of customers have at least one defaulted loan.",
              "- Schema: raw loan + customer columns, `*_missing` flags, `bureau_*` (customer-grain snapshot), "
              "`bureau_tf_*` (time-filtered sensitivity variant), `rep_*` (account-grain repayment), `col_*` (collections, analytics only), "
              "context ratios (`emi_to_income`, `loan_to_income`, `customer_n_loans`, `customer_total_exposure`) and bands "
              "(`income_band`, `age_band`). Full definitions: `reports/feature_dictionary.md`.",
              "- Exposure definition used throughout: `loan_amount` (originated amount). `rep_balance_m12` is shown as the "
              "outstanding balance at the prediction point where relevant. See assumption A6."]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "data_quality_report.md").write_text("\n".join(lines))


def write_feature_dictionary() -> None:
    lines = ["# Feature Dictionary", "", "Generated from `src/features.py`. Grain states the level at which the feature is "
             "computed before joining to the loan table.", "", "| Feature | Grain | Definition | Rationale |", "|---|---|---|---|"]
    for name, grain, defn, why in features.FEATURE_DOCS:
        lines.append(f"| `{name}` | {grain} | {defn} | {why} |")
    lines += ["", "Raw application fields used directly by the model: `loan_type`, `loan_purpose`, `loan_amount`, "
              "`interest_rate_pct`, `tenure_months`, `disbursal_channel`, `collateral_flag`, `age`, `marital_status`, "
              "`education_level`, `employment_type`, `occupation_sector`, `years_employed`, `annual_income`, `region`, "
              "`city_tier`, `num_dependents`, `housing_type`, `years_at_residence` plus the `*_missing` flags.", "",
              "Deliberately excluded from the model: `gender` (fairness — used only for the post-hoc disparity check), "
              "`customer_since` (inconsistent with origination dates), all `col_*` collections features (post-decision "
              "information with unreliable dates), `rep_balance_m1`/`rep_total_*` (redundant with loan_amount and paid_ratio), "
              "and identifiers."]
    (REPORTS_DIR / "feature_dictionary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    df, summary = build()
    print(f"Built loan-level dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(json.dumps(summary["reconciliation"], indent=2))
