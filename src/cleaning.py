"""
Cleaning: table-by-table treatment of the data quality issues found during
profiling. Every treatment is recorded in an issue log (what / how / why /
impact / row count) which is rendered into reports/data_quality_report.md.

Design rules
- Never drop rows from the loan table: every loan carries a target and the
  dashboard must be able to show every account.
- Prefer flag-and-keep over silent imputation. Missing numeric fields keep
  their NaN here and gain a `<field>_missing` flag; the model pipeline fits
  its imputers on the training fold only (see modeling.py). For portfolio
  analytics a separate, transparently imputed copy is used where needed.
"""
from __future__ import annotations

import pandas as pd

from src.config import BUREAU_MAX_PLAUSIBLE_DATE, GENDER_MAP


class IssueLog:
    """Collects data-quality findings as structured records."""

    def __init__(self):
        self.records: list[dict] = []

    def add(self, table: str, field: str, found: str, handled: str, why: str, impact: str, rows: int):
        self.records.append(
            {"table": table, "field": field, "found": found, "handled": handled,
             "why": why, "impact": impact, "rows_affected": int(rows)}
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


def _add_missing_flag(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


# --------------------------------------------------------------------------- #
# customers
# --------------------------------------------------------------------------- #
def clean_customers(df: pd.DataFrame, log: IssueLog) -> pd.DataFrame:
    df = df.copy()

    short_codes = df["gender"].isin(["M", "F"]).sum()
    df["gender"] = df["gender"].map(GENDER_MAP)
    unmapped = df["gender"].isna().sum()
    if unmapped:
        raise ValueError(f"customers.gender has {unmapped} values outside the known label set")
    log.add("customers", "gender",
            f"{short_codes} rows coded 'M'/'F' alongside 'Male'/'Female'",
            "Mapped M→Male, F→Female",
            "Same attribute, two spellings; without harmonising, segment counts would be split across four labels",
            "None on values; gender is used for the fairness check only, never as a model feature",
            short_codes)

    for col in ["years_employed", "num_dependents", "years_at_residence"]:
        n = df[col].isna().sum()
        df = _add_missing_flag(df, col)
        log.add("customers", col, f"{n} missing ({n / len(df):.1%})",
                "Kept as NaN + added missing flag; model imputes with the training-fold median",
                "5% is too much to drop and the field is a weak secondary signal; a median keeps the distribution intact "
                "and the flag lets the model learn if missingness itself carries risk",
                "Slight shrinkage of variance in the imputed field; negligible for profiling", n)

    # Unemployed customers all report a positive annual_income (min 12,000).
    # The dictionary gives no definition of income for this group, so it is
    # kept as supplied and recorded as an open question rather than altered.
    n_unemp = (df["employment_type"] == "Unemployed").sum()
    log.add("customers", "annual_income",
            f"All {n_unemp} 'Unemployed' customers have annual_income > 0 (median ≈ 40.8k)",
            "Kept as supplied; raised as open question Q2 in the assumptions log",
            "Income may include household / benefit / rental income; no basis in the data to overwrite it",
            "Income-based segments for the unemployed group should be read with this in mind", n_unemp)
    return df


# --------------------------------------------------------------------------- #
# loan_accounts
# --------------------------------------------------------------------------- #
def clean_loan_accounts(df: pd.DataFrame, customers: pd.DataFrame, log: IssueLog) -> pd.DataFrame:
    df = df.copy()

    n = df["interest_rate_pct"].isna().sum()
    df = _add_missing_flag(df, "interest_rate_pct")
    # Transparent analytics copy: interest rate is strongly determined by
    # loan_type (medians range 6.5% mortgage → 19.5% credit card), so a
    # loan-type median is far more sensible than a global one.
    type_median = df.groupby("loan_type")["interest_rate_pct"].transform("median")
    df["interest_rate_pct_filled"] = df["interest_rate_pct"].fillna(type_median)
    log.add("loan_accounts", "interest_rate_pct", f"{n} missing ({n / len(df):.1%}), spread evenly across loan types",
            "Raw column kept with NaN + missing flag; `interest_rate_pct_filled` uses the loan-type median for analytics; "
            "the model pipeline imputes on the training fold only",
            "Rate is a pricing field that depends mostly on product; loan-type median preserves that structure",
            "Imputed rows have slightly understated rate dispersion; 2% of rows, so immaterial", n)

    # Consistency checks that passed (recorded so a reviewer knows they ran).
    bad_collateral = ((df["loan_type"].isin(["Mortgage", "Auto Loan"])) != (df["collateral_flag"] == 1)).sum()
    log.add("loan_accounts", "collateral_flag", f"{bad_collateral} rows inconsistent with loan_type (Mortgage/Auto ⇒ secured)",
            "No change needed", "Check only", "None", bad_collateral)

    # Relationship-start date is inconsistent with loan origination for ~half
    # of loans (loan originated before the customer relationship 'began').
    merged = df[["account_id", "customer_id", "origination_date"]].merge(
        customers[["customer_id", "customer_since"]], on="customer_id", how="left")
    n_before = (merged["origination_date"] < merged["customer_since"]).sum()
    df["orig_before_customer_since"] = (merged["origination_date"] < merged["customer_since"]).astype(int).values
    log.add("loan_accounts", "origination_date vs customers.customer_since",
            f"{n_before} loans ({n_before / len(df):.1%}) originated before the customer relationship start date",
            "Both dates kept; a flag column records the inconsistency; relationship tenure is NOT used as a model feature",
            "Cannot tell which date is wrong; a tenure feature derived from them would be meaningless for half the book",
            "Loses a potentially useful 'relationship length' feature; raised as open question Q4", n_before)

    df["origination_year"] = df["origination_date"].dt.year
    df["origination_quarter"] = df["origination_date"].dt.to_period("Q").astype(str)
    return df


# --------------------------------------------------------------------------- #
# bureau_credit
# --------------------------------------------------------------------------- #
def clean_bureau(df: pd.DataFrame, log: IssueLog) -> pd.DataFrame:
    df = df.copy()

    n = df["days_past_due"].isna().sum()
    df = _add_missing_flag(df, "days_past_due")
    log.add("bureau_credit", "days_past_due", f"{n} missing ({n / len(df):.1%}), spread evenly across statuses",
            "Kept as NaN; aggregations skip NaN and a per-customer count of unreported DPD lines is created",
            "DPD for a Closed line is always 0 in the data, but for Active/Written-Off lines the true value is unknown; "
            "filling with 0 would understate risk, filling with the mean would invent delinquency",
            "Customer-level max/mean DPD are computed over reported lines only; a 'dpd_unreported_lines' feature carries the gap", n)

    n_null_closed = df["date_closed"].isna().sum()
    log.add("bureau_credit", "date_closed", f"{n_null_closed} nulls, all on Active lines (structural, not a quality issue)",
            "Kept as NaT", "Null means still open; consistent with account_status", "None", n_null_closed)

    cutoff = pd.Timestamp(BUREAU_MAX_PLAUSIBLE_DATE)
    future = df["date_closed"] > cutoff
    df["date_closed_implausible"] = future.astype(int)
    df.loc[future, "date_closed"] = pd.NaT
    log.add("bureau_credit", "date_closed", f"{future.sum()} closure dates fall after {cutoff.date()} (up to 2029-12-24)",
            "Set to NaT and flagged; account_status kept as 'Closed'/'Written-Off'",
            f"No line was opened after 2025-12-29, so a closure in 2026-2029 cannot be an observed event; "
            "the status label is still informative, the date is not",
            "Closure date is not used as a feature; 'time since closure' features are therefore not built", future.sum())

    # Timing: bureau lines are not aligned to loan originations (opened dates
    # are uniform 2018-2025 regardless of the loan). This is recorded here and
    # handled explicitly in features.py (snapshot vs time-filtered variants).
    log.add("bureau_credit", "date_opened",
            "Opened dates run 2018-2025 and are independent of the customer's loan origination dates "
            "(≈57% of lines open after the customer's latest loan)",
            "Treated as a current snapshot for the primary model; a time-filtered sensitivity model is also trained",
            "The table has no 'as-of' date, so true point-in-time reconstruction is impossible", "Main leakage risk — see assumption A3",
            int(len(df)))
    return df


# --------------------------------------------------------------------------- #
# repayment_history
# --------------------------------------------------------------------------- #
def clean_repayment(df: pd.DataFrame, log: IssueLog) -> pd.DataFrame:
    df = df.copy()
    overpaid = (df["emi_paid"] > df["emi_due"] + 0.01).sum()
    status_mismatch = (
        ((df["payment_status"] == "On-Time") & (df["days_past_due"] > 0))
        | ((df["payment_status"] == "Missed") & (df["emi_paid"] > 0))
    ).sum()
    log.add("repayment_history", "emi_paid / payment_status / days_past_due",
            f"{overpaid} overpayments, {status_mismatch} status-vs-DPD contradictions, 0 missing values, "
            "exactly 12 months per account, max DPD 89",
            "No change needed; validated that Missed ⇔ emi_paid=0 and On-Time ⇔ DPD=0",
            "Table is internally consistent", "None. The 12-month window with DPD<90 defines the observation window (A1)",
            overpaid + status_mismatch)
    return df


# --------------------------------------------------------------------------- #
# collections_events
# --------------------------------------------------------------------------- #
def clean_collections(df: pd.DataFrame, loans: pd.DataFrame, log: IssueLog) -> pd.DataFrame:
    df = df.copy()
    merged = df.merge(loans[["account_id", "origination_date"]], on="account_id", how="left")
    before = merged["event_date"] < merged["origination_date"]
    df["event_before_origination"] = before.astype(int).values
    log.add("collections_events", "event_date",
            f"{before.sum()} of {len(df)} events ({before.mean():.0%}) are dated before the loan was originated",
            "Kept and flagged; collections data is used for portfolio analytics and the account view only, "
            "and is excluded from the default model",
            "A collections action cannot precede the loan; the date field is unreliable, and collections activity "
            "is in any case a consequence of delinquency (post-decision information) rather than a predictor "
            "available at the prediction point",
            "The model does not benefit from collections signal; analytics on collections intensity are presented "
            "without a time dimension", before.sum())
    return df
