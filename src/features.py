"""
Feature engineering. Each one-to-many table is aggregated to the grain at
which it joins the loan table BEFORE joining, so the loan table is never
multiplied:

    bureau_credit       → one row per customer_id   (joins on customer_id)
    repayment_history   → one row per account_id    (joins on account_id)
    collections_events  → one row per account_id    (analytics only)

Feature definitions are documented in reports/feature_dictionary.md, which
is generated from FEATURE_DOCS below so the code and the documentation
cannot drift apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import OBSERVATION_WINDOW_MONTHS

# (feature, grain, definition, rationale)
FEATURE_DOCS: list[tuple[str, str, str, str]] = []


def _doc(name, grain, definition, rationale):
    FEATURE_DOCS.append((name, grain, definition, rationale))


# --------------------------------------------------------------------------- #
# Bureau (customer grain)
# --------------------------------------------------------------------------- #
def _aggregate_bureau_rows(bureau: pd.DataFrame, prefix: str) -> pd.DataFrame:
    b = bureau.copy()
    b["is_active"] = (b["account_status"] == "Active").astype(int)
    b["is_written_off"] = (b["account_status"] == "Written-Off").astype(int)
    b["is_fintech_nbfc"] = b["lender_category"].isin(["Fintech", "NBFC"]).astype(int)
    b["dpd_30plus"] = (b["days_past_due"] >= 30).astype(int)
    b["active_outstanding"] = b["outstanding_amount"].where(b["is_active"] == 1, 0.0)
    b["active_sanctioned"] = b["sanctioned_amount"].where(b["is_active"] == 1, 0.0)

    g = b.groupby("customer_id")
    out = pd.DataFrame({
        f"{prefix}n_lines": g.size(),
        f"{prefix}n_active": g["is_active"].sum(),
        f"{prefix}n_written_off": g["is_written_off"].sum(),
        f"{prefix}total_sanctioned": g["sanctioned_amount"].sum(),
        f"{prefix}total_outstanding": g["outstanding_amount"].sum(),
        f"{prefix}active_outstanding": g["active_outstanding"].sum(),
        f"{prefix}active_sanctioned": g["active_sanctioned"].sum(),
        f"{prefix}max_dpd": g["days_past_due"].max(),
        f"{prefix}mean_dpd": g["days_past_due"].mean(),
        f"{prefix}lines_dpd_30plus": g["dpd_30plus"].sum(),
        f"{prefix}dpd_unreported_lines": g["days_past_due"].apply(lambda s: s.isna().sum()),
        f"{prefix}share_fintech_nbfc": g["is_fintech_nbfc"].mean(),
    })
    out[f"{prefix}written_off_share"] = out[f"{prefix}n_written_off"] / out[f"{prefix}n_lines"]
    out[f"{prefix}utilisation"] = np.where(
        out[f"{prefix}active_sanctioned"] > 0,
        out[f"{prefix}active_outstanding"] / out[f"{prefix}active_sanctioned"], np.nan)
    return out.reset_index()


def build_bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    """Snapshot bureau features: all lines on the customer's bureau file."""
    feats = _aggregate_bureau_rows(bureau, "bureau_")
    for name, defn, why in [
        ("bureau_n_lines", "count of external credit lines", "breadth of credit history"),
        ("bureau_n_active", "count with account_status = Active", "current external obligations"),
        ("bureau_n_written_off", "count with account_status = Written-Off", "prior defaults elsewhere — strongest bureau signal in the data"),
        ("bureau_written_off_share", "n_written_off / n_lines", "write-offs relative to history length"),
        ("bureau_total_sanctioned", "sum of sanctioned_amount over all lines", "total external credit ever granted"),
        ("bureau_total_outstanding", "sum of outstanding_amount", "external indebtedness (incl. written-off balances)"),
        ("bureau_active_outstanding", "sum of outstanding_amount on Active lines", "live external debt"),
        ("bureau_utilisation", "active_outstanding / active_sanctioned (NaN if no active line)", "how heavily external limits are drawn"),
        ("bureau_max_dpd", "max days_past_due over lines that report DPD", "worst current external delinquency"),
        ("bureau_mean_dpd", "mean days_past_due over lines that report DPD", "typical external delinquency"),
        ("bureau_lines_dpd_30plus", "count of lines with DPD ≥ 30", "number of seriously delinquent external lines"),
        ("bureau_dpd_unreported_lines", "count of lines with missing DPD", "carries the information that DPD was missing instead of imputing it"),
        ("bureau_share_fintech_nbfc", "share of lines from Fintech or NBFC lenders", "lender-mix proxy for credit-seeking behaviour"),
    ]:
        _doc(name, "customer", defn, why)
    return feats


def build_bureau_features_time_filtered(bureau: pd.DataFrame, loans: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity variant (loan grain): only lines opened on or before the
    loan's prediction point (origination + 12 months). Status/DPD remain the
    snapshot values because the table carries no history — see A3."""
    pp = loans[["account_id", "customer_id", "origination_date"]].copy()
    pp["prediction_point"] = pp["origination_date"] + pd.DateOffset(months=OBSERVATION_WINDOW_MONTHS)
    x = pp.merge(bureau, on="customer_id", how="inner")
    x = x[x["date_opened"] <= x["prediction_point"]]
    # Re-use the customer aggregator by temporarily keying on account_id.
    x = x.drop(columns=["customer_id"]).rename(columns={"account_id": "customer_id"})
    feats = _aggregate_bureau_rows(x, "bureau_tf_").rename(columns={"customer_id": "account_id"})
    return feats


# --------------------------------------------------------------------------- #
# Repayment (account grain)
# --------------------------------------------------------------------------- #
def build_repayment_features(rep: pd.DataFrame) -> pd.DataFrame:
    r = rep.sort_values(["account_id", "payment_month"]).copy()
    r["missed"] = (r["payment_status"] == "Missed").astype(int)
    r["late"] = (r["payment_status"] == "Late").astype(int)
    last3 = r["payment_month"] > OBSERVATION_WINDOW_MONTHS - 3

    g = r.groupby("account_id")
    out = pd.DataFrame({
        "rep_months_observed": g.size(),
        "rep_max_dpd": g["days_past_due"].max(),
        "rep_mean_dpd": g["days_past_due"].mean(),
        "rep_n_missed": g["missed"].sum(),
        "rep_n_late": g["late"].sum(),
        "rep_total_due": g["emi_due"].sum(),
        "rep_total_paid": g["emi_paid"].sum(),
        "rep_emi_due": g["emi_due"].first(),
        "rep_balance_m1": g["outstanding_balance"].first(),
        "rep_balance_m12": g["outstanding_balance"].last(),
    })
    out["rep_paid_ratio"] = out["rep_total_paid"] / out["rep_total_due"]
    out["rep_n_delinquent"] = out["rep_n_missed"] + out["rep_n_late"]

    gl = r[last3].groupby("account_id")
    out["rep_missed_last3"] = gl["missed"].sum()
    out["rep_max_dpd_last3"] = gl["days_past_due"].max()
    out["rep_mean_dpd_last3"] = gl["days_past_due"].mean()
    gf = r[~last3].groupby("account_id")
    out["rep_mean_dpd_first9"] = gf["days_past_due"].mean()
    # Positive values mean delinquency is worsening towards the prediction point.
    out["rep_dpd_trend"] = out["rep_mean_dpd_last3"] - out["rep_mean_dpd_first9"]

    last_missed = r[r["missed"] == 1].groupby("account_id")["payment_month"].max()
    out["rep_months_since_last_missed"] = (OBSERVATION_WINDOW_MONTHS - last_missed).reindex(out.index)
    out["rep_months_since_last_missed"] = out["rep_months_since_last_missed"].fillna(OBSERVATION_WINDOW_MONTHS + 1)

    def _max_consecutive_delinquent(s: pd.Series) -> int:
        best = run = 0
        for v in s:
            run = run + 1 if v else 0
            best = max(best, run)
        return best

    out["rep_max_consecutive_delinquent"] = g["days_past_due"].apply(lambda s: _max_consecutive_delinquent(s.gt(0)))
    out = out.reset_index()

    for name, defn, why in [
        ("rep_max_dpd", "max days_past_due over months 1-12", "worst delinquency in the observation window"),
        ("rep_mean_dpd", "mean days_past_due over months 1-12", "average delinquency level"),
        ("rep_n_missed", "count of months with payment_status = Missed (emi_paid = 0)", "missed instalments are the strongest behavioural signal"),
        ("rep_n_late", "count of months with payment_status = Late", "minor delinquency frequency"),
        ("rep_n_delinquent", "n_missed + n_late", "any-delinquency frequency"),
        ("rep_paid_ratio", "sum(emi_paid) / sum(emi_due)", "repayment completeness"),
        ("rep_missed_last3", "missed months within months 10-12", "recent missed payments weigh more than old ones"),
        ("rep_max_dpd_last3", "max DPD in months 10-12", "recent worst delinquency"),
        ("rep_dpd_trend", "mean DPD months 10-12 minus mean DPD months 1-9", "is the account deteriorating or recovering?"),
        ("rep_months_since_last_missed", "12 − last missed month (13 if never missed)", "recency of the last missed payment"),
        ("rep_max_consecutive_delinquent", "longest run of consecutive months with DPD > 0", "persistence of delinquency"),
        ("rep_balance_m12", "outstanding_balance at month 12", "current exposure at the prediction point"),
        ("rep_emi_due", "monthly instalment amount", "used for the EMI-to-income ratio"),
    ]:
        _doc(name, "loan account", defn, why)
    return out


# --------------------------------------------------------------------------- #
# Collections (account grain, analytics only)
# --------------------------------------------------------------------------- #
def build_collections_features(col: pd.DataFrame) -> pd.DataFrame:
    c = col.copy()
    c["is_legal"] = c["action_type"].isin(["Legal Notice", "Written-Off Referral"]).astype(int)
    g = c.groupby("account_id")
    out = pd.DataFrame({
        "col_n_events": g.size(),
        "col_n_legal_or_writeoff": g["is_legal"].sum(),
        "col_total_recovered": g["amount_recovered"].sum(),
        "col_n_events_before_orig": g["event_before_origination"].sum(),
    }).reset_index()
    for name, defn in [
        ("col_n_events", "count of collections actions on the account"),
        ("col_n_legal_or_writeoff", "count of Legal Notice or Written-Off Referral actions"),
        ("col_total_recovered", "sum of amount_recovered"),
        ("col_n_events_before_orig", "events dated before origination (data quality flag)"),
    ]:
        _doc(name, "loan account (analytics only, NOT a model input)", defn, "collections is post-decision information; see A4")
    return out


# --------------------------------------------------------------------------- #
# Loan-level context features
# --------------------------------------------------------------------------- #
def add_loan_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["emi_to_income"] = (df["rep_emi_due"] * 12) / df["annual_income"]
    df["loan_to_income"] = df["loan_amount"] / df["annual_income"]
    df["customer_n_loans"] = df.groupby("customer_id")["account_id"].transform("count")
    df["customer_total_exposure"] = df.groupby("customer_id")["loan_amount"].transform("sum")
    df["has_bureau_file"] = df["bureau_n_lines"].notna().astype(int)
    for name, defn, why in [
        ("emi_to_income", "annualised EMI / annual_income", "affordability"),
        ("loan_to_income", "loan_amount / annual_income", "leverage relative to earnings"),
        ("customer_n_loans", "number of loan accounts the customer holds in this portfolio", "internal multi-borrowing"),
        ("customer_total_exposure", "sum of loan_amount across the customer's accounts", "internal concentration"),
        ("has_bureau_file", "1 if the customer has any bureau line", "thin-file indicator; bureau features are NaN otherwise"),
    ]:
        _doc(name, "loan account", defn, why)
    return df
