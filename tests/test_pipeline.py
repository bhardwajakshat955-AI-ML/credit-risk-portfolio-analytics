"""
Tests for the critical data transformations, model behaviour and the
dashboard. Run with:  pytest -q

Fast tests use small synthetic frames; the integration tests read the
processed outputs (run `python run_pipeline.py` first) and are skipped
if those files are absent.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import cleaning, features
from src.anomaly import RULES, assign_peer_segment, detect_anomalies
from src.config import MODELS_DIR, PROCESSED_DIR, RAW_DIR, REPORTS_DIR
from src.ingestion import load_raw_tables, validate_tables
from src.modeling import evaluate, ks_statistic

PROCESSED_READY = (PROCESSED_DIR / "loan_level_dataset.parquet").exists() and (PROCESSED_DIR / "scored_loans.parquet").exists()
needs_pipeline = pytest.mark.skipif(not PROCESSED_READY, reason="run `python run_pipeline.py` first")


# ---------------------------------------------------------------- ingestion
def test_raw_tables_load_and_validate():
    tables = load_raw_tables(RAW_DIR)
    assert set(tables) == {"customers", "loan_accounts", "bureau_credit", "repayment_history", "collections_events"}
    findings = validate_tables(tables)
    assert all(v["duplicate_key_rows"] == 0 for v in findings["primary_keys"].values())
    assert all(v["orphan_rows"] == 0 for v in findings["foreign_keys"].values())
    assert findings["repayment_months_per_account"] == {12: 14685}


def test_validate_rejects_orphan_foreign_key():
    tables = load_raw_tables(RAW_DIR)
    broken = tables["repayment_history"].copy()
    broken.loc[0, "account_id"] = "ACC999999"
    with pytest.raises(ValueError):
        validate_tables({**tables, "repayment_history": broken})


# ----------------------------------------------------------------- cleaning
def test_gender_harmonised_and_missing_flags():
    log = cleaning.IssueLog()
    df = pd.DataFrame({"gender": ["M", "F", "Male", "Female"], "employment_type": ["Salaried"] * 4,
                       "years_employed": [1.0, np.nan, 3.0, 4.0], "num_dependents": [0, 1, np.nan, 2],
                       "years_at_residence": [1.0, 2.0, 3.0, np.nan]})
    out = cleaning.clean_customers(df, log)
    assert set(out["gender"]) == {"Male", "Female"}
    assert out["years_employed_missing"].tolist() == [0, 1, 0, 0]
    assert out["years_employed"].isna().sum() == 1  # kept as NaN, not imputed here
    assert len(log.records) >= 4


def test_interest_rate_filled_by_loan_type_median():
    log = cleaning.IssueLog()
    loans = pd.DataFrame({"account_id": ["a", "b", "c", "d"], "customer_id": ["c1"] * 4,
                          "loan_type": ["Mortgage", "Mortgage", "Credit Card", "Credit Card"],
                          "interest_rate_pct": [6.0, np.nan, 20.0, np.nan], "collateral_flag": [1, 1, 0, 0],
                          "origination_date": pd.to_datetime(["2020-01-01"] * 4)})
    customers = pd.DataFrame({"customer_id": ["c1"], "customer_since": pd.to_datetime(["2019-01-01"])})
    out = cleaning.clean_loan_accounts(loans, customers, log)
    assert out.loc[1, "interest_rate_pct_filled"] == 6.0
    assert out.loc[3, "interest_rate_pct_filled"] == 20.0
    assert out["interest_rate_pct"].isna().sum() == 2


def test_bureau_future_close_dates_nulled():
    log = cleaning.IssueLog()
    b = pd.DataFrame({"bureau_id": ["x", "y"], "customer_id": ["c", "c"], "account_status": ["Closed", "Closed"],
                      "days_past_due": [0.0, np.nan], "date_opened": pd.to_datetime(["2020-01-01", "2021-01-01"]),
                      "date_closed": pd.to_datetime(["2024-01-01", "2028-06-01"])})
    out = cleaning.clean_bureau(b, log)
    assert out["date_closed_implausible"].tolist() == [0, 1]
    assert pd.isna(out.loc[1, "date_closed"]) and out.loc[1, "account_status"] == "Closed"


# ----------------------------------------------------------------- features
def _rep(account, dpds, paid_frac=1.0):
    return pd.DataFrame({"account_id": account, "payment_month": range(1, 13), "emi_due": 100.0,
                         "emi_paid": [0.0 if d >= 30 else 100.0 * paid_frac for d in dpds], "days_past_due": dpds,
                         "payment_status": ["Missed" if d >= 30 else ("Late" if d > 0 else "On-Time") for d in dpds],
                         "outstanding_balance": np.linspace(1000, 500, 12)})


def test_repayment_features_one_row_per_account_and_values():
    rep = pd.concat([_rep("A", [0] * 12), _rep("B", [0] * 9 + [40, 45, 50])])
    f = features.build_repayment_features(rep).set_index("account_id")
    assert len(f) == 2
    assert f.loc["A", "rep_n_missed"] == 0 and f.loc["A", "rep_paid_ratio"] == 1.0
    assert f.loc["A", "rep_months_since_last_missed"] == 13
    assert f.loc["B", "rep_n_missed"] == 3 and f.loc["B", "rep_max_dpd"] == 50
    assert f.loc["B", "rep_missed_last3"] == 3 and f.loc["B", "rep_months_since_last_missed"] == 0
    assert f.loc["B", "rep_dpd_trend"] == pytest.approx(45.0)
    assert f.loc["B", "rep_max_consecutive_delinquent"] == 3
    assert f.loc["B", "rep_paid_ratio"] == pytest.approx(9 / 12)


def test_bureau_features_aggregate_to_customer_grain():
    b = pd.DataFrame({"customer_id": ["c1", "c1", "c2"], "account_status": ["Active", "Written-Off", "Closed"],
                      "lender_category": ["Bank", "NBFC", "Bank"], "sanctioned_amount": [100.0, 50.0, 10.0],
                      "outstanding_amount": [40.0, 50.0, 0.0], "days_past_due": [10.0, np.nan, 0.0]})
    f = features.build_bureau_features(b).set_index("customer_id")
    assert len(f) == 2
    assert f.loc["c1", "bureau_n_written_off"] == 1 and f.loc["c1", "bureau_written_off_share"] == 0.5
    assert f.loc["c1", "bureau_max_dpd"] == 10 and f.loc["c1", "bureau_dpd_unreported_lines"] == 1
    assert f.loc["c1", "bureau_utilisation"] == pytest.approx(0.4)
    assert np.isnan(f.loc["c2", "bureau_utilisation"])  # no active line


# ------------------------------------------------------------------ metrics
def test_ks_and_gini_definitions():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    perfect = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    m = evaluate(y, perfect)
    assert m["roc_auc"] == 1.0 and m["gini"] == 1.0 and ks_statistic(y, perfect) == 1.0
    random = np.array([0.5] * 8)
    m = evaluate(y, random)
    assert m["gini"] == pytest.approx(2 * m["roc_auc"] - 1)


# ------------------------------------------------------------------ anomaly
def test_anomaly_rules_fire_on_constructed_cases():
    n = 150
    base = pd.DataFrame({"account_id": [f"A{i}" for i in range(n)], "loan_type": "Personal Loan", "income_band": "60-80k",
                         "rep_mean_dpd_last3": 0.0, "rep_mean_dpd_first9": 0.0, "rep_max_dpd_last3": 0.0,
                         "rep_n_missed": 0, "rep_paid_ratio": 1.0, "rep_total_paid": 1200.0, "rep_total_due": 1200.0,
                         "bureau_n_written_off": 0.0, "bureau_max_dpd": 0.0, "rep_max_dpd": 0.0})
    base.loc[0, ["rep_mean_dpd_last3", "rep_max_dpd_last3"]] = [45.0, 60.0]        # R1
    base.loc[1, "rep_n_missed"] = 4                                                # R2 (peer p95 is 0)
    base.loc[2, ["rep_paid_ratio", "rep_total_paid"]] = [0.6, 720.0]               # R3
    base.loc[3, ["bureau_n_written_off", "bureau_max_dpd"]] = [2.0, 70.0]          # R4 (internal clean)
    base.loc[4, ["bureau_n_written_off", "rep_max_dpd"]] = [3.0, 40.0]             # not R4: internal not clean
    flags = detect_anomalies(base)
    by_rule = flags.groupby("rule")["account_id"].apply(set).to_dict()
    assert "A0" in by_rule["R1"] and "A1" in by_rule["R2"] and "A2" in by_rule["R3"] and "A3" in by_rule["R4"]
    assert "A4" not in by_rule.get("R4", set())
    assert assign_peer_segment(base).iloc[0] == "Personal Loan | 60-80k"
    assert set(RULES) == {"R1", "R2", "R3", "R4"}


# ------------------------------------------------------- integration (slow)
@needs_pipeline
def test_processed_dataset_grain_and_reconciliation():
    df = pd.read_parquet(PROCESSED_DIR / "loan_level_dataset.parquet")
    raw_loans = pd.read_csv(RAW_DIR / "loan_accounts.csv")
    assert len(df) == len(raw_loans) == df["account_id"].nunique()
    assert df["target_default"].sum() == raw_loans["target_default"].sum()
    assert df["customer_id"].nunique() == 10000
    assert (df["rep_months_observed"] == 12).all()
    assert df.loc[df["has_bureau_file"] == 0, "bureau_n_lines"].isna().all()
    assert df["loan_amount"].sum() == pytest.approx(raw_loans["loan_amount"].sum())


@needs_pipeline
def test_model_artefacts_and_no_customer_leakage():
    scored = pd.read_parquet(PROCESSED_DIR / "scored_loans.parquet")
    tr = set(scored.loc[scored["split"] == "train", "customer_id"])
    te = set(scored.loc[scored["split"] == "test", "customer_id"])
    assert not (tr & te)
    assert scored["pd_lightgbm"].between(0, 1).all()
    assert set(scored["risk_segment"]) == {"Low", "Medium", "High"}
    results = json.loads((REPORTS_DIR / "model_results.json").read_text())
    for m in ["logistic_regression", "lightgbm"]:
        t = results["test_metrics"][m]
        assert 0.5 < t["roc_auc"] < 1.0
        assert t["gini"] == pytest.approx(2 * t["roc_auc"] - 1)
        assert 0 < t["ks"] < 1
    meta = json.loads((MODELS_DIR / "model_metadata.json").read_text())
    assert "gender" not in meta["categorical_features"] and not any(c.startswith("col_") for c in meta["numeric_features"])
    shap_df = pd.read_parquet(PROCESSED_DIR / "shap_values.parquet")
    assert len(shap_df) == len(scored)


@needs_pipeline
def test_dashboard_runs_filter_and_account_lookup():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROCESSED_DIR.parents[1] / "app" / "dashboard.py"), default_timeout=180).run()
    assert not at.exception
    before = at.main.metric[0].value
    at.sidebar.multiselect[0].select("South").run()
    assert not at.exception and at.main.metric[0].value != before and at.main.metric[0].value == "3,600"
    at.sidebar.multiselect[0].unselect("South")
    at.sidebar.radio[0].set_value("Account").run()
    at.main.text_input[0].set_value("ACC999999").run()
    assert at.main.error and "not in the portfolio" in at.main.error[0].value
    at.main.text_input[0].set_value("ACC200001").run()
    assert not at.exception and any("probability of default" in m.label for m in at.main.metric)
    at.main.text_input[0].set_value("").run()
    at.sidebar.radio[0].set_value("Anomalies (AI feature)").run()
    assert not at.exception and any(m.label == "Accounts flagged" for m in at.main.metric)
