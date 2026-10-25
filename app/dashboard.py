"""
Credit Risk Portfolio Dashboard (Streamlit).

Run from the project root:  streamlit run app/dashboard.py

Pages
  Portfolio  – size, exposure, default rates, risk segments, dimensions, trends, insights
  Account    – one loan: identifiers, PD estimate, risk segment, SHAP drivers, repayment / bureau / collections
  Anomalies  – embedded AI feature (rule-based anomaly detection) with human review queue
  Method     – definitions, model metrics and data notes

All numbers come from data/processed/* produced by the pipeline; nothing is hardcoded.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.anomaly import RULES, detect_anomalies, summarise_flags  # noqa: E402
from src.config import PROFILING_DIMENSIONS  # noqa: E402

PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

st.set_page_config(page_title="Credit Risk Portfolio Analytics", layout="wide", page_icon="📊")

FILTER_DIMS = {"region": "Region", "income_band": "Income band", "employment_type": "Employment type",
               "loan_type": "Loan type", "risk_segment": "Risk segment"}
DIM_LABELS = {"income_band": "Income band", "employment_type": "Employment type", "region": "Region",
              "age_band": "Age band", "loan_type": "Loan type"}
SEG_ORDER = ["Low", "Medium", "High"]
SEG_COLOURS = {"Low": "#7fb77e", "Medium": "#e8b04b", "High": "#c0392b"}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading processed data…")
def load_data():
    required = ["loan_level_dataset.parquet", "scored_loans.parquet", "shap_values.parquet",
                "repayment_history_clean.parquet", "bureau_credit_clean.parquet", "collections_events_clean.parquet"]
    missing = [f for f in required if not (PROC / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed files {missing}. Run `python run_pipeline.py` first.")
    df = pd.read_parquet(PROC / "loan_level_dataset.parquet")
    scored = pd.read_parquet(PROC / "scored_loans.parquet")
    df = df.merge(scored[["account_id", "pd_lightgbm", "pd_logistic", "risk_segment", "split"]], on="account_id", how="left")
    for c in ["income_band", "age_band"]:
        df[c] = df[c].astype(str)
    shap_df = pd.read_parquet(PROC / "shap_values.parquet").set_index("account_id")
    rep = pd.read_parquet(PROC / "repayment_history_clean.parquet")
    bureau = pd.read_parquet(PROC / "bureau_credit_clean.parquet")
    col = pd.read_parquet(PROC / "collections_events_clean.parquet")
    meta = json.loads((MODELS / "model_metadata.json").read_text())
    results = json.loads((REPORTS / "model_results.json").read_text())
    flags = detect_anomalies(df)
    return df, shap_df, rep, bureau, col, meta, results, flags


@st.cache_data
def load_insights_markdown() -> str:
    path = REPORTS / "portfolio_findings.md"
    if not path.exists():
        return "_Run `python -m src.analytics` to generate the findings._"
    text = path.read_text()
    m = re.search(r"## Insights\n(.*)", text, re.S)
    return m.group(1) if m else text


def money(x: float) -> str:
    if pd.isna(x):
        return "—"
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.1f}m"
    if abs(x) >= 1e3:
        return f"${x / 1e3:,.0f}k"
    return f"${x:,.0f}"


def pct(x: float, d: int = 1) -> str:
    return "—" if pd.isna(x) else f"{x * 100:.{d}f}%"


# --------------------------------------------------------------------------- #
# Sidebar: navigation + global filters
# --------------------------------------------------------------------------- #
try:
    DF, SHAP, REP, BUREAU, COL, META, RESULTS, FLAGS = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.sidebar.title("Credit Risk Portfolio")
page = st.sidebar.radio("View", ["Portfolio", "Account", "Anomalies (AI feature)", "Method & data notes"], label_visibility="collapsed")
st.sidebar.markdown("### Filters")
st.sidebar.caption("Filters apply to the Portfolio, Account list and Anomaly views.")
selections = {}
for col_name, label in FILTER_DIMS.items():
    options = sorted(DF[col_name].dropna().unique().tolist(), key=lambda v: (SEG_ORDER.index(v) if v in SEG_ORDER else 99, str(v)))
    if col_name == "income_band":
        options = [o for o in ["<40k", "40-60k", "60-80k", "80-100k", "100k+"] if o in options]
    selections[col_name] = st.sidebar.multiselect(label, options, default=[])

mask = pd.Series(True, index=DF.index)
for col_name, chosen in selections.items():
    if chosen:
        mask &= DF[col_name].isin(chosen)
F = DF[mask]
active_filters = {k: v for k, v in selections.items() if v}
st.sidebar.caption(f"{len(F):,} of {len(DF):,} loans selected" + (" (no filters)" if not active_filters else ""))


# --------------------------------------------------------------------------- #
# Shared chart helpers
# --------------------------------------------------------------------------- #
def profile(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(dim, observed=True)
    out = pd.DataFrame({"Loans": g.size(), "Customers": g["customer_id"].nunique(), "Exposure": g["loan_amount"].sum(),
                        "Loan default rate": g["target_default"].mean(),
                        "Exposure-weighted default rate": g.apply(lambda x: x.loc[x["target_default"] == 1, "loan_amount"].sum() / x["loan_amount"].sum(), include_groups=False)})
    cust = df.groupby([dim, "customer_id"], observed=True)["target_default"].max().reset_index()
    out["Customer default rate"] = cust.groupby(dim, observed=True)["target_default"].mean()
    out["Exposure share"] = out["Exposure"] / out["Exposure"].sum()
    order = {"income_band": ["<40k", "40-60k", "60-80k", "80-100k", "100k+"], "age_band": ["21-30", "31-40", "41-50", "51-60", "61+"],
             "risk_segment": SEG_ORDER}.get(dim)
    if order:
        out = out.reindex([o for o in order if o in out.index])
    return out


def dimension_chart(prof: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=prof.index.astype(str), y=prof["Exposure"], name="Exposure", marker_color="#9fb3c8", yaxis="y")
    fig.add_scatter(x=prof.index.astype(str), y=prof["Loan default rate"] * 100, name="Loan default rate (%)",
                    mode="lines+markers", marker_color="#b23a48", yaxis="y2")
    fig.update_layout(title=title, yaxis=dict(title="Exposure ($)"), yaxis2=dict(title="Default rate (%)", overlaying="y", side="right", rangemode="tozero"),
                      legend=dict(orientation="h", y=-0.25), height=360, margin=dict(t=50, b=20))
    return fig


def repayment_chart(account_id: str) -> go.Figure:
    r = REP[REP["account_id"] == account_id].sort_values("payment_month")
    fig = go.Figure()
    fig.add_bar(x=r["payment_month"], y=r["emi_due"], name="EMI due", marker_color="#d0d7de")
    fig.add_bar(x=r["payment_month"], y=r["emi_paid"], name="EMI paid", marker_color="#4a6fa5")
    fig.add_scatter(x=r["payment_month"], y=r["days_past_due"], name="Days past due", yaxis="y2", mode="lines+markers", marker_color="#b23a48")
    fig.update_layout(barmode="overlay", xaxis=dict(title="Month since origination", dtick=1), yaxis=dict(title="$"),
                      yaxis2=dict(title="DPD", overlaying="y", side="right", range=[0, 95]), height=320, margin=dict(t=30, b=20),
                      legend=dict(orientation="h", y=-0.3))
    return fig


# --------------------------------------------------------------------------- #
# Portfolio view
# --------------------------------------------------------------------------- #
def page_portfolio():
    st.title("Portfolio view")
    if F.empty:
        st.warning("No loans match the current filters. Clear one or more filters in the sidebar.")
        return
    if active_filters:
        st.info("Showing: " + "; ".join(f"{FILTER_DIMS[k]} = {', '.join(v)}" for k, v in active_filters.items()))

    n_loans, n_cust = len(F), F["customer_id"].nunique()
    exposure = F["loan_amount"].sum()
    loan_dr = F["target_default"].mean()
    cust_dr = F.groupby("customer_id")["target_default"].max().mean()
    exp_dr = F.loc[F["target_default"] == 1, "loan_amount"].sum() / exposure
    c = st.columns(6)
    c[0].metric("Loan accounts", f"{n_loans:,}")
    c[1].metric("Customers", f"{n_cust:,}")
    c[2].metric("Originated exposure", money(exposure))
    c[3].metric("Loan default rate", pct(loan_dr))
    c[4].metric("Customer default rate", pct(cust_dr))
    c[5].metric("Exposure in default", pct(exp_dr))
    st.caption("Loan default rate = defaulted loans ÷ loans. Customer default rate = customers with ≥1 defaulted loan ÷ customers "
               "(higher because 39% of customers hold several loans). Exposure in default = originated amount of defaulted loans ÷ "
               "originated amount of all loans. Exposure = `loan_amount` (originated amount).")

    # Risk segments
    st.subheader("Model risk segments")
    seg = profile(F, "risk_segment")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig = go.Figure()
        fig.add_bar(x=seg.index, y=seg["Loans"], name="Loans", marker_color=[SEG_COLOURS[s] for s in seg.index], text=seg["Loans"], textposition="outside")
        fig.update_layout(height=300, title="Loans by risk segment", margin=dict(t=40, b=10), showlegend=False)
        st.plotly_chart(fig, width='stretch')
    with c2:
        show = seg[["Loans", "Exposure", "Exposure share", "Loan default rate"]].copy()
        show["Exposure"] = show["Exposure"].map(money)
        for k in ["Exposure share", "Loan default rate"]:
            show[k] = show[k].map(pct)
        st.dataframe(show, width='stretch')
        cut = META["risk_segment_cutoffs"]
        st.caption(f"Segments are cut on the LightGBM predicted probability using training-set quantiles: High ≥ {cut['High']:.3f} "
                   f"(top 10% of training scores), Medium ≥ {cut['Medium']:.3f} (next 20%), Low below. The default rate shown is the "
                   "observed historical rate for loans in each segment; 80% of loans were used to train the model, so it is partly in-sample.")

    # Dimension profile
    st.subheader("Exposure and default rate by dimension")
    dim = st.selectbox("Dimension", PROFILING_DIMENSIONS, format_func=lambda d: DIM_LABELS[d])
    prof = profile(F, dim)
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.plotly_chart(dimension_chart(prof, f"By {DIM_LABELS[dim].lower()}"), width='stretch')
    with c2:
        show = prof.copy()
        show["Exposure"] = show["Exposure"].map(money)
        for k in ["Exposure share", "Loan default rate", "Exposure-weighted default rate", "Customer default rate"]:
            show[k] = show[k].map(pct)
        st.dataframe(show, width='stretch')
        small = prof[prof["Loans"] < 30]
        if not small.empty:
            st.caption(f"Segments with fewer than 30 loans ({', '.join(map(str, small.index))}) — rates are not reliable.")

    # Trend
    st.subheader("Trend by origination vintage")
    freq = st.radio("Granularity", ["Year", "Quarter"], horizontal=True)
    key = "origination_year" if freq == "Year" else "origination_quarter"
    t = F.groupby(key).agg(Loans=("account_id", "size"), Exposure=("loan_amount", "sum"), rate=("target_default", "mean"))
    fig = go.Figure()
    fig.add_bar(x=t.index.astype(str), y=t["Exposure"], name="Originated exposure", marker_color="#9fb3c8")
    fig.add_scatter(x=t.index.astype(str), y=t["rate"] * 100, name="Loan default rate (%)", yaxis="y2", mode="lines+markers", marker_color="#b23a48")
    fig.update_layout(yaxis=dict(title="Exposure ($)"), yaxis2=dict(title="Default rate (%)", overlaying="y", side="right", rangemode="tozero"),
                      height=320, margin=dict(t=30, b=20), legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, width='stretch')
    st.caption("Only origination dates support a time trend. Repayment history is indexed by month-since-origination and bureau / collections "
               "dates are not aligned to the loan, so no calendar-time delinquency trend is available (see data quality report).")

    # Cross-dimensional cut
    st.subheader("Cross-dimensional cut")
    pairs = {"Income band × Employment type": ("income_band", "employment_type"), "Region × Loan type": ("region", "loan_type"),
             "Age band × Income band": ("age_band", "income_band")}
    pair = st.selectbox("Cut", list(pairs))
    r_dim, c_dim = pairs[pair]
    rate = F.pivot_table(index=r_dim, columns=c_dim, values="target_default", aggfunc="mean", observed=True)
    n = F.pivot_table(index=r_dim, columns=c_dim, values="target_default", aggfunc="size", observed=True)
    order_r = {"income_band": ["<40k", "40-60k", "60-80k", "80-100k", "100k+"], "age_band": ["21-30", "31-40", "41-50", "51-60", "61+"]}
    if r_dim in order_r:
        rate = rate.reindex([o for o in order_r[r_dim] if o in rate.index]); n = n.reindex(rate.index)
    if c_dim in order_r:
        rate = rate[[o for o in order_r[c_dim] if o in rate.columns]]; n = n[rate.columns]
    masked = rate.where(n >= 30)
    text = np.where(n.values >= 30, (rate.values * 100).round(0).astype("object").astype(str) + "%<br>n=" + n.values.astype(int).astype(str),
                    "n<30")
    fig = go.Figure(data=go.Heatmap(z=masked.values * 100, x=masked.columns.astype(str), y=masked.index.astype(str), text=text, texttemplate="%{text}",
                                    colorscale="Reds", zmin=5, zmax=45, colorbar=dict(title="Default %")))
    fig.update_layout(height=380, margin=dict(t=20, b=20))
    st.plotly_chart(fig, width='stretch')
    st.caption("Cells with fewer than 30 loans are blanked rather than shown, because a rate on a handful of loans is not evidence.")

    st.subheader("Key insights (full portfolio)")
    st.caption("These findings are computed on the full book by `src/analytics.py`; they do not change with the sidebar filters.")
    st.markdown(load_insights_markdown())


# --------------------------------------------------------------------------- #
# Account view
# --------------------------------------------------------------------------- #
def page_account():
    st.title("Account view")
    st.caption("Select an account from the filtered list or type an account ID. Predictions are model estimates, not guarantees.")
    c1, c2 = st.columns([1, 1])
    with c1:
        options = F["account_id"].sort_values().tolist()
        chosen = st.selectbox("Account (within current filters)", options, index=0 if options else None, placeholder="No accounts match the filters")
    with c2:
        typed = st.text_input("…or type an account ID (e.g. ACC200001)").strip().upper()
    account_id = typed or chosen
    if not account_id:
        st.warning("No account selected.")
        return
    if account_id not in DF["account_id"].values:
        st.error(f"Account '{account_id}' is not in the portfolio. IDs run from ACC200001 to ACC{200000 + len(DF)}.")
        return
    row = DF.loc[DF["account_id"] == account_id].iloc[0]

    st.subheader(f"{account_id} — {row['loan_type']} for customer {row['customer_id']}")
    a, b, c, d = st.columns(4)
    a.metric("Loan amount", money(row["loan_amount"]))
    b.metric("Balance at month 12", money(row["rep_balance_m12"]))
    rate_txt = f"{row['interest_rate_pct']:.2f}%" if pd.notna(row["interest_rate_pct"]) else f"{row['interest_rate_pct_filled']:.2f}% (imputed)"
    c.metric("Interest rate", rate_txt)
    d.metric("Tenure", f"{int(row['tenure_months'])} months")
    st.markdown(
        f"**Originated:** {row['origination_date'].date()} via {row['disbursal_channel']} · **Purpose:** {row['loan_purpose']} · "
        f"**Region:** {row['region']} ({row['city_tier']}) · **Income band:** {row['income_band']} · "
        f"**Employment:** {row['employment_type']} · **Age band:** {row['age_band']}")
    st.caption("Personal identifiers beyond the account and customer IDs are not shown; gender is excluded from the model and the view.")

    # Score
    st.subheader("Default risk estimate")
    s1, s2, s3 = st.columns(3)
    s1.metric("Estimated probability of default (LightGBM)", pct(row["pd_lightgbm"]))
    s2.metric("Risk segment", str(row["risk_segment"]))
    s3.metric("Baseline logistic estimate", pct(row["pd_logistic"]))
    sample_note = "This loan was in the model's **training set**, so its score is in-sample." if row["split"] == "train" else \
        "This loan was in the **held-out test set**; its score is a genuine out-of-sample estimate."
    st.caption(f"{sample_note} The probability is an estimate of default after the 12-month observation window; the portfolio base rate is "
               f"{pct(DF['target_default'].mean())}. The logistic estimate is class-weighted and not calibrated — use it for ranking only.")
    with st.expander("Historical outcome (label from the supplied data)"):
        st.write("Defaulted" if row["target_default"] == 1 else "Performing")

    # Drivers
    st.subheader("Key drivers of this score")
    if account_id in SHAP.index:
        sv = SHAP.loc[account_id].sort_values(key=np.abs, ascending=False).head(8)
        base = META["shap_base_value"]
        drivers = pd.DataFrame({"Feature": sv.index, "Contribution": sv.values})
        drivers["Value"] = [row[f] if f in row.index else "" for f in drivers["Feature"]]
        drivers["Value"] = drivers["Value"].map(lambda v: f"{v:,.2f}" if isinstance(v, (int, float, np.floating)) and not pd.isna(v) else ("missing" if (isinstance(v, float) and pd.isna(v)) else str(v)))
        fig = go.Figure(go.Bar(x=drivers["Contribution"][::-1], y=drivers["Feature"][::-1], orientation="h",
                               marker_color=["#c0392b" if v > 0 else "#2e86ab" for v in drivers["Contribution"][::-1]],
                               text=drivers["Value"][::-1], textposition="outside"))
        fig.update_layout(height=340, margin=dict(t=10, b=10), xaxis_title="SHAP contribution (log-odds); red raises risk, blue lowers it")
        st.plotly_chart(fig, width='stretch')
        st.caption(f"SHAP values from the LightGBM model (exact TreeSHAP). Base log-odds {base:.2f} ≈ {1 / (1 + np.exp(-base)):.1%} probability; "
                   "the eight largest contributions are shown with the feature's value for this loan. Feature definitions: reports/feature_dictionary.md.")
    else:
        st.info("No driver breakdown is available for this account.")

    # Repayment / bureau / collections
    st.subheader("Repayment history (12-month observation window)")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Max DPD", f"{int(row['rep_max_dpd'])} days")
    r2.metric("Missed months", int(row["rep_n_missed"]))
    r3.metric("Late months", int(row["rep_n_late"]))
    r4.metric("Paid / due", pct(row["rep_paid_ratio"]))
    st.plotly_chart(repayment_chart(account_id), width='stretch')

    st.subheader("Bureau (external credit) summary")
    b = BUREAU[BUREAU["customer_id"] == row["customer_id"]]
    if b.empty:
        st.info("No bureau file for this customer (thin file). Bureau features are missing for the model, which treats this as its own signal.")
    else:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Bureau lines", len(b))
        b2.metric("Written-off lines", int((b["account_status"] == "Written-Off").sum()))
        b3.metric("Max bureau DPD", f"{int(b['days_past_due'].max())}" if b["days_past_due"].notna().any() else "not reported")
        b4.metric("Active outstanding", money(b.loc[b["account_status"] == "Active", "outstanding_amount"].sum()))
        show = b[["credit_type", "lender_category", "account_status", "sanctioned_amount", "outstanding_amount", "days_past_due", "date_opened"]].copy()
        show["date_opened"] = show["date_opened"].dt.date
        st.dataframe(show, width='stretch', hide_index=True)
        st.caption("Bureau values are a snapshot at last reporting (no as-of date in the source); missing DPD is shown as blank, not zero.")

    st.subheader("Collections activity")
    ce = COL[COL["account_id"] == account_id].sort_values("event_date")
    if ce.empty:
        st.write("No collections events recorded for this account.")
    else:
        show = ce[["event_date", "action_type", "outcome", "amount_recovered", "event_before_origination"]].copy()
        show["event_date"] = show["event_date"].dt.date
        st.dataframe(show, width='stretch', hide_index=True)
        if ce["event_before_origination"].sum():
            st.warning(f"{int(ce['event_before_origination'].sum())} event(s) are dated before the loan was originated — a data quality issue in the source; "
                       "collections data is therefore not used by the model.")

    fl = FLAGS[FLAGS["account_id"] == account_id]
    if not fl.empty:
        st.subheader("Anomaly flags on this account")
        st.dataframe(fl[["rule", "rule_name", "evidence", "comparison_group"]], width='stretch', hide_index=True)
        st.caption("Flags are prompts for analyst review, not findings of default or fraud. See the Anomalies page.")


# --------------------------------------------------------------------------- #
# Anomaly detection (embedded AI feature)
# --------------------------------------------------------------------------- #
def page_anomalies():
    st.title("Anomaly detection — embedded AI feature")
    st.markdown(
        "Flags accounts whose repayment or bureau pattern is **unusual relative to their own history or their peer segment**. "
        "A flag is a prompt for a human analyst to look at the account; it is **not** a default, a fraud finding or a decision.")

    with st.expander("How it works: triggers, thresholds and comparison groups", expanded=False):
        rules_tbl = pd.DataFrame([{"Rule": k, "Name": v["name"], "Trigger": v["trigger"], "Compared against": v["comparison"]} for k, v in RULES.items()])
        st.dataframe(rules_tbl, width='stretch', hide_index=True)
        st.markdown(
            "- **Own-history comparison (R1)** uses the account's first nine months as its baseline; every account has exactly 12 months, so the baseline exists for all of them.\n"
            "- **Peer comparison (R2, R3)** uses the loan type × income band segment; segments with fewer than 100 loans fall back to loan type only.\n"
            "- **R4** compares the external bureau snapshot with the internal repayment record and flags external stress that has not yet shown internally.\n"
            "- Thresholds are fixed in `src/anomaly.py` and were chosen so that roughly one account in ten carries at least one flag; they are a starting point for the risk team to tune.")

    flagged_all = summarise_flags(FLAGS, DF)
    flagged = flagged_all[flagged_all["account_id"].isin(F["account_id"])]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accounts in selection", f"{len(F):,}")
    k2.metric("Accounts flagged", f"{len(flagged):,}")
    k3.metric("Share flagged", pct(len(flagged) / len(F)) if len(F) else "—")
    k4.metric("Flagged with 2+ rules", f"{int((flagged['n_flags'] >= 2).sum()):,}" if not flagged.empty else "0")

    if not flagged.empty:
        unflagged_rate = F.loc[~F["account_id"].isin(flagged["account_id"]), "target_default"].mean()
        flagged_rate = flagged["target_default"].mean()
        per_rule = FLAGS[FLAGS["account_id"].isin(F["account_id"])].merge(DF[["account_id", "target_default"]], on="account_id") \
            .groupby("rule")["target_default"].agg(["size", "mean"]).rename(columns={"size": "Flags", "mean": "Observed default rate"})
        per_rule["Observed default rate"] = per_rule["Observed default rate"].map(pct)
        st.markdown(f"**Sanity check against historical outcomes:** flagged accounts defaulted at {pct(flagged_rate)} vs {pct(unflagged_rate)} for "
                    "unflagged accounts in this selection. This shows the flags are enriched for risk, not that a flag *is* a default — "
                    "most flagged accounts did not default, and most defaults were not flagged.")
        st.dataframe(per_rule, width='stretch')

    st.subheader("Flagged accounts")
    if flagged.empty:
        st.info("No flagged accounts in the current selection.")
        return
    rule_filter = st.multiselect("Show accounts with rule", list(RULES), default=[], format_func=lambda r: f"{r} — {RULES[r]['name']}")
    if rule_filter:
        keep = FLAGS[FLAGS["rule"].isin(rule_filter)]["account_id"].unique()
        flagged = flagged[flagged["account_id"].isin(keep)]
    show = flagged[["account_id", "customer_id", "n_flags", "rules", "loan_type", "region", "income_band", "loan_amount", "rep_max_dpd", "rep_n_missed", "bureau_n_written_off"]].copy()
    show["loan_amount"] = show["loan_amount"].map(money)
    st.dataframe(show, width='stretch', hide_index=True, height=300)

    st.subheader("Review a flagged account")
    sel = st.selectbox("Flagged account", flagged["account_id"].tolist())
    fl = FLAGS[FLAGS["account_id"] == sel]
    row = DF.loc[DF["account_id"] == sel].iloc[0]
    st.dataframe(fl[["rule", "rule_name", "measured", "threshold", "comparison_group", "evidence"]], width='stretch', hide_index=True)
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.plotly_chart(repayment_chart(sel), width='stretch')
    with c2:
        st.markdown(f"**Model PD estimate:** {pct(row['pd_lightgbm'])} · **Risk segment:** {row['risk_segment']}  \n"
                    f"**Bureau:** {int(row['bureau_n_lines']) if pd.notna(row['bureau_n_lines']) else 0} lines, "
                    f"{int(row['bureau_n_written_off']) if pd.notna(row['bureau_n_written_off']) else 0} written off, max DPD "
                    f"{int(row['bureau_max_dpd']) if pd.notna(row['bureau_max_dpd']) else 'n/a'}  \n"
                    f"**Historical outcome:** {'Defaulted' if row['target_default'] == 1 else 'Performing'}")
        st.markdown("**Human review (required before any action)**")
        if "review_queue" not in st.session_state:
            st.session_state.review_queue = {}
        decision = st.radio("Analyst decision", ["Pending", "Confirmed — escalate to collections", "Dismissed — explained / false positive", "Needs more information"],
                            index=["Pending", "Confirmed — escalate to collections", "Dismissed — explained / false positive", "Needs more information"].index(
                                st.session_state.review_queue.get(sel, {}).get("decision", "Pending")))
        note = st.text_input("Reviewer note", value=st.session_state.review_queue.get(sel, {}).get("note", ""))
        if st.button("Save review"):
            st.session_state.review_queue[sel] = {"decision": decision, "note": note, "rules": ", ".join(fl["rule"])}
            st.success("Saved to this session's review queue.")
    if st.session_state.get("review_queue"):
        q = pd.DataFrame.from_dict(st.session_state.review_queue, orient="index").rename_axis("account_id").reset_index()
        st.dataframe(q, width='stretch', hide_index=True)
        st.download_button("Download review queue (CSV)", q.to_csv(index=False), "anomaly_review_queue.csv", "text/csv")
        st.caption("The queue lives in this browser session only; a production version would write to an audited case-management system.")

    with st.expander("False positives, false negatives, limitations and safeguards"):
        st.markdown(
            "**False positive** — an account is flagged but nothing is wrong: e.g. R1 fires because a customer paid two instalments 25 days late "
            "around a job change and is now current; R3 fires because a borrower negotiated a reduced instalment; R4 fires because an old bureau "
            "write-off has been settled but the bureau snapshot was not updated. Cost: analyst time and, if acted on without review, an unjustified "
            "collections contact or limit cut.\n\n"
            "**False negative** — an account with a real problem is not flagged: a borrower who was late every single month from month 1 has no "
            "'spike' (R1) and may sit under the peer percentile cut (R2/R3); a customer whose external distress is on a lender that does not report "
            "to the bureau is invisible to R4. The anomaly feature therefore complements the model score; it does not replace it.\n\n"
            "**Human in the loop** — every flag lands in the review queue above with its evidence. An analyst must record a decision "
            "(confirm / dismiss / needs information) before any consequential action; a confirmed flag goes to the collections or credit team under "
            "their normal procedures. No limit change, contact or reporting is triggered automatically by this feature.\n\n"
            "**Limitations** — 12 months of history is a short baseline; bureau data has no as-of date so R4 may reflect stale information; "
            "thresholds are fixed and untuned against outcomes; the historical sanity-check above uses the supplied labels and would not be "
            "available in real time. Flags are not fraud detection: the data contains no transaction-level behaviour.")


# --------------------------------------------------------------------------- #
# Method & data notes
# --------------------------------------------------------------------------- #
def page_method():
    st.title("Method & data notes")
    st.subheader("Model performance (held-out test set)")
    tm = RESULTS["test_metrics"]
    tbl = pd.DataFrame({k: {"ROC-AUC": v["roc_auc"], "Gini (2·AUC−1)": v["gini"], "KS": v["ks"], "Brier": v["brier"]} for k, v in tm.items()}).T.round(4)
    tbl.index = ["Logistic regression (baseline)", "LightGBM"]
    st.dataframe(tbl, width='stretch')
    s = RESULTS["split"]
    st.caption(f"Customer-grouped 80/20 split: {s['train_loans']:,} training loans / {s['test_loans']:,} test loans; "
               f"customer overlap between folds = {s['customer_overlap']}. Full report: reports/model_evaluation.md.")
    oot = RESULTS["out_of_time_2023"]
    st.caption(f"Out-of-time (2023 vintage): LightGBM AUC {oot['lightgbm']['roc_auc']:.3f}, logistic AUC {oot['logistic_regression']['roc_auc']:.3f}.")

    st.subheader("Definitions")
    st.markdown(
        "- **Grain:** one row per loan account; a customer with several loans appears once per loan. Customer-level rates de-duplicate on customer.\n"
        "- **Exposure:** originated `loan_amount`. Outstanding balance at month 12 is shown at account level.\n"
        "- **Prediction point:** end of month 12 of the repayment history (behavioural scorecard); `target_default` is taken as the outcome after that point.\n"
        "- **Risk segments:** Low / Medium / High from LightGBM probability at training-set quantiles (70th, 90th).\n"
        "- **Excluded from the model:** gender, collections events, customer_since.")
    st.subheader("Reports")
    for f in ["data_quality_report.md", "portfolio_findings.md", "model_evaluation.md", "model_evaluation_notes.md", "feature_dictionary.md", "assumptions_log.md"]:
        p = REPORTS / f
        if p.exists():
            with st.expander(f):
                st.markdown(p.read_text().replace("](figures/", "](reports/figures/"))


PAGES = {"Portfolio": page_portfolio, "Account": page_account, "Anomalies (AI feature)": page_anomalies, "Method & data notes": page_method}
PAGES[page]()
