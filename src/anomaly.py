"""
Embedded AI feature (assignment option A): anomaly detection on repayment
and bureau patterns.

An anomaly is an account whose observed pattern is unusual relative to
(a) its own first nine months of history or (b) its peer segment. It is
NOT a default, a fraud finding or a decision — it is a prompt for a
human analyst to look. Every flag carries the measured value, the
threshold, and the comparison group so the analyst can see the evidence.

Rules (all deterministic; thresholds in RULES)
  R1 DPD_SPIKE_VS_OWN_HISTORY   mean DPD in months 10-12 exceeds the account's
                                 own months 1-9 mean by ≥ 20 days AND the
                                 worst month in 10-12 is ≥ 30 DPD.
  R2 MISSED_PAYMENTS_VS_PEERS   missed-payment count strictly above the 95th percentile of
                                 the peer segment (loan_type × income_band)
                                 and at least 2 missed months.
  R3 UNDERPAYMENT_VS_PEERS      paid/due ratio strictly below the 5th percentile of peers.
  R4 EXTERNAL_STRESS_HIDDEN     bureau shows severe stress (≥ 2 written-off
                                 lines or max bureau DPD ≥ 60) while the
                                 internal account looks clean (max DPD ≤ 15):
                                 external risk not yet visible internally.

Peer segments with fewer than MIN_PEER_SIZE loans fall back to loan_type
only so that percentiles are not computed on tiny groups.
"""
from __future__ import annotations

import pandas as pd

MIN_PEER_SIZE = 100

RULES = {
    "R1": {"name": "DPD spike vs own history", "min_increase_days": 20, "min_recent_max_dpd": 30,
           "comparison": "own months 1-9",
           "trigger": "mean DPD in months 10-12 is at least 20 days above the account's months 1-9 mean, and a month in 10-12 reached ≥ 30 DPD"},
    "R2": {"name": "Missed payments vs peers", "peer_quantile": 0.95, "min_missed": 2,
           "comparison": "peer segment (loan type × income band)",
           "trigger": "missed-payment count above the peer 95th percentile and ≥ 2 missed months"},
    "R3": {"name": "Underpayment vs peers", "peer_quantile": 0.05,
           "comparison": "peer segment (loan type × income band)",
           "trigger": "paid/due ratio below the peer 5th percentile"},
    "R4": {"name": "External stress not visible internally", "min_written_off": 2, "min_bureau_dpd": 60, "max_internal_dpd": 15,
           "comparison": "bureau snapshot vs own repayment",
           "trigger": "≥ 2 written-off bureau lines or bureau max DPD ≥ 60, while internal max DPD ≤ 15"},
}


def assign_peer_segment(df: pd.DataFrame) -> pd.Series:
    seg = df["loan_type"].astype(str) + " | " + df["income_band"].astype(str)
    sizes = seg.map(seg.value_counts())
    return seg.where(sizes >= MIN_PEER_SIZE, df["loan_type"].astype(str) + " | all incomes")


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (account, rule) flag with the supporting evidence.

    Expects the loan-level dataset (rep_*, bureau_* and income_band columns).
    """
    d = df.copy()
    d["peer_segment"] = assign_peer_segment(d)
    flags = []

    # R1 — own history
    r1 = RULES["R1"]
    inc = d["rep_mean_dpd_last3"] - d["rep_mean_dpd_first9"]
    m1 = (inc >= r1["min_increase_days"]) & (d["rep_max_dpd_last3"] >= r1["min_recent_max_dpd"])
    for _, row in d[m1].iterrows():
        flags.append({"account_id": row["account_id"], "rule": "R1", "rule_name": r1["name"],
                      "measured": round(float(row["rep_mean_dpd_last3"] - row["rep_mean_dpd_first9"]), 1),
                      "threshold": float(r1["min_increase_days"]), "comparison_group": r1["comparison"],
                      "evidence": f"mean DPD months 1-9 = {row['rep_mean_dpd_first9']:.1f}, months 10-12 = {row['rep_mean_dpd_last3']:.1f}, "
                                  f"worst recent month = {row['rep_max_dpd_last3']:.0f} DPD"})

    # R2 / R3 — peer percentiles
    r2, r3 = RULES["R2"], RULES["R3"]
    p95_missed = d.groupby("peer_segment")["rep_n_missed"].transform(lambda s: s.quantile(r2["peer_quantile"]))
    p05_paid = d.groupby("peer_segment")["rep_paid_ratio"].transform(lambda s: s.quantile(r3["peer_quantile"]))
    peer_n = d.groupby("peer_segment")["account_id"].transform("count")
    m2 = (d["rep_n_missed"] > p95_missed) & (d["rep_n_missed"] >= r2["min_missed"])
    for i, row in d[m2].iterrows():
        flags.append({"account_id": row["account_id"], "rule": "R2", "rule_name": r2["name"],
                      "measured": float(row["rep_n_missed"]), "threshold": float(p95_missed[i]),
                      "comparison_group": f"{row['peer_segment']} (n={int(peer_n[i])})",
                      "evidence": f"{int(row['rep_n_missed'])} missed months vs peer 95th percentile {p95_missed[i]:.0f}"})
    m3 = d["rep_paid_ratio"] < p05_paid
    for i, row in d[m3].iterrows():
        flags.append({"account_id": row["account_id"], "rule": "R3", "rule_name": r3["name"],
                      "measured": round(float(row["rep_paid_ratio"]), 3), "threshold": round(float(p05_paid[i]), 3),
                      "comparison_group": f"{row['peer_segment']} (n={int(peer_n[i])})",
                      "evidence": f"paid {row['rep_total_paid']:,.0f} of {row['rep_total_due']:,.0f} due ({row['rep_paid_ratio']:.1%}) vs peer 5th percentile {p05_paid[i]:.1%}"})

    # R4 — external stress, clean internal history
    r4 = RULES["R4"]
    ext = (d["bureau_n_written_off"] >= r4["min_written_off"]) | (d["bureau_max_dpd"] >= r4["min_bureau_dpd"])
    m4 = ext.fillna(False) & (d["rep_max_dpd"] <= r4["max_internal_dpd"])
    for _, row in d[m4].iterrows():
        flags.append({"account_id": row["account_id"], "rule": "R4", "rule_name": r4["name"],
                      "measured": float(max(row["bureau_n_written_off"], 0)), "threshold": float(r4["min_written_off"]),
                      "comparison_group": r4["comparison"],
                      "evidence": f"{int(row['bureau_n_written_off'])} written-off bureau lines, bureau max DPD {row['bureau_max_dpd']:.0f}, "
                                  f"internal max DPD {row['rep_max_dpd']:.0f}"})

    out = pd.DataFrame(flags, columns=["account_id", "rule", "rule_name", "measured", "threshold", "comparison_group", "evidence"])
    return out.sort_values(["account_id", "rule"]).reset_index(drop=True)


def summarise_flags(flags: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """One row per flagged account: rules triggered, count, and context used
    by the dashboard list (exposure, PD, risk segment, observed default)."""
    if flags.empty:
        return pd.DataFrame(columns=["account_id", "n_flags", "rules"])
    g = flags.groupby("account_id").agg(n_flags=("rule", "size"), rules=("rule", lambda s: ", ".join(sorted(s))))
    cols = ["account_id", "customer_id", "loan_type", "region", "income_band", "employment_type", "loan_amount",
            "rep_max_dpd", "rep_n_missed", "bureau_n_written_off", "target_default"]
    cols = [c for c in cols if c in df.columns]
    return g.reset_index().merge(df[cols], on="account_id", how="left").sort_values(["n_flags", "loan_amount"], ascending=[False, False])
