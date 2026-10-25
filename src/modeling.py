"""
Default prediction: logistic regression baseline and LightGBM, with
leakage-aware feature selection, customer-grouped splitting, an
out-of-time check, SHAP explanations and sensitivity runs.

Prediction set-up (see assumptions A1-A4)
- Grain: one row per loan account.
- Prediction point: end of month 12 of the loan (behavioural scorecard).
- Inputs: application fields, customer profile, 12-month repayment
  behaviour, bureau snapshot. Collections data is excluded.
- Target: target_default as supplied (assumed to describe the outcome
  after the observation window).

Run:  python -m src.modeling
"""
from __future__ import annotations

import json
import warnings

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, RANDOM_SEED, REPORTS_DIR, RISK_SEGMENT_QUANTILES  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

APPLICATION_NUMERIC = ["loan_amount", "interest_rate_pct", "interest_rate_pct_missing", "tenure_months", "collateral_flag",
                       "age", "years_employed", "years_employed_missing", "annual_income", "num_dependents",
                       "num_dependents_missing", "years_at_residence", "years_at_residence_missing",
                       "loan_to_income", "customer_n_loans", "customer_total_exposure"]
APPLICATION_CATEGORICAL = ["loan_type", "loan_purpose", "disbursal_channel", "marital_status", "education_level",
                           "employment_type", "occupation_sector", "region", "city_tier", "housing_type"]
BUREAU_FEATURES = ["has_bureau_file", "bureau_n_lines", "bureau_n_active", "bureau_n_written_off", "bureau_written_off_share",
                   "bureau_total_sanctioned", "bureau_total_outstanding", "bureau_active_outstanding", "bureau_utilisation",
                   "bureau_max_dpd", "bureau_mean_dpd", "bureau_lines_dpd_30plus", "bureau_dpd_unreported_lines",
                   "bureau_share_fintech_nbfc"]
REPAYMENT_FEATURES = ["rep_max_dpd", "rep_mean_dpd", "rep_n_missed", "rep_n_late", "rep_n_delinquent", "rep_paid_ratio",
                      "rep_missed_last3", "rep_max_dpd_last3", "rep_dpd_trend", "rep_months_since_last_missed",
                      "rep_max_consecutive_delinquent", "rep_balance_m12", "emi_to_income"]

FEATURE_SETS = {
    "full": APPLICATION_NUMERIC + BUREAU_FEATURES + REPAYMENT_FEATURES,
    "application_only": APPLICATION_NUMERIC,
    "no_bureau": APPLICATION_NUMERIC + REPAYMENT_FEATURES,
    "no_repayment": APPLICATION_NUMERIC + BUREAU_FEATURES,
}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def ks_statistic(y_true, y_score) -> float:
    """KS = max separation between the cumulative distributions of scores
    for defaulters and non-defaulters = max(TPR − FPR) over thresholds."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def evaluate(y_true, y_score) -> dict:
    auc = roc_auc_score(y_true, y_score)
    return {"roc_auc": float(auc), "gini": float(2 * auc - 1), "ks": ks_statistic(y_true, y_score),
            "brier": float(brier_score_loss(y_true, y_score))}


def threshold_metrics(y_true, y_score, threshold: float) -> dict:
    pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"threshold": float(threshold), "flagged_share": float(pred.mean()),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0, "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #
def make_preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first" if scale else None, sparse_output=False), categorical),
    ], verbose_feature_names_out=False)


def make_logistic(numeric, categorical, C: float = 0.1) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(numeric, categorical, scale=True)),
                     ("model", LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED))])


def make_lightgbm(numeric, categorical, num_leaves: int = 15, n_estimators: int = 400) -> Pipeline:
    # Median imputation for NaN would hide the 'no bureau file' pattern, so
    # LightGBM gets the raw NaNs and handles them natively.
    prep = ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ], verbose_feature_names_out=False)
    model = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.03, num_leaves=num_leaves,
                               min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               reg_lambda=1.0, random_state=RANDOM_SEED, verbose=-1)
    return Pipeline([("prep", prep), ("model", model)])


def grouped_cv_auc(make, X, y, groups, n_splits: int = 5) -> float:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    for tr, va in cv.split(X, y, groups):
        p = make().fit(X.iloc[tr], y.iloc[tr])
        scores.append(roc_auc_score(y.iloc[va], p.predict_proba(X.iloc[va])[:, 1]))
    return float(np.mean(scores))


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def customer_grouped_split(df: pd.DataFrame, test_size: float = 0.2):
    """Hold out 20% of CUSTOMERS (all of their loans together) so that a
    multi-loan customer never appears on both sides of the split."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_SEED)
    tr, te = next(gss.split(df, df["target_default"], groups=df["customer_id"]))
    return df.index[tr], df.index[te]


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_roc_ks(y, scores: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, s):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate"); axes[0].set_title("ROC — test set"); axes[0].legend()
    for name, s in scores.items():
        fpr, tpr, thr = roc_curve(y, s)
        axes[1].plot(thr[1:], (tpr - fpr)[1:], label=f"{name} (KS {np.max(tpr - fpr):.3f})")
    axes[1].set_xlim(0, 1); axes[1].set_xlabel("Score threshold"); axes[1].set_ylabel("TPR − FPR"); axes[1].set_title("KS curve — test set"); axes[1].legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "model_roc_ks.png", dpi=130); plt.close(fig)


def plot_calibration(y, s, name: str) -> pd.DataFrame:
    dec = pd.qcut(s, 10, labels=False, duplicates="drop")
    tab = pd.DataFrame({"y": y.values, "s": s, "decile": dec}).groupby("decile").agg(
        loans=("y", "size"), observed_default_rate=("y", "mean"), mean_predicted=("s", "mean"))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(tab["mean_predicted"], tab["observed_default_rate"], marker="o", label=name)
    ax.plot([0, tab.max().max()], [0, tab.max().max()], "k--", lw=0.8)
    ax.set_xlabel("Mean predicted probability (decile)"); ax.set_ylabel("Observed default rate"); ax.set_title("Calibration — test set"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "model_calibration.png", dpi=130); plt.close(fig)
    return tab


def plot_importance(imp: pd.Series) -> None:
    top = imp.sort_values().tail(20)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(top.index, top.values, color="#4a6fa5")
    ax.set_title("LightGBM feature importance (gain, top 20)")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "model_feature_importance.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run() -> None:
    MODELS_DIR.mkdir(exist_ok=True); FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROCESSED_DIR / "loan_level_dataset.parquet")
    for c in ["income_band", "age_band"]:
        df[c] = df[c].astype(str)
    y = df["target_default"]
    train_idx, test_idx = customer_grouped_split(df)
    df["split"] = "train"; df.loc[test_idx, "split"] = "test"
    X_tr, X_te, y_tr, y_te = df.loc[train_idx], df.loc[test_idx], y.loc[train_idx], y.loc[test_idx]
    groups_tr = X_tr["customer_id"]
    numeric, categorical = FEATURE_SETS["full"], APPLICATION_CATEGORICAL
    report: dict = {"split": {"train_loans": int(len(X_tr)), "test_loans": int(len(X_te)),
                              "train_customers": int(X_tr["customer_id"].nunique()), "test_customers": int(X_te["customer_id"].nunique()),
                              "train_default_rate": float(y_tr.mean()), "test_default_rate": float(y_te.mean()),
                              "customer_overlap": int(len(set(X_tr["customer_id"]) & set(X_te["customer_id"])))}}

    # ---- Hyper-parameters chosen by grouped CV on the training fold only --
    lr_cv = {C: grouped_cv_auc(lambda C=C: make_logistic(numeric, categorical, C), X_tr, y_tr, groups_tr) for C in [0.01, 0.1, 1.0]}
    best_C = max(lr_cv, key=lr_cv.get)
    lgb_cv = {}
    # The signal is close to additive (see CV results), so the grid leans
    # towards shallow trees; deeper settings over-fit the training fold.
    for nl in [3, 4, 7, 15]:
        for ne in [150, 300]:
            lgb_cv[f"leaves={nl},trees={ne}"] = grouped_cv_auc(lambda nl=nl, ne=ne: make_lightgbm(numeric, categorical, nl, ne), X_tr, y_tr, groups_tr)
    best_lgb = max(lgb_cv, key=lgb_cv.get)
    nl, ne = (int(v.split("=")[1]) for v in best_lgb.split(","))
    report["cv"] = {"logistic_C": lr_cv, "logistic_best_C": best_C, "lightgbm": lgb_cv, "lightgbm_best": best_lgb}

    # ---- Final fits -------------------------------------------------------
    lr = make_logistic(numeric, categorical, best_C).fit(X_tr, y_tr)
    gbm = make_lightgbm(numeric, categorical, nl, ne).fit(X_tr, y_tr)
    p_lr, p_gbm = lr.predict_proba(X_te)[:, 1], gbm.predict_proba(X_te)[:, 1]
    p_lr_tr, p_gbm_tr = lr.predict_proba(X_tr)[:, 1], gbm.predict_proba(X_tr)[:, 1]
    report["test_metrics"] = {"logistic_regression": evaluate(y_te, p_lr), "lightgbm": evaluate(y_te, p_gbm)}
    report["train_metrics"] = {"logistic_regression": evaluate(y_tr, p_lr_tr), "lightgbm": evaluate(y_tr, p_gbm_tr)}

    # ---- Risk segment cut-offs from TRAINING scores of the selected model --
    cutoffs = {k: float(np.quantile(p_gbm_tr, q)) for k, q in RISK_SEGMENT_QUANTILES.items()}
    report["risk_segment_cutoffs"] = cutoffs
    report["threshold_metrics"] = {
        "lightgbm_high_cutoff": threshold_metrics(y_te, p_gbm, cutoffs["High"]),
        "lightgbm_medium_cutoff": threshold_metrics(y_te, p_gbm, cutoffs["Medium"]),
        "logistic_top10pct": threshold_metrics(y_te, p_lr, float(np.quantile(p_lr_tr, 0.9))),
        "logistic_top30pct": threshold_metrics(y_te, p_lr, float(np.quantile(p_lr_tr, 0.7))),
    }
    seg_te = pd.cut(p_gbm, [-np.inf, cutoffs["Medium"], cutoffs["High"], np.inf], labels=["Low", "Medium", "High"])
    report["segment_default_rate_test"] = pd.DataFrame({"seg": seg_te, "y": y_te.values}).groupby("seg", observed=True)["y"].agg(["size", "mean"]).rename(columns={"size": "loans", "mean": "default_rate"}).to_dict("index")

    plot_roc_ks(y_te, {"Logistic regression": p_lr, "LightGBM": p_gbm})
    report["calibration_lightgbm"] = plot_calibration(y_te, p_gbm, "LightGBM").to_dict("index")

    # ---- Out-of-time check: train on 2018-2022 vintages, test on 2023 ------
    oot_tr, oot_te = df["origination_year"] < 2023, df["origination_year"] == 2023
    gbm_oot = make_lightgbm(numeric, categorical, nl, ne).fit(df[oot_tr], y[oot_tr])
    lr_oot = make_logistic(numeric, categorical, best_C).fit(df[oot_tr], y[oot_tr])
    report["out_of_time_2023"] = {"train_loans": int(oot_tr.sum()), "test_loans": int(oot_te.sum()),
                                  "lightgbm": evaluate(y[oot_te], gbm_oot.predict_proba(df[oot_te])[:, 1]),
                                  "logistic_regression": evaluate(y[oot_te], lr_oot.predict_proba(df[oot_te])[:, 1])}

    # ---- Sensitivity runs (LightGBM, same split) -------------------------
    sens = {}
    for name, feats in FEATURE_SETS.items():
        if name == "full":
            continue
        m = make_lightgbm(feats, categorical, nl, ne).fit(X_tr, y_tr)
        sens[name] = evaluate(y_te, m.predict_proba(X_te)[:, 1])
    tf_feats = APPLICATION_NUMERIC + REPAYMENT_FEATURES + [c for c in df.columns if c.startswith("bureau_tf_")]
    m = make_lightgbm(tf_feats, categorical, nl, ne).fit(X_tr, y_tr)
    sens["time_filtered_bureau"] = evaluate(y_te, m.predict_proba(X_te)[:, 1])
    report["sensitivity_lightgbm"] = sens

    # ---- Importance + SHAP -----------------------------------------------
    feat_names = list(gbm.named_steps["prep"].get_feature_names_out())
    imp = pd.Series(gbm.named_steps["model"].booster_.feature_importance("gain"), index=feat_names)
    imp = imp / imp.sum()
    plot_importance(imp)
    imp.sort_values(ascending=False).to_csv(REPORTS_DIR / "tables" / "lightgbm_feature_importance.csv", header=["gain_share"])

    X_all_t = pd.DataFrame(gbm.named_steps["prep"].transform(df), columns=feat_names, index=df.index)
    explainer = shap.TreeExplainer(gbm.named_steps["model"])
    shap_all = explainer.shap_values(X_all_t)
    if isinstance(shap_all, list):
        shap_all = shap_all[1]
    shap_df = pd.DataFrame(shap_all, columns=feat_names, index=df.index)
    shap_df.insert(0, "account_id", df["account_id"].values)
    shap_df.to_parquet(PROCESSED_DIR / "shap_values.parquet", index=False)
    base_value = float(np.ravel(explainer.expected_value)[-1])
    shap_test = shap_all[df.index.get_indexer(test_idx)]
    mean_abs = pd.Series(np.abs(shap_test).mean(axis=0), index=feat_names).sort_values(ascending=False)
    mean_abs.to_csv(REPORTS_DIR / "tables" / "shap_mean_abs_test.csv", header=["mean_abs_shap"])
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_test, X_all_t.loc[test_idx], max_display=20, show=False)
    plt.title("SHAP summary — test set (LightGBM)")
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "model_shap_summary.png", dpi=130, bbox_inches="tight"); plt.close()
    report["top_features_gain"] = imp.sort_values(ascending=False).head(15).round(4).to_dict()
    report["top_features_shap"] = mean_abs.head(15).round(4).to_dict()

    # ---- Logistic coefficients (for the baseline's own explanation) -------
    lr_names = list(lr.named_steps["prep"].get_feature_names_out())
    coef = pd.Series(lr.named_steps["model"].coef_[0], index=lr_names)
    coef.sort_values(key=np.abs, ascending=False).to_csv(REPORTS_DIR / "tables" / "logistic_coefficients.csv", header=["coefficient_standardised"])
    report["top_logistic_coefficients"] = coef.sort_values(key=np.abs, ascending=False).head(15).round(3).to_dict()

    # ---- Fairness / disparity check (gender is NOT a model input) ---------
    scored = df[["account_id", "customer_id", "split", "gender", "age_band", "employment_type"]].copy()
    scored["pd_lightgbm"] = gbm.predict_proba(df)[:, 1]
    scored["pd_logistic"] = lr.predict_proba(df)[:, 1]
    scored["risk_segment"] = pd.cut(scored["pd_lightgbm"], [-np.inf, cutoffs["Medium"], cutoffs["High"], np.inf], labels=["Low", "Medium", "High"]).astype(str)
    scored["target_default"] = y.values
    fair = {}
    te = scored[scored["split"] == "test"]
    for dim in ["gender", "age_band"]:
        rows = {}
        for g, sub in te.groupby(dim):
            rows[str(g)] = {"loans": int(len(sub)), "observed_default_rate": float(sub["target_default"].mean()),
                            "mean_pd": float(sub["pd_lightgbm"].mean()), "high_risk_share": float((sub["risk_segment"] == "High").mean()),
                            "auc_within_group": float(roc_auc_score(sub["target_default"], sub["pd_lightgbm"])) if sub["target_default"].nunique() == 2 else None}
        fair[dim] = rows
    report["fairness_check_test"] = fair

    # ---- Persist artefacts -------------------------------------------------
    joblib.dump(lr, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(gbm, MODELS_DIR / "lightgbm.joblib")
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump({"numeric_features": numeric, "categorical_features": categorical, "transformed_feature_names": feat_names,
                   "risk_segment_cutoffs": cutoffs, "shap_base_value": base_value, "random_seed": RANDOM_SEED,
                   "lightgbm_params": {"num_leaves": nl, "n_estimators": ne}, "logistic_C": best_C}, f, indent=2)
    scored.drop(columns=["gender", "age_band", "employment_type"]).to_parquet(PROCESSED_DIR / "scored_loans.parquet", index=False)
    with open(REPORTS_DIR / "model_results.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    write_model_report(report)
    print(json.dumps(report["test_metrics"], indent=2))


def write_model_report(r: dict) -> None:
    tm, tr, oot, s = r["test_metrics"], r["train_metrics"], r["out_of_time_2023"], r["split"]
    def row(name, m): return f"| {name} | {m['roc_auc']:.4f} | {m['gini']:.4f} | {m['ks']:.4f} | {m['brier']:.4f} |"
    th = r["threshold_metrics"]
    def throw(name, t): return (f"| {name} | {t['threshold']:.3f} | {t['flagged_share'] * 100:.1f}% | {t['precision'] * 100:.1f}% | "
                                f"{t['recall'] * 100:.1f}% | {t['tp']} | {t['fp']} | {t['fn']} | {t['tn']} |")
    L = ["# Model Evaluation Report", "", "Generated by `python -m src.modeling`. All metrics are computed on held-out data; nothing is hardcoded.", "",
         "## Set-up", "",
         "- **Grain / target:** one row per loan account; `target_default` as supplied (1 = defaulted).",
         "- **Prediction point:** end of the 12-month observation window (behavioural scorecard). Inputs are application fields, "
         "customer profile, 12-month repayment aggregates and the bureau snapshot. Collections data and `gender` are excluded "
         "(see `reports/feature_dictionary.md` and assumptions A1-A4).",
         f"- **Split:** customers (not loans) are split 80/20 with a fixed seed, so a customer's loans are all in train or all in test. "
         f"Train: {s['train_loans']:,} loans / {s['train_customers']:,} customers (default rate {s['train_default_rate']:.2%}); "
         f"test: {s['test_loans']:,} loans / {s['test_customers']:,} customers ({s['test_default_rate']:.2%}); customer overlap: {s['customer_overlap']}.",
         "- **Preprocessing:** median imputer + standard scaler + one-hot (logistic), one-hot only with native NaN handling (LightGBM). "
         "All fitted on the training fold only. Hyper-parameters chosen by 5-fold customer-grouped stratified CV on the training fold.",
         "- **Class imbalance:** 16% positive rate. Logistic regression uses `class_weight='balanced'`; LightGBM is unweighted so that its "
         "probabilities stay close to calibrated for the dashboard (see calibration below). Both are evaluated on the untouched test fold.",
         "- **Metric definitions:** ROC-AUC; **Gini = 2 × AUC − 1** (ROC-based definition); **KS** = max(TPR − FPR) across thresholds; Brier score as a calibration check.", "",
         "## Model comparison (test set)", "", "| Model | ROC-AUC | Gini | KS | Brier |", "|---|---|---|---|---|",
         row("Logistic regression (baseline)", tm["logistic_regression"]), row("LightGBM", tm["lightgbm"]), "",
         "Training-fold metrics, for over-fit inspection:", "", "| Model | ROC-AUC | Gini | KS | Brier |", "|---|---|---|---|---|",
         row("Logistic regression", tr["logistic_regression"]), row("LightGBM", tr["lightgbm"]), "",
         f"Out-of-time check — trained on 2018-2022 originations ({oot['train_loans']:,} loans), tested on the 2023 vintage ({oot['test_loans']:,} loans):", "",
         "| Model | ROC-AUC | Gini | KS | Brier |", "|---|---|---|---|---|",
         row("Logistic regression", oot["logistic_regression"]), row("LightGBM", oot["lightgbm"]), "",
         "![ROC and KS](figures/model_roc_ks.png)", "",
         "### Cross-validation (training fold only)", "",
         f"Logistic C: {', '.join(f'{k}: {v:.4f}' for k, v in r['cv']['logistic_C'].items())} → selected C = {r['cv']['logistic_best_C']}.  ",
         f"LightGBM: {', '.join(f'{k}: {v:.4f}' for k, v in r['cv']['lightgbm'].items())} → selected {r['cv']['lightgbm_best']}.", "",
         "## Threshold-dependent metrics (test set)", "",
         f"Risk segments are defined on the LightGBM score using training-set quantiles: **High** ≥ {r['risk_segment_cutoffs']['High']:.3f} "
         f"(top 10% of training scores), **Medium** ≥ {r['risk_segment_cutoffs']['Medium']:.3f} (next 20%), else **Low**.", "",
         "| Operating point | Threshold | Flagged | Precision | Recall | TP | FP | FN | TN |", "|---|---|---|---|---|---|---|---|---|",
         throw("LightGBM — High cut-off", th["lightgbm_high_cutoff"]), throw("LightGBM — Medium cut-off", th["lightgbm_medium_cutoff"]),
         throw("Logistic — top 10% of train scores", th["logistic_top10pct"]), throw("Logistic — top 30% of train scores", th["logistic_top30pct"]), "",
         "Observed default rate by risk segment (test set):", "", "| Segment | Loans | Default rate |", "|---|---|---|"]
    for seg, v in r["segment_default_rate_test"].items():
        L.append(f"| {seg} | {int(v['loans'])} | {v['default_rate']:.1%} |")
    L += ["", "### Calibration (LightGBM, test deciles)", "", "| Decile | Loans | Mean predicted | Observed |", "|---|---|---|---|"]
    for d, v in r["calibration_lightgbm"].items():
        L.append(f"| {d} | {int(v['loans'])} | {v['mean_predicted']:.3f} | {v['observed_default_rate']:.3f} |")
    L += ["", "![Calibration](figures/model_calibration.png)", "",
          "## Sensitivity: what the engineered features add (LightGBM, same split)", "",
          "| Feature set | ROC-AUC | Gini | KS |", "|---|---|---|---|", f"| Full (application + bureau + repayment) | {tm['lightgbm']['roc_auc']:.4f} | {tm['lightgbm']['gini']:.4f} | {tm['lightgbm']['ks']:.4f} |"]
    for k, v in r["sensitivity_lightgbm"].items():
        L.append(f"| {k.replace('_', ' ')} | {v['roc_auc']:.4f} | {v['gini']:.4f} | {v['ks']:.4f} |")
    L += ["", "The `time_filtered_bureau` row replaces the snapshot bureau features with ones computed only from lines opened before "
          "the loan's prediction point (assumption A3). The gap between it and the full model bounds how much of the bureau "
          "contribution could be attributable to information that post-dates the prediction point.", "",
          "## Feature importance and SHAP (LightGBM)", "",
          "Top features by gain share:", "", "| Feature | Gain share |", "|---|---|"]
    for k, v in r["top_features_gain"].items():
        L.append(f"| `{k}` | {v:.3f} |")
    L += ["", "Top features by mean |SHAP| on the test set:", "", "| Feature | Mean abs SHAP |", "|---|---|"]
    for k, v in r["top_features_shap"].items():
        L.append(f"| `{k}` | {v:.4f} |")
    L += ["", "![Importance](figures/model_feature_importance.png)", "", "![SHAP](figures/model_shap_summary.png)", "",
          "Per-loan SHAP values for every account are stored in `data/processed/shap_values.parquet` and drive the account-level "
          "'key drivers' in the dashboard. Logistic regression coefficients (standardised) are in `reports/tables/logistic_coefficients.csv`; "
          "the five largest are: " + ", ".join(f"`{k}` {v:+.2f}" for k, v in list(r["top_logistic_coefficients"].items())[:5]) + ".", "",
          "## Fairness / disparity check (test set, LightGBM)", "",
          "`gender` is not a model input; `age` is (it carries signal, but it is a protected characteristic in many jurisdictions — "
          "flagged for the compliance team, assumption A9). The table reports observed vs predicted default and the share flagged High per group.", ""]
    for dim, rows in r["fairness_check_test"].items():
        L += [f"**{dim}**", "", "| Group | Loans | Observed default | Mean PD | High-risk share | AUC within group |", "|---|---|---|---|---|---|"]
        for g, v in rows.items():
            auc = f"{v['auc_within_group']:.3f}" if v["auc_within_group"] is not None else "n/a"
            L.append(f"| {g} | {v['loans']} | {v['observed_default_rate']:.1%} | {v['mean_pd']:.3f} | {v['high_risk_share']:.1%} | {auc} |")
        L.append("")
    L += ["## Model selection recommendation", "", "_See the narrative in `reports/model_evaluation_notes.md`, written after reviewing these numbers._", "",
          "## Post-deployment monitoring (proposed — not implemented here)", "",
          "Once live, the score should be monitored monthly. **PSI** on the predicted-PD distribution (training population vs the "
          "latest scored month, 10 score bins) and **CSI** on each input feature's distribution (same bins) detect population drift; "
          "PSI/CSI below 0.10 is stable, 0.10-0.25 warrants investigation, above 0.25 triggers escalation to the model-risk team. "
          "Input-distribution checks should include missing-value rates (a jump in `interest_rate_pct_missing` or in customers with no "
          "bureau file would signal an upstream feed problem). When outcomes mature — here, once each loan's post-window default status "
          "is known — observed default rate by score decile, AUC, KS and Gini are recomputed quarterly and compared with the test-set "
          f"benchmarks (AUC {tm['lightgbm']['roc_auc']:.3f}, KS {tm['lightgbm']['ks']:.3f}); a drop of more than 0.05 AUC or a decile "
          "ranking inversion is an alert. Retraining is proposed annually, or earlier if PSI > 0.25, AUC falls below the alert level, "
          "or a material product/policy change occurs. Any retrained model goes through the same held-out and out-of-time validation, "
          "a fairness check, independent model-validation sign-off and a documented rollback plan (previous model kept deployable) "
          "before it replaces the production scorer. **In this assignment** only the one-off validation above exists; no monitoring "
          "job is implemented because there is no post-deployment data to monitor."]
    (REPORTS_DIR / "model_evaluation.md").write_text("\n".join(L))


if __name__ == "__main__":
    run()
