# Model Evaluation Notes — reading the results

These notes interpret the numbers in `model_evaluation.md` / `model_results.json` (seeded run, `RANDOM_SEED = 42`). If the pipeline is re-run in a different environment the figures may move in the third decimal; the conclusions below do not depend on that.

## 1. How much signal is there?

Both models land at a test ROC-AUC of about 0.68 (Gini ≈ 0.37, KS ≈ 0.27), and the out-of-time check on the 2023 vintage gives essentially the same picture (AUC 0.69-0.70). That is a usable but modest separation: the top-decile "High" segment defaults at 36% against an 11% rate in "Low", so the score ranks risk three-to-one across segments, but it is nowhere near strong enough to make automated decline decisions. The data dictionary describes the synthetic signal as "clear but not trivial", and that is what we see.

Three things support that the result is genuine rather than an artefact:

- The train/test gap is small for both models (LightGBM 0.71 → 0.68; logistic 0.69 → 0.68), so neither is over-fitting the training customers.
- The customer-grouped split means no borrower's loans are split across folds, and the out-of-time vintage test is consistent with the random split.
- The sensitivity runs behave as a credit analyst would expect: removing repayment features costs ~0.02 AUC, removing bureau features ~0.006, and an application-only model drops to 0.65. Engineered behavioural and bureau features therefore add real, if modest, lift over raw application fields.

## 2. Logistic regression vs LightGBM

On discrimination the two are tied (0.6842 vs 0.6840 AUC; KS 0.276 vs 0.271). The cross-validation grid tells the story: LightGBM's best configuration was the shallowest one tried (3 leaves, 300 trees), and every deeper setting scored worse. The relationships in this data are close to additive and monotonic — income, employment tenure, age, delinquency counts and bureau DPD all move risk in one direction — which is exactly the situation where a well-regularised logistic model matches a gradient-boosted one.

The differences are elsewhere:

| Consideration | Logistic regression | LightGBM |
|---|---|---|
| Discrimination (AUC / KS / Gini) | 0.684 / 0.276 / 0.368 | 0.684 / 0.271 / 0.368 |
| Probability calibration | Not calibrated as fitted (class-weighted; Brier 0.22) — would need a Platt/isotonic step | Close to calibrated out of the box (Brier 0.13; decile table tracks the diagonal) |
| Missing data | Needs explicit imputation; "no bureau file" collapses to median values + flag | Handles NaN natively; thin-file pattern learned directly |
| Explainability | Global: standardised coefficients. Per-loan: coefficient × value, fully transparent | Per-loan SHAP values (exact for trees); global importance by gain and mean |SHAP| |
| Governance familiarity | Standard scorecard form; easiest to validate and document | Widely accepted but requires more validation effort |

## 3. Recommendation

**For the dashboard in this assignment: LightGBM.** The account view needs a probability a risk manager can read as "roughly 1-in-4" and a per-loan driver breakdown; LightGBM delivers both without an extra calibration layer, and its SHAP values are exact. Risk segments (Low / Medium / High) are cut on its score.

**For a production scorecard: start with the logistic model, keep LightGBM as a challenger.** Given no measurable accuracy advantage, the simpler model wins on validation cost, auditability and regulatory familiarity. A calibration step would be added before deployment. LightGBM should be retained as a monitored challenger; if future data (longer repayment history, time-stamped bureau pulls) introduces non-linear or interaction effects, it will show up as a widening gap in the quarterly performance comparison.

**Either way, the score is decision support, not a decision.** At the High cut-off, 36% precision means almost two of every three flagged accounts would not default; at the Medium cut-off the model catches half of defaults by flagging 30% of the book. Those trade-offs must be set by the risk team against collections capacity and tolerance for false positives, and the chosen threshold must be documented with the reasons.

## 4. What the drivers say

The top features by mean |SHAP| are annual income, years employed, the count of delinquent months in the first year, age and mean bureau DPD. This agrees with the portfolio analytics (income gradient, unemployed segment, bureau write-offs/DPD) and with the logistic coefficients (`annual_income` −0.23, `years_employed` −0.18, `employment_type_Unemployed` +0.17, `age` −0.17, `bureau_mean_dpd` +0.13 on standardised inputs). The agreement between a linear model, a tree model and simple cross-tabs is reassuring: the drivers are features of the data, not of one algorithm.

## 5. Caveats carried into the recommendation

- Age is a model input and is also a protected characteristic in many jurisdictions. It is the fourth-strongest driver. The disparity table shows younger borrowers are flagged High roughly four times as often as 41-50s — consistent with their observed default rate, but the compliance team must decide whether age may be used at all (assumption A9). A version without `age` should be validated before any production use.
- The bureau snapshot is not time-stamped (A3). The time-filtered sensitivity model loses only 0.001 AUC, which suggests the leakage exposure is small, but it cannot be ruled out from this data.
- `target_default` timing is assumed (A1-A2). If defaults can occur inside the 12-month window, repayment features partially describe the outcome rather than predict it.
