# Assumptions Log

Every assumption or approximation made in this project, why it was needed, where it bites, how it could be validated, and what a real bank's IT, data, risk and compliance teams would need to provide before any of this goes to production. Unresolved questions are listed separately at the end rather than silently assumed.

## A. Assumptions made

| # | Assumption / approximation | Why it was necessary | Where it affects the work | Possible impact | How to validate |
|---|---|---|---|---|---|
| A1 | **Observation window = repayment months 1-12; prediction point = end of month 12.** Every account has exactly 12 monthly rows and no DPD reaches 90, so the window is treated as a pre-default behavioural period. | The repayment table has no calendar dates and no stated as-of date. | Feature engineering (`rep_*`), model framing (behavioural scorecard), dashboard wording ("estimate of default after the window"). | If defaults could occur inside the window, repayment features partly describe the outcome and the AUC is optimistic. | Confirm with the data owner how `target_default` is defined and when it is observed relative to origination. |
| A2 | **`target_default` is a loan-level outcome observed after the window, taken as supplied.** | No definition (90+ DPD? write-off? at what horizon?) is given. | Everything downstream: default rates, model, risk segments. | Different definitions (e.g. 30 vs 90 DPD, 12 vs 24-month horizon) would change the base rate and the drivers. | Obtain the bank's default definition (ideally the regulatory one) and the performance horizon. |
| A3 | **Bureau data is a current snapshot available at the prediction point.** Lines opened 2018-2025 are independent of loan origination dates; ~57% open after the customer's latest loan. | The bureau table has no as-of or reporting date; point-in-time reconstruction is impossible. | `bureau_*` model features, R4 anomaly rule, insight 4. | Possible look-ahead leakage. The time-filtered sensitivity model (lines opened ≤ prediction point) loses only 0.001 AUC, so the exposure appears small, but status/DPD values still come from the snapshot. | Source bureau pulls with a report date and re-build features as of each loan's prediction point. |
| A4 | **Collections events are excluded from the model** and used only descriptively. | 45% of events are dated before the loan was originated, and collections activity is a consequence of delinquency (post-decision information). | Model feature set; account view shows collections with a data-quality warning. | The model forgoes any genuine early-collections signal. | Fix the event-date feed; if dates are trustworthy, test "events within the observation window" as a feature. |
| A5 | **Bureau `date_closed` after 2025-12-31 is implausible** and is nulled (3,149 rows); the status label is kept. | No line was opened after 2025-12-29, so closures in 2026-2029 cannot have been observed. | No closure-date features are built. | Lose "time since closure" information. | Confirm the extraction date of the bureau file. |
| A6 | **Exposure = originated `loan_amount`.** | No field for current outstanding on the loan table; the repayment table gives month-12 balance only (`rep_balance_m12`), and credit-card "loan_amount" is presumably a limit. | All exposure shares, exposure-weighted default rates, dashboard KPIs. | Mortgage share of exposure (52%) is on originated amounts; EAD would differ. | Obtain outstanding balance / EAD as at a reporting date; confirm the meaning of `loan_amount` for credit cards. |
| A7 | **Income is annual, in the same currency as loan amounts (USD per dictionary), and is self-reported at application.** | Dictionary gives no further definition. | Income bands, loan-to-income, EMI-to-income, insights 2-3. | Band boundaries (<40k, 40-60k, …) are judgement calls; verified vs declared income could differ materially. | Confirm income definition, verification status and currency. |
| A8 | **Risk segment cut-offs at the 70th and 90th percentile of training-set LightGBM scores** (Medium ≥ 0.180, High ≥ 0.278). | The assignment requires a documented segmentation rule; no bank policy was supplied. | Dashboard risk segments, anomaly context, confusion matrices. | Thresholds are illustrative; business thresholds should reflect collections capacity and loss appetite. | Risk team sets thresholds by expected-loss and capacity analysis; document the rationale. |
| A9 | **Age is used as a model feature; gender is not.** Age carries clear signal in the data. | Decision needed on protected characteristics without a bank policy. | Model (4th-strongest driver), disparity check in the model report. | Use of age may be prohibited or restricted by law/policy in the deployment jurisdiction. | Compliance/legal review; validate an age-free model variant before production. |
| A10 | **Customer-grouped random split (80/20) is the primary validation; the 2023-vintage out-of-time test is secondary.** | Only origination dates are available for a time split and all vintages have the same 12-month window. | Reported metrics. | Both splits agree (AUC 0.68-0.70), which supports the result, but neither replicates a true deployment gap. | Validate on a later, fully matured vintage once available. |
| A11 | **Anomaly thresholds (R1-R4) are set by judgement** so that ~10% of accounts carry a flag. | No labelled anomaly / fraud outcomes exist in the data. | Anomaly feature volumes and the review queue. | Thresholds may be too loose or tight for the bank's analyst capacity. | Tune on reviewed outcomes after a pilot; track confirmed/dismissed rates per rule. |
| A12 | **"Unemployed" customers' positive income is kept as supplied.** | No definition of income for this group. | Insight 3, income × employment cut. | If income is household or benefit income, affordability ratios for this group are misleading. | See Q2. |
| A13 | **Relationship tenure (`customer_since`) is not used.** | Half of loans are originated before the customer relationship "began". | One potential feature dropped. | Minor loss of signal. | See Q4. |
| A14 | **The dataset is synthetic** (per `DATA_DICTIONARY.md`); all findings describe this dataset only. | Stated by the data provider. | Every insight and metric. | Nothing here transfers to a real book without re-estimation. | n/a |

## B. Open questions (not assumed)

| # | Question | Who can answer |
|---|---|---|
| Q1 | What is the precise definition and observation horizon of `target_default`? | Risk / data owner |
| Q2 | What does `annual_income` represent for customers with `employment_type = Unemployed` (household income, benefits, rental)? | Data owner / credit policy |
| Q3 | What is the as-of date of the bureau snapshot, and are historical bureau pulls available? | Data / bureau integration team |
| Q4 | Which of `customer_since` and `origination_date` is wrong for the 7,293 inconsistent loans? | Data owner |
| Q5 | Why are 45% of collections events dated before origination — a feed/time-zone defect or a different date semantics? | Collections systems team |
| Q6 | For credit cards, is `loan_amount` the limit or the drawn balance, and is `tenure_months` meaningful? | Product / data owner |
| Q7 | May age be used as a model input in the target jurisdiction? | Compliance / legal |

## C. What would be needed from a real bank before production

**IT / data engineering**
- Authoritative, time-stamped extracts of the five tables (plus loan outstanding balance / EAD) with an agreed as-of date and a documented refresh schedule.
- Point-in-time bureau data (report date per pull) and a corrected collections event feed.
- A feature store or scheduled job to recompute the aggregated features; a model-serving endpoint or batch scorer; secure storage of model artefacts with versioning.
- Access controls: the dashboard shows account-level data and must sit behind the bank's SSO with role-based access (analyst vs manager), audit logging of who viewed what, and no personal identifiers beyond what the role needs.
- Secrets management for any future LLM or external API integration (none is used here; no credentials exist in the repository).

**Risk / model governance**
- The bank's default definition, loss data (LGD/EAD) to move from PD ranking to expected-loss ranking, and a policy on threshold setting.
- Independent model validation, challenger process, PSI/CSI monitoring with agreed alert thresholds, and a documented retraining and rollback procedure (proposed design in `model_evaluation.md`).
- Sign-off on the anomaly rules, the human-review workflow and the case-management system that would replace the session-level review queue.

**Compliance / legal**
- Ruling on protected characteristics (age; and any proxies for gender, ethnicity or marital status) in the model.
- Data-protection assessment for the dashboard (purpose limitation, retention, access) and an explainability standard for adverse decisions.
- Confirmation that model-based outputs are used as decision support with human review, and that the audit trail (score, drivers, reviewer decision) is retained.

**Nothing in this repository has been assessed for regulatory compliance; the items above are prerequisites, not achievements.**
