# Feature Dictionary

Generated from `src/features.py`. Grain states the level at which the feature is computed before joining to the loan table.

| Feature | Grain | Definition | Rationale |
|---|---|---|---|
| `bureau_n_lines` | customer | count of external credit lines | breadth of credit history |
| `bureau_n_active` | customer | count with account_status = Active | current external obligations |
| `bureau_n_written_off` | customer | count with account_status = Written-Off | prior defaults elsewhere — strongest bureau signal in the data |
| `bureau_written_off_share` | customer | n_written_off / n_lines | write-offs relative to history length |
| `bureau_total_sanctioned` | customer | sum of sanctioned_amount over all lines | total external credit ever granted |
| `bureau_total_outstanding` | customer | sum of outstanding_amount | external indebtedness (incl. written-off balances) |
| `bureau_active_outstanding` | customer | sum of outstanding_amount on Active lines | live external debt |
| `bureau_utilisation` | customer | active_outstanding / active_sanctioned (NaN if no active line) | how heavily external limits are drawn |
| `bureau_max_dpd` | customer | max days_past_due over lines that report DPD | worst current external delinquency |
| `bureau_mean_dpd` | customer | mean days_past_due over lines that report DPD | typical external delinquency |
| `bureau_lines_dpd_30plus` | customer | count of lines with DPD ≥ 30 | number of seriously delinquent external lines |
| `bureau_dpd_unreported_lines` | customer | count of lines with missing DPD | carries the information that DPD was missing instead of imputing it |
| `bureau_share_fintech_nbfc` | customer | share of lines from Fintech or NBFC lenders | lender-mix proxy for credit-seeking behaviour |
| `rep_max_dpd` | loan account | max days_past_due over months 1-12 | worst delinquency in the observation window |
| `rep_mean_dpd` | loan account | mean days_past_due over months 1-12 | average delinquency level |
| `rep_n_missed` | loan account | count of months with payment_status = Missed (emi_paid = 0) | missed instalments are the strongest behavioural signal |
| `rep_n_late` | loan account | count of months with payment_status = Late | minor delinquency frequency |
| `rep_n_delinquent` | loan account | n_missed + n_late | any-delinquency frequency |
| `rep_paid_ratio` | loan account | sum(emi_paid) / sum(emi_due) | repayment completeness |
| `rep_missed_last3` | loan account | missed months within months 10-12 | recent missed payments weigh more than old ones |
| `rep_max_dpd_last3` | loan account | max DPD in months 10-12 | recent worst delinquency |
| `rep_dpd_trend` | loan account | mean DPD months 10-12 minus mean DPD months 1-9 | is the account deteriorating or recovering? |
| `rep_months_since_last_missed` | loan account | 12 − last missed month (13 if never missed) | recency of the last missed payment |
| `rep_max_consecutive_delinquent` | loan account | longest run of consecutive months with DPD > 0 | persistence of delinquency |
| `rep_balance_m12` | loan account | outstanding_balance at month 12 | current exposure at the prediction point |
| `rep_emi_due` | loan account | monthly instalment amount | used for the EMI-to-income ratio |
| `col_n_events` | loan account (analytics only, NOT a model input) | count of collections actions on the account | collections is post-decision information; see A4 |
| `col_n_legal_or_writeoff` | loan account (analytics only, NOT a model input) | count of Legal Notice or Written-Off Referral actions | collections is post-decision information; see A4 |
| `col_total_recovered` | loan account (analytics only, NOT a model input) | sum of amount_recovered | collections is post-decision information; see A4 |
| `col_n_events_before_orig` | loan account (analytics only, NOT a model input) | events dated before origination (data quality flag) | collections is post-decision information; see A4 |
| `emi_to_income` | loan account | annualised EMI / annual_income | affordability |
| `loan_to_income` | loan account | loan_amount / annual_income | leverage relative to earnings |
| `customer_n_loans` | loan account | number of loan accounts the customer holds in this portfolio | internal multi-borrowing |
| `customer_total_exposure` | loan account | sum of loan_amount across the customer's accounts | internal concentration |
| `has_bureau_file` | loan account | 1 if the customer has any bureau line | thin-file indicator; bureau features are NaN otherwise |

Raw application fields used directly by the model: `loan_type`, `loan_purpose`, `loan_amount`, `interest_rate_pct`, `tenure_months`, `disbursal_channel`, `collateral_flag`, `age`, `marital_status`, `education_level`, `employment_type`, `occupation_sector`, `years_employed`, `annual_income`, `region`, `city_tier`, `num_dependents`, `housing_type`, `years_at_residence` plus the `*_missing` flags.

Deliberately excluded from the model: `gender` (fairness — used only for the post-hoc disparity check), `customer_since` (inconsistent with origination dates), all `col_*` collections features (post-decision information with unreliable dates), `rep_balance_m1`/`rep_total_*` (redundant with loan_amount and paid_ratio), and identifiers.