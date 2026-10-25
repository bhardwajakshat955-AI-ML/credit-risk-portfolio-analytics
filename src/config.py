"""
Central configuration for the credit-risk portfolio analytics project.

Everything that other modules need to agree on (paths, seeds, banding rules,
segment definitions, timing assumptions) lives here so that the pipeline,
the model, the dashboard and the tests all use one set of definitions.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_SEED = 42

RAW_FILES = {
    "customers": "customers.csv",
    "loan_accounts": "loan_accounts.csv",
    "bureau_credit": "bureau_credit.csv",
    "repayment_history": "repayment_history.csv",
    "collections_events": "collections_events.csv",
}

# Expected primary keys per table (validated at ingestion).
PRIMARY_KEYS = {
    "customers": ["customer_id"],
    "loan_accounts": ["account_id"],
    "bureau_credit": ["bureau_id"],
    "repayment_history": ["account_id", "payment_month"],
    "collections_events": ["collection_event_id"],
}

DATE_COLUMNS = {
    "customers": ["customer_since"],
    "loan_accounts": ["origination_date"],
    "bureau_credit": ["date_opened", "date_closed"],
    "collections_events": ["event_date"],
}

# ---------------------------------------------------------------------------
# Timing assumptions (see reports/assumptions_log.md, A1-A4)
# ---------------------------------------------------------------------------
# Repayment history holds exactly 12 monthly observations per account, with
# days_past_due never reaching 90. We therefore treat months 1-12 as the
# observation window and the end of month 12 as the prediction point.
OBSERVATION_WINDOW_MONTHS = 12

# Bureau `date_closed` values beyond this date are treated as implausible
# (they extend to 2029, years after the latest `date_opened` of 2025-12-29).
BUREAU_MAX_PLAUSIBLE_DATE = "2025-12-31"

# ---------------------------------------------------------------------------
# Segmentation / banding rules used consistently across analytics & dashboard
# ---------------------------------------------------------------------------
INCOME_BANDS = {
    "bins": [0, 40_000, 60_000, 80_000, 100_000, float("inf")],
    "labels": ["<40k", "40-60k", "60-80k", "80-100k", "100k+"],
}
AGE_BANDS = {
    "bins": [0, 30, 40, 50, 60, 200],
    "labels": ["21-30", "31-40", "41-50", "51-60", "61+"],
}

# Risk segments assigned from the model's predicted default probability.
# Cut-offs are chosen on the training set so that roughly the top decile
# of scores is "High" and the next ~two deciles are "Medium"; the actual
# numeric cut-offs are stored with the model artefacts, not hardcoded here.
RISK_SEGMENT_LABELS = ["Low", "Medium", "High"]
RISK_SEGMENT_QUANTILES = {"Medium": 0.70, "High": 0.90}

# Canonical label mappings for inconsistent categorical values.
GENDER_MAP = {"M": "Male", "F": "Female", "Male": "Male", "Female": "Female"}

PROFILING_DIMENSIONS = ["income_band", "employment_type", "region", "age_band", "loan_type"]
