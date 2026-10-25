"""
Ingestion: load the five raw tables and validate their keys and relationships.

The raw files are read-only inputs. Nothing here mutates them; all derived
data is written to data/processed by build_dataset.py.
"""
from __future__ import annotations

import pandas as pd

from src.config import DATE_COLUMNS, PRIMARY_KEYS, RAW_DIR, RAW_FILES


def load_raw_tables(raw_dir=RAW_DIR) -> dict[str, pd.DataFrame]:
    """Read all five CSVs. Date columns are parsed with errors='coerce' so an
    unparseable date becomes NaT instead of failing the load; the count of
    coerced values is checked in `validate_tables`."""
    tables = {}
    for name, filename in RAW_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required source file missing: {path}")
        df = pd.read_csv(path)
        for col in DATE_COLUMNS.get(name, []):
            raw_non_null = df[col].notna().sum()
            df[col] = pd.to_datetime(df[col], errors="coerce", format="%Y-%m-%d")
            coerced = raw_non_null - df[col].notna().sum()
            if coerced:
                # Keep going but make it visible; the DQ report records it.
                df.attrs[f"{col}_unparseable"] = int(coerced)
        tables[name] = df
    return tables


def validate_tables(tables: dict[str, pd.DataFrame]) -> dict:
    """Check primary-key uniqueness and foreign-key integrity.

    Returns a dictionary of findings that feeds the data quality report and
    the tests. Raises only for failures that would make a join unsafe.
    """
    findings = {"row_counts": {k: int(len(v)) for k, v in tables.items()}, "primary_keys": {}, "foreign_keys": {}}

    for name, keys in PRIMARY_KEYS.items():
        dup = int(tables[name].duplicated(subset=keys).sum())
        findings["primary_keys"][name] = {"keys": keys, "duplicate_key_rows": dup}
        if dup:
            raise ValueError(f"{name}: {dup} duplicate rows on primary key {keys}; joins would be unsafe")

    cust = tables["customers"]
    loans = tables["loan_accounts"]
    fk_checks = {
        "loan_accounts.customer_id -> customers": (loans["customer_id"], cust["customer_id"]),
        "bureau_credit.customer_id -> customers": (tables["bureau_credit"]["customer_id"], cust["customer_id"]),
        "repayment_history.account_id -> loan_accounts": (tables["repayment_history"]["account_id"], loans["account_id"]),
        "collections_events.account_id -> loan_accounts": (tables["collections_events"]["account_id"], loans["account_id"]),
    }
    for label, (child, parent) in fk_checks.items():
        orphans = int((~child.isin(parent)).sum())
        findings["foreign_keys"][label] = {"orphan_rows": orphans}
        if orphans:
            raise ValueError(f"{label}: {orphans} child rows with no parent; cannot join safely")

    # Coverage in the other direction (parents with no children) is not an
    # error, but it matters for feature engineering (e.g. 'no bureau file').
    findings["coverage"] = {
        "customers_without_loans": int((~cust["customer_id"].isin(loans["customer_id"])).sum()),
        "customers_without_bureau": int((~cust["customer_id"].isin(tables["bureau_credit"]["customer_id"])).sum()),
        "loans_without_repayment": int((~loans["account_id"].isin(tables["repayment_history"]["account_id"])).sum()),
        "loans_with_collections": int(loans["account_id"].isin(tables["collections_events"]["account_id"]).sum()),
    }
    findings["repayment_months_per_account"] = (
        tables["repayment_history"].groupby("account_id").size().value_counts().to_dict()
    )
    findings["loans_per_customer"] = loans.groupby("customer_id").size().value_counts().sort_index().to_dict()
    return findings
