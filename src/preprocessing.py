"""
Cleans raw transaction data. Every cleaning decision is logged with the row
count it affected, so the cleaning process is auditable -- a reviewer (or
your professor) can see exactly what was removed and why, rather than
trusting a black-box "df.dropna()".
"""

import logging
import pandas as pd

from config import (
    COL_INVOICE, COL_QUANTITY, COL_UNIT_PRICE, COL_CUSTOMER_ID, COL_DESCRIPTION,
    MIN_VALID_QUANTITY, MIN_VALID_PRICE,
    OUTLIER_QUANTITY_PERCENTILE, OUTLIER_PRICE_PERCENTILE,
    CLEANED_TRANSACTIONS_FILE,
)

logger = logging.getLogger(__name__)


def _log_step(df_before: pd.DataFrame, df_after: pd.DataFrame, step_name: str):
    removed = len(df_before) - len(df_after)
    pct = removed / len(df_before) * 100 if len(df_before) else 0
    logger.info(f"[{step_name}] removed {removed:,} rows ({pct:.2f}%) "
                f"-> {len(df_after):,} rows remain")


def remove_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """Invoice numbers prefixed with 'C' are cancellations in the Online
    Retail schema. We drop them rather than net them against the original
    order: for segmentation purposes we care about realised purchase
    behaviour, and cancelled orders introduce negative-quantity noise that
    distorts monetary/frequency features."""
    before = df
    mask = ~df[COL_INVOICE].astype(str).str.startswith("C")
    after = df[mask].copy()
    _log_step(before, after, "remove_cancellations")
    return after


def remove_invalid_quantities_and_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Drops non-positive quantities (returns/adjustments that slipped through)
    and non-positive prices (data entry errors, free promotional items that
    would zero out monetary calculations)."""
    before = df
    after = df[(df[COL_QUANTITY] >= MIN_VALID_QUANTITY) &
               (df[COL_UNIT_PRICE] >= MIN_VALID_PRICE)].copy()
    _log_step(before, after, "remove_invalid_quantities_and_prices")
    return after


def remove_missing_customer_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Rows without a Customer ID (guest checkouts) can't be attributed to a
    customer and are unusable for customer-level segmentation, so they're
    dropped here rather than imputed -- imputing an identity is not
    meaningful."""
    before = df
    after = df.dropna(subset=[COL_CUSTOMER_ID]).copy()
    after[COL_CUSTOMER_ID] = after[COL_CUSTOMER_ID].astype(int)
    _log_step(before, after, "remove_missing_customer_ids")
    return after


def remove_non_product_stockcodes(df: pd.DataFrame) -> pd.DataFrame:
    """The real dataset contains administrative pseudo-products (e.g.
    'POSTAGE', 'BANK CHARGES', 'DOTCOM POSTAGE', 'Manual', 'Adjust bad debt')
    that aren't genuine purchases and would distort basket/category features."""
    before = df
    non_product_keywords = ["POSTAGE", "BANK CHARGES", "MANUAL", "ADJUST",
                             "CARRIAGE", "DOTCOM", "SAMPLES", "AMAZON FEE"]
    pattern = "|".join(non_product_keywords)
    mask = ~df[COL_DESCRIPTION].astype(str).str.upper().str.contains(pattern, na=False)
    after = df[mask].copy()
    _log_step(before, after, "remove_non_product_stockcodes")
    return after


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Caps (winsorizes) extreme bulk-order quantities and prices at the
    99.5th percentile rather than deleting them -- these are usually genuine
    wholesale-style bulk buyers, not errors, but left uncapped they would let
    a single transaction dominate a customer's monetary/frequency features
    and destabilise K-Means centroids (which are sensitive to outliers)."""
    q_cap = df[COL_QUANTITY].quantile(OUTLIER_QUANTITY_PERCENTILE)
    p_cap = df[COL_UNIT_PRICE].quantile(OUTLIER_PRICE_PERCENTILE)
    df = df.copy()
    n_capped_q = (df[COL_QUANTITY] > q_cap).sum()
    n_capped_p = (df[COL_UNIT_PRICE] > p_cap).sum()
    df[COL_QUANTITY] = df[COL_QUANTITY].clip(upper=q_cap)
    df[COL_UNIT_PRICE] = df[COL_UNIT_PRICE].clip(upper=p_cap)
    logger.info(f"[cap_outliers] capped {n_capped_q:,} quantity outliers at {q_cap:.1f}, "
                f"{n_capped_p:,} price outliers at {p_cap:.2f}")
    return df


def add_line_total(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LineTotal"] = df[COL_QUANTITY] * df[COL_UNIT_PRICE]
    return df


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Runs the full cleaning sequence in order. Order matters: cancellations
    and non-product rows are removed before outlier capping so the outlier
    thresholds are computed only on genuine sales data."""
    logger.info(f"Starting cleaning pipeline on {len(df):,} raw rows")
    df = remove_cancellations(df)
    df = remove_invalid_quantities_and_prices(df)
    df = remove_non_product_stockcodes(df)
    df = remove_missing_customer_ids(df)
    df = cap_outliers(df)
    df = add_line_total(df)
    logger.info(f"Cleaning complete: {len(df):,} clean rows, "
                f"{df[COL_CUSTOMER_ID].nunique():,} unique customers")
    return df


if __name__ == "__main__":
    from data_loader import load_raw_transactions
    raw = load_raw_transactions()
    clean = clean_pipeline(raw)
    CLEANED_TRANSACTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEANED_TRANSACTIONS_FILE, index=False)
    print(f"Saved cleaned transactions to {CLEANED_TRANSACTIONS_FILE}")
