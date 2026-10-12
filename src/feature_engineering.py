"""
Transforms cleaned line-item transactions into one row per customer, with
engineered behavioural features. This is the highest-leverage step in the
whole project: K-Means only ever sees what we hand it here, so feature
quality bounds cluster quality far more than the choice of algorithm does.

Feature groups:
  1. Classic RFM       -- Recency, Frequency, Monetary (industry standard
                           baseline for retail segmentation, used by every
                           real CRM/marketing platform)
  2. Order-shape        -- avg order value, avg basket size (how they buy)
  3. Behavioural rhythm -- purchase regularity, tenure (how consistently /
                           how long they've been a customer)
  4. Assortment         -- category/product diversity (breadth of interest)
"""

import logging
import numpy as np
import pandas as pd

from config import (
    COL_INVOICE, COL_STOCKCODE, COL_INVOICE_DATE,
    COL_CUSTOMER_ID, RECENCY_SNAPSHOT_BUFFER_DAYS, CUSTOMER_FEATURES_FILE,
)

logger = logging.getLogger(__name__)


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: cleaned line-item transactions (one row per product per invoice),
        must already have a 'LineTotal' column.
    Returns: one row per customer with engineered features.
    """
    snapshot_date = df[COL_INVOICE_DATE].max() + pd.Timedelta(days=RECENCY_SNAPSHOT_BUFFER_DAYS)
    logger.info(f"Using snapshot date {snapshot_date.date()} for recency calculation")

    # --- Order-level aggregation first (needed for basket size / regularity) ---
    order_level = (
        df.groupby([COL_CUSTOMER_ID, COL_INVOICE])
        .agg(order_value=("LineTotal", "sum"),
             order_date=(COL_INVOICE_DATE, "min"),
             basket_size=(COL_STOCKCODE, "nunique"))
        .reset_index()
    )

    features = order_level.groupby(COL_CUSTOMER_ID).agg(
        frequency=("Invoice", "nunique"),
        monetary_total=("order_value", "sum"),
        avg_order_value=("order_value", "mean"),
        avg_basket_size=("basket_size", "mean"),
        first_purchase=("order_date", "min"),
        last_purchase=("order_date", "max"),
    ).reset_index()

    # --- Recency ---
    features["recency_days"] = (snapshot_date - features["last_purchase"]).dt.days

    # --- Tenure: how long they've been a customer (in days) ---
    features["tenure_days"] = (features["last_purchase"] - features["first_purchase"]).dt.days
    features["tenure_days"] = features["tenure_days"].clip(lower=1)  # avoid div-by-zero downstream

    # --- Purchase regularity: avg days between orders (lower = more frequent/predictable) ---
    # For single-order customers this is genuinely undefined -- there is no
    # observed gap between orders. A naive fallback to their own (near-zero)
    # tenure_days would wrongly imply they buy extremely often (tenure=0 for
    # a one-off order made and never repeated within the data window), which
    # then blows up any downstream rate-based estimate like CLV. Instead we
    # impute with the FULL observation window of the dataset: the honest
    # statement is "we have zero evidence of a repeat cadence, so treat their
    # implied purchase rate as being at most one order across the whole
    # window we observed", which is the conservative, defensible choice.
    observation_window_days = (df[COL_INVOICE_DATE].max() - df[COL_INVOICE_DATE].min()).days
    observation_window_days = max(observation_window_days, 1)

    features["purchase_regularity"] = np.where(
        features["frequency"] > 1,
        features["tenure_days"] / (features["frequency"] - 1),
        observation_window_days,
    )

    # --- Category diversity: distinct products purchased across all orders ---
    category_diversity = (
        df.groupby(COL_CUSTOMER_ID)[COL_STOCKCODE].nunique()
        .rename("category_diversity")
        .reset_index()
    )
    features = features.merge(category_diversity, on=COL_CUSTOMER_ID, how="left")

    features = features.drop(columns=["first_purchase", "last_purchase"])
    features = features.rename(columns={COL_CUSTOMER_ID: "customer_id"})

    logger.info(f"Built feature table for {len(features):,} customers, "
                f"{features.shape[1]-1} features")
    return features


if __name__ == "__main__":
    from config import CLEANED_TRANSACTIONS_FILE
    df = pd.read_csv(CLEANED_TRANSACTIONS_FILE, parse_dates=[COL_INVOICE_DATE])
    feats = build_customer_features(df)
    CUSTOMER_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(CUSTOMER_FEATURES_FILE, index=False)
    print(feats.describe().T)
    print(f"\nSaved to {CUSTOMER_FEATURES_FILE}")
