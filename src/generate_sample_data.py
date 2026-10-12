"""
Generates a schema-accurate SYNTHETIC sample that mimics the real
Online Retail II dataset (same columns, same messiness: cancellations,
missing Customer IDs, returns, free promo items).

This exists ONLY so the pipeline can be built, tested, and demoed without
network access. For the actual submission, download the real file from:
https://archive.ics.uci.edu/dataset/502/online+retail+ii
and place it at data/raw/online_retail_ii.csv (see README for the exact
column-mapping step). The rest of the pipeline does not change.
"""

import numpy as np
import pandas as pd

from config import RAW_TRANSACTIONS_FILE, DATA_RAW_DIR

RNG = np.random.default_rng(42)

N_CUSTOMERS = 1500
COUNTRIES = ["United Kingdom", "France", "Germany", "EIRE", "Spain",
             "Netherlands", "Belgium", "Portugal", "Australia"]
COUNTRY_WEIGHTS = [0.82, 0.04, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01]

PRODUCTS = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
    ("71053", "WHITE METAL LANTERN", 3.39),
    ("84406B", "CREAM CUPID HEARTS COAT HANGER", 2.75),
    ("21730", "GLASS STAR FROSTED T-LIGHT HOLDER", 4.25),
    ("22423", "REGENCY CAKESTAND 3 TIER", 12.75),
    ("21754", "HOME BUILDING BLOCK WORD", 5.95),
    ("22960", "JAM MAKING SET WITH JARS", 4.25),
    ("21212", "PACK OF 72 RETROSPOT CAKE CASES", 0.55),
    ("22138", "BAKING SET 9 PIECE RETROSPOT", 4.95),
    ("20725", "LUNCH BAG RED RETROSPOT", 1.65),
    ("22383", "LUNCH BAG SUKI DESIGN", 1.65),
    ("47566", "PARTY BUNTING", 4.95),
    ("84879", "ASSORTED COLOUR BIRD ORNAMENT", 1.69),
    ("21931", "JUMBO STORAGE BAG SUKI", 2.08),
    ("22086", "PAPER CHAIN KIT 50'S CHRISTMAS", 2.55),
]


def _customer_archetype(rng):
    """Assign each synthetic customer a hidden 'true' behavioural archetype
    so the generated data actually contains separable clusters -- otherwise
    testing a clustering pipeline on pure noise proves nothing."""
    return rng.choice(
        ["champion", "loyal", "at_risk", "new", "dormant", "bargain"],
        p=[0.10, 0.20, 0.15, 0.20, 0.20, 0.15],
    )


def generate(n_customers=N_CUSTOMERS, start_date="2010-12-01", end_date="2011-12-09"):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    total_days = (end - start).days

    rows = []
    invoice_counter = 536365

    for cust_idx in range(n_customers):
        customer_id = 12000 + cust_idx
        archetype = _customer_archetype(RNG)
        country = RNG.choice(COUNTRIES, p=COUNTRY_WEIGHTS)

        # Archetype -> behavioural parameters (this is the "ground truth"
        # signal the clustering algorithm should be able to recover)
        if archetype == "champion":
            n_orders = RNG.integers(15, 30)
            recency_bias = 0.9   # orders skew recent
            price_mult = RNG.uniform(1.1, 1.6)
        elif archetype == "loyal":
            n_orders = RNG.integers(8, 15)
            recency_bias = 0.7
            price_mult = RNG.uniform(0.9, 1.2)
        elif archetype == "at_risk":
            n_orders = RNG.integers(5, 12)
            recency_bias = 0.15   # orders skew old -> high recency_days now
            price_mult = RNG.uniform(0.9, 1.3)
        elif archetype == "new":
            n_orders = RNG.integers(1, 3)
            recency_bias = 0.95
            price_mult = RNG.uniform(0.8, 1.1)
        elif archetype == "dormant":
            n_orders = RNG.integers(1, 4)
            recency_bias = 0.05
            price_mult = RNG.uniform(0.7, 1.0)
        else:  # bargain hunters: frequent but low value
            n_orders = RNG.integers(6, 14)
            recency_bias = 0.5
            price_mult = RNG.uniform(0.4, 0.7)

        order_days = np.sort(
            start.value / 1e9 + (RNG.beta(2 * recency_bias + 0.5, 2 * (1 - recency_bias) + 0.5, n_orders) * total_days * 86400)
        )

        for od in order_days:
            invoice_counter += 1
            invoice_no = str(invoice_counter)
            order_ts = pd.Timestamp(od, unit="s")
            n_items = RNG.integers(1, 8)
            is_cancelled = RNG.random() < 0.02  # ~2% of orders are cancellations

            for _ in range(n_items):
                code, desc, base_price = PRODUCTS[RNG.integers(0, len(PRODUCTS))]
                qty = int(RNG.integers(1, 12))
                price = round(base_price * price_mult * RNG.uniform(0.9, 1.1), 2)

                rows.append({
                    "Invoice": ("C" + invoice_no) if is_cancelled else invoice_no,
                    "StockCode": code,
                    "Description": desc,
                    "Quantity": -qty if is_cancelled else qty,
                    "InvoiceDate": order_ts,
                    "Price": price,
                    "Customer ID": customer_id,
                    "Country": country,
                })

    # Inject realistic messiness: some rows with missing Customer ID
    # (guest checkouts) and a few corrupt price/quantity rows
    n_missing = int(0.08 * len(rows))
    missing_idx = RNG.choice(len(rows), size=n_missing, replace=False)
    for i in missing_idx:
        rows[i]["Customer ID"] = np.nan

    n_bad = int(0.005 * len(rows))
    bad_idx = RNG.choice(len(rows), size=n_bad, replace=False)
    for i in bad_idx:
        rows[i]["Price"] = 0.0

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)  # shuffle
    return df


if __name__ == "__main__":
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(RAW_TRANSACTIONS_FILE, index=False)
    print(f"Synthetic sample written: {RAW_TRANSACTIONS_FILE}")
    print(f"Rows: {len(df):,} | Unique customers: {df['Customer ID'].nunique():,}")
    print(df.head())
