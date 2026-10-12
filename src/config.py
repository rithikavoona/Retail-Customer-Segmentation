"""
Central configuration for the Retail Customer Segmentation project.
Keeping every path, threshold, and hyperparameter here means the rest of the
codebase never hardcodes a "magic number" -- change behaviour in one place.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"
DASHBOARD_DIR = ROOT_DIR / "dashboard"

RAW_TRANSACTIONS_FILE = (
    DATA_RAW_DIR / "online_retail_real.csv"
    if (DATA_RAW_DIR / "online_retail_real.csv").exists()
    else DATA_RAW_DIR / "online_retail_ii.csv"
)
CLEANED_TRANSACTIONS_FILE = DATA_PROCESSED_DIR / "cleaned_transactions.csv"
CUSTOMER_FEATURES_FILE = DATA_PROCESSED_DIR / "customer_features.csv"
CLUSTER_ASSIGNMENTS_FILE = DATA_PROCESSED_DIR / "cluster_assignments.csv"
SEGMENT_SUMMARY_JSON = DASHBOARD_DIR / "segment_summary.json"

# ---------------------------------------------------------------------------
# Column names expected in the raw Online Retail II export
# (matches the UCI schema: https://archive.ics.uci.edu/dataset/502)
# ---------------------------------------------------------------------------
COL_INVOICE = "Invoice"
COL_STOCKCODE = "StockCode"
COL_DESCRIPTION = "Description"
COL_QUANTITY = "Quantity"
COL_INVOICE_DATE = "InvoiceDate"
COL_UNIT_PRICE = "Price"
COL_CUSTOMER_ID = "Customer ID"
COL_COUNTRY = "Country"

# ---------------------------------------------------------------------------
# Data cleaning thresholds
# ---------------------------------------------------------------------------
MIN_VALID_QUANTITY = 1          # transactions must be positive quantity (sales, not returns)
MIN_VALID_PRICE = 0.01          # zero/negative price rows are data errors or free promo items
OUTLIER_QUANTITY_PERCENTILE = 0.995   # clip extreme bulk-order outliers at this percentile
OUTLIER_PRICE_PERCENTILE = 0.995
MIN_TRANSACTIONS_PER_CUSTOMER = 1     # customers need at least this many valid orders to be segmented

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
# The "snapshot date" for Recency is one day after the last transaction in the dataset
RECENCY_SNAPSHOT_BUFFER_DAYS = 1

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
K_SEARCH_RANGE = range(2, 11)     # candidate k values to evaluate for K-Means
KMEANS_N_INIT = 20                # number of centroid initializations (avoid local minima)
FINAL_K = None                    # if None, chosen automatically (see MIN_BUSINESS_K note)

# The statistically "optimal" k (max silhouette) sometimes lands on a coarse
# split (e.g. k=2) that is technically well-separated but useless for
# differentiated marketing -- you can't run 6 campaigns off 2 segments.
# We therefore select the best-silhouette k within a business-usable range,
# and report this trade-off transparently rather than silently overriding it.
MIN_BUSINESS_K = 3
MAX_BUSINESS_K = 8

# Features used for clustering (after engineering) -- kept explicit and named
# rather than "all numeric columns" so the model's inputs are auditable.
CLUSTERING_FEATURES = [
    "recency_days",
    "frequency",
    "monetary_total",
    "avg_order_value",
    "avg_basket_size",
    "purchase_regularity",
    "category_diversity",
    "tenure_days",
]

# ---------------------------------------------------------------------------
# Business layer
# ---------------------------------------------------------------------------
# Simple CLV proxy: expected future value over a fixed horizon.
# CLV = avg_order_value * purchase_frequency_per_month * horizon_months * gross_margin
CLV_HORIZON_MONTHS = 12
ASSUMED_GROSS_MARGIN = 0.30
