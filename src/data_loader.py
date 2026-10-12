"""
Loads raw transaction data and performs schema validation.
Separating "loading" from "cleaning" (preprocessing.py) keeps each function
single-purpose and testable -- a core software-engineering practice that's
often skipped in ML notebooks but matters once a pipeline goes to production.
"""

import logging
import pandas as pd

from config import (
    RAW_TRANSACTIONS_FILE, COL_INVOICE, COL_STOCKCODE, COL_DESCRIPTION,
    COL_QUANTITY, COL_INVOICE_DATE, COL_UNIT_PRICE, COL_CUSTOMER_ID, COL_COUNTRY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    COL_INVOICE, COL_STOCKCODE, COL_DESCRIPTION, COL_QUANTITY,
    COL_INVOICE_DATE, COL_UNIT_PRICE, COL_CUSTOMER_ID, COL_COUNTRY,
]


class SchemaValidationError(Exception):
    """Raised when the input file doesn't match the expected transaction schema."""


def load_raw_transactions(path=RAW_TRANSACTIONS_FILE) -> pd.DataFrame:
    """
    Loads the raw transaction file (csv or xlsx) and validates its schema.

    If you're using the REAL Online Retail II file from UCI, it ships as
    .xlsx with two sheets (Year 2009-2010, Year 2010-2011). See README.md
    for the one-time conversion step to a single csv with these column names.
    """
    path = str(path)
    logger.info(f"Loading raw transactions from {path}")

    if path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # Normalize common column aliases (e.g. InvoiceNo vs Invoice, UnitPrice vs Price, CustomerID vs Customer ID)
    alias_map = {
        "invoiceno": COL_INVOICE,
        "invoice_no": COL_INVOICE,
        "invoice": COL_INVOICE,
        "stockcode": COL_STOCKCODE,
        "stock_code": COL_STOCKCODE,
        "description": COL_DESCRIPTION,
        "quantity": COL_QUANTITY,
        "qty": COL_QUANTITY,
        "invoicedate": COL_INVOICE_DATE,
        "invoice_date": COL_INVOICE_DATE,
        "unitprice": COL_UNIT_PRICE,
        "unit_price": COL_UNIT_PRICE,
        "price": COL_UNIT_PRICE,
        "customerid": COL_CUSTOMER_ID,
        "customer_id": COL_CUSTOMER_ID,
        "country": COL_COUNTRY,
    }
    rename_dict = {}
    for col in df.columns:
        norm_key = str(col).strip().lower()
        if norm_key in alias_map and col != alias_map[norm_key]:
            rename_dict[col] = alias_map[norm_key]
    if rename_dict:
        df = df.rename(columns=rename_dict)
        logger.info(f"Normalized column aliases: {rename_dict}")

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise SchemaValidationError(
            f"Input file is missing required columns: {missing_cols}. "
            f"Expected schema: {REQUIRED_COLUMNS}"
        )

    df[COL_INVOICE_DATE] = pd.to_datetime(df[COL_INVOICE_DATE])
    logger.info(f"Loaded {len(df):,} raw transaction rows, "
                f"{df[COL_CUSTOMER_ID].nunique():,} unique (non-null) customer IDs")
    return df


if __name__ == "__main__":
    df = load_raw_transactions()
    print(df.info())
