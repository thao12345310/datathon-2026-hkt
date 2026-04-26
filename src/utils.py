"""
Datathon 2026 — Utility functions for data loading, merging, and common operations.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
RANDOM_SEED = 42


def load_all_data(data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Load all 14 CSV files and return as a dictionary of DataFrames."""
    d = Path(data_dir) if data_dir else DATA_DIR

    data = {
        # Master
        'products':   pd.read_csv(d / 'products.csv'),
        'customers':  pd.read_csv(d / 'customers.csv', parse_dates=['signup_date']),
        'geography':  pd.read_csv(d / 'geography.csv'),
        'promotions': pd.read_csv(d / 'promotions.csv', parse_dates=['start_date', 'end_date']),
        # Transaction
        'orders':      pd.read_csv(d / 'orders.csv', parse_dates=['order_date']),
        'order_items': pd.read_csv(d / 'order_items.csv'),
        'payments':    pd.read_csv(d / 'payments.csv'),
        'shipments':   pd.read_csv(d / 'shipments.csv', parse_dates=['ship_date', 'delivery_date']),
        'returns':     pd.read_csv(d / 'returns.csv', parse_dates=['return_date']),
        'reviews':     pd.read_csv(d / 'reviews.csv', parse_dates=['review_date']),
        # Analytical
        'sales_train': pd.read_csv(d / 'sales.csv', parse_dates=['Date']),
        'sample_sub':  pd.read_csv(d / 'sample_submission.csv', parse_dates=['Date']),
        # Operational
        'inventory':   pd.read_csv(d / 'inventory.csv', parse_dates=['snapshot_date']),
        'web_traffic': pd.read_csv(d / 'web_traffic.csv', parse_dates=['date']),
    }
    return data


def add_gross_margin(products_df: pd.DataFrame) -> pd.DataFrame:
    """Add gross_margin column to products DataFrame."""
    df = products_df.copy()
    df['gross_margin'] = (df['price'] - df['cogs']) / df['price']
    return df


def compute_net_revenue(order_items_df: pd.DataFrame) -> pd.DataFrame:
    """Compute net revenue for each order item."""
    df = order_items_df.copy()
    df['net_revenue'] = (df['unit_price'] * df['quantity']) - df['discount_amount']
    return df


def merge_orders_full(data: dict) -> pd.DataFrame:
    """
    Merge orders with order_items, products, customers, geography, payments.
    Returns a wide DataFrame for comprehensive analysis.
    """
    df = data['orders'].merge(data['order_items'], on='order_id', how='inner')
    df = df.merge(data['products'], on='product_id', how='left')
    df = df.merge(data['customers'], on='customer_id', how='left', suffixes=('', '_cust'))
    df = df.merge(data['geography'], on='zip', how='left', suffixes=('', '_geo'))
    df = df.merge(data['payments'], on='order_id', how='left', suffixes=('', '_pay'))
    df['net_revenue'] = (df['unit_price'] * df['quantity']) - df['discount_amount']
    df['gross_margin'] = (df['price'] - df['cogs']) / df['price']
    return df
