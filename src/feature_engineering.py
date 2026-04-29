"""
Datathon 2026 — Feature Engineering for Revenue & COGS Forecasting.

This module provides all feature-engineering functions used in Phase 4 (Modeling).
Features are grouped into:
  1. Calendar / time features
  2. Vietnamese holiday & sale-season markers
  3. Lag features (for Revenue and COGS)
  4. Rolling window statistics
  5. External signals (web_traffic, inventory, promotions, orders)
"""

import pandas as pd
import numpy as np
from pathlib import Path

RANDOM_SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# 1. Calendar / time features
# ---------------------------------------------------------------------------

def create_time_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Extract calendar features from the date column."""
    df = df.copy()
    dt = df[date_col]
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["day_of_week"] = dt.dt.dayofweek          # 0=Mon … 6=Sun
    df["day_of_month"] = dt.dt.day
    df["day_of_year"] = dt.dt.dayofyear
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)
    # Cyclical encoding for month & day_of_week
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


# ---------------------------------------------------------------------------
# 2. Vietnamese holiday & sale-season markers
# ---------------------------------------------------------------------------

def create_holiday_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Mark Vietnamese Tet, major e-commerce sale days, and shopping seasons.
    """
    df = df.copy()
    dt = df[date_col]
    month = dt.dt.month
    day = dt.dt.day

    # Tet season (Jan-Feb, roughly lunar new year period)
    df["is_tet_season"] = month.isin([1, 2]).astype(int)

    # Major sale events (approximate Gregorian dates)
    # 9/9 (Sep 9), 10/10, 11/11, 12/12
    df["is_sale_99"] = ((month == 9) & (day >= 7) & (day <= 11)).astype(int)
    df["is_sale_1010"] = ((month == 10) & (day >= 8) & (day <= 12)).astype(int)
    df["is_sale_1111"] = ((month == 11) & (day >= 9) & (day <= 13)).astype(int)
    df["is_sale_1212"] = ((month == 12) & (day >= 10) & (day <= 14)).astype(int)
    # Black Friday window (last week of Nov)
    df["is_black_friday"] = ((month == 11) & (day >= 22) & (day <= 28)).astype(int)
    # Year-end holiday season (Dec 20-31)
    df["is_yearend"] = ((month == 12) & (day >= 20)).astype(int)
    # Vietnamese holidays: 30/4, 1/5, 2/9
    df["is_national_holiday"] = (
        ((month == 4) & (day == 30)) |
        ((month == 5) & (day == 1)) |
        ((month == 9) & (day == 2))
    ).astype(int)
    # Overall sale-season flag
    df["is_sale_season"] = (
        df[["is_sale_99", "is_sale_1010", "is_sale_1111", "is_sale_1212",
            "is_black_friday", "is_yearend", "is_tet_season"]].max(axis=1)
    )
    return df


# ---------------------------------------------------------------------------
# 3. Lag features
# ---------------------------------------------------------------------------

def create_lag_features(
    df: pd.DataFrame,
    target_cols: list[str] = ["Revenue", "COGS"],
    lags: list[int] = [1, 2, 3, 7, 14, 21, 28, 30, 60, 90, 365],
    date_col: str = "Date",
) -> pd.DataFrame:
    """Create lag features for each target column."""
    df = df.sort_values(date_col).copy()
    for col in target_cols:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


# ---------------------------------------------------------------------------
# 4. Rolling window statistics
# ---------------------------------------------------------------------------

def create_rolling_features(
    df: pd.DataFrame,
    target_cols: list[str] = ["Revenue", "COGS"],
    windows: list[int] = [7, 14, 30, 60, 90],
    date_col: str = "Date",
) -> pd.DataFrame:
    """Create rolling mean, std, min, max for each target with a 1-day shift (no leakage)."""
    df = df.sort_values(date_col).copy()
    for col in target_cols:
        shifted = df[col].shift(1)  # avoid leaking current day
        for w in windows:
            df[f"{col}_rmean_{w}"] = shifted.rolling(w, min_periods=1).mean()
            df[f"{col}_rstd_{w}"] = shifted.rolling(w, min_periods=2).std()
            df[f"{col}_rmin_{w}"] = shifted.rolling(w, min_periods=1).min()
            df[f"{col}_rmax_{w}"] = shifted.rolling(w, min_periods=1).max()
    return df


# ---------------------------------------------------------------------------
# 5. Expanding (cumulative) statistics
# ---------------------------------------------------------------------------

def create_expanding_features(
    df: pd.DataFrame,
    target_cols: list[str] = ["Revenue", "COGS"],
    date_col: str = "Date",
) -> pd.DataFrame:
    """Create expanding mean & std (historical average up to t-1)."""
    df = df.sort_values(date_col).copy()
    for col in target_cols:
        shifted = df[col].shift(1)
        df[f"{col}_exp_mean"] = shifted.expanding(min_periods=1).mean()
        df[f"{col}_exp_std"] = shifted.expanding(min_periods=2).std()
    return df


# ---------------------------------------------------------------------------
# 6. External signals — aggregate daily features from auxiliary tables
# ---------------------------------------------------------------------------

def aggregate_daily_orders(orders_path: str | Path | None = None) -> pd.DataFrame:
    """Aggregate daily order counts and status breakdown from orders.csv."""
    path = Path(orders_path) if orders_path else DATA_DIR / "orders.csv"
    orders = pd.read_csv(path, parse_dates=["order_date"])
    # Daily total orders
    daily = orders.groupby("order_date").agg(
        daily_orders=("order_id", "count"),
        daily_delivered=("order_status", lambda x: (x == "delivered").sum()),
        daily_cancelled=("order_status", lambda x: (x == "cancelled").sum()),
        daily_returned=("order_status", lambda x: (x == "returned").sum()),
        n_devices=("device_type", "nunique"),
        n_sources=("order_source", "nunique"),
    ).reset_index()
    daily.rename(columns={"order_date": "Date"}, inplace=True)
    daily["cancel_rate"] = daily["daily_cancelled"] / daily["daily_orders"]
    daily["return_rate"] = daily["daily_returned"] / daily["daily_orders"]
    return daily


def aggregate_daily_web_traffic(wt_path: str | Path | None = None) -> pd.DataFrame:
    """Aggregate daily web-traffic metrics (sum across traffic sources)."""
    path = Path(wt_path) if wt_path else DATA_DIR / "web_traffic.csv"
    wt = pd.read_csv(path, parse_dates=["date"])
    daily = wt.groupby("date").agg(
        total_sessions=("sessions", "sum"),
        total_visitors=("unique_visitors", "sum"),
        total_page_views=("page_views", "sum"),
        avg_bounce_rate=("bounce_rate", "mean"),
        avg_session_duration=("avg_session_duration_sec", "mean"),
        n_traffic_sources=("traffic_source", "nunique"),
    ).reset_index()
    daily.rename(columns={"date": "Date"}, inplace=True)
    return daily


def aggregate_monthly_inventory(inv_path: str | Path | None = None) -> pd.DataFrame:
    """Aggregate monthly inventory health metrics (average across products)."""
    path = Path(inv_path) if inv_path else DATA_DIR / "inventory.csv"
    inv = pd.read_csv(path, parse_dates=["snapshot_date"])
    monthly = inv.groupby("snapshot_date").agg(
        avg_fill_rate=("fill_rate", "mean"),
        total_stock=("stock_on_hand", "sum"),
        total_sold=("units_sold", "sum"),
        pct_stockout=("stockout_flag", "mean"),
        pct_overstock=("overstock_flag", "mean"),
        avg_sell_through=("sell_through_rate", "mean"),
    ).reset_index()
    monthly.rename(columns={"snapshot_date": "Date"}, inplace=True)
    # We'll forward-fill these monthly values to daily
    return monthly


def create_promo_daily_flags(promo_path: str | Path | None = None,
                              date_range: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Create daily promo flags: number of active promos and avg discount."""
    path = Path(promo_path) if promo_path else DATA_DIR / "promotions.csv"
    promos = pd.read_csv(path, parse_dates=["start_date", "end_date"])

    if date_range is None:
        min_d = promos["start_date"].min()
        max_d = promos["end_date"].max()
        date_range = pd.date_range(min_d, max_d, freq="D")

    records = []
    for d in date_range:
        active = promos[(promos["start_date"] <= d) & (promos["end_date"] >= d)]
        records.append({
            "Date": d,
            "n_active_promos": len(active),
            "avg_promo_discount": active["discount_value"].mean() if len(active) > 0 else 0.0,
            "max_promo_discount": active["discount_value"].max() if len(active) > 0 else 0.0,
            "any_promo_active": int(len(active) > 0),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 7. Master pipeline: build the full feature DataFrame
# ---------------------------------------------------------------------------

def build_feature_dataframe(
    sales_df: pd.DataFrame,
    include_external: bool = True,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Build the complete feature matrix from sales_train + external signals.

    Parameters
    ----------
    sales_df : DataFrame with columns ['Date', 'Revenue', 'COGS']
    include_external : whether to merge external signals (orders, web, inventory, promos)
    data_dir : override data directory

    Returns
    -------
    DataFrame with all features (rows sorted by Date, NaN rows from lags at start)
    """
    d = Path(data_dir) if data_dir else DATA_DIR
    df = sales_df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # 1. Time features
    df = create_time_features(df)

    # 2. Holiday features
    df = create_holiday_features(df)

    # 3. Lag features
    df = create_lag_features(df)

    # 4. Rolling features
    df = create_rolling_features(df)

    # 5. Expanding features
    df = create_expanding_features(df)

    # 6. Margin ratio feature
    df["margin_ratio"] = (df["Revenue"] - df["COGS"]) / df["Revenue"]
    df["margin_ratio_lag1"] = df["margin_ratio"].shift(1)

    if include_external:
        # 6a. Daily orders
        try:
            daily_orders = aggregate_daily_orders(d / "orders.csv")
            df = df.merge(daily_orders, on="Date", how="left")
        except Exception as e:
            print(f"Warning: Could not load orders.csv — {e}")

        # 6b. Web traffic
        try:
            daily_wt = aggregate_daily_web_traffic(d / "web_traffic.csv")
            df = df.merge(daily_wt, on="Date", how="left")
        except Exception as e:
            print(f"Warning: Could not load web_traffic.csv — {e}")

        # 6c. Monthly inventory → forward-fill to daily
        try:
            monthly_inv = aggregate_monthly_inventory(d / "inventory.csv")
            df = df.merge(monthly_inv, on="Date", how="left")
            inv_cols = ["avg_fill_rate", "total_stock", "total_sold",
                        "pct_stockout", "pct_overstock", "avg_sell_through"]
            for c in inv_cols:
                if c in df.columns:
                    df[c] = df[c].ffill()
        except Exception as e:
            print(f"Warning: Could not load inventory.csv — {e}")

        # 6d. Promo flags
        try:
            promo_flags = create_promo_daily_flags(
                d / "promotions.csv",
                date_range=pd.DatetimeIndex(df["Date"])
            )
            df = df.merge(promo_flags, on="Date", how="left")
        except Exception as e:
            print(f"Warning: Could not load promotions.csv — {e}")

    return df
