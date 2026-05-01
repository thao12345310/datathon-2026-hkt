"""
Datathon 2026 — Forecasting v6: Built on v2 (best Kaggle score) with genuine improvements.

Strategy: v2 was best because it let the model learn freely without artificial constraints.
v5 failed because it tried to match the sample_submission baseline (not ground truth).

v6 improvements over v2:
  1. More features: EWM, Fourier, seasonal index, interaction terms, external signals
  2. Multi-seed ensemble (3 algos × 3 seeds = 9 models) for robustness
  3. Better hyperparameters: slightly deeper trees, more regularization
  4. External data integration (web traffic, daily orders) via forward-fill
  5. Business constraint: COGS < Revenue using learned ratio
  6. NO sample weighting, NO log-transform, NO drift dampening
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgbm
import catboost as cb
import shap
import json

SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "submission"
RPT_DIR = Path(__file__).resolve().parent.parent / "reports"
RPT_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════
# EXTERNAL DATA AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════

def load_daily_orders():
    """Aggregate daily order stats."""
    try:
        orders = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["order_date"])
        daily = orders.groupby("order_date").agg(
            daily_orders=("order_id", "count"),
            daily_delivered=("order_status", lambda x: (x == "delivered").sum()),
            daily_cancelled=("order_status", lambda x: (x == "cancelled").sum()),
        ).reset_index()
        daily.rename(columns={"order_date": "Date"}, inplace=True)
        daily["cancel_rate"] = daily["daily_cancelled"] / daily["daily_orders"].clip(lower=1)
        return daily
    except Exception as e:
        print(f"  ⚠ orders.csv: {e}")
        return None


def load_daily_web_traffic():
    """Aggregate daily web traffic."""
    try:
        wt = pd.read_csv(DATA_DIR / "web_traffic.csv", parse_dates=["date"])
        daily = wt.groupby("date").agg(
            total_sessions=("sessions", "sum"),
            total_visitors=("unique_visitors", "sum"),
            total_page_views=("page_views", "sum"),
            avg_bounce_rate=("bounce_rate", "mean"),
        ).reset_index()
        daily.rename(columns={"date": "Date"}, inplace=True)
        return daily
    except Exception as e:
        print(f"  ⚠ web_traffic.csv: {e}")
        return None


def load_monthly_inventory():
    """Aggregate monthly inventory stats."""
    try:
        inv = pd.read_csv(DATA_DIR / "inventory.csv", parse_dates=["snapshot_date"])
        monthly = inv.groupby("snapshot_date").agg(
            avg_fill_rate=("fill_rate", "mean"),
            total_stock=("stock_on_hand", "sum"),
            total_sold=("units_sold", "sum"),
            pct_stockout=("stockout_flag", "mean"),
        ).reset_index()
        monthly.rename(columns={"snapshot_date": "Date"}, inplace=True)
        return monthly
    except Exception as e:
        print(f"  ⚠ inventory.csv: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING v6: v2 base + enrichments
# ═══════════════════════════════════════════════════════════════════════════

def build_features(df, target="Revenue", include_external=True):
    """Build features from Date + target history + optional external data."""
    df = df.sort_values("Date").reset_index(drop=True).copy()
    dt = df["Date"]

    # ── Calendar (same as v2) ──
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["day_of_year"] = dt.dt.dayofyear
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)
    df["is_quarter_end"] = dt.dt.is_quarter_end.astype(int)

    # ── Cyclical + Fourier (v2 base + extras) ──
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # Fourier harmonics — annual
    for k in [1, 2, 3, 4, 6]:
        df[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * df["day_of_year"] / 365.25)
        df[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * df["day_of_year"] / 365.25)
    # Fourier — weekly
    for k in [1, 2, 3]:
        df[f"weekly_sin_{k}"] = np.sin(2 * np.pi * k * df["day_of_week"] / 7)
        df[f"weekly_cos_{k}"] = np.cos(2 * np.pi * k * df["day_of_week"] / 7)

    # ── Vietnamese seasons (same as v2) ──
    m, d = df["month"], df["day_of_month"]
    df["is_tet"] = m.isin([1, 2]).astype(int)
    df["is_sale_1111"] = ((m == 11) & (d >= 9) & (d <= 13)).astype(int)
    df["is_sale_1212"] = ((m == 12) & (d >= 10) & (d <= 14)).astype(int)
    df["is_yearend"] = ((m == 12) & (d >= 20)).astype(int)
    df["is_black_friday"] = ((m == 11) & (d >= 22) & (d <= 28)).astype(int)
    df["is_99"] = ((m == 9) & (d >= 7) & (d <= 11)).astype(int)
    df["is_1010"] = ((m == 10) & (d >= 8) & (d <= 12)).astype(int)
    df["is_holiday"] = (
        ((m == 4) & (d == 30)) | ((m == 5) & (d == 1)) | ((m == 9) & (d == 2))
    ).astype(int)

    # NEW: End-of-month spike window (last 3 days ~1.6x revenue)
    days_in_month = dt.dt.days_in_month
    df["is_eom_window"] = (df["day_of_month"] >= days_in_month - 2).astype(int)
    # Month progress (0-1)
    df["month_progress"] = df["day_of_month"] / days_in_month

    # ── Lag features (same as v2 + lag_5 and lag_180) ──
    for lag in [1, 2, 3, 5, 7, 14, 21, 28, 30, 60, 90, 180, 365]:
        df[f"lag_{lag}"] = df[target].shift(lag)

    # ── Rolling features (same as v2) ──
    shifted = df[target].shift(1)
    for w in [7, 14, 30, 60, 90]:
        df[f"rmean_{w}"] = shifted.rolling(w, min_periods=1).mean()
        df[f"rstd_{w}"] = shifted.rolling(w, min_periods=2).std()
        df[f"rmin_{w}"] = shifted.rolling(w, min_periods=1).min()
        df[f"rmax_{w}"] = shifted.rolling(w, min_periods=1).max()

    # NEW: Exponential moving averages (capture recent momentum better)
    for span in [7, 14, 30]:
        df[f"ewm_{span}"] = shifted.ewm(span=span, min_periods=1).mean()

    # ── Expanding mean (same as v2) ──
    df["exp_mean"] = shifted.expanding(min_periods=1).mean()

    # ── Diff features (same as v2) ──
    df["diff_1"] = df[target].diff(1)
    df["diff_7"] = df[target].diff(7)

    # ── Trend index (same as v2) ──
    df["trend_idx"] = np.arange(len(df))

    # NEW: Seasonal index (historical day-of-year average / overall)
    valid = df[target].notna()
    if valid.sum() > 60:
        doy_avg = df.loc[valid].groupby("day_of_year")[target].mean()
        overall = df.loc[valid, target].mean()
        df["seasonal_idx"] = df["day_of_year"].map(doy_avg) / (overall if overall != 0 else 1)
        df["seasonal_idx"] = df["seasonal_idx"].fillna(1.0)

        # Month-DayOfWeek average
        wm_avg = df.loc[valid].groupby(["month", "day_of_week"])[target].mean()
        df["dow_month_avg"] = df.apply(
            lambda r: wm_avg.get((r["month"], r["day_of_week"]), overall), axis=1
        )
    else:
        df["seasonal_idx"] = 1.0
        df["dow_month_avg"] = df[target].mean() if valid.any() else 0

    # NEW: Interaction features
    df["weekend_x_month"] = df["is_weekend"] * df["month"]
    df["eom_x_quarter"] = df["is_eom_window"] * df["quarter"]

    # ── External data (forward-filled into test period) ──
    if include_external:
        ext_orders = load_daily_orders()
        if ext_orders is not None:
            df = df.merge(ext_orders, on="Date", how="left")
            for c in ["daily_orders", "daily_delivered", "daily_cancelled", "cancel_rate"]:
                if c in df.columns:
                    df[c] = df[c].ffill().fillna(0)

        ext_wt = load_daily_web_traffic()
        if ext_wt is not None:
            df = df.merge(ext_wt, on="Date", how="left")
            for c in ["total_sessions", "total_visitors", "total_page_views", "avg_bounce_rate"]:
                if c in df.columns:
                    df[c] = df[c].ffill().fillna(0)

        ext_inv = load_monthly_inventory()
        if ext_inv is not None:
            df = df.merge(ext_inv, on="Date", how="left")
            for c in ["avg_fill_rate", "total_stock", "total_sold", "pct_stockout"]:
                if c in df.columns:
                    df[c] = df[c].ffill().fillna(0)

    return df


def get_feat_cols(df):
    """Return numeric feature columns."""
    drop = {"Date", "Revenue", "COGS", "diff_1", "diff_7"}
    return [c for c in df.columns
            if c not in drop and df[c].dtype in ("float64", "int64", "int32", "float32")]


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING: v2 hyperparameters + multi-seed ensemble
# ═══════════════════════════════════════════════════════════════════════════

def train_models(sales, target="Revenue"):
    """Train multi-seed ensemble using v2-proven hyperparameters."""
    print(f"\n{'=' * 60}")
    print(f"Training for {target}")
    print(f"{'=' * 60}")

    # Build features WITH external data for training
    df = build_features(sales.copy(), target=target, include_external=True)
    df = df.dropna(subset=["lag_365"]).reset_index(drop=True)
    feat_cols = get_feat_cols(df)
    X = df[feat_cols].values
    y = df[target].values
    print(f"  {len(df)} rows, {len(feat_cols)} features")

    # ── CV evaluation (v2 hyperparameters) ──
    tscv = TimeSeriesSplit(n_splits=5)

    cv_configs = {
        "XGB": xgb.XGBRegressor(
            n_estimators=1500, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=2.0, min_child_weight=10,
            random_state=SEED, n_jobs=-1, early_stopping_rounds=50,
        ),
        "LGB": lgbm.LGBMRegressor(
            n_estimators=1500, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=2.0, min_child_weight=10,
            random_state=SEED, n_jobs=-1, verbosity=-1,
        ),
        "CB": cb.CatBoostRegressor(
            iterations=1500, depth=7, learning_rate=0.03,
            l2_leaf_reg=5.0, random_seed=SEED, verbose=0,
            early_stopping_rounds=50,
        ),
    }

    for name, model in cv_configs.items():
        maes, r2s = [], []
        for fold, (ti, vi) in enumerate(tscv.split(X)):
            Xt, Xv, yt, yv = X[ti], X[vi], y[ti], y[vi]
            if name == "XGB":
                model.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
            elif name == "LGB":
                model.fit(Xt, yt, eval_set=[(Xv, yv)])
            else:
                model.fit(Xt, yt, eval_set=(Xv, yv), verbose=False)
            p = np.maximum(model.predict(Xv), 0)
            maes.append(mean_absolute_error(yv, p))
            r2s.append(r2_score(yv, p))
        print(f"  {name}: MAE={np.mean(maes):,.0f}±{np.std(maes):,.0f}  R²={np.mean(r2s):.4f}")

    # ── Train final multi-seed models ──
    seeds = [42, 2024, 7]
    models = {}

    for seed in seeds:
        for name in ["XGB", "LGB", "CB"]:
            key = f"{name}_s{seed}"
            if name == "XGB":
                m = xgb.XGBRegressor(
                    n_estimators=2000, max_depth=7, learning_rate=0.02,
                    subsample=0.8, colsample_bytree=0.7,
                    reg_alpha=0.5, reg_lambda=2.0, min_child_weight=10,
                    random_state=seed, n_jobs=-1,
                )
                m.fit(X, y, verbose=False)
            elif name == "LGB":
                m = lgbm.LGBMRegressor(
                    n_estimators=2000, max_depth=7, learning_rate=0.02,
                    subsample=0.8, colsample_bytree=0.7,
                    reg_alpha=0.5, reg_lambda=2.0, min_child_weight=10,
                    random_state=seed, n_jobs=-1, verbosity=-1,
                )
                m.fit(X, y)
            else:
                m = cb.CatBoostRegressor(
                    iterations=2000, depth=7, learning_rate=0.02,
                    l2_leaf_reg=5.0, random_seed=seed, verbose=0,
                )
                m.fit(X, y, verbose=False)
            models[key] = m

    print(f"  ✅ {len(models)} models trained (3 algos × {len(seeds)} seeds)")
    return models, feat_cols, df


# ═══════════════════════════════════════════════════════════════════════════
# RECURSIVE PREDICTION — v2 style, no drift dampening
# ═══════════════════════════════════════════════════════════════════════════

def predict_recursive(models, train_sales, test_dates, feat_cols, target="Revenue"):
    """
    Recursive day-by-day prediction — pure model output, no corrections.
    Same approach as v2 (best Kaggle score) but with multi-seed ensemble.
    """
    print(f"\n  Recursive prediction for {target} ({len(test_dates)} days)...")

    history = train_sales[["Date", target]].copy()
    predictions = []

    for i, date in enumerate(test_dates):
        new_row = pd.DataFrame({"Date": [date], target: [np.nan]})
        temp = pd.concat([history, new_row], ignore_index=True)

        # Build features (without external data for speed in recursion)
        featured = build_features(temp, target=target, include_external=False)
        last = featured.iloc[-1:]

        fc = [c for c in feat_cols if c in last.columns]
        X = last[fc].values
        X = np.nan_to_num(X, nan=0.0)

        # Pure ensemble average — no dampening, no corrections
        preds = [max(m.predict(X)[0], 0) for m in models.values()]
        pred = np.mean(preds)
        predictions.append(pred)

        # Update history for next day's lags
        new_row[target] = pred
        history = pd.concat([history, new_row], ignore_index=True)

        if (i + 1) % 50 == 0:
            print(f"    Day {i + 1}/{len(test_dates)}...")

    print(
        f"    ✅ Range: {min(predictions):,.0f}—{max(predictions):,.0f}, "
        f"mean={np.mean(predictions):,.0f}, std={np.std(predictions):,.0f}"
    )
    return dict(zip(test_dates, predictions))


# ═══════════════════════════════════════════════════════════════════════════
# SHAP
# ═══════════════════════════════════════════════════════════════════════════

def do_shap(models, df, feat_cols, target):
    xgb_key = [k for k in models if k.startswith("XGB")][0]
    model = models[xgb_key]
    X = df[feat_cols].values[:500]
    sv = shap.TreeExplainer(model).shap_values(X)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X, feature_names=feat_cols, max_display=20, show=False)
    plt.title(f"SHAP — {target} (v6)", fontsize=14)
    plt.tight_layout()
    plt.savefig(RPT_DIR / f"shap_v6_{target.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: reports/shap_v6_{target.lower()}.png")

    mean_shap = np.abs(sv).mean(axis=0)
    top = np.argsort(mean_shap)[::-1][:10]
    print(f"  Top 10 features ({target}):")
    for i, idx in enumerate(top):
        print(f"    {i + 1:2d}. {feat_cols[idx]:30s} {mean_shap[idx]:>12,.0f}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("🚀 DATATHON 2026 — Forecasting v6 (v2 base + better features + multi-seed)")
    print("=" * 60)

    sales = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["Date"])
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"])
    test_dates = sample_sub["Date"].sort_values().values
    print(f"Train: {len(sales)} rows ({sales['Date'].min().date()} → {sales['Date'].max().date()})")
    print(f"Test:  {len(test_dates)} days")

    # ── Revenue ──
    rev_models, rev_feats, rev_df = train_models(sales, "Revenue")
    rev_preds = predict_recursive(rev_models, sales, test_dates, rev_feats, "Revenue")

    # ── COGS ──
    cogs_models, cogs_feats, cogs_df = train_models(sales, "COGS")
    cogs_preds = predict_recursive(cogs_models, sales, test_dates, cogs_feats, "COGS")

    # ── SHAP ──
    do_shap(rev_models, rev_df, rev_feats, "Revenue")
    do_shap(cogs_models, cogs_df, cogs_feats, "COGS")

    # ── Build submission ──
    sub = sample_sub[["Date"]].copy()
    sub["Revenue"] = sub["Date"].map(rev_preds)
    sub["COGS"] = sub["Date"].map(cogs_preds)

    # Business constraint: COGS < Revenue (use learned ratio)
    ratio_train = (sales["COGS"] / sales["Revenue"]).mean()  # ~0.875
    mask = sub["COGS"] > sub["Revenue"]
    if mask.any():
        print(f"  ⚠ Fixing {mask.sum()} rows where COGS > Revenue (using ratio {ratio_train:.3f})")
        sub.loc[mask, "COGS"] = sub.loc[mask, "Revenue"] * ratio_train

    # Ensure no negatives
    sub["Revenue"] = sub["Revenue"].clip(lower=0)
    sub["COGS"] = sub["COGS"].clip(lower=0)

    assert sub["Revenue"].notna().all(), "Missing Revenue!"
    assert sub["COGS"].notna().all(), "Missing COGS!"

    sub.to_csv(OUT_DIR / "submission.csv", index=False)

    # ── Diagnostics ──
    mae_r = np.abs(sub["Revenue"] - sample_sub["Revenue"]).mean()
    mae_c = np.abs(sub["COGS"] - sample_sub["COGS"]).mean()
    ratio = (sub["COGS"] / sub["Revenue"]).mean()

    print(f"\n{'=' * 60}")
    print(f"✅ submission/submission.csv ({len(sub)} rows)")
    print(f"  Revenue: mean={sub['Revenue'].mean():,.0f}, std={sub['Revenue'].std():,.0f}")
    print(f"  COGS:    mean={sub['COGS'].mean():,.0f}, std={sub['COGS'].std():,.0f}")
    print(f"  COGS/Revenue ratio: {ratio:.3f}")
    print(f"  MAE vs sample_sub (baseline only, NOT Kaggle score): Rev={mae_r:,.0f}, COGS={mae_c:,.0f}")

    # Drift check
    n = len(sub)
    e1 = np.abs(sub["Revenue"].iloc[:100] - sample_sub["Revenue"].iloc[:100]).mean()
    e2 = np.abs(sub["Revenue"].iloc[-100:] - sample_sub["Revenue"].iloc[-100:]).mean()
    print(f"  Drift: early_100={e1:,.0f}, late_100={e2:,.0f}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    for ax, col, c in zip(axes, ["Revenue", "COGS"], ["#FF6B6B", "#4ECDC4"]):
        ax.plot(sales["Date"].tail(365), sales[col].tail(365), color="gray", alpha=0.5, label="Train")
        ax.plot(sub["Date"], sub[col], color=c, lw=2, label="v6 Prediction")
        ax.plot(sample_sub["Date"], sample_sub[col], "k--", alpha=0.4, label="Sample Baseline")
        ax.set_title(f"{col} (v6)", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RPT_DIR / "forecast_v6.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: reports/forecast_v6.png")

    # Save results
    results = {
        "version": "v6",
        "mae_vs_sample_revenue": float(mae_r),
        "mae_vs_sample_cogs": float(mae_c),
        "cogs_ratio": float(ratio),
        "revenue_mean": float(sub["Revenue"].mean()),
        "cogs_mean": float(sub["COGS"].mean()),
        "n_models": len(rev_models),
        "approach": "v2 base (best Kaggle) + extra features + multi-seed ensemble",
    }
    with open(RPT_DIR / "results_v6.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("🏁 v6 complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
