"""
Datathon 2026 — Forecasting v5: Major improvements over v4.

Key improvements:
  1. Recent-year sample weighting (2019-2022 matter more than 2012-2018)
  2. Joint Revenue→COGS modeling via learned COGS/Revenue ratio
  3. Multi-seed ensemble for robustness (3 models × 3 seeds = 9 models)
  4. Improved feature engineering: interaction features, better trend capture
  5. Adaptive drift correction with exponential decay
  6. Proper COGS/Revenue ratio enforcement (train ratio ~0.875)
  7. Log-transform target to stabilize variance
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
# FEATURE ENGINEERING v5
# ═══════════════════════════════════════════════════════════════════════════

def build_features(df, target="Revenue"):
    """Build comprehensive features for a single target."""
    df = df.sort_values("Date").reset_index(drop=True).copy()
    dt = df["Date"]

    # ── Calendar features ──
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
    df["days_in_month"] = dt.dt.days_in_month
    # Position within month (0-1)
    df["month_progress"] = df["day_of_month"] / df["days_in_month"]

    # ── Cyclical encodings ──
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # ── Fourier terms (annual + weekly) ──
    for k in [1, 2, 3, 4, 6]:
        df[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * df["day_of_year"] / 365.25)
        df[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * df["day_of_year"] / 365.25)
    for k in [1, 2, 3]:
        df[f"wk_sin_{k}"] = np.sin(2 * np.pi * k * df["day_of_week"] / 7)
        df[f"wk_cos_{k}"] = np.cos(2 * np.pi * k * df["day_of_week"] / 7)

    # ── Vietnamese holidays & sale seasons ──
    m, d = df["month"], df["day_of_month"]
    df["is_tet"] = m.isin([1, 2]).astype(int)
    df["is_1111"] = ((m == 11) & (d >= 9) & (d <= 13)).astype(int)
    df["is_1212"] = ((m == 12) & (d >= 10) & (d <= 14)).astype(int)
    df["is_yearend"] = ((m == 12) & (d >= 20)).astype(int)
    df["is_bf"] = ((m == 11) & (d >= 22) & (d <= 28)).astype(int)
    df["is_99"] = ((m == 9) & (d >= 7) & (d <= 11)).astype(int)
    df["is_1010"] = ((m == 10) & (d >= 8) & (d <= 12)).astype(int)
    df["is_holiday"] = (
        ((m == 4) & (d == 30)) | ((m == 5) & (d == 1)) | ((m == 9) & (d == 2))
    ).astype(int)
    # End-of-month spike indicator (last 3 days tend to have ~1.6x revenue)
    df["is_eom_window"] = (df["day_of_month"] >= df["days_in_month"] - 2).astype(int)
    # Combined sale flag
    df["any_sale"] = df[["is_1111", "is_1212", "is_bf", "is_99", "is_1010"]].max(axis=1)

    # ── Lag features ──
    for lag in [1, 2, 3, 5, 7, 14, 21, 28, 30, 60, 90, 180, 365]:
        df[f"lag_{lag}"] = df[target].shift(lag)

    # ── Rolling stats (shifted to avoid leakage) ──
    sh = df[target].shift(1)
    for w in [7, 14, 30, 60, 90]:
        df[f"rmean_{w}"] = sh.rolling(w, min_periods=1).mean()
        df[f"rstd_{w}"] = sh.rolling(w, min_periods=2).std()
        df[f"rmin_{w}"] = sh.rolling(w, min_periods=1).min()
        df[f"rmax_{w}"] = sh.rolling(w, min_periods=1).max()
    # Exponential moving averages (more weight to recent)
    for span in [7, 14, 30]:
        df[f"ewm_{span}"] = sh.ewm(span=span, min_periods=1).mean()

    # ── Expanding mean ──
    df["exp_mean"] = sh.expanding(min_periods=1).mean()

    # ── Trend index ──
    df["trend_idx"] = np.arange(len(df))

    # ── Seasonal indices (computed only from known target values) ──
    valid = df[target].notna()
    if valid.sum() > 60:
        # Day-of-year seasonal index
        doy_avg = df.loc[valid].groupby("day_of_year")[target].mean()
        overall = df.loc[valid, target].mean()
        df["seasonal_idx"] = df["day_of_year"].map(doy_avg) / (overall if overall != 0 else 1)
        df["seasonal_idx"] = df["seasonal_idx"].fillna(1.0)

        # Month-DayOfWeek average
        wm_avg = df.loc[valid].groupby(["month", "day_of_week"])[target].mean()
        df["dow_month_avg"] = df.apply(
            lambda r: wm_avg.get((r["month"], r["day_of_week"]), overall), axis=1
        )

        # Recent-year seasonal (last 3 years only)
        recent_mask = valid & (df["year"] >= df.loc[valid, "year"].max() - 2)
        if recent_mask.sum() > 30:
            recent_doy = df.loc[recent_mask].groupby("day_of_year")[target].mean()
            recent_overall = df.loc[recent_mask, target].mean()
            df["recent_seasonal_idx"] = df["day_of_year"].map(recent_doy) / (
                recent_overall if recent_overall != 0 else 1
            )
            df["recent_seasonal_idx"] = df["recent_seasonal_idx"].fillna(1.0)
        else:
            df["recent_seasonal_idx"] = 1.0
    else:
        df["seasonal_idx"] = 1.0
        df["dow_month_avg"] = df[target].mean() if valid.any() else 0
        df["recent_seasonal_idx"] = 1.0

    # ── Interaction features ──
    df["weekend_x_month"] = df["is_weekend"] * df["month"]
    df["eom_x_month"] = df["is_eom_window"] * df["month"]

    return df


def get_feat_cols(df):
    """Return numeric feature columns, excluding targets and date."""
    drop = {"Date", "Revenue", "COGS", "days_in_month"}
    return [
        c for c in df.columns
        if c not in drop and df[c].dtype in ("float64", "int64", "int32", "float32")
    ]


# ═══════════════════════════════════════════════════════════════════════════
# SAMPLE WEIGHTING — Recent years matter more
# ═══════════════════════════════════════════════════════════════════════════

def compute_sample_weights(df, decay=0.85):
    """
    Give more weight to recent years. Exponential decay from newest to oldest.
    decay=0.85 means each older year gets 85% of the next year's weight.
    """
    years = df["year"].values
    max_year = years.max()
    weights = decay ** (max_year - years)
    # Normalize so mean weight = 1
    weights = weights / weights.mean()
    return weights


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING WITH MULTI-SEED ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════

def train_models(sales, target="Revenue", use_log=True):
    """Train XGB + LGB + CB with sample weights and optional log-transform."""
    print(f"\n{'=' * 60}")
    print(f"Training for {target}")
    print(f"{'=' * 60}")

    df = build_features(sales.copy(), target=target)
    df = df.dropna(subset=["lag_365"]).reset_index(drop=True)
    feat_cols = get_feat_cols(df)
    X = df[feat_cols].values
    y = df[target].values

    if use_log:
        y_train = np.log1p(y)
    else:
        y_train = y

    weights = compute_sample_weights(df)
    print(f"  {len(df)} rows, {len(feat_cols)} features")
    print(f"  Sample weights: min={weights.min():.3f}, max={weights.max():.3f}")

    # CV evaluation
    tscv = TimeSeriesSplit(n_splits=5)
    seeds = [42, 2024, 7]

    for name_base in ["XGB", "LGB", "CB"]:
        maes = []
        for ti, vi in tscv.split(X):
            w_tr = weights[ti]
            if name_base == "XGB":
                m = xgb.XGBRegressor(
                    n_estimators=2000, max_depth=6, learning_rate=0.02,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
                    random_state=SEED, n_jobs=-1,
                )
                m.fit(X[ti], y_train[ti], sample_weight=w_tr,
                      eval_set=[(X[vi], y_train[vi])], verbose=False)
            elif name_base == "LGB":
                m = lgbm.LGBMRegressor(
                    n_estimators=2000, max_depth=6, learning_rate=0.02,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
                    random_state=SEED, n_jobs=-1, verbosity=-1,
                )
                m.fit(X[ti], y_train[ti], sample_weight=w_tr,
                      eval_set=[(X[vi], y_train[vi])])
            else:
                m = cb.CatBoostRegressor(
                    iterations=2000, depth=6, learning_rate=0.02,
                    l2_leaf_reg=3.0, random_seed=SEED, verbose=0,
                )
                m.fit(X[ti], y_train[ti], sample_weight=w_tr,
                      eval_set=(X[vi], y_train[vi]), verbose=False)

            p = m.predict(X[vi])
            if use_log:
                p = np.expm1(p)
            p = np.maximum(p, 0)
            maes.append(mean_absolute_error(y[vi], p))
        print(f"  {name_base}: MAE={np.mean(maes):,.0f}±{np.std(maes):,.0f}")

    # Train final multi-seed models on full data
    models = {}
    for seed in seeds:
        for name_base in ["XGB", "LGB", "CB"]:
            key = f"{name_base}_s{seed}"
            if name_base == "XGB":
                m = xgb.XGBRegressor(
                    n_estimators=2500, max_depth=6, learning_rate=0.015,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
                    random_state=seed, n_jobs=-1,
                )
                m.fit(X, y_train, sample_weight=weights, verbose=False)
            elif name_base == "LGB":
                m = lgbm.LGBMRegressor(
                    n_estimators=2500, max_depth=6, learning_rate=0.015,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
                    random_state=seed, n_jobs=-1, verbosity=-1,
                )
                m.fit(X, y_train, sample_weight=weights)
            else:
                m = cb.CatBoostRegressor(
                    iterations=2500, depth=6, learning_rate=0.015,
                    l2_leaf_reg=3.0, random_seed=seed, verbose=0,
                )
                m.fit(X, y_train, sample_weight=weights, verbose=False)
            models[key] = m

    print(f"  ✅ {len(models)} models trained (3 algos × {len(seeds)} seeds)")
    return models, feat_cols, df, use_log


# ═══════════════════════════════════════════════════════════════════════════
# RECURSIVE PREDICTION WITH ADAPTIVE DRIFT CORRECTION
# ═══════════════════════════════════════════════════════════════════════════

def predict_recursive_v5(
    models, train_sales, test_dates, feat_cols, target="Revenue",
    use_log=True, dampen_alpha=0.20
):
    """
    Recursive prediction with:
    - Multi-model averaging (9 models)
    - Adaptive drift correction using recent-year monthly patterns
    - Reasonable range clipping
    """
    print(f"\n  Recursive prediction for {target} ({len(test_dates)} days)...")

    # Compute monthly reference from last 3 years of training
    recent = train_sales[train_sales["Date"].dt.year >= train_sales["Date"].dt.year.max() - 2]
    monthly_ref = recent.groupby(recent["Date"].dt.month)[target].agg(["mean", "std"]).to_dict("index")

    # Per-day-of-year averages (last 3 years)
    recent_copy = recent.copy()
    recent_copy["doy"] = recent_copy["Date"].dt.dayofyear
    doy_means = recent_copy.groupby("doy")[target].mean().to_dict()
    recent_mean = recent[target].mean()

    history = train_sales[["Date", target]].copy()
    predictions = []

    for i, date in enumerate(test_dates):
        new_row = pd.DataFrame({"Date": [date], target: [np.nan]})
        temp = pd.concat([history, new_row], ignore_index=True)
        featured = build_features(temp, target=target)
        last = featured.iloc[-1:]
        fc = [c for c in feat_cols if c in last.columns]
        X = last[fc].values
        X = np.nan_to_num(X, nan=0.0)

        # Multi-model ensemble (average all 9 models)
        raw_preds = []
        for m in models.values():
            p = m.predict(X)[0]
            if use_log:
                p = np.expm1(p)
            raw_preds.append(max(p, 0))
        raw_pred = np.mean(raw_preds)

        # Adaptive drift correction using recent-year seasonal pattern
        doy = pd.Timestamp(date).dayofyear
        historical_ref = doy_means.get(doy, recent_mean)
        pred = (1 - dampen_alpha) * raw_pred + dampen_alpha * historical_ref

        # Clip to reasonable range based on monthly stats
        month = pd.Timestamp(date).month
        if month in monthly_ref:
            m_mean = monthly_ref[month]["mean"]
            m_std = monthly_ref[month]["std"]
            pred = np.clip(pred, m_mean - 3 * m_std, m_mean + 3 * m_std)

        pred = max(pred, 0)
        predictions.append(pred)

        new_row[target] = pred
        history = pd.concat([history, new_row], ignore_index=True)

        if (i + 1) % 100 == 0:
            print(f"    Day {i + 1}/{len(test_dates)}...")

    print(
        f"    ✅ Range: {min(predictions):,.0f}—{max(predictions):,.0f}, "
        f"mean={np.mean(predictions):,.0f}, std={np.std(predictions):,.0f}"
    )
    return dict(zip(test_dates, predictions))


# ═══════════════════════════════════════════════════════════════════════════
# COGS POST-PROCESSING: Enforce realistic COGS/Revenue ratio
# ═══════════════════════════════════════════════════════════════════════════

def fix_cogs_ratio(sub, train_sales, blend_weight=0.5):
    """
    Blend independent COGS predictions with ratio-based COGS.
    Training data shows COGS/Revenue ≈ 0.875 on average.
    """
    # Compute monthly COGS/Revenue ratio from recent training
    recent = train_sales[train_sales["Date"].dt.year >= train_sales["Date"].dt.year.max() - 2]
    monthly_ratio = recent.groupby(recent["Date"].dt.month).apply(
        lambda x: (x["COGS"] / x["Revenue"]).mean()
    ).to_dict()
    overall_ratio = (recent["COGS"] / recent["Revenue"]).mean()

    sub = sub.copy()
    sub["month_temp"] = sub["Date"].dt.month
    sub["ratio_cogs"] = sub.apply(
        lambda r: r["Revenue"] * monthly_ratio.get(r["month_temp"], overall_ratio), axis=1
    )

    # Blend: weight between independent COGS prediction and ratio-based
    sub["COGS"] = blend_weight * sub["COGS"] + (1 - blend_weight) * sub["ratio_cogs"]

    # Hard constraint: COGS < Revenue
    mask = sub["COGS"] > sub["Revenue"]
    if mask.any():
        print(f"  ⚠ Fixing {mask.sum()} rows where COGS > Revenue")
        sub.loc[mask, "COGS"] = sub.loc[mask, "Revenue"] * overall_ratio

    sub = sub.drop(columns=["month_temp", "ratio_cogs"])
    return sub


# ═══════════════════════════════════════════════════════════════════════════
# SHAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def do_shap(models, df, feat_cols, target):
    """SHAP analysis using the first XGB model."""
    print(f"\n  SHAP analysis for {target}...")
    xgb_key = [k for k in models if k.startswith("XGB")][0]
    model = models[xgb_key]
    X = df[feat_cols].values[:500]
    sv = shap.TreeExplainer(model).shap_values(X)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X, feature_names=feat_cols, max_display=15, show=False)
    plt.title(f"SHAP — {target} (v5)", fontsize=14)
    plt.tight_layout()
    plt.savefig(RPT_DIR / f"shap_v5_{target.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: reports/shap_v5_{target.lower()}.png")

    mean_shap = np.abs(sv).mean(axis=0)
    top = np.argsort(mean_shap)[::-1][:10]
    print(f"  Top 10 features ({target}):")
    for i, idx in enumerate(top):
        print(f"    {i + 1:2d}. {feat_cols[idx]:30s} {mean_shap[idx]:>12,.0f}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("🚀 DATATHON 2026 — Forecasting v5 (Multi-seed + Ratio fix + Log-transform)")
    print("=" * 60)

    sales = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["Date"])
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"])
    test_dates = sample_sub["Date"].sort_values().values
    print(f"Train: {len(sales)} rows ({sales['Date'].min().date()} → {sales['Date'].max().date()})")
    print(f"Test:  {len(test_dates)} days ({pd.Timestamp(test_dates[0]).date()} → {pd.Timestamp(test_dates[-1]).date()})")

    # ── Revenue ──
    rev_models, rev_feats, rev_df, rev_log = train_models(sales, "Revenue", use_log=True)
    rev_preds = predict_recursive_v5(
        rev_models, sales, test_dates, rev_feats, "Revenue",
        use_log=rev_log, dampen_alpha=0.20,
    )

    # ── COGS ──
    cogs_models, cogs_feats, cogs_df, cogs_log = train_models(sales, "COGS", use_log=True)
    cogs_preds = predict_recursive_v5(
        cogs_models, sales, test_dates, cogs_feats, "COGS",
        use_log=cogs_log, dampen_alpha=0.20,
    )

    # ── SHAP ──
    do_shap(rev_models, rev_df, rev_feats, "Revenue")
    do_shap(cogs_models, cogs_df, cogs_feats, "COGS")

    # ── Build submission ──
    sub = sample_sub[["Date"]].copy()
    sub["Revenue"] = sub["Date"].map(rev_preds)
    sub["COGS"] = sub["Date"].map(cogs_preds)

    # Fix COGS/Revenue ratio
    sub = fix_cogs_ratio(sub, sales, blend_weight=0.4)

    # Ensure no negatives
    sub["Revenue"] = sub["Revenue"].clip(lower=0)
    sub["COGS"] = sub["COGS"].clip(lower=0)

    assert sub["Revenue"].notna().all(), "Missing Revenue predictions!"
    assert sub["COGS"].notna().all(), "Missing COGS predictions!"

    sub.to_csv(OUT_DIR / "submission.csv", index=False)

    # ── Diagnostics ──
    mae_r = np.abs(sub["Revenue"] - sample_sub["Revenue"]).mean()
    mae_c = np.abs(sub["COGS"] - sample_sub["COGS"]).mean()
    ratio = (sub["COGS"] / sub["Revenue"]).mean()

    print(f"\n{'=' * 60}")
    print(f"✅ submission/submission.csv ({len(sub)} rows)")
    print(f"  Revenue: mean={sub['Revenue'].mean():,.0f}, std={sub['Revenue'].std():,.0f}")
    print(f"  COGS:    mean={sub['COGS'].mean():,.0f}, std={sub['COGS'].std():,.0f}")
    print(f"  COGS/Revenue ratio: {ratio:.3f} (train: {(sales['COGS']/sales['Revenue']).mean():.3f})")
    print(f"  MAE vs sample: Revenue={mae_r:,.0f}, COGS={mae_c:,.0f}")

    # Monthly breakdown
    sub["month"] = sub["Date"].dt.month
    sample_sub["month_tmp"] = sample_sub["Date"].dt.month
    print(f"\n  Monthly MAE (Revenue):")
    for m in sorted(sub["month"].unique()):
        s = sub[sub["month"] == m]
        sp = sample_sub[sample_sub["month_tmp"] == m]
        rev_mae = np.abs(s["Revenue"].values - sp["Revenue"].values).mean()
        print(f"    Month {m:2d}: MAE={rev_mae:>12,.0f}  Pred={s['Revenue'].mean():>12,.0f}  Sample={sp['Revenue'].mean():>12,.0f}")

    # Drift check
    n = len(sub)
    e1 = np.abs(sub["Revenue"].iloc[:100] - sample_sub["Revenue"].iloc[:100]).mean()
    e2 = np.abs(sub["Revenue"].iloc[-100:] - sample_sub["Revenue"].iloc[-100:]).mean()
    print(f"\n  Drift check: early_100={e1:,.0f}, late_100={e2:,.0f}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    for ax, col, c in zip(axes, ["Revenue", "COGS"], ["#FF6B6B", "#4ECDC4"]):
        ax.plot(sales["Date"].tail(365), sales[col].tail(365), color="gray", alpha=0.5, label="Train (last yr)")
        ax.plot(sub["Date"], sub[col], color=c, lw=2, label="v5 Prediction")
        ax.plot(sample_sub["Date"], sample_sub[col], "k--", alpha=0.4, label="Sample Submission")
        ax.set_title(f"{col} Forecast (v5)", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RPT_DIR / "forecast_v5.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: reports/forecast_v5.png")

    # Save detailed results
    results = {
        "version": "v5",
        "mae_revenue": float(mae_r),
        "mae_cogs": float(mae_c),
        "cogs_ratio": float(ratio),
        "revenue_mean": float(sub["Revenue"].mean()),
        "cogs_mean": float(sub["COGS"].mean()),
        "n_models": len(rev_models),
        "improvements": [
            "recent-year sample weighting (decay=0.85)",
            "log1p target transform",
            "multi-seed ensemble (3x3=9 models)",
            "COGS/Revenue ratio blending",
            "adaptive drift correction (alpha=0.20)",
            "interaction features",
            "EWM features",
        ],
    }
    with open(RPT_DIR / "results_v5.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: reports/results_v5.json")

    print(f"\n{'=' * 60}")
    print("🏁 Pipeline v5 complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
