"""
Datathon 2026 — Improved Forecasting Pipeline v2.
Key fix: Recursive day-by-day prediction with lag feature updates.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgbm
import catboost as cb
import shap, json

SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "submission"
RPT_DIR = Path(__file__).resolve().parent.parent / "reports"
RPT_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True)
np.random.seed(SEED)

# ── Feature Engineering (vectorized, no external) ────────────────────────────

def build_features(df, target="Revenue"):
    """Build features from Date + target history. No external data needed."""
    df = df.sort_values("Date").reset_index(drop=True).copy()
    dt = df["Date"]

    # Calendar
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

    # Cyclical
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # Fourier terms for seasonality (multiple harmonics)
    for k in [1, 2, 3, 4, 6]:
        df[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * df["day_of_year"] / 365.25)
        df[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * df["day_of_year"] / 365.25)

    # Weekly fourier
    for k in [1, 2, 3]:
        df[f"weekly_sin_{k}"] = np.sin(2 * np.pi * k * df["day_of_week"] / 7)
        df[f"weekly_cos_{k}"] = np.cos(2 * np.pi * k * df["day_of_week"] / 7)

    # Vietnamese seasons
    m, d = df["month"], df["day_of_month"]
    df["is_tet"] = m.isin([1, 2]).astype(int)
    df["is_sale_1111"] = ((m == 11) & (d >= 9) & (d <= 13)).astype(int)
    df["is_sale_1212"] = ((m == 12) & (d >= 10) & (d <= 14)).astype(int)
    df["is_yearend"] = ((m == 12) & (d >= 20)).astype(int)
    df["is_black_friday"] = ((m == 11) & (d >= 22) & (d <= 28)).astype(int)

    # Lag features
    for lag in [1, 2, 3, 7, 14, 21, 28, 30, 60, 90, 365]:
        df[f"lag_{lag}"] = df[target].shift(lag)

    # Rolling features (shifted by 1 to avoid leakage)
    shifted = df[target].shift(1)
    for w in [7, 14, 30, 60, 90]:
        df[f"rmean_{w}"] = shifted.rolling(w, min_periods=1).mean()
        df[f"rstd_{w}"] = shifted.rolling(w, min_periods=2).std()
        df[f"rmin_{w}"] = shifted.rolling(w, min_periods=1).min()
        df[f"rmax_{w}"] = shifted.rolling(w, min_periods=1).max()

    # Expanding
    df["exp_mean"] = shifted.expanding(min_periods=1).mean()

    # Diff features
    df["diff_1"] = df[target].diff(1)
    df["diff_7"] = df[target].diff(7)

    # Trend: linear index
    df["trend_idx"] = np.arange(len(df))

    return df


def get_feat_cols(df):
    drop = {"Date", "Revenue", "COGS", "diff_1", "diff_7"}
    return [c for c in df.columns if c not in drop and df[c].dtype in ("float64","int64","int32","float32")]


# ── Train & CV ───────────────────────────────────────────────────────────────

def train_and_cv(sales, target="Revenue"):
    print(f"\n{'='*60}\nTraining pipeline for {target}\n{'='*60}")

    df = build_features(sales.copy(), target=target)
    df = df.dropna(subset=["lag_365"]).reset_index(drop=True)
    feat_cols = get_feat_cols(df)
    print(f"  {len(df)} rows, {len(feat_cols)} features")

    X = df[feat_cols].values
    y = df[target].values

    # CV
    tscv = TimeSeriesSplit(n_splits=5)
    cv_results = {}

    configs = {
        "XGB": xgb.XGBRegressor(
            n_estimators=1500, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
            min_child_weight=10, random_state=SEED, n_jobs=-1, early_stopping_rounds=50),
        "LGB": lgbm.LGBMRegressor(
            n_estimators=1500, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
            min_child_weight=10, random_state=SEED, n_jobs=-1, verbosity=-1),
        "CB": cb.CatBoostRegressor(
            iterations=1500, depth=7, learning_rate=0.03,
            l2_leaf_reg=5.0, random_seed=SEED, verbose=0, early_stopping_rounds=50),
    }

    for name, model in configs.items():
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
        cv_results[name] = {"mae": np.mean(maes), "r2": np.mean(r2s)}

    # Train final models on full data
    models = {}
    for name, model in configs.items():
        # Re-create without early stopping for full training
        if name == "XGB":
            m = xgb.XGBRegressor(n_estimators=2000, max_depth=7, learning_rate=0.02,
                subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
                min_child_weight=10, random_state=SEED, n_jobs=-1)
        elif name == "LGB":
            m = lgbm.LGBMRegressor(n_estimators=2000, max_depth=7, learning_rate=0.02,
                subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
                min_child_weight=10, random_state=SEED, n_jobs=-1, verbosity=-1)
        else:
            m = cb.CatBoostRegressor(iterations=2000, depth=7, learning_rate=0.02,
                l2_leaf_reg=5.0, random_seed=SEED, verbose=0)
        m.fit(X, y)
        models[name] = m

    print("  ✅ Final models trained")
    return models, feat_cols, df, cv_results


# ── Recursive Prediction ─────────────────────────────────────────────────────

def predict_recursive(models, train_sales, test_dates, feat_cols, target="Revenue"):
    """
    Predict day-by-day, appending each prediction to history
    so lag/rolling features are always up-to-date.
    """
    print(f"\n  Recursive prediction for {target} ({len(test_dates)} days)...")

    # Start with full training history
    history = train_sales[["Date", target]].copy()

    predictions = []
    for i, date in enumerate(test_dates):
        # Append placeholder
        new_row = pd.DataFrame({"Date": [date], target: [np.nan]})
        temp = pd.concat([history, new_row], ignore_index=True)

        # Build features for the entire history + new row
        featured = build_features(temp, target=target)
        last = featured.iloc[-1:]

        # Get features, fill NaN with 0
        X = last[feat_cols].values
        X = np.nan_to_num(X, nan=0.0)

        # Ensemble prediction
        preds = [max(m.predict(X)[0], 0) for m in models.values()]
        pred = np.mean(preds)
        predictions.append(pred)

        # Update history with prediction
        new_row[target] = pred
        history = pd.concat([history, new_row], ignore_index=True)

        if (i + 1) % 50 == 0:
            print(f"    Day {i+1}/{len(test_dates)} done...")

    print(f"    ✅ Done. Range: {min(predictions):,.0f} — {max(predictions):,.0f}, std={np.std(predictions):,.0f}")
    return dict(zip(test_dates, predictions))


# ── SHAP ─────────────────────────────────────────────────────────────────────

def do_shap(model, X, feat_names, target):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X[:500])  # sample for speed
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, X[:500], feature_names=feat_names, max_display=20, show=False)
    plt.title(f"SHAP — {target}", fontsize=14)
    plt.tight_layout()
    plt.savefig(RPT_DIR / f"shap_v2_{target.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: reports/shap_v2_{target.lower()}.png")

    mean_shap = np.abs(sv).mean(axis=0)
    top = np.argsort(mean_shap)[::-1][:10]
    print(f"  Top 10 features ({target}):")
    for i, idx in enumerate(top):
        print(f"    {i+1:2d}. {feat_names[idx]:25s} {mean_shap[idx]:>12,.0f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("🚀 DATATHON 2026 — Forecasting v2 (Recursive)")
    print("=" * 60)

    sales = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["Date"])
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"])
    test_dates = sample_sub["Date"].sort_values().values

    print(f"Train: {len(sales)} days | Test: {len(test_dates)} days")

    # Revenue
    rev_models, rev_feats, rev_df, rev_cv = train_and_cv(sales, "Revenue")
    rev_preds = predict_recursive(rev_models, sales, test_dates, rev_feats, "Revenue")

    # COGS
    cogs_models, cogs_feats, cogs_df, cogs_cv = train_and_cv(sales, "COGS")
    cogs_preds = predict_recursive(cogs_models, sales, test_dates, cogs_feats, "COGS")

    # SHAP
    do_shap(rev_models["XGB"], rev_df[rev_feats].values, rev_feats, "Revenue")
    do_shap(cogs_models["XGB"], cogs_df[cogs_feats].values, cogs_feats, "COGS")

    # Generate submission
    sub = sample_sub[["Date"]].copy()
    sub["Revenue"] = sub["Date"].map(rev_preds)
    sub["COGS"] = sub["Date"].map(cogs_preds)
    assert sub.notna().all().all()
    sub.to_csv(OUT_DIR / "submission.csv", index=False)

    print(f"\n{'='*60}")
    print(f"✅ submission/submission.csv saved ({len(sub)} rows)")
    print(f"  Revenue: mean={sub['Revenue'].mean():,.0f}, std={sub['Revenue'].std():,.0f}")
    print(f"  COGS:    mean={sub['COGS'].mean():,.0f}, std={sub['COGS'].std():,.0f}")

    # Comparison with sample
    mae_r = np.abs(sub["Revenue"] - sample_sub["Revenue"]).mean()
    mae_c = np.abs(sub["COGS"] - sample_sub["COGS"]).mean()
    print(f"  MAE vs sample: Revenue={mae_r:,.0f}, COGS={mae_c:,.0f}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    for ax, col, color in zip(axes, ["Revenue","COGS"], ["#FF6B6B","#4ECDC4"]):
        ax.plot(sales["Date"].tail(365), sales[col].tail(365), color="gray", alpha=0.5, label="Train (last yr)")
        ax.plot(sub["Date"], sub[col], color=color, linewidth=2, label="Prediction v2")
        ax.plot(sample_sub["Date"], sample_sub[col], color="black", linestyle="--", alpha=0.4, label="Sample")
        ax.set_title(f"{col} Forecast v2", fontsize=14, fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RPT_DIR / "forecast_v2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: reports/forecast_v2.png")

    # Save CV
    with open(RPT_DIR / "cv_results_v2.json", "w") as f:
        json.dump({"Revenue": rev_cv, "COGS": cogs_cv}, f, indent=2, default=lambda x: float(x))


if __name__ == "__main__":
    main()
