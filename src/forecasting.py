"""
Datathon 2026 — Revenue & COGS Forecasting Pipeline (Phase 4).

End-to-end pipeline:
  1. Load & prepare data
  2. Feature engineering
  3. Train multiple models (XGBoost, LightGBM, CatBoost)
  4. TimeSeriesSplit cross-validation
  5. Ensemble & final prediction
  6. SHAP explainability
  7. Generate submission.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgbm
import catboost as cb
import shap
import json

from feature_engineering import build_feature_dataframe, create_time_features, create_holiday_features

RANDOM_SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "submission"
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(RANDOM_SEED)


# ── Helpers ──────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  {label:20s} MAE={mae:,.0f}  RMSE={rmse:,.0f}  R²={r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def get_feature_cols(df, target_cols=["Revenue", "COGS"], exclude_cols=["Date", "margin_ratio"]):
    """Return list of feature columns (exclude targets, date, etc.)."""
    drop = set(target_cols + exclude_cols)
    return [c for c in df.columns if c not in drop and df[c].dtype in ["float64", "int64", "int32", "float32"]]


# ── Step 1: Load & Build Features ────────────────────────────────────────────

def load_and_prepare():
    print("=" * 60)
    print("STEP 1: Loading data & building features")
    print("=" * 60)
    sales = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["Date"])
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"])

    print(f"  Train: {sales.shape[0]} days ({sales['Date'].min().date()} → {sales['Date'].max().date()})")
    print(f"  Test:  {sample_sub.shape[0]} days ({sample_sub['Date'].min().date()} → {sample_sub['Date'].max().date()})")

    df = build_feature_dataframe(sales, include_external=True, data_dir=DATA_DIR)

    # Drop rows with NaN from lag features (first ~365 rows)
    n_before = len(df)
    feature_cols = get_feature_cols(df)
    df = df.dropna(subset=[c for c in feature_cols if "lag_365" in c]).reset_index(drop=True)
    print(f"  Dropped {n_before - len(df)} rows with NaN lags → {len(df)} rows remain")

    return df, sample_sub, feature_cols


# ── Step 2: Cross-Validation ─────────────────────────────────────────────────

def cross_validate_models(df, feature_cols, target="Revenue", n_splits=5):
    print(f"\n{'=' * 60}")
    print(f"STEP 2: Cross-validation for {target}")
    print(f"{'=' * 60}")

    X = df[feature_cols].values
    y = df[target].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {}

    models_config = {
        "XGBoost": xgb.XGBRegressor(
            n_estimators=1000, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=RANDOM_SEED, n_jobs=-1,
            early_stopping_rounds=50,
        ),
        "LightGBM": lgbm.LGBMRegressor(
            n_estimators=1000, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1,
        ),
        "CatBoost": cb.CatBoostRegressor(
            iterations=1000, depth=6, learning_rate=0.05,
            l2_leaf_reg=3.0, random_seed=RANDOM_SEED,
            verbose=0, early_stopping_rounds=50,
        ),
    }

    for name, model in models_config.items():
        print(f"\n  ── {name} ──")
        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            if name == "XGBoost":
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            elif name == "LightGBM":
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
            elif name == "CatBoost":
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

            preds = model.predict(X_val)
            preds = np.maximum(preds, 0)  # Revenue/COGS can't be negative
            m = evaluate(y_val, preds, f"Fold {fold+1}")
            fold_metrics.append(m)

        avg = {k: np.mean([f[k] for f in fold_metrics]) for k in fold_metrics[0]}
        std = {k: np.std([f[k] for f in fold_metrics]) for k in fold_metrics[0]}
        print(f"  → Avg MAE={avg['mae']:,.0f}±{std['mae']:,.0f}  "
              f"RMSE={avg['rmse']:,.0f}±{std['rmse']:,.0f}  "
              f"R²={avg['r2']:.4f}±{std['r2']:.4f}")
        results[name] = {"avg": avg, "std": std}

    return results


# ── Step 3: Train Final Models ────────────────────────────────────────────────

def train_final_models(df, feature_cols, target="Revenue"):
    print(f"\n{'=' * 60}")
    print(f"STEP 3: Training final models on full train set — {target}")
    print(f"{'=' * 60}")

    X = df[feature_cols].values
    y = df[target].values

    models = {}

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=1500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    xgb_model.fit(X, y, verbose=False)
    models["XGBoost"] = xgb_model

    # LightGBM
    lgb_model = lgbm.LGBMRegressor(
        n_estimators=1500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1,
    )
    lgb_model.fit(X, y)
    models["LightGBM"] = lgb_model

    # CatBoost
    cb_model = cb.CatBoostRegressor(
        iterations=1500, depth=6, learning_rate=0.03,
        l2_leaf_reg=3.0, random_seed=RANDOM_SEED, verbose=0,
    )
    cb_model.fit(X, y)
    models["CatBoost"] = cb_model

    print("  ✅ All models trained.")
    return models


# ── Step 4: Recursive Prediction for Test Period ──────────────────────────────

def predict_test_recursive(models, train_df, sample_sub, feature_cols, target="Revenue"):
    """
    Predict the test period day-by-day, updating lag/rolling features recursively.
    For external features that are unavailable in the test period, use last known values.
    """
    print(f"\n  Predicting {target} for test period (recursive)...")

    # Start with the full training data and append test dates one by one
    full_df = train_df.copy()

    test_dates = sample_sub["Date"].sort_values().values
    predictions = {}

    for date in test_dates:
        # Create a new row for this date
        new_row = pd.DataFrame({"Date": [pd.Timestamp(date)], target: [np.nan]})
        if target == "Revenue":
            new_row["COGS"] = np.nan
        else:
            new_row["Revenue"] = np.nan

        # Append to full_df and rebuild features for the last row
        full_df = pd.concat([full_df[["Date", "Revenue", "COGS"]], new_row], ignore_index=True)
        full_df = full_df.sort_values("Date").reset_index(drop=True)

        # Rebuild features (only need the last row but need full history for lags)
        featured = build_feature_dataframe(full_df, include_external=False, data_dir=DATA_DIR)

        # Get the last row's features
        last_row = featured.iloc[-1:]
        X_pred = last_row[feature_cols].values

        # Handle NaN in features (use 0 as fallback)
        X_pred = np.nan_to_num(X_pred, nan=0.0)

        # Ensemble prediction (average of 3 models)
        preds = []
        for name, model in models.items():
            p = model.predict(X_pred)[0]
            preds.append(max(p, 0))  # floor at 0

        ensemble_pred = np.mean(preds)
        predictions[pd.Timestamp(date)] = ensemble_pred

        # Update the target value in full_df for future lags
        full_df.loc[full_df["Date"] == pd.Timestamp(date), target] = ensemble_pred

    return predictions


def predict_test_batch(models, train_df, sample_sub, feature_cols, target="Revenue"):
    """
    Batch prediction: combine train+test, build all features (incl. external),
    forward-fill external signals into test period, then predict.
    """
    print(f"\n  Predicting {target} for test period (batch)...")

    # Build combined DataFrame with NaN targets for test dates
    test_dates_df = sample_sub[["Date"]].copy()
    test_dates_df["Revenue"] = np.nan
    test_dates_df["COGS"] = np.nan

    combined = pd.concat([
        train_df[["Date", "Revenue", "COGS"]],
        test_dates_df
    ], ignore_index=True).sort_values("Date").reset_index(drop=True)

    # Build features WITH external signals (they'll be NaN for test dates → ffill)
    featured = build_feature_dataframe(combined, include_external=True, data_dir=DATA_DIR)

    # Forward-fill all external columns into test period
    external_cols = [c for c in feature_cols if c in featured.columns]
    for c in external_cols:
        featured[c] = featured[c].ffill()

    # Get test rows
    test_mask = featured["Date"].isin(sample_sub["Date"])
    test_featured = featured[test_mask].copy()

    # Ensure all feature_cols exist (fill missing with 0)
    for c in feature_cols:
        if c not in test_featured.columns:
            test_featured[c] = 0.0

    X_test = test_featured[feature_cols].values
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Ensemble
    all_preds = []
    for name, model in models.items():
        p = model.predict(X_test)
        p = np.maximum(p, 0)
        all_preds.append(p)

    ensemble = np.mean(all_preds, axis=0)

    predictions = dict(zip(test_featured["Date"], ensemble))
    return predictions


# ── Step 5: SHAP Explainability ───────────────────────────────────────────────

def compute_shap(model, X, feature_names, target="Revenue", top_n=20):
    print(f"\n{'=' * 60}")
    print(f"STEP 5: SHAP analysis for {target}")
    print(f"{'=' * 60}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      max_display=top_n, show=False)
    plt.title(f"SHAP Feature Importance — {target}", fontsize=14)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"shap_summary_{target.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: reports/shap_summary_{target.lower()}.png")

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      plot_type="bar", max_display=top_n, show=False)
    plt.title(f"SHAP Feature Importance (Bar) — {target}", fontsize=14)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"shap_bar_{target.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: reports/shap_bar_{target.lower()}.png")

    # Top features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
    print(f"\n  Top {top_n} features by mean |SHAP|:")
    for i, idx in enumerate(top_idx):
        print(f"    {i+1:2d}. {feature_names[idx]:30s}  {mean_abs_shap[idx]:>12,.0f}")

    return shap_values


# ── Step 6: Generate Submission ───────────────────────────────────────────────

def generate_submission(rev_preds, cogs_preds, sample_sub):
    print(f"\n{'=' * 60}")
    print("STEP 6: Generating submission.csv")
    print(f"{'=' * 60}")

    submission = sample_sub[["Date"]].copy()
    submission["Revenue"] = submission["Date"].map(rev_preds)
    submission["COGS"] = submission["Date"].map(cogs_preds)

    # Sanity checks
    assert submission["Revenue"].notna().all(), "Missing Revenue predictions!"
    assert submission["COGS"].notna().all(), "Missing COGS predictions!"
    assert (submission["Revenue"] >= 0).all(), "Negative Revenue!"
    assert (submission["COGS"] >= 0).all(), "Negative COGS!"

    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    print(f"  ✅ Saved: submission/submission.csv ({len(submission)} rows)")
    print(f"  Revenue range: {submission['Revenue'].min():,.0f} — {submission['Revenue'].max():,.0f}")
    print(f"  COGS range:    {submission['COGS'].min():,.0f} — {submission['COGS'].max():,.0f}")
    return submission


# ── Visualization helpers ─────────────────────────────────────────────────────

def plot_cv_results(cv_results, target="Revenue"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["mae", "rmse", "r2"]
    titles = ["MAE ↓", "RMSE ↓", "R² ↑"]

    for ax, metric, title in zip(axes, metrics, titles):
        names = list(cv_results.keys())
        avgs = [cv_results[n]["avg"][metric] for n in names]
        stds = [cv_results[n]["std"][metric] for n in names]
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        bars = ax.bar(names, avgs, yerr=stds, capsize=5, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(metric.upper())
        for bar, v in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{v:,.0f}" if metric != "r2" else f"{v:.4f}",
                    ha="center", va="bottom", fontsize=10)

    plt.suptitle(f"Model Comparison — {target}", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"cv_comparison_{target.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: reports/cv_comparison_{target.lower()}.png")


def plot_predictions(submission, sample_sub, train_df):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=False)

    for ax, col, color in zip(axes, ["Revenue", "COGS"], ["#FF6B6B", "#4ECDC4"]):
        # Train
        ax.plot(train_df["Date"].tail(365), train_df[col].tail(365),
                label="Train (last year)", alpha=0.6, color="gray", linewidth=1)
        # Prediction
        ax.plot(submission["Date"], submission[col],
                label="Prediction", color=color, linewidth=2)
        # Sample sub (baseline)
        ax.plot(sample_sub["Date"], sample_sub[col],
                label="Sample Submission", color="black", linewidth=1, linestyle="--", alpha=0.5)
        ax.set_title(f"{col} Forecast", fontsize=14, fontweight="bold")
        ax.set_ylabel(col)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "forecast_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: reports/forecast_results.png")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("🚀 DATATHON 2026 — Revenue & COGS Forecasting Pipeline")
    print("=" * 60)

    # Step 1: Load & prepare
    df, sample_sub, _ = load_and_prepare()
    feature_cols = get_feature_cols(df)
    print(f"  Features: {len(feature_cols)} columns")

    # Separate feature types for diagnostics
    lag_cols = [c for c in feature_cols if "lag_" in c or "rmean_" in c or "rstd_" in c
                or "rmin_" in c or "rmax_" in c or "exp_" in c or "margin_ratio_lag" in c]
    non_lag_cols = [c for c in feature_cols if c not in lag_cols]
    print(f"  Lag-dependent features: {len(lag_cols)}")
    print(f"  Lag-independent features: {len(non_lag_cols)}")

    # Step 2: Cross-validate (all features — shows best achievable performance)
    rev_cv = cross_validate_models(df, feature_cols, target="Revenue")
    plot_cv_results(rev_cv, "Revenue")
    cogs_cv = cross_validate_models(df, feature_cols, target="COGS")
    plot_cv_results(cogs_cv, "COGS")

    # Step 3 & 4: Train final models on NON-LAG features for test prediction
    # (lag features are NaN/stale for most test rows in batch mode)
    print(f"\n{'=' * 60}")
    print("Using non-lag features for test prediction (no data leakage)")
    print(f"{'=' * 60}")
    rev_models = train_final_models(df, non_lag_cols, target="Revenue")
    cogs_models = train_final_models(df, non_lag_cols, target="COGS")

    rev_preds = predict_test_batch(rev_models, df, sample_sub, non_lag_cols, target="Revenue")
    cogs_preds = predict_test_batch(cogs_models, df, sample_sub, non_lag_cols, target="COGS")

    # Step 5: SHAP (on non-lag model)
    X_train = df[non_lag_cols].values
    compute_shap(rev_models["XGBoost"], X_train, non_lag_cols, target="Revenue")
    compute_shap(cogs_models["XGBoost"], X_train, non_lag_cols, target="COGS")

    # Step 6: Generate submission
    submission = generate_submission(rev_preds, cogs_preds, sample_sub)

    # Plots
    plot_predictions(submission, sample_sub, df)

    # Save CV results
    all_results = {"Revenue": rev_cv, "COGS": cogs_cv}
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return obj

    with open(REPORT_DIR / "cv_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\n{'=' * 60}")
    print("🏁 Pipeline complete!")
    print(f"{'=' * 60}")
    print(f"  📁 submission/submission.csv  — ready for Kaggle upload")
    print(f"  📊 reports/                   — CV results, SHAP plots, forecast plots")


if __name__ == "__main__":
    main()

