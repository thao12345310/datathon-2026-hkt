"""
Datathon 2026 — Forecasting v4: v2 base + anti-drift + seasonal correction.
Changes from v2: seasonal index features, drift dampening, monthly recalibration.
"""
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgbm
import catboost as cb
import shap, json

SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR  = Path(__file__).resolve().parent.parent / "submission"
RPT_DIR  = Path(__file__).resolve().parent.parent / "reports"
RPT_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True)
np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════════════════════
# FEATURES — same as v2 + seasonal index + Fourier
# ═══════════════════════════════════════════════════════════════════════════

def build_features(df, target="Revenue"):
    df = df.sort_values("Date").reset_index(drop=True).copy()
    dt = df["Date"]

    # Calendar
    df["year"]         = dt.dt.year
    df["month"]        = dt.dt.month
    df["quarter"]      = dt.dt.quarter
    df["day_of_week"]  = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["day_of_year"]  = dt.dt.dayofyear
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"]   = dt.dt.dayofweek.isin([5, 6]).astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"]   = dt.dt.is_month_end.astype(int)

    # Cyclical
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    df["dow_sin"]   = np.sin(2*np.pi*df["day_of_week"]/7)
    df["dow_cos"]   = np.cos(2*np.pi*df["day_of_week"]/7)
    df["doy_sin"]   = np.sin(2*np.pi*df["day_of_year"]/365.25)
    df["doy_cos"]   = np.cos(2*np.pi*df["day_of_year"]/365.25)

    # Fourier (annual seasonality)
    for k in [1, 2, 3, 4, 6]:
        df[f"fourier_sin_{k}"] = np.sin(2*np.pi*k*df["day_of_year"]/365.25)
        df[f"fourier_cos_{k}"] = np.cos(2*np.pi*k*df["day_of_year"]/365.25)
    for k in [1, 2, 3]:
        df[f"wk_sin_{k}"] = np.sin(2*np.pi*k*df["day_of_week"]/7)
        df[f"wk_cos_{k}"] = np.cos(2*np.pi*k*df["day_of_week"]/7)

    # Vietnamese seasons
    m, d = df["month"], df["day_of_month"]
    df["is_tet"]     = m.isin([1,2]).astype(int)
    df["is_1111"]    = ((m==11)&(d>=9)&(d<=13)).astype(int)
    df["is_1212"]    = ((m==12)&(d>=10)&(d<=14)).astype(int)
    df["is_yearend"] = ((m==12)&(d>=20)).astype(int)
    df["is_bf"]      = ((m==11)&(d>=22)&(d<=28)).astype(int)
    df["is_99"]      = ((m==9)&(d>=7)&(d<=11)).astype(int)
    df["is_1010"]    = ((m==10)&(d>=8)&(d<=12)).astype(int)
    df["is_holiday"] = (((m==4)&(d==30))|((m==5)&(d==1))|((m==9)&(d==2))).astype(int)

    # Lag features
    for lag in [1, 2, 3, 5, 7, 14, 21, 28, 30, 60, 90, 180, 365]:
        df[f"lag_{lag}"] = df[target].shift(lag)

    # Rolling stats
    sh = df[target].shift(1)
    for w in [7, 14, 30, 60, 90]:
        df[f"rmean_{w}"] = sh.rolling(w, min_periods=1).mean()
        df[f"rstd_{w}"]  = sh.rolling(w, min_periods=2).std()
        df[f"rmin_{w}"]  = sh.rolling(w, min_periods=1).min()
        df[f"rmax_{w}"]  = sh.rolling(w, min_periods=1).max()

    # Expanding mean
    df["exp_mean"] = sh.expanding(min_periods=1).mean()

    # Trend
    df["trend_idx"] = np.arange(len(df))

    # Seasonal index: historical avg for this day-of-year / overall avg
    # (only computed on non-NaN target rows)
    valid = df[target].notna()
    if valid.sum() > 30:
        doy_avg = df.loc[valid].groupby("day_of_year")[target].mean()
        overall = df.loc[valid, target].mean()
        df["seasonal_idx"] = df["day_of_year"].map(doy_avg) / overall
        # Same weekday-month average
        wm_avg = df.loc[valid].groupby(["month","day_of_week"])[target].mean()
        df["dow_month_avg"] = df.set_index(["month","day_of_week"]).index.map(
            lambda x: wm_avg.get(x, overall))
    else:
        df["seasonal_idx"] = 1.0
        df["dow_month_avg"] = df[target].mean() if valid.any() else 0

    return df


def get_feat_cols(df):
    drop = {"Date", "Revenue", "COGS"}
    return [c for c in df.columns if c not in drop
            and df[c].dtype in ("float64","int64","int32","float32")]


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING — v2 hyperparameters (proven on Kaggle)
# ═══════════════════════════════════════════════════════════════════════════

def train_models(sales, target="Revenue"):
    print(f"\n{'='*60}\nTraining for {target}\n{'='*60}")

    df = build_features(sales.copy(), target=target)
    df = df.dropna(subset=["lag_365"]).reset_index(drop=True)
    feat_cols = get_feat_cols(df)
    X, y = df[feat_cols].values, df[target].values
    print(f"  {len(df)} rows, {len(feat_cols)} features")

    # Same hyperparameters as v2 (proven on Kaggle)
    configs = {
        "XGB": xgb.XGBRegressor(
            n_estimators=2000, max_depth=6, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            min_child_weight=5, random_state=SEED, n_jobs=-1),
        "LGB": lgbm.LGBMRegressor(
            n_estimators=2000, max_depth=6, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            min_child_weight=5, random_state=SEED, n_jobs=-1, verbosity=-1),
        "CB": cb.CatBoostRegressor(
            iterations=2000, depth=6, learning_rate=0.02,
            l2_leaf_reg=3.0, random_seed=SEED, verbose=0),
    }

    # CV
    tscv = TimeSeriesSplit(n_splits=5)
    for name, model in configs.items():
        maes = []
        for ti, vi in tscv.split(X):
            mc = type(model)(**model.get_params()) if hasattr(model,'get_params') else model
            if name == "XGB":
                mc.fit(X[ti], y[ti], eval_set=[(X[vi],y[vi])], verbose=False)
            elif name == "LGB":
                mc.fit(X[ti], y[ti], eval_set=[(X[vi],y[vi])])
            else:
                mc.fit(X[ti], y[ti], eval_set=(X[vi],y[vi]), verbose=False)
            maes.append(mean_absolute_error(y[vi], np.maximum(mc.predict(X[vi]),0)))
        print(f"  {name}: MAE={np.mean(maes):,.0f}±{np.std(maes):,.0f}")

    # Train on full data
    models = {}
    for name, model in configs.items():
        if name == "XGB":
            model.fit(X, y, verbose=False)
        elif name == "LGB":
            model.fit(X, y)
        else:
            model.fit(X, y, verbose=False)
        models[name] = model
    print("  ✅ Final models trained")

    return models, feat_cols, df


# ═══════════════════════════════════════════════════════════════════════════
# RECURSIVE PREDICTION WITH DRIFT DAMPENING
# ═══════════════════════════════════════════════════════════════════════════

def compute_monthly_stats(train_sales, target):
    """Compute historical monthly averages for drift correction."""
    df = train_sales.copy()
    df["month"] = df["Date"].dt.month
    monthly = df.groupby("month")[target].agg(["mean", "std"]).to_dict("index")
    return monthly


def predict_recursive_v4(models, train_sales, test_dates, feat_cols, target="Revenue"):
    """
    Recursive prediction with equal-weight ensemble + drift dampening.
    After each month of predictions, recalibrate towards historical monthly mean.
    """
    print(f"\n  Recursive prediction for {target} ({len(test_dates)} days)...")

    monthly_stats = compute_monthly_stats(train_sales, target)
    history = train_sales[["Date", target]].copy()

    # Pre-compute historical per-day-of-year averages for blending
    train_copy = train_sales.copy()
    train_copy["doy"] = train_copy["Date"].dt.dayofyear
    doy_means = train_copy.groupby("doy")[target].mean().to_dict()
    overall_mean = train_sales[target].mean()

    predictions = []
    dampen_alpha = 0.15  # blend 15% towards historical seasonal pattern

    for i, date in enumerate(test_dates):
        new_row = pd.DataFrame({"Date": [date], target: [np.nan]})
        temp = pd.concat([history, new_row], ignore_index=True)
        featured = build_features(temp, target=target)
        last = featured.iloc[-1:]
        fc = [c for c in feat_cols if c in last.columns]
        X = last[fc].values
        X = np.nan_to_num(X, nan=0.0)

        # Equal-weight ensemble (proven better than weighted on Kaggle)
        raw_pred = np.mean([max(m.predict(X)[0], 0) for m in models.values()])

        # Drift dampening: blend with historical seasonal pattern
        doy = pd.Timestamp(date).dayofyear
        historical_ref = doy_means.get(doy, overall_mean)
        pred = (1 - dampen_alpha) * raw_pred + dampen_alpha * historical_ref

        # Clip to reasonable range (based on historical month)
        month = pd.Timestamp(date).month
        if month in monthly_stats:
            m_mean = monthly_stats[month]["mean"]
            m_std = monthly_stats[month]["std"]
            pred = np.clip(pred, m_mean - 3*m_std, m_mean + 3*m_std)

        pred = max(pred, 0)
        predictions.append(pred)

        new_row[target] = pred
        history = pd.concat([history, new_row], ignore_index=True)

        if (i+1) % 100 == 0:
            print(f"    Day {i+1}/{len(test_dates)}...")

    print(f"    ✅ Range: {min(predictions):,.0f}—{max(predictions):,.0f}, "
          f"std={np.std(predictions):,.0f}")
    return dict(zip(test_dates, predictions))


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("🚀 DATATHON 2026 — Forecasting v4 (v2 base + anti-drift)")
    print("="*60)

    sales = pd.read_csv(DATA_DIR/"sales.csv", parse_dates=["Date"])
    sample_sub = pd.read_csv(DATA_DIR/"sample_submission.csv", parse_dates=["Date"])
    test_dates = sample_sub["Date"].sort_values().values
    print(f"Train: {len(sales)} | Test: {len(test_dates)}")

    # Revenue
    rev_m, rev_f, rev_df = train_models(sales, "Revenue")
    rev_preds = predict_recursive_v4(rev_m, sales, test_dates, rev_f, "Revenue")

    # COGS
    cogs_m, cogs_f, cogs_df = train_models(sales, "COGS")
    cogs_preds = predict_recursive_v4(cogs_m, sales, test_dates, cogs_f, "COGS")

    # SHAP
    for models, df, feats, tgt in [(rev_m,rev_df,rev_f,"Revenue"),(cogs_m,cogs_df,cogs_f,"COGS")]:
        X = df[feats].values[:500]
        sv = shap.TreeExplainer(models["XGB"]).shap_values(X)
        plt.figure(figsize=(10,8))
        shap.summary_plot(sv, X, feature_names=feats, max_display=15, show=False)
        plt.title(f"SHAP — {tgt}"); plt.tight_layout()
        plt.savefig(RPT_DIR/f"shap_v4_{tgt.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Submission
    sub = sample_sub[["Date"]].copy()
    sub["Revenue"] = sub["Date"].map(rev_preds)
    sub["COGS"] = sub["Date"].map(cogs_preds)

    # Business constraint: COGS < Revenue
    mask = sub["COGS"] > sub["Revenue"]
    if mask.any():
        print(f"  ⚠ Fixing {mask.sum()} rows where COGS > Revenue")
        sub.loc[mask, "COGS"] = sub.loc[mask, "Revenue"] * 0.85

    sub.to_csv(OUT_DIR/"submission.csv", index=False)

    mae_r = np.abs(sub["Revenue"] - sample_sub["Revenue"]).mean()
    mae_c = np.abs(sub["COGS"] - sample_sub["COGS"]).mean()

    print(f"\n{'='*60}")
    print(f"✅ submission/submission.csv ({len(sub)} rows)")
    print(f"  Revenue: mean={sub['Revenue'].mean():,.0f}, std={sub['Revenue'].std():,.0f}")
    print(f"  COGS:    mean={sub['COGS'].mean():,.0f}, std={sub['COGS'].std():,.0f}")
    print(f"  MAE vs sample: Rev={mae_r:,.0f}, COGS={mae_c:,.0f}")

    # Error drift check
    n = len(sub)
    e1 = np.abs(sub['Revenue'].iloc[:100]-sample_sub['Revenue'].iloc[:100]).mean()
    e2 = np.abs(sub['Revenue'].iloc[-100:]-sample_sub['Revenue'].iloc[-100:]).mean()
    print(f"  Drift check: early={e1:,.0f}, late={e2:,.0f}")

    # Plot
    fig, axes = plt.subplots(2,1,figsize=(16,10))
    for ax,col,c in zip(axes,["Revenue","COGS"],["#FF6B6B","#4ECDC4"]):
        ax.plot(sales["Date"].tail(365),sales[col].tail(365),color="gray",alpha=0.5,label="Train")
        ax.plot(sub["Date"],sub[col],color=c,lw=2,label="v4 Prediction")
        ax.plot(sample_sub["Date"],sample_sub[col],"k--",alpha=0.4,label="Sample")
        ax.set_title(f"{col}",fontsize=14,fontweight="bold"); ax.legend(); ax.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(RPT_DIR/"forecast_v4.png",dpi=150,bbox_inches="tight"); plt.close()
    print(f"  Saved: reports/forecast_v4.png")

if __name__ == "__main__":
    main()
