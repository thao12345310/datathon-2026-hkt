"""
Datathon 2026 — Forecasting v3: Optuna tuning + weighted ensemble + joint prediction.
"""
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgbm
import catboost as cb
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
import shap, json

SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR  = Path(__file__).resolve().parent.parent / "submission"
RPT_DIR  = Path(__file__).resolve().parent.parent / "reports"
RPT_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True)
np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING v3
# ═══════════════════════════════════════════════════════════════════════════════

def build_features(df, target="Revenue"):
    df = df.sort_values("Date").reset_index(drop=True).copy()
    dt = df["Date"]

    # ── Calendar ──
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

    # ── Cyclical + Fourier ──
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    df["dow_sin"]   = np.sin(2*np.pi*df["day_of_week"]/7)
    df["dow_cos"]   = np.cos(2*np.pi*df["day_of_week"]/7)
    for k in [1,2,3,4,6]:
        df[f"yr_sin_{k}"] = np.sin(2*np.pi*k*df["day_of_year"]/365.25)
        df[f"yr_cos_{k}"] = np.cos(2*np.pi*k*df["day_of_year"]/365.25)
    for k in [1,2,3]:
        df[f"wk_sin_{k}"] = np.sin(2*np.pi*k*df["day_of_week"]/7)
        df[f"wk_cos_{k}"] = np.cos(2*np.pi*k*df["day_of_week"]/7)

    # ── Vietnamese seasons ──
    m, d = df["month"], df["day_of_month"]
    df["is_tet"]   = m.isin([1,2]).astype(int)
    df["is_1111"]  = ((m==11)&(d>=9)&(d<=13)).astype(int)
    df["is_1212"]  = ((m==12)&(d>=10)&(d<=14)).astype(int)
    df["is_yearend"] = ((m==12)&(d>=20)).astype(int)
    df["is_bf"]    = ((m==11)&(d>=22)&(d<=28)).astype(int)
    df["is_99"]    = ((m==9)&(d>=7)&(d<=11)).astype(int)
    df["is_1010"]  = ((m==10)&(d>=8)&(d<=12)).astype(int)
    df["is_holiday"] = (((m==4)&(d==30))|((m==5)&(d==1))|((m==9)&(d==2))).astype(int)

    # ── Lag features ──
    for lag in [1,2,3,5,7,14,21,28,30,60,90,180,365]:
        df[f"lag_{lag}"] = df[target].shift(lag)

    # ── Rolling stats ──
    sh = df[target].shift(1)
    for w in [3,7,14,30,60,90]:
        df[f"rm_{w}"]   = sh.rolling(w, min_periods=1).mean()
        df[f"rs_{w}"]   = sh.rolling(w, min_periods=2).std()
        df[f"rmin_{w}"] = sh.rolling(w, min_periods=1).min()
        df[f"rmax_{w}"] = sh.rolling(w, min_periods=1).max()
        df[f"rmed_{w}"] = sh.rolling(w, min_periods=1).median()

    # ── Expanding ──
    df["exp_mean"] = sh.expanding(min_periods=1).mean()

    # ── Momentum / rate-of-change ──
    df["mom_7"]  = df[target] / df[target].shift(7) - 1
    df["mom_30"] = df[target] / df[target].shift(30) - 1
    df["mom_365"]= df[target] / df[target].shift(365) - 1

    # ── Seasonal index: same-day-of-year avg from history ──
    yearly_avg = df.groupby("day_of_year")[target].transform("mean")
    overall_avg = df[target].mean()
    df["seasonal_idx"] = yearly_avg / overall_avg if overall_avg != 0 else 1.0

    # ── Same-weekday-in-same-month average ──
    df["dow_month_avg"] = df.groupby(["month","day_of_week"])[target].transform("mean")

    # ── Trend index ──
    df["trend_idx"] = np.arange(len(df))

    # ── Ratio features ──
    df["lag1_to_rm7"]  = df["lag_1"] / df["rm_7"].replace(0, np.nan)
    df["lag1_to_rm30"] = df["lag_1"] / df["rm_30"].replace(0, np.nan)

    return df

def get_feat_cols(df):
    drop = {"Date","Revenue","COGS","mom_7","mom_30","mom_365"}
    return [c for c in df.columns if c not in drop
            and df[c].dtype in ("float64","int64","int32","float32")]


# ═══════════════════════════════════════════════════════════════════════════════
# OPTUNA HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════════════════════

def optuna_tune(X, y, n_trials=40):
    """Tune XGBoost hyperparameters with Optuna."""
    print(f"  Optuna tuning ({n_trials} trials)...")
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 800, 2500),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("lr", 0.005, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample", 0.5, 0.9),
            "reg_alpha": trial.suggest_float("alpha", 0.01, 10, log=True),
            "reg_lambda": trial.suggest_float("lambda", 0.1, 10, log=True),
            "min_child_weight": trial.suggest_int("mcw", 3, 30),
            "random_state": SEED, "n_jobs": -1, "early_stopping_rounds": 50,
        }
        maes = []
        for ti, vi in tscv.split(X):
            m = xgb.XGBRegressor(**params)
            m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])], verbose=False)
            p = np.maximum(m.predict(X[vi]), 0)
            maes.append(mean_absolute_error(y[vi], p))
        return np.mean(maes)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print(f"  Best MAE: {study.best_value:,.0f}")
    print(f"  Best params: depth={best.get('max_depth')}, lr={best.get('lr'):.4f}, "
          f"n_est={best.get('n_estimators')}")
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING WITH OPTIMIZED ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

def train_optimized(sales, target="Revenue", tune=True):
    print(f"\n{'='*60}\nTraining for {target}\n{'='*60}")

    df = build_features(sales.copy(), target=target)
    df = df.dropna(subset=["lag_365"]).reset_index(drop=True)
    feat_cols = get_feat_cols(df)
    X, y = df[feat_cols].values, df[target].values
    print(f"  {len(df)} rows, {len(feat_cols)} features")

    # Optuna tune XGBoost
    if tune:
        best_xgb = optuna_tune(X, y, n_trials=40)
    else:
        best_xgb = {"n_estimators":2000,"max_depth":7,"lr":0.02,
                     "subsample":0.8,"colsample":0.7,"alpha":0.5,
                     "lambda":2.0,"mcw":10}

    # Train 3 models
    configs = {
        "XGB": xgb.XGBRegressor(
            n_estimators=best_xgb.get("n_estimators",2000),
            max_depth=best_xgb.get("max_depth",7),
            learning_rate=best_xgb.get("lr",0.02),
            subsample=best_xgb.get("subsample",0.8),
            colsample_bytree=best_xgb.get("colsample",0.7),
            reg_alpha=best_xgb.get("alpha",0.5),
            reg_lambda=best_xgb.get("lambda",2.0),
            min_child_weight=best_xgb.get("mcw",10),
            random_state=SEED, n_jobs=-1),
        "LGB": lgbm.LGBMRegressor(
            n_estimators=2000, max_depth=7, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=2.0,
            min_child_weight=10, random_state=SEED, n_jobs=-1, verbosity=-1),
        "CB": cb.CatBoostRegressor(
            iterations=2000, depth=7, learning_rate=0.02,
            l2_leaf_reg=5.0, random_seed=SEED, verbose=0),
    }

    # CV with optimized weights
    tscv = TimeSeriesSplit(n_splits=5)
    val_preds_all = {n: [] for n in configs}
    val_true_all = []

    models = {}
    for name, model in configs.items():
        maes = []
        for fold, (ti, vi) in enumerate(tscv.split(X)):
            m = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
            if name == "XGB":
                m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])], verbose=False)
            elif name == "LGB":
                m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])])
            else:
                m.fit(X[ti], y[ti], eval_set=(X[vi], y[vi]), verbose=False)
            p = np.maximum(m.predict(X[vi]), 0)
            maes.append(mean_absolute_error(y[vi], p))
            if fold == tscv.get_n_splits() - 1:  # last fold for weight optimization
                val_preds_all[name] = p
                val_true_all = y[vi]
        print(f"  {name}: MAE={np.mean(maes):,.0f}±{np.std(maes):,.0f}")

    # Optimize ensemble weights on last fold
    best_w, best_mae = [1/3,1/3,1/3], float("inf")
    names = list(configs.keys())
    for w0 in np.arange(0.1, 0.8, 0.05):
        for w1 in np.arange(0.1, 0.8-w0, 0.05):
            w2 = 1.0 - w0 - w1
            if w2 < 0.05: continue
            blend = w0*val_preds_all[names[0]] + w1*val_preds_all[names[1]] + w2*val_preds_all[names[2]]
            mae = mean_absolute_error(val_true_all, blend)
            if mae < best_mae:
                best_mae = mae
                best_w = [w0, w1, w2]
    weights = dict(zip(names, best_w))
    print(f"  Ensemble weights: {', '.join(f'{k}={v:.2f}' for k,v in weights.items())}")
    print(f"  Ensemble MAE (last fold): {best_mae:,.0f}")

    # Train final models on full data
    for name, model in configs.items():
        if name == "XGB":
            model.fit(X, y, verbose=False)
        elif name == "LGB":
            model.fit(X, y)
        else:
            model.fit(X, y, verbose=False)
        models[name] = model
    print("  ✅ Final models trained")

    return models, weights, feat_cols, df


# ═══════════════════════════════════════════════════════════════════════════════
# RECURSIVE PREDICTION WITH WEIGHTED ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

def predict_recursive(models, weights, train_sales, test_dates, feat_cols,
                      target="Revenue", other_preds=None):
    """
    Recursive day-by-day prediction with weighted ensemble.
    other_preds: dict of {date: value} for the other target (e.g., Revenue when predicting COGS)
    """
    print(f"\n  Recursive prediction for {target} ({len(test_dates)} days)...")
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

        # Weighted ensemble
        pred = sum(weights[n] * max(m.predict(X)[0], 0) for n, m in models.items())
        predictions.append(pred)

        new_row[target] = pred
        history = pd.concat([history, new_row], ignore_index=True)

        if (i+1) % 100 == 0:
            print(f"    Day {i+1}/{len(test_dates)}...")

    print(f"    ✅ Range: {min(predictions):,.0f}—{max(predictions):,.0f}, std={np.std(predictions):,.0f}")
    return dict(zip(test_dates, predictions))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("🚀 DATATHON 2026 — Forecasting v3 (Optuna + Weighted Ensemble)")
    print("="*60)

    sales = pd.read_csv(DATA_DIR/"sales.csv", parse_dates=["Date"])
    sample_sub = pd.read_csv(DATA_DIR/"sample_submission.csv", parse_dates=["Date"])
    test_dates = sample_sub["Date"].sort_values().values
    print(f"Train: {len(sales)} | Test: {len(test_dates)}")

    # Revenue
    rev_m, rev_w, rev_f, rev_df = train_optimized(sales, "Revenue", tune=True)
    rev_preds = predict_recursive(rev_m, rev_w, sales, test_dates, rev_f, "Revenue")

    # COGS
    cogs_m, cogs_w, cogs_f, cogs_df = train_optimized(sales, "COGS", tune=True)
    cogs_preds = predict_recursive(cogs_m, cogs_w, sales, test_dates, cogs_f, "COGS")

    # SHAP
    for models, df, feats, tgt in [(rev_m,rev_df,rev_f,"Revenue"),(cogs_m,cogs_df,cogs_f,"COGS")]:
        X = df[feats].values[:500]
        sv = shap.TreeExplainer(models["XGB"]).shap_values(X)
        plt.figure(figsize=(10,8))
        shap.summary_plot(sv, X, feature_names=feats, max_display=15, show=False)
        plt.title(f"SHAP — {tgt}"); plt.tight_layout()
        plt.savefig(RPT_DIR/f"shap_v3_{tgt.lower()}.png", dpi=150, bbox_inches="tight"); plt.close()

    # Submission
    sub = sample_sub[["Date"]].copy()
    sub["Revenue"] = sub["Date"].map(rev_preds)
    sub["COGS"] = sub["Date"].map(cogs_preds)

    # Post-processing: ensure COGS < Revenue (business constraint)
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

    # Plot
    fig, axes = plt.subplots(2,1,figsize=(16,10))
    for ax,col,c in zip(axes,["Revenue","COGS"],["#FF6B6B","#4ECDC4"]):
        ax.plot(sales["Date"].tail(365),sales[col].tail(365),color="gray",alpha=0.5,label="Train")
        ax.plot(sub["Date"],sub[col],color=c,lw=2,label="v3 Prediction")
        ax.plot(sample_sub["Date"],sample_sub[col],"k--",alpha=0.4,label="Sample")
        ax.set_title(f"{col}",fontsize=14,fontweight="bold"); ax.legend(); ax.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(RPT_DIR/"forecast_v3.png",dpi=150,bbox_inches="tight"); plt.close()
    print(f"  Saved: reports/forecast_v3.png")

if __name__ == "__main__":
    main()
