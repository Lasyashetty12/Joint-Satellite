"""
m4_forecast_xgboost.py - XGBoost "direct" multi-step forecaster on lag/rolling/Fourier features.

Direct strategy: one model per (channel, horizon) predicts the CHANGE
    delta = y[t+h] - y[t]
from the no-look-ahead feature vector at time t (all 8 channels' features -> captures
cross-channel physics) plus orbit covariates at t+h that are known in advance
(sunlit flag and orbit phase from the ephemeris). Predicting the change instead of the level lets trees handle
slow trends (battery fade) they could not extrapolate otherwise.

Training rows: train split, every 3rd origin (speed), keeping only rows whose input
window and target are normal (no labelled anomaly, no missing value).
Early stopping: on a subsample of the VALIDATION split (val is for tuning; test is untouched).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m4_ts_common as C  # noqa: E402
from m4_forecast_naive import HORIZONS  # noqa: E402
from m4_ts_features import build_features  # noqa: E402

PARAMS = dict(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.6,
              tree_method="hist", early_stopping_rounds=25, random_state=42, n_jobs=-1, verbosity=0)
TRAIN_STRIDE = 3
VAL_STRIDE = 6


def window_ok(normal: np.ndarray, lookback: int) -> np.ndarray:
    """ok[t] = all of normal[t-lookback+1 .. t] are True."""
    bad = (~normal).astype(np.int32)
    cs = np.concatenate([[0], np.cumsum(bad)])
    idx = np.arange(len(normal))
    lo = np.maximum(0, idx - lookback + 1)
    return (cs[idx + 1] - cs[lo]) == 0


def fut_known(df: pd.DataFrame, h: int) -> np.ndarray:
    """Covariates at the TARGET time t+h that are known in advance from the orbit:
    sunlit flag and orbit phase (sin/cos). Not look-ahead: they come from the ephemeris."""
    cfg = C.load_config()
    ts_future = df["timestamp"] + pd.Timedelta(seconds=h * cfg["cadence_s"])
    f = C.fourier_terms(ts_future, cfg["orbit_period_min"] * 60, 1, start=pd.Timestamp(cfg["start_time"])).values
    period_s = cfg["orbit_period_min"] * 60
    phase = ((ts_future - pd.Timestamp(cfg["start_time"])).dt.total_seconds().values % period_s) / period_s
    sunlit_future = (phase < 1.0 - cfg["eclipse_fraction"]).astype(np.float32)
    return np.column_stack([sunlit_future, f]).astype(np.float32)


def fit_predict(df: pd.DataFrame, masks: dict, normal: np.ndarray, horizons=HORIZONS,
                channels=C.CHANNELS, seed: int = 42, verbose=True, return_models=False, **_):
    import xgboost as xgb

    horizons = list(horizons)
    X = build_features(df).astype(np.float32)
    feat_names = list(X.columns)
    Xv = X.values
    n = len(df)
    in_ok = window_ok(normal, 191)                                   # longest lag/rolling window = 190
    pred = np.full((n, len(horizons), len(channels)), np.nan, dtype=np.float32)
    models, info = {}, {}
    params = {**PARAMS, "random_state": seed}
    for j, c in enumerate(channels):
        y = df[c].values.astype(np.float32)
        for k, h in enumerate(horizons):
            tgt = np.arange(n) + h
            valid_t = np.zeros(n, bool); valid_t[:n - h] = normal[tgt[:n - h]]
            delta = np.full(n, np.nan, np.float32); delta[:n - h] = y[h:] - y[:n - h]
            rows_ok = in_ok & valid_t & ~np.isnan(delta)
            Xh = np.column_stack([Xv, fut_known(df, h)])            # + known-future orbit covariates
            tr = np.where(masks["train"] & rows_ok)[0][::TRAIN_STRIDE]
            va = np.where(masks["val"] & rows_ok)[0][::VAL_STRIDE]
            t0 = time.time()
            m = xgb.XGBRegressor(**params)
            m.fit(Xh[tr], delta[tr], eval_set=[(Xh[va], delta[va])], verbose=False)
            pred[:, k, j] = y + m.predict(Xh)
            info[f"{c}_h{h}"] = {"best_iteration": int(m.best_iteration), "train_rows": int(len(tr)),
                                 "fit_s": round(time.time() - t0, 1)}
            if return_models:
                models[(c, h)] = m
        if verbose:
            print(f"  XGBoost {c:<10} horizons {horizons}  done")
    return (pred, info, models, feat_names) if return_models else (pred, info)


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description="XGBoost forecaster (library). Run mlts/m4_forecast_eval.py --models xgboost").parse_args()
    print("Use m4_forecast_eval.py --models xgboost")
