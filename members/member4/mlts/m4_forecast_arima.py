"""
m4_forecast_arima.py - ARIMAX: regression on Fourier orbit terms with AR(p) errors.

    y_t = b0 + sum_j [a_j sin(2*pi*j*t/P) + b_j cos(2*pi*j*t/P)] + g * sunlit_t + u_t   (P = 95 min, j = 1..K)
    u_t = phi_1 u_{t-1} + ... + phi_p u_{t-p} + e_t                             (ARIMA(p, d=0, q=0) errors)

Why this form (handbook note on long seasonal periods): a full SARIMA with period 190
is very slow, so the orbit seasonality enters as Fourier exogenous regressors (K = 3)
plus the sunlit/eclipse flag (sharp steps that a few smooth waves cannot draw; the
eclipse schedule is known in advance from the orbit, so it is not look-ahead)
and only short-memory AR errors are estimated. d = 0 because Milestone 3 showed every
channel is stationary once the orbit pattern is removed. p comes from the PACF
(results/m4_arima_orders.json), capped at 6.

Fitting: statsmodels SARIMAX (exact MLE, NaN-tolerant) on the LAST 7 DAYS OF TRAIN,
labelled train anomalies set to NaN (we model normal behaviour).
Forecasting: the fitted equations are applied to every origin at once (vectorised
h-step recursion), which equals the Kalman forecast whenever the last p values exist.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m4_ts_common as C  # noqa: E402
from m4_forecast_naive import HORIZONS  # noqa: E402

P_CAP = 6
FIT_DAYS = 7
FOURIER_K = 3


def design(df: pd.DataFrame, k: int = FOURIER_K) -> np.ndarray:
    """[1, Fourier orbit terms, sunlit]. All are known in advance from the orbit (no look-ahead):
    the eclipse schedule is predictable, so sunlit at the TARGET time is a legitimate input."""
    cfg = C.load_config()
    f = C.fourier_terms(df["timestamp"], cfg["orbit_period_min"] * 60, k, start=pd.Timestamp(cfg["start_time"])).values
    return np.column_stack([np.ones(len(df)), f, df["sunlit"].values.astype(float)])


def fit_channel(y: np.ndarray, X: np.ndarray, p: int):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(y, exog=X, order=(p, 0, 0), trend="n").fit(disp=False, maxiter=200)
    k = X.shape[1]
    beta = np.asarray(res.params[:k], dtype=float)
    phi = np.asarray(res.params[k:k + p], dtype=float)
    return beta, phi, res


def forecast_all(y: np.ndarray, X: np.ndarray, beta: np.ndarray, phi: np.ndarray, horizons) -> np.ndarray:
    """Vectorised h-step forecasts for every origin t. Returns [N, len(horizons)]."""
    n, p = len(y), len(phi)
    u = y - X @ beta                                       # error process (NaN where y is NaN)
    hist = [np.concatenate([np.full(i, np.nan), u[:n - i]]) for i in range(p)]   # hist[i][t] = u[t-i]
    out = np.full((n, len(horizons)), np.nan)
    hmax = max(horizons)
    fut = []
    for h in range(1, hmax + 1):
        lagged = fut[::-1] + hist                          # u_{t+h-1}, u_{t+h-2}, ...
        u_next = sum(phi[i] * lagged[i] for i in range(p))
        fut.append(u_next)
        if h in horizons:
            tgt = np.arange(n) + h
            ok = tgt < n
            col = horizons.index(h)
            out[ok, col] = X[tgt[ok]] @ beta + u_next[ok]
    return out


def fit_predict(df: pd.DataFrame, masks: dict, normal: np.ndarray, horizons=HORIZONS,
                channels=C.CHANNELS, orders_path: Path = C.RESULTS / "m4_arima_orders.json", verbose=True, **_):
    horizons = list(horizons)
    orders = json.loads(Path(orders_path).read_text()) if Path(orders_path).exists() else {}
    tr = np.where(masks["train"])[0][-FIT_DAYS * C.samples_per_day():]
    X_all = design(df)
    pred = np.full((len(df), len(horizons), len(channels)), np.nan, dtype=np.float32)
    info = {}
    for j, c in enumerate(channels):
        p = min(P_CAP, orders.get(c, {}).get("p_from_pacf_of_stl_remainder", 3))
        y_fit = df[c].values[tr].astype(float).copy()
        y_fit[~normal[tr]] = np.nan
        t0 = time.time()
        beta, phi, res = fit_channel(y_fit, X_all[tr], p)
        pred[:, :, j] = forecast_all(df[c].values.astype(float), X_all, beta, phi, horizons)
        info[c] = {"p": p, "aic": round(float(res.aic), 1), "phi": np.round(phi, 4).tolist(),
                   "fit_s": round(time.time() - t0, 1)}
        if verbose:
            print(f"  ARIMAX {c:<10} p={p}  AIC={res.aic:,.0f}  ({time.time() - t0:.0f}s)")
    return pred, info


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description="ARIMAX forecaster (library). Run mlts/m4_forecast_eval.py --models arimax").parse_args()
    print("Use m4_forecast_eval.py --models arimax")
