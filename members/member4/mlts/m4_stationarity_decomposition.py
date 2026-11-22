"""
m4_stationarity_decomposition.py - STL decomposition, ADF/KPSS stationarity tests, ACF/PACF.

Concepts (MLTS syllabus):
  * STL splits a series into TREND (slow change, e.g. battery ageing) + SEASONAL
    (the repeating orbit pattern, period = 190 samples = 95 min) + REMAINDER (the rest).
  * Stationary = statistics do not drift over time. Many classical models assume it.
      ADF  : H0 = "has a unit root (NOT stationary)"  -> small p (< 0.05) => stationary
      KPSS : H0 = "IS stationary"                     -> small p (< 0.05) => NOT stationary
    We test raw, first-differenced and orbit-differenced (lag 190) versions.
  * ACF/PACF show how a value depends on its own past -> choose lags and the AR order p.

Analysis window: the CLEANEST 7-DAY BLOCK OF THE TRAIN SPLIT (20,160 samples), normal points,
small gaps interpolated for the tests only. Rationale: tests on 155k points are slow
and over-powered (everything becomes "significant"); 7 days = 106 orbits is plenty.

Outputs:
  results/stationarity_report.csv, results/m4_seasonal_strength.csv, results/m4_arima_orders.json
  reports/figures/stl_<channel>.png (3 key channels), acf_pacf_channels.png

Run from the repository root:
    python members/member4/mlts/m4_stationarity_decomposition.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m4_ts_common as C  # noqa: E402
import m4_plot_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf  # noqa: E402

STL_PLOT_CHANNELS = ["SoC", "panel_temp", "bus_V"]


def analysis_window(df: pd.DataFrame, masks: dict, labels: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    """The CLEANEST `days`-day block inside the train split (fewest anomalous or missing
    samples), so that removing anomalies does not leave long interpolated stretches."""
    per_day = C.samples_per_day()
    tr_idx = np.where(masks["train"])[0]
    bad = (C.label_mask(df, labels) == 1) | df[C.CHANNELS].isna().any(axis=1).values
    bad_per_day = np.add.reduceat(bad[tr_idx].astype(int), np.arange(0, len(tr_idx), per_day))
    n_days = len(bad_per_day)
    scores = [bad_per_day[d:d + days].sum() for d in range(n_days - days + 1)]
    d0 = int(np.argmin(scores))
    sel = tr_idx[d0 * per_day:(d0 + days) * per_day]
    w = df.iloc[sel].copy()
    w.loc[bad[sel], C.CHANNELS] = np.nan                  # remove labelled anomalies / gaps
    w[C.CHANNELS] = w[C.CHANNELS].interpolate(limit_direction="both")
    w.attrs["start_day"], w.attrs["bad_samples"] = d0, int(scores[d0])
    print(f"analysis window: train days {d0}-{d0 + days - 1}, {scores[d0]} anomalous/missing samples removed")
    return w.reset_index(drop=True)


def stationarity_tests(x: np.ndarray) -> dict:
    x = x[~np.isnan(x)]
    if np.nanstd(x) < 1e-12:
        return {"adf_stat": np.nan, "adf_p": np.nan, "kpss_stat": np.nan, "kpss_p": np.nan, "verdict": "constant"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf = adfuller(x, maxlag=30, autolag="AIC")
        kp = kpss(x, regression="c", nlags="auto")
    adf_st, kpss_st = adf[1] < 0.05, kp[1] >= 0.05
    verdict = ("stationary" if adf_st and kpss_st else
               "non-stationary" if not adf_st and not kpss_st else
               "trend-stationary / mixed" if adf_st else "difference-stationary / mixed")
    return {"adf_stat": round(adf[0], 3), "adf_p": round(adf[1], 4),
            "kpss_stat": round(kp[0], 3), "kpss_p": round(kp[1], 4), "verdict": verdict}


def seasonal_strength(res) -> tuple[float, float]:
    """Hyndman's strengths: 1 = strong, 0 = none."""
    r, s, t = res.resid, res.seasonal, res.trend
    fs = max(0.0, 1 - np.var(r) / np.var(s + r))
    ft = max(0.0, 1 - np.var(r) / np.var(t + r))
    return round(fs, 3), round(ft, 3)


def suggest_p(pacf_vals: np.ndarray, n: int, max_p: int = 10) -> int:
    """Last lag (<= max_p) whose PACF is outside the 95 % band before the first insignificant one."""
    band = 1.96 / np.sqrt(n)
    p = 0
    for k in range(1, min(max_p, len(pacf_vals) - 1) + 1):
        if abs(pacf_vals[k]) > band:
            p = k
        else:
            break
    return max(1, p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Member 4: STL, ADF/KPSS and ACF/PACF on the telemetry.")
    ap.add_argument("--days", type=int, default=7, help="analysis window length (cleanest block of the train split)")
    args = ap.parse_args(argv)
    S.apply()
    period = C.samples_per_orbit()
    df = C.load_telemetry()
    masks = C.split_masks(df)
    w = analysis_window(df, masks, C.load_labels(), args.days)
    hours = np.arange(len(w)) * 0.5 / 60

    rows, strength, orders = [], [], {}
    fig_acf, axes = plt.subplots(len(C.CHANNELS), 2, figsize=(11, 15))
    for i, c in enumerate(C.CHANNELS):
        x = w[c].values
        res = STL(x, period=period, robust=True).fit()
        fs, ft = seasonal_strength(res)
        strength.append({"channel": c, "seasonal_strength": fs, "trend_strength": ft})
        for name, y in [("raw", x), ("diff1", np.diff(x)), (f"seasonal_diff{period}", x[period:] - x[:-period]),
                        ("stl_remainder", res.resid)]:
            rows.append({"channel": c, "transform": name, **stationarity_tests(y)})

        a = acf(x, nlags=2 * period + 10, fft=True, missing="drop")
        pr = pacf(res.resid, nlags=40, method="ywm")
        orders[c] = {"p_from_pacf_of_stl_remainder": suggest_p(pr, len(x)),
                     "acf_at_orbit_lag": round(float(a[period]), 3)}
        ax1, ax2 = axes[i]
        ax1.bar(np.arange(len(a)), a, width=1.0, color=S.SERIES[0])
        ax1.axvline(period, color=S.SERIES[1], ls="--", lw=1)
        ax1.set_title(f"{c}: ACF (dashed = 1 orbit, lag {period}); value {a[period]:.2f}")
        band = 1.96 / np.sqrt(len(x))
        ax2.bar(np.arange(len(pr)), pr, width=0.8, color=S.SERIES[0])
        ax2.axhspan(-band, band, color=S.SHADE, lw=0)
        ax2.set_title(f"{c}: PACF of STL remainder → suggested p = {orders[c]['p_from_pacf_of_stl_remainder']}")
        if c in STL_PLOT_CHANNELS:
            f, axs = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
            for ax, (lab, y) in zip(axs, [("observed", x), ("trend", res.trend), ("seasonal (orbit)", res.seasonal),
                                          ("remainder", res.resid)]):
                ax.plot(hours, y, color=S.SERIES[0], lw=0.7)
                ax.set_ylabel(lab)
            axs[-1].set_xlabel("hours (cleanest 7-day block of train)")
            f.suptitle(f"STL decomposition of {c} [{C.UNITS[c]}], period = {period} samples (95 min); "
                       f"seasonal strength {fs}, trend strength {ft}", color=S.INK)
            S.save(f, C.FIGURES / f"stl_{c}.png")
    axes[-1, 0].set_xlabel("lag (samples, 30 s)"); axes[-1, 1].set_xlabel("lag (samples)")
    S.save(fig_acf, C.FIGURES / "acf_pacf_channels.png")

    C.RESULTS.mkdir(parents=True, exist_ok=True)
    rep = pd.DataFrame(rows)
    rep.to_csv(C.RESULTS / "stationarity_report.csv", index=False)
    pd.DataFrame(strength).to_csv(C.RESULTS / "m4_seasonal_strength.csv", index=False)
    (C.RESULTS / "m4_arima_orders.json").write_text(json.dumps(orders, indent=2), encoding="utf-8")
    print(rep.to_string(index=False))
    print("\n", pd.DataFrame(strength).to_string(index=False))
    print("\nSuggested AR orders:", json.dumps(orders))
    return 0


if __name__ == "__main__":
    sys.exit(main())
