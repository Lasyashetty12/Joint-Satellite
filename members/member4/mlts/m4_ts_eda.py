"""
m4_ts_eda.py - time-series exploratory data analysis of Member 4's telemetry.

Figures (reports/figures/):
    eda_overview.png       all 8 channels over 90 days (hourly mean + min-max band), split blocks shaded
    eda_orbit_fold.png     every channel folded by ORBIT PHASE (median + 5-95 % band) -> the seasonality
    eda_histograms.png     value distributions on the train split
    eda_missing_map.png    fraction of missing values per day x channel (telemetry dropouts)
    corr_channels.png      correlation matrix between channels (train, normal points)
    corr_crosslag.png      cross-correlation vs lag for physically linked channel pairs
Table (results/):
    m4_eda_summary.csv     per-channel stats + date/time components check

Run from the repository root (after m4_make_splits.py):
    python members/member4/mlts/m4_ts_eda.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m4_ts_common as C  # noqa: E402
import m4_plot_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

PAIRS = [("solar_I", "panel_temp"), ("batt_I", "bus_V"), ("batt_I", "batt_temp"), ("solar_I", "SoC")]


def add_time_components(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Date/time components used throughout the MLTS pipeline."""
    out = df.copy()
    t0 = pd.Timestamp(cfg["start_time"])
    sec = (out["timestamp"] - t0).dt.total_seconds()
    period = cfg["orbit_period_min"] * 60
    out["day_index"] = (sec // 86400).astype(int)
    out["orbit_index"] = (sec // period).astype(int)
    out["orbit_phase"] = (sec % period) / period
    out["hour_utc"] = out["timestamp"].dt.hour
    return out


def plot_overview(df, masks, path):
    hourly = df.set_index("timestamp")[C.CHANNELS].resample("1h")
    mean, lo, hi = hourly.mean(), hourly.min(), hourly.max()
    days = (mean.index - mean.index[0]).total_seconds() / 86400
    fig, axes = plt.subplots(len(C.CHANNELS), 1, figsize=(11, 13), sharex=True)
    split_days = {s: (np.where(m)[0][[0, -1]] / C.samples_per_day()) for s, m in masks.items()}
    for ax, c in zip(axes, C.CHANNELS):
        ax.fill_between(days, lo[c], hi[c], color=S.SERIES[0], alpha=0.18, linewidth=0)
        ax.plot(days, mean[c], color=S.SERIES[0], lw=0.9)
        ax.set_ylabel(f"{c}\n[{C.UNITS[c]}]")
        for s, (a, b) in split_days.items():
            if s != "train":
                ax.axvspan(a, b + 1 / C.samples_per_day(), color=S.SHADE, zorder=0, lw=0)
    for s, (a, b) in split_days.items():
        axes[0].text((a + b) / 2, 1.04, s, transform=axes[0].get_xaxis_transform(), ha="center", color=S.INK_2)
    axes[-1].set_xlabel("day")
    fig.suptitle("Telemetry over 90 days: hourly mean (line) and min–max (band); val/test blocks shaded", color=S.INK)
    return S.save(fig, path)


def plot_orbit_fold(df, mask, path, bins=95):
    d = df[mask].copy()
    d["phase_bin"] = (d["orbit_phase"] * bins).astype(int).clip(0, bins - 1)
    g = d.groupby("phase_bin")[C.CHANNELS]
    med, q05, q95 = g.median(), g.quantile(0.05), g.quantile(0.95)
    x = (med.index + 0.5) / bins
    ecl = d.groupby("phase_bin")["sunlit"].mean() < 0.5
    fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True)
    for ax, c in zip(axes.ravel(), C.CHANNELS):
        ax.fill_between(x, 0, 1, where=ecl.values, transform=ax.get_xaxis_transform(), color=S.SHADE, lw=0)
        ax.fill_between(x, q05[c], q95[c], color=S.SERIES[0], alpha=0.2, lw=0, label="5–95 %")
        ax.plot(x, med[c], color=S.SERIES[0], label="median")
        ax.set_title(f"{c} [{C.UNITS[c]}]")
    axes[0, 0].legend(loc="lower left")
    for ax in axes[-1]:
        ax.set_xlabel("orbit phase (0 → 1 = one 95-min orbit; grey = eclipse)")
    fig.suptitle("Seasonality: every train orbit folded on top of each other (normal points only)", color=S.INK)
    return S.save(fig, path)


def plot_histograms(df, mask, path):
    fig, axes = plt.subplots(2, 4, figsize=(12, 5))
    for ax, c in zip(axes.ravel(), C.CHANNELS):
        ax.hist(df.loc[mask, c].dropna(), bins=60, color=S.SERIES[0], alpha=0.85)
        ax.set_title(f"{c} [{C.UNITS[c]}]")
        ax.set_yticks([])
    fig.suptitle("Value distributions (train split). Two-peaked shapes = sunlight vs eclipse", color=S.INK)
    return S.save(fig, path)


def plot_missing(df, path):
    day = (np.arange(len(df)) // C.samples_per_day())
    frac = df[C.CHANNELS].isna().groupby(day).mean()
    fig, ax = plt.subplots(figsize=(11, 3.2))
    im = ax.imshow(frac.T.values, aspect="auto", cmap=S.SEQUENTIAL, vmin=0, vmax=max(0.05, frac.values.max()),
                   interpolation="nearest")
    ax.set_yticks(range(len(C.CHANNELS)), C.CHANNELS)
    ax.set_xlabel("day"); ax.grid(False)
    fig.colorbar(im, ax=ax, label="fraction missing")
    ax.set_title("Missing-data map: telemetry dropouts per day and channel")
    return S.save(fig, path)


def plot_corr(df, mask, path):
    corr = df.loc[mask, C.CHANNELS].corr()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(corr.values, cmap=S.DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(C.CHANNELS)), C.CHANNELS, rotation=45, ha="right")
    ax.set_yticks(range(len(C.CHANNELS)), C.CHANNELS)
    ax.grid(False)
    for i in range(len(C.CHANNELS)):
        for j in range(len(C.CHANNELS)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color=S.INK)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Channel correlation (train, normal points)")
    return S.save(fig, path), corr


def crosscorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> np.ndarray:
    """corr(a[t], b[t+lag]) for lag in [-max_lag, max_lag]; positive lag = b responds after a."""
    out = []
    for L in range(-max_lag, max_lag + 1):
        x, y = (a[:-L], b[L:]) if L > 0 else ((a[-L:], b[:len(b) + L]) if L < 0 else (a, b))
        ok = ~(np.isnan(x) | np.isnan(y))
        out.append(np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 10 else np.nan)
    return np.array(out)


def plot_crosslag(df, mask, path, max_lag=95):
    d = df[mask]
    lags = np.arange(-max_lag, max_lag + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    peaks = {}
    for ax, (a, b) in zip(axes.ravel(), PAIRS):
        cc = crosscorr(d[a].values, d[b].values, max_lag)
        k = int(np.nanargmax(np.abs(cc)))
        peaks[f"{a}->{b}"] = {"peak_lag_samples": int(lags[k]), "peak_lag_min": lags[k] * 0.5, "r": round(float(cc[k]), 3)}
        ax.plot(lags * 0.5, cc, color=S.SERIES[0])
        ax.axvline(lags[k] * 0.5, color=S.SERIES[1], lw=1, ls="--")
        ax.set_title(f"{a} → {b}: peak r={cc[k]:.2f} at {lags[k] * 0.5:+.1f} min")
        ax.axhline(0, color=S.INK_2, lw=0.6)
    for ax in axes[-1]:
        ax.set_xlabel("lag (minutes; + = second channel responds later)")
    fig.suptitle("Cross-correlation of physically linked channels (train)", color=S.INK)
    return S.save(fig, path), peaks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Member 4 time-series EDA (figures + summary table).")
    ap.parse_args(argv)
    cfg = C.load_config()
    df = add_time_components(C.load_telemetry(), cfg)
    masks = C.split_masks(df)
    normal = C.label_mask(df, C.load_labels()) == 0
    train_normal = masks["train"] & normal
    S.apply()

    figs = [plot_overview(df, masks, C.FIGURES / "eda_overview.png"),
            plot_orbit_fold(df, train_normal, C.FIGURES / "eda_orbit_fold.png"),
            plot_histograms(df, masks["train"], C.FIGURES / "eda_histograms.png"),
            plot_missing(df, C.FIGURES / "eda_missing_map.png")]
    f, corr = plot_corr(df, train_normal, C.FIGURES / "corr_channels.png"); figs.append(f)
    f, peaks = plot_crosslag(df, train_normal, C.FIGURES / "corr_crosslag.png"); figs.append(f)

    tr = df[masks["train"]]
    summary = pd.DataFrame({
        "unit": [C.UNITS[c] for c in C.CHANNELS],
        "train_mean": tr[C.CHANNELS].mean().round(4).values,
        "train_std": tr[C.CHANNELS].std().round(4).values,
        "train_min": tr[C.CHANNELS].min().round(4).values,
        "train_max": tr[C.CHANNELS].max().round(4).values,
        "missing_frac_all": df[C.CHANNELS].isna().mean().round(5).values,
        "sunlit_mean": tr[tr.sunlit == 1][C.CHANNELS].mean().round(4).values,
        "eclipse_mean": tr[tr.sunlit == 0][C.CHANNELS].mean().round(4).values,
    }, index=pd.Index(C.CHANNELS, name="channel"))
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(C.RESULTS / "m4_eda_summary.csv")
    pd.DataFrame(peaks).T.to_csv(C.RESULTS / "m4_crosslag_peaks.csv")

    print("Date/time components: day_index 0..%d, orbits %d, samples/orbit %d, eclipse fraction %.4f"
          % (df.day_index.max(), df.orbit_index.nunique(), C.samples_per_orbit(cfg), 1 - df.sunlit.mean()))
    print(summary.to_string())
    print("\nCross-correlation peaks:", peaks)
    for p in figs:
        print("saved", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
