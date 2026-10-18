"""
m4_plot_forecasts.py - figures for the forecasting ladder (Milestone 4).

    forecast_skill_heatmap.png   skill vs naive = 1 - MAE/MAE_naive (val), models x channels, per horizon
    forecast_actual_vs_pred.png  one validation day: actual vs best model at h = 10, with residual band
    forecast_gru_loss.png        GRU train/val loss per epoch

Run from the repository root (after mlts/m4_forecast_eval.py):
    python members/member4/explainability/m4_plot_forecasts.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mlts"))
import m4_ts_common as C  # noqa: E402
import m4_plot_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

MODEL_ORDER = ["naive", "snaive", "arimax", "xgboost", "gru"]
SHOW = ["SoC", "bus_V", "panel_temp", "batt_I"]


def skill_heatmap(path):
    sk = pd.read_csv(C.RESULTS / "m4_forecast_skill.csv")
    hs = sorted(sk.horizon.unique())
    fig, axes = plt.subplots(1, len(hs), figsize=(5.2 * len(hs), 4.0), sharey=True)
    for ax, h in zip(np.atleast_1d(axes), hs):
        t = sk[sk.horizon == h].pivot_table(index="model", columns="channel", values="skill_vs_naive")
        t = t.reindex([m for m in MODEL_ORDER if m in t.index])[C.CHANNELS]
        im = ax.imshow(t.values, cmap=S.DIVERGING.reversed(), vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(C.CHANNELS)), C.CHANNELS, rotation=60, ha="right")
        ax.set_yticks(range(len(t.index)), t.index)
        ax.grid(False)
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                v = t.values[i, j]
                ax.text(j, i, f"{v:+.2f}" if np.isfinite(v) else "–", ha="center", va="center", fontsize=6.5, color=S.INK)
        ax.set_title(f"h = {h} ({h * 30} s ahead)")
    fig.colorbar(im, ax=axes, label="skill vs naive (+ = better)", shrink=0.8)
    fig.suptitle("Forecast skill on validation: 1 − MAE / MAE(naive)", color=S.INK)
    return S.save(fig, path)


def best_model(h: int) -> str:
    sk = pd.read_csv(C.RESULTS / "m4_forecast_skill.csv")
    m = sk[sk.horizon == h].groupby("model")["skill_vs_naive"].mean()
    return m.idxmax()


def actual_vs_pred(path, h=10, day=60):
    model = best_model(h)
    df = C.load_telemetry()
    z = np.load(C.GEN_DIR / "forecasts" / f"m4_pred_{model}.npz")
    pred, hz = z["pred"], list(z["horizons"])
    k = hz.index(h)
    masks = C.split_masks(df)
    per = C.samples_per_day()
    a, b = day * per, (day + 1) * per
    tr = np.where(masks["train"])[0]
    hrs = np.arange(a, b) * 0.5 / 60 - day * 24
    fig, axes = plt.subplots(len(SHOW), 1, figsize=(11, 9), sharex=True)
    for ax, c in zip(axes, SHOW):
        j = C.CHANNELS.index(c)
        y = df[c].values
        p_shift = np.full(len(y), np.nan); p_shift[h:] = pred[:-h, k, j]       # forecast placed at its target time
        res_tr = y[tr] - p_shift[tr]
        sd = np.nanstd(res_tr)
        ax.fill_between(hrs, p_shift[a:b] - 2 * sd, p_shift[a:b] + 2 * sd, color=S.SERIES[1], alpha=0.18, lw=0,
                        label="forecast ± 2·sd (train residuals)")
        ax.plot(hrs, y[a:b], color=S.SERIES[0], lw=0.9, label="actual")
        ax.plot(hrs, p_shift[a:b], color=S.SERIES[1], lw=0.9, ls="--", label=f"{model} forecast")
        ax.set_ylabel(f"{c}\n[{C.UNITS[c]}]")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3)
    axes[-1].set_xlabel(f"hours into validation day {day}")
    fig.suptitle(f"Actual vs {model} forecast made {h * 30 // 60} min earlier (h = {h})", color=S.INK, y=1.02)
    return S.save(fig, path)


def gru_loss(path):
    info = json.loads((C.RESULTS / "m4_forecast_info.json").read_text())
    hist = pd.DataFrame(info.get("gru", {}).get("history", []))
    if hist.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(hist.epoch, hist.train_loss, color=S.SERIES[0], marker="o", ms=3, label="train")
    ax.plot(hist.epoch, hist.val_loss, color=S.SERIES[1], marker="o", ms=3, label="validation")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (standardised)"); ax.legend()
    ax.set_title("GRU forecaster: loss per epoch")
    return S.save(fig, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Member 4 forecasting figures.")
    ap.add_argument("--day", type=int, default=60, help="validation day to plot (55-71)")
    args = ap.parse_args(argv)
    S.apply()
    for p in [skill_heatmap(C.FIGURES / "forecast_skill_heatmap.png"),
              actual_vs_pred(C.FIGURES / "forecast_actual_vs_pred.png", day=args.day),
              gru_loss(C.FIGURES / "forecast_gru_loss.png")]:
        if p:
            print("saved", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
