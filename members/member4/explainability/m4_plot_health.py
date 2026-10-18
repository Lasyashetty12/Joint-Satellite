"""
m4_plot_health.py - figures for health-anomaly detection (Milestone 5).

    health_score_timeline.png   selected detector's smoothed score over val + test, threshold,
                                labelled events shaded (what an operator would see)
    health_per_type_recall.png  recall per anomaly type x detector (all splits, diagnostic)
    health_detector_f1.png      event-F1 on val and test per detector

Run from the repository root (after mlts/m4_residual_anomaly.py):
    python members/member4/explainability/m4_plot_health.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mlts"))
import m4_ts_common as C  # noqa: E402
import m4_plot_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["limit_check", "isolation_forest", "residual_naive", "residual_snaive", "residual_arimax",
         "residual_xgboost", "residual_gru", "residual_ensemble"]


def timeline(path):
    sc = pd.read_csv(C.GEN_DIR / "m4_anomaly_scores.csv.gz", parse_dates=["timestamp"])
    t0 = pd.Timestamp(C.load_config()["start_time"])
    day = (sc.timestamp - t0).dt.total_seconds() / 86400
    tau = sc.threshold.iloc[0]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    lab = sc.label.values.astype(bool)
    ax.fill_between(day, 0, 1, where=lab, transform=ax.get_xaxis_transform(), color=S.ALERT, alpha=0.18, lw=0,
                    label="labelled anomaly")
    for k, split in enumerate(("val", "test")):                       # separate lines: no bridge over the gap day
        m = (sc.split == split).values
        ax.plot(day[m], sc.score[m].clip(upper=tau * 6), color=S.SERIES[0], lw=0.6,
                label="smoothed score" if k == 0 else None)
    ax.axhline(tau, color=S.SERIES[1], ls="--", lw=1, label=f"threshold τ = {tau:.2f} (from val)")
    gap = day[sc.split == "test"].min()
    ax.axvline(gap, color=S.INK_2, lw=0.8)
    ax.text(gap + 0.2, 0.95, "test →", transform=ax.get_xaxis_transform(), color=S.INK_2, va="top")
    ax.text(day.min() + 0.2, 0.95, "validation", transform=ax.get_xaxis_transform(), color=S.INK_2, va="top")
    ax.set_xlabel("day"); ax.set_ylabel("anomaly score (clipped)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    ax.set_title(f"Health monitor: {sc.model.iloc[0]} score, validation and test")
    return S.save(fig, path)


def per_type(path):
    pt = pd.read_csv(C.RESULTS / "m4_health_per_type.csv", index_col=0)
    pt = pt[[c for c in ORDER if c in pt.columns]]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    im = ax.imshow(pt.values, cmap=S.SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(pt.shape[1]), [c.replace("residual_", "res. ") for c in pt.columns], rotation=35, ha="right")
    ax.set_yticks(range(pt.shape[0]), pt.index)
    ax.grid(False)
    for i in range(pt.shape[0]):
        for j in range(pt.shape[1]):
            v = pt.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else S.INK)
    fig.colorbar(im, ax=ax, label="recall")
    ax.set_title("Which detector catches which fault? Recall per anomaly type (all 59 events)")
    return S.save(fig, path)


def f1_bars(path):
    hm = pd.read_csv(C.RESULTS / "health_metrics.csv")
    hm = hm[hm.dataset == "simulated_v1"]
    models = [m for m in ORDER if m in set(hm.model)]
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for k, (split, col) in enumerate([("val", S.SERIES[0]), ("test", S.SERIES[1])]):
        v = hm[hm.split == split].set_index("model").reindex(models)["event_F1"].values
        b = ax.bar(x + (k - 0.5) * w, v, width=w - 0.04, color=col, label=split)
        ax.bar_label(b, fmt="%.2f", fontsize=7, color=S.INK_2, padding=2)
    ax.set_xticks(x, [m.replace("residual_", "res. ") for m in models], rotation=25, ha="right")
    ax.set_ylim(0, 1.1); ax.set_ylabel("event F1")
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Event-F1 per detector (threshold chosen on validation; 12 val / 9 test events)")
    return S.save(fig, path)


def main(argv=None) -> int:
    argparse.ArgumentParser(description="Member 4 health-anomaly figures.").parse_args(argv)
    S.apply()
    for p in (timeline(C.FIGURES / "health_score_timeline.png"), per_type(C.FIGURES / "health_per_type_recall.png"),
              f1_bars(C.FIGURES / "health_detector_f1.png")):
        print("saved", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
