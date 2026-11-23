"""
m4_plot_fusion.py - paper figures for Milestones 6-11 (EO baselines, fusion, robustness, alpha, ablations).

    eo_baselines.png            F1 / AUPRC of optical-only, SAR-only, optical+SAR CNN (test, Bolivia)
    fusion_comparison.png       test F1 per model and task, all inputs present (mean ± std over seeds)
    robustness_missing.png      flood and health F1 when inputs go missing (simple vs proposed fusion)
    robustness_cloud.png        flood F1 vs cloud cover on the optical image
    alpha_behaviour.png         gated-fusion trust weights per situation (RQ4)
    reconstruction.png          telemetry channel reconstruction error and its effect on detection (RQ3)
    ablations.png               proposed model vs its ablations

If the EO inputs are the synthetic stand-in, every figure is stamped "SYNTHETIC EO - PIPELINE CHECK".

Run from the repository root (after training/m4_train_fusion.py):
    python members/member4/explainability/m4_plot_fusion.py
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

FUSION_ORDER = ["tel_only", "opt_only", "sar_only", "early_fusion", "late_fusion", "equal_weights", "gated_fusion"]
ABLATIONS = ["gated_fusion", "gated_no_quality", "gated_zero_fill", "gated_no_tel_in_gate", "gated_no_dropout", "gated_no_residuals",
             "gated_no_multitask"]
NICE = {"tel_only": "telemetry only", "opt_only": "optical only", "sar_only": "SAR only", "early_fusion": "early fusion",
        "late_fusion": "late fusion", "equal_weights": "equal weights", "gated_fusion": "gated fusion (ours)",
        "gated_zero_fill": "− missing token", "gated_no_tel_in_gate": "− telemetry in gate",
        "gated_no_dropout": "− modality dropout", "gated_no_residuals": "− forecast residuals",
        "gated_no_multitask": "− multi-task loss", "gated_no_quality": "− quality check in gate"}
COMPARE = [("early_fusion", S.SERIES[1]), ("late_fusion", S.SERIES[2]), ("gated_fusion", S.SERIES[0])]


def stamp(fig, synthetic: bool):
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC EO – PIPELINE CHECK, NOT A RESULT", ha="center", va="center", rotation=20,
                 fontsize=22, color=S.ALERT, alpha=0.25, weight="bold")


def is_synth(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and len(df) and str(df[col].iloc[0]).startswith("synthetic")


def eo_baselines(path):
    f = C.RESULTS / "eo_baseline_results.csv"
    if not f.exists():
        return None
    r = pd.read_csv(f)
    models = [m for m in ("optical_only", "sar_only", "optical_sar_cnn") if m in set(r.model)]
    splits = [s for s in ("test", "bolivia") if s in set(r.split)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, metric in zip(axes, ("F1", "AUPRC")):
        x = np.arange(len(splits)); w = 0.8 / len(models)
        for k, (m, col) in enumerate(zip(models, S.SERIES)):
            v = r[r.model == m].set_index("split").reindex(splits)[metric].values
            b = ax.bar(x + (k - (len(models) - 1) / 2) * w, v, width=w - 0.03, color=col, label=m.replace("_", " "))
            ax.bar_label(b, fmt="%.2f", fontsize=7, color=S.INK_2, padding=2)
        ax.set_xticks(x, ["test" if s == "test" else "Bolivia (unseen event)" for s in splits]); ax.set_title(metric)
        ax.set_ylim(0, 1.12)
    axes[0].legend(loc="lower left")
    fig.suptitle("Earth-observation baselines: patch-level flood detection", color=S.INK)
    stamp(fig, is_synth(r, "dataset"))
    return S.save(fig, path)


def _fusion():
    f = C.RESULTS / "fusion_results.csv"
    return pd.read_csv(f) if f.exists() else None


def fusion_comparison(path, fr):
    d = fr[fr.scenario == "all_present"]
    g = d.groupby(["model", "task"])["F1"].agg(["mean", "std"]).reset_index()
    models = [m for m in FUSION_ORDER if m in set(g.model)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, task, title in zip(axes, ("health", "eo"), ("Satellite-health detection", "Flood detection")):
        t = g[g.task == task].set_index("model").reindex(models)
        cols = [S.SERIES[0] if m == "gated_fusion" else "#b9b8b3" for m in models]
        b = ax.barh(range(len(models)), t["mean"], xerr=t["std"].fillna(0), color=cols, height=0.65,
                    error_kw={"ecolor": S.INK_2, "lw": 0.8})
        ax.bar_label(b, fmt="%.3f", fontsize=7, padding=3, color=S.INK_2)
        ax.set_yticks(range(len(models)), [NICE[m] for m in models]); ax.invert_yaxis()
        ax.set_xlim(0, 1.12); ax.set_title(title); ax.set_xlabel("test F1 (mean ± std, 3 seeds)")
    fig.suptitle("All inputs present: proposed gated fusion vs baselines", color=S.INK)
    stamp(fig, is_synth(fr, "eo_dataset"))
    return S.save(fig, path)


def robustness_missing(path, fr):
    scen = ["all_present", "missing_optical", "missing_sar", "missing_telemetry", "only_optical", "only_sar", "only_telemetry"]
    lab = ["all", "− optical", "− SAR", "− telemetry", "optical only", "SAR only", "telemetry only"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, task, title in zip(axes, ("eo", "health"), ("Flood F1", "Health F1")):
        x = np.arange(len(scen)); w = 0.27
        for k, (m, col) in enumerate(COMPARE):
            d = fr[(fr.model == m) & (fr.task == task)].groupby("scenario")["F1"].agg(["mean", "std"]).reindex(scen)
            ax.bar(x + (k - 1) * w, d["mean"], yerr=d["std"].fillna(0), width=w - 0.03, color=col, label=NICE[m],
                   error_kw={"ecolor": S.INK_2, "lw": 0.7})
        ax.set_xticks(x, lab, rotation=25, ha="right"); ax.set_title(title); ax.set_ylim(0, 1.1)
    axes[0].set_ylabel("test F1 (mean ± std over seeds)"); axes[0].legend(loc="lower left")
    fig.suptitle("Robustness when inputs are missing", color=S.INK)
    stamp(fig, is_synth(fr, "eo_dataset"))
    return S.save(fig, path)


def robustness_cloud(path, fr):
    scen = ["all_present", "cloud_25", "cloud_50", "cloud_75", "cloud_100"]
    x = [0, 25, 50, 75, 100]
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, col in COMPARE + [("opt_only", S.INK_2)]:
        d = fr[(fr.model == m) & (fr.task == "eo")].groupby("scenario")["F1"].agg(["mean", "std"]).reindex(scen)
        ls = "--" if m == "opt_only" else "-"
        ax.errorbar(x, d["mean"], yerr=d["std"].fillna(0), color=col, ls=ls, marker="o", ms=4, lw=1.4,
                    capsize=2, label=NICE[m])
        ax.annotate(NICE[m], (x[-1], d["mean"].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=7, color=S.INK_2, va="center")
    ax.set_xlabel("extra cloud cover on the optical image (%)"); ax.set_ylabel("flood F1 (test)")
    ax.set_xlim(-5, 125); ax.legend(loc="lower left")
    ax.set_title("Flood detection as clouds hide the ground")
    stamp(fig, is_synth(fr, "eo_dataset"))
    return S.save(fig, path)


def alpha_behaviour(path):
    f = C.RESULTS / "m4_alpha_summary.csv"
    if not f.exists():
        return None
    a = pd.read_csv(f)
    a = a[(a.model == "gated_fusion") & (a.gate == "eo")]
    rows = [("all_present", "clean", "clear image"), ("all_present", "cloudy", "cloudy image"),
            ("all_present", "coupled_payload_fault", "payload fault\n(damaged image)"),
            ("cloud_50", "all", "+50 % cloud"), ("cloud_100", "all", "+100 % cloud"),
            ("missing_optical", "all", "optical missing"), ("missing_sar", "all", "SAR missing")]
    vals, labels = [], []
    for scen, sub, lab in rows:
        d = a[(a.scenario == scen) & (a.subset == sub)]
        if len(d):
            vals.append(d[["alpha_tel", "alpha_opt", "alpha_sar"]].mean().values); labels.append(lab)
    if not vals:
        return None
    v = np.array(vals)
    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = np.zeros(len(v))
    for k, (name, col) in enumerate(zip(("α telemetry", "α optical", "α SAR"), (S.SERIES[1], S.SERIES[0], S.SERIES[2]))):
        b = ax.bar(range(len(v)), v[:, k], bottom=bottom, color=col, width=0.62, label=name, edgecolor=S.SURFACE, lw=1)
        for i, val in enumerate(v[:, k]):
            if val > 0.08:
                ax.text(i, bottom[i] + val / 2, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white")
        bottom += v[:, k]
    ax.set_xticks(range(len(v)), labels, fontsize=8); ax.set_ylim(0, 1.05); ax.set_ylabel("mean trust weight")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.set_title("RQ4: how much the flood decision trusts each input (gated fusion, test passes)")
    stamp(fig, is_synth(pd.read_csv(f), "eo_dataset"))
    return S.save(fig, path)


def reconstruction(path, fr):
    f = C.RESULTS / "m4_reconstruction_results.csv"
    if not f.exists():
        return None
    r = pd.read_csv(f).pivot(index="channel", columns="method", values="MAE").reindex(C.CHANNELS)
    rel = r.div(r["mean_fill"], axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]; x = np.arange(len(rel)); w = 0.27
    for k, (m, col) in enumerate(zip(("mean_fill", "ffill", "cross_channel"), (S.INK_2, S.SERIES[1], S.SERIES[0]))):
        ax.bar(x + (k - 1) * w, rel[m], width=w - 0.03, color=col, label={"mean_fill": "train mean", "ffill": "last value",
                                                                          "cross_channel": "cross-channel (ours)"}[m])
    ax.set_xticks(x, rel.index, rotation=30, ha="right"); ax.set_yscale("log")
    ax.set_ylabel("MAE relative to train-mean fill (log)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3)
    ax.set_title("Reconstructing a lost channel (test, 60-min gaps)", pad=28)
    ax = axes[1]
    if fr is not None:
        ks = [1, 2, 4]
        for fill, col, lab in (("zero", S.SERIES[1], "channels zero-filled"), ("recon", S.SERIES[0], "channels reconstructed")):
            d = fr[(fr.model == "gated_fusion") & (fr.task == "health")]
            m = [d[d.scenario == f"tel_drop{k}_{fill}"]["F1"].mean() for k in ks]
            s = [d[d.scenario == f"tel_drop{k}_{fill}"]["F1"].std() for k in ks]
            ax.errorbar(ks, m, yerr=np.nan_to_num(s), color=col, marker="o", ms=4, capsize=2, label=lab)
        base = fr[(fr.model == "gated_fusion") & (fr.task == "health") & (fr.scenario == "all_present")]["F1"].mean()
        ax.axhline(base, color=S.INK_2, ls=":", lw=1); ax.text(4.05, base, "no channel lost", fontsize=7, color=S.INK_2, va="center")
        ax.set_xticks(ks); ax.set_xlabel("telemetry channels lost per pass"); ax.set_ylabel("health F1 (gated fusion, test)")
        ax.legend(loc="lower left"); ax.set_title("RQ3: effect on health detection")
    return S.save(fig, path)


def ablations(path, fr):
    models = [m for m in ABLATIONS if m in set(fr.model)]
    scen_sets = {"all inputs": ["all_present"],
                 "inputs missing (avg)": ["missing_optical", "missing_sar", "missing_telemetry", "only_optical", "only_sar"]}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, task, title in zip(axes, ("eo", "health"), ("Flood F1", "Health F1")):
        x = np.arange(len(models)); w = 0.38
        for k, (lab, sc) in enumerate(scen_sets.items()):
            v = [fr[(fr.model == m) & (fr.task == task) & fr.scenario.isin(sc)]["F1"].mean() for m in models]
            b = ax.bar(x + (k - 0.5) * w, v, width=w - 0.03, color=S.SERIES[k], label=lab)
            ax.bar_label(b, fmt="%.2f", fontsize=6.5, padding=2, color=S.INK_2)
        ax.set_xticks(x, [NICE[m] for m in models], rotation=30, ha="right"); ax.set_title(title); ax.set_ylim(0, 1.12)
    axes[0].legend(loc="lower left")
    fig.suptitle("Ablations of the proposed model (test, mean over seeds)", color=S.INK)
    stamp(fig, is_synth(fr, "eo_dataset"))
    return S.save(fig, path)


def main(argv=None) -> int:
    argparse.ArgumentParser(description="Member 4 Phase-2 figures.").parse_args(argv)
    S.apply()
    out = [eo_baselines(C.FIGURES / "eo_baselines.png")]
    fr = _fusion()
    if fr is not None:
        out += [fusion_comparison(C.FIGURES / "fusion_comparison.png", fr),
                robustness_missing(C.FIGURES / "robustness_missing.png", fr),
                robustness_cloud(C.FIGURES / "robustness_cloud.png", fr),
                ablations(C.FIGURES / "ablations.png", fr)]
    out += [alpha_behaviour(C.FIGURES / "alpha_behaviour.png"), reconstruction(C.FIGURES / "reconstruction.png", fr)]
    for p in out:
        if p:
            print("saved", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
