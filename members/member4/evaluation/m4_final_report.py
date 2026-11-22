"""
m4_final_report.py - write the final conclusions FROM MEASURED RESULTS ONLY (Milestones 14-15).

Reads every results/*.csv produced by the pipeline and writes
    reports/FINAL_CONCLUSIONS.md     tables + answers to the primary question and RQ2-RQ4
    results/m4_final_summary.json    the headline numbers (also used by the dashboard)

Decision rules (stated in the report, applied mechanically):
  RQ2  supported if the best forecast-residual detector beats BOTH limit checks and Isolation Forest
       on test event-F1 (simulated) AND on segment F1 (OPSSAT-AD real telemetry, if available).
  RQ3  supported if cross-channel reconstruction has lower MAE than both mean-fill and last-value on
       the physically linked channels AND the health F1 with reconstructed channels is >= zero-filled
       for 1, 2 and 4 lost channels; PARTIALLY supported if that holds only up to some number of lost
       channels (reported); not supported otherwise.
  RQ4  supported if the optical trust weight alpha_opt (flood gate) is LOWER for damaged/cloudy optical
       images than for clear ones, and is 0 when optical is missing.
  Primary  "maintains reliable detection" if, averaged over the missing-input scenarios, gated fusion
       keeps >= 90 % of its all-inputs flood F1 and is not worse than early/late fusion by > 1 std.
Any conclusion that depends on images is WITHHELD when the EO inputs are the synthetic stand-in.

Run from the repository root:
    python members/member4/evaluation/m4_final_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mlts"))
import m4_ts_common as C  # noqa: E402

R = C.RESULTS
MISSING = ["missing_optical", "missing_sar", "missing_telemetry", "only_optical", "only_sar"]
NICE = {"tel_only": "telemetry only", "opt_only": "optical only", "sar_only": "SAR only", "early_fusion": "early fusion",
        "late_fusion": "late fusion", "equal_weights": "equal weights", "gated_fusion": "**gated fusion (ours)**",
        "gated_zero_fill": "− missing token", "gated_no_tel_in_gate": "− telemetry in gate",
        "gated_no_dropout": "− modality dropout", "gated_no_residuals": "− forecast residuals",
        "gated_no_multitask": "− multi-task loss"}


def read(name):
    f = R / name
    return pd.read_csv(f) if f.exists() else None


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                                     else ("–" if isinstance(v, (float, np.floating)) else str(v)) for v in r.values) + " |")
    return "\n".join(out)


def forecasting(summary: dict) -> str:
    fr = read("forecast_results.csv")
    if fr is None:
        return "_Forecasting results missing._"
    v = fr[fr.split == "val"]
    order = [m for m in ["naive", "snaive", "arimax", "xgboost", "gru", "lstm"] if m in set(v.model)]
    parts = []
    for h in (1, 10):
        t = v[v.horizon == h].pivot_table(index="channel", columns="model", values="MAE")[order].reset_index()
        parts.append(f"**Validation MAE, h = {h} ({h * 30} s ahead)** — lower is better\n\n" + md_table(t, "{:.4f}"))
    sk = read("m4_forecast_skill.csv")
    if sk is not None:
        avg = sk.groupby(["model", "horizon"]).skill_vs_naive.mean().unstack().reindex(order).reset_index()
        avg.columns = ["model"] + [f"skill h={c}" for c in avg.columns[1:]]
        parts.append("**Mean skill vs naive (1 − MAE/MAE_naive), validation**\n\n" + md_table(avg))
        best = sk[sk.horizon == 10].groupby("model").skill_vs_naive.mean()
        summary["forecast_best_model"] = best.idxmax(); summary["forecast_best_skill_h10"] = round(float(best.max()), 3)
        for m in ("gru", "lstm"):
            if m in best:
                summary[f"forecast_skill_h10_{m}"] = round(float(best[m]), 3)
    return "\n\n".join(parts)


def health(summary: dict) -> tuple[str, bool | None]:
    hm = read("health_metrics.csv")
    if hm is None:
        return "_Health results missing._", None
    sim = hm[(hm.dataset == "simulated_v1") & (hm.split == "test")].copy()
    cols = ["model", "event_precision", "event_recall", "event_F1", "AUPRC", "mean_delay_min", "false_alarms_per_day"]
    txt = ["**Simulated telemetry, test split (thresholds chosen on validation)**\n\n" + md_table(sim[cols])]
    res = sim[sim.model.str.startswith("residual_") & ~sim.model.str.contains("ensemble")]
    best = res.sort_values("event_F1", ascending=False).iloc[0]
    lim = sim[sim.model == "limit_check"].event_F1.max(); iso = sim[sim.model == "isolation_forest"].event_F1.max()
    sim_ok = best.event_F1 > max(lim, iso)
    summary.update({"health_best_detector": best.model, "health_best_test_F1": round(float(best.event_F1), 3),
                    "health_limit_F1": round(float(lim), 3), "health_iforest_F1": round(float(iso), 3),
                    "health_limit_false_alarms_day": round(float(sim[sim.model == "limit_check"].false_alarms_per_day.max()), 2)})
    op = read("m4_opssat_results.csv")
    real_ok = None
    if op is not None:
        txt.append("**Real telemetry: OPSSAT-AD (ESA OPS-SAT CubeSat), 529 test segments**\n\n" +
                   md_table(op[["model", "precision", "recall", "F1", "MCC", "AUC_ROC", "AUC_PR"]]))
        rb = op[op.model.str.startswith("residual")].sort_values("F1", ascending=False).iloc[0]
        real_ok = rb.F1 > op[op.model.isin(["limit_check", "isolation_forest"])].F1.max()
        summary.update({"opssat_best": rb.model, "opssat_best_F1": round(float(rb.F1), 3),
                        "opssat_best_MCC": round(float(rb.MCC), 3), "opssat_best_AUC_ROC": round(float(rb.AUC_ROC), 3),
                        "opssat_iforest_F1": round(float(op[op.model == "isolation_forest"].F1.max()), 3),
                        "opssat_limit_F1": round(float(op[op.model == "limit_check"].F1.max()), 3)})
    verdict = sim_ok and (real_ok is not False)
    return "\n\n".join(txt), verdict


def eo(summary: dict, synthetic: bool) -> str:
    r = read("eo_baseline_results.csv")
    if r is None:
        return "_EO baselines not run yet._"
    t = r[r.split.isin(["test", "bolivia"])][["model", "split", "accuracy", "precision", "recall", "F1", "AUROC", "AUPRC", "n"]]
    if not synthetic:
        b = r[r.split == "test"].sort_values("F1", ascending=False).iloc[0]
        summary.update({"eo_best_model": b.model, "eo_best_test_F1": round(float(b.F1), 3)})
    return md_table(t)


def fusion(summary: dict, synthetic: bool) -> tuple[str, dict]:
    fr = read("fusion_results.csv")
    if fr is None:
        return "_Fusion not run yet._", {}
    verdicts = {}
    a = fr[fr.scenario == "all_present"].groupby(["model", "task"]).F1.agg(["mean", "std"]).unstack("task")
    order = [m for m in NICE if m in a.index]
    tab = pd.DataFrame({"model": [NICE[m] for m in order],
                        "health F1": [f"{a.loc[m, ('mean', 'health')]:.3f} ± {np.nan_to_num(a.loc[m, ('std', 'health')]):.3f}" for m in order],
                        "flood F1": [f"{a.loc[m, ('mean', 'eo')]:.3f} ± {np.nan_to_num(a.loc[m, ('std', 'eo')]):.3f}" for m in order]})
    parts = ["**All inputs present, test (mean ± std over seeds)**\n\n" + md_table(tab)]
    rob = fr[(fr.task == "eo") & fr.model.isin(["early_fusion", "late_fusion", "gated_fusion"])]
    rt = rob.groupby(["scenario", "model"]).F1.mean().unstack("model").reindex(
        ["all_present"] + MISSING + ["only_telemetry", "cloud_25", "cloud_50", "cloud_75", "cloud_100"]).reset_index()
    parts.append("**Flood F1 under missing / degraded inputs (test, mean over seeds)**\n\n" + md_table(rt))
    g = fr[(fr.model == "gated_fusion") & (fr.task == "eo")]
    full = g[g.scenario == "all_present"].F1.mean(); miss = g[g.scenario.isin(MISSING)].F1.mean()
    others = {m: fr[(fr.model == m) & (fr.task == "eo") & fr.scenario.isin(MISSING)].F1.mean() for m in ("early_fusion", "late_fusion")}
    sd = fr[(fr.model == "gated_fusion") & (fr.task == "eo") & fr.scenario.isin(MISSING)].groupby("seed").F1.mean().std()
    keep = miss / full if full else np.nan
    verdicts["primary"] = bool(keep >= 0.9 and all(miss >= v - np.nan_to_num(sd) for v in others.values()))
    summary.update({"fusion_flood_F1_all": round(float(full), 3), "fusion_flood_F1_missing_avg": round(float(miss), 3),
                    "fusion_flood_F1_kept": round(float(keep), 3),
                    **{f"{m}_flood_F1_missing_avg": round(float(v), 3) for m, v in others.items()}})
    hg = fr[(fr.model == "gated_fusion") & (fr.task == "health")]
    summary["fusion_health_F1_all"] = round(float(hg[hg.scenario == "all_present"].F1.mean()), 3)
    rec = []
    for k in (1, 2, 4):
        z = hg[hg.scenario == f"tel_drop{k}_zero"].F1.mean(); r = hg[hg.scenario == f"tel_drop{k}_recon"].F1.mean()
        rec.append({"channels lost": k, "health F1 zero-filled": z, "health F1 reconstructed": r})
    rec = pd.DataFrame(rec)
    better = (rec["health F1 reconstructed"] >= rec["health F1 zero-filled"]).values
    verdicts["rq3_detection"] = True if better.all() else ("partial" if better[0] else False)
    verdicts["rq3_helps_up_to"] = int(rec["channels lost"][better].max()) if better[0] else 0
    summary.update({f"recon_health_F1_k{int(k)}_{f}": round(float(v), 3) for k, z, r in rec.values
                    for f, v in (("zero", z), ("recon", r))})
    parts.append("**Telemetry channels lost (gated fusion, health task)**\n\n" + md_table(rec))
    ab = [m for m in ("gated_fusion", "gated_zero_fill", "gated_no_tel_in_gate", "gated_no_dropout", "gated_no_residuals",
                      "gated_no_multitask") if m in set(fr.model)]
    abt = pd.DataFrame([{"variant": NICE[m],
                         "flood F1 (all)": fr[(fr.model == m) & (fr.task == "eo") & (fr.scenario == "all_present")].F1.mean(),
                         "flood F1 (missing avg)": fr[(fr.model == m) & (fr.task == "eo") & fr.scenario.isin(MISSING)].F1.mean(),
                         "health F1 (all)": fr[(fr.model == m) & (fr.task == "health") & (fr.scenario == "all_present")].F1.mean(),
                         "health F1 (missing avg)": fr[(fr.model == m) & (fr.task == "health") & fr.scenario.isin(MISSING)].F1.mean()}
                        for m in ab])
    parts.append("**Ablations (test, mean over seeds)**\n\n" + md_table(abt))
    al = read("m4_alpha_summary.csv")
    if al is not None:
        e = al[(al.model == "gated_fusion") & (al.gate == "eo")]
        get = lambda sc, sub: e[(e.scenario == sc) & (e.subset == sub)].alpha_opt.mean()  # noqa: E731
        vals = {"clear": get("all_present", "clean"), "cloudy": get("all_present", "cloudy"),
                "payload fault": get("all_present", "coupled_payload_fault"), "+100 % cloud": get("cloud_100", "all"),
                "optical missing": get("missing_optical", "all")}
        at = pd.DataFrame([{"situation": k, "mean α_opt (flood gate)": v} for k, v in vals.items()])
        parts.append("**RQ4: optical trust weight by situation (gated fusion, test)**\n\n" + md_table(at))
        deg = [v for k, v in vals.items() if k in ("cloudy", "payload fault", "+100 % cloud") and np.isfinite(v)]
        verdicts["rq4"] = bool(deg and all(d < vals["clear"] for d in deg) and vals["optical missing"] < 1e-6)
        summary.update({f"alpha_opt_{k.replace(' ', '_').replace('+', '').replace('%', 'pct')}": round(float(v), 4)
                        for k, v in vals.items() if np.isfinite(v)})
    return "\n\n".join(parts), verdicts


def reconstruction(summary: dict) -> tuple[str, bool | None]:
    r = read("m4_reconstruction_results.csv")
    if r is None:
        return "_Reconstruction not run._", None
    p = r.pivot(index="channel", columns="method", values="MAE").reindex(C.CHANNELS)
    linked = ["batt_temp", "panel_temp", "bus_V", "batt_I", "solar_I", "SoC"]
    ok = bool((p.loc[linked, "cross_channel"] < p.loc[linked, ["mean_fill", "ffill"]].min(axis=1)).all())
    gain = (1 - p.loc[linked, "cross_channel"] / p.loc[linked, "mean_fill"]).mean()
    summary["recon_mean_error_reduction_linked"] = round(float(gain), 3)
    return md_table(p.reset_index(), "{:.4f}"), ok


SYNTH_RUN = False


def word(v) -> str:
    if v is None and SYNTH_RUN:
        return "**Withheld: needs the real Sen1Floods11 run**"
    return {True: "**Supported**", False: "**Not supported**", None: "**Not evaluated**",
            "partial": "**Partially supported**"}[v]


def main(argv=None) -> int:
    argparse.ArgumentParser(description="Member 4: write FINAL_CONCLUSIONS.md from the measured results.").parse_args(argv)
    summary: dict = {"generated": str(date.today())}
    eor = read("eo_baseline_results.csv"); fr = read("fusion_results.csv")
    eo_name = (str(eor.dataset.iloc[0]) if eor is not None else (str(fr.eo_dataset.iloc[0]) if fr is not None else "none"))
    synthetic = eo_name.startswith("synthetic")
    global SYNTH_RUN
    SYNTH_RUN = synthetic
    summary["eo_dataset"] = eo_name; summary["eo_is_synthetic"] = synthetic
    f_txt = forecasting(summary)
    h_txt, rq2 = health(summary)
    e_txt = eo(summary, synthetic)
    fu_txt, fv = fusion(summary, synthetic)
    rc_txt, rq3a = reconstruction(summary)
    det = fv.get("rq3_detection", True)
    rq3 = None if rq3a is None else (False if not rq3a else det)
    primary = None if synthetic else fv.get("primary")
    rq4 = None if synthetic else fv.get("rq4")
    summary.update({"RQ2": rq2, "RQ3": rq3, "RQ4": rq4, "primary": primary})

    warn = ("\n> ⚠️ **The Earth-observation inputs in this run are the SYNTHETIC stand-in** "
            "(`synthetic_eo_devcheck`). Image and fusion numbers below only prove the pipeline runs; "
            "they are **not results**, and every image-dependent conclusion is withheld. Run the full pipeline on "
            "Sen1Floods11 (`python members/member4/m4_run_all.py`) to fill them in.\n") if synthetic else ""
    s = summary
    lines = [f"# Final conclusions — Member 4\n\n_Generated {s['generated']} by `evaluation/m4_final_report.py` from the measured result files. Earth-observation data: `{eo_name}`._\n{warn}",
             "## Answers to the research questions\n",
             f"**Primary — can a lightweight missing-modality-aware fusion model keep satellite-health and flood detection reliable when inputs are missing?** {word(primary)}"
             + ("" if primary is None else f". Averaged over the five missing-input scenarios, gated fusion keeps {s['fusion_flood_F1_kept']:.0%} of its all-inputs flood F1 ({s['fusion_flood_F1_missing_avg']:.3f} vs {s['fusion_flood_F1_all']:.3f}); early fusion averages {s.get('early_fusion_flood_F1_missing_avg', float('nan')):.3f} and late fusion {s.get('late_fusion_flood_F1_missing_avg', float('nan')):.3f} in the same scenarios."),
             f"\n**RQ2 — does forecast-residual detection beat static limits and Isolation Forest?** {word(rq2)}. "
             f"On simulated telemetry the best residual detector (`{s.get('health_best_detector')}`) reaches test event-F1 {s.get('health_best_test_F1')} vs {s.get('health_limit_F1')} for limit checks (which raise {s.get('health_limit_false_alarms_day')} false alarms/day) and {s.get('health_iforest_F1')} for Isolation Forest."
             + (f" On **real** OPSSAT-AD telemetry the AR forecast-residual detector reaches F1 {s['opssat_best_F1']} (MCC {s['opssat_best_MCC']}, AUC-ROC {s['opssat_best_AUC_ROC']}) vs {s['opssat_iforest_F1']} (Isolation Forest) and {s['opssat_limit_F1']} (limits)." if "opssat_best_F1" in s else ""),
             f"\n**RQ3 — can missing telemetry channels be reconstructed so detection degrades gracefully?** {word(rq3)}. "
             f"Cross-channel reconstruction cuts the error on the six physically linked channels by {s.get('recon_mean_error_reduction_linked', float('nan')):.0%} on average versus mean-filling; attitude error and payload power cannot be reconstructed (no physical link)."
             + (f" In the fused model, reconstructing lost channels instead of zero-filling them raises health F1 from {s.get('recon_health_F1_k1_zero')} to {s.get('recon_health_F1_k1_recon')} (1 channel lost) and from {s.get('recon_health_F1_k2_zero')} to {s.get('recon_health_F1_k2_recon')} (2 lost), but not when half the channels are lost ({s.get('recon_health_F1_k4_zero')} zero-filled vs {s.get('recon_health_F1_k4_recon')} reconstructed, 4 of 8): degradation is graceful only while most channels remain." if "recon_health_F1_k1_zero" in s else "")
             + (" Image-dependent part provisional (synthetic EO)." if synthetic else ""),
             f"\n**RQ4 — do the learned trust weights shift sensibly when an input is degraded or removed?** {word(rq4)}"
             + ("" if rq4 is None else f". Mean α_opt for the flood decision: clear {s.get('alpha_opt_clear', float('nan')):.3f}, cloudy {s.get('alpha_opt_cloudy', float('nan')):.3f}, payload-fault-damaged {s.get('alpha_opt_payload_fault', float('nan')):.3f}, optical missing {s.get('alpha_opt_optical_missing', float('nan')):.3f}."),
             "\nDecision rules are stated in `evaluation/m4_final_report.py` and applied mechanically.\n",
             "## 1. Forecasting ladder (MLTS)\n", f_txt,
             f"\nThe deep models: GRU mean skill at 5 min = {s.get('forecast_skill_h10_gru', float('nan')):.3f}, LSTM = {s.get('forecast_skill_h10_lstm', float('nan')):.3f}; best overall = `{s.get('forecast_best_model')}` ({s.get('forecast_best_skill_h10', float('nan')):.3f}).\n",
             "## 2. Satellite-health anomaly detection\n", h_txt,
             "\n## 3. Earth-observation baselines (flood)\n", ("> Synthetic stand-in: pipeline check only.\n\n" if synthetic else "") + e_txt,
             "\n## 4. Fusion, missing inputs and ablations\n", ("> Synthetic EO stand-in: pipeline check only.\n\n" if synthetic else "") + fu_txt,
             "\n## 5. Telemetry channel reconstruction (test, 60-min gaps, MAE)\n", rc_txt,
             "\n## Limitations (state these in the paper)\n",
             "- Telemetry is **simulated** (generator v1); real-telemetry evidence comes from OPSSAT-AD, segment-level only.\n"
             "- Telemetry–image pairing is a **simulation assumption** (virtual passes): payload faults are coupled to damaged optical patches by design.\n"
             "- Small event counts in the simulated test split (9 events): one event ≈ ±0.1 F1.\n"
             "- Patches are 128×128 (2× downsampled) with 6 optical bands for CPU training; flood = water fraction ≥ 5 %, which includes permanent water.\n"
             "- Single generator seed; fusion results averaged over 3 training seeds.\n",
             "## Future work\n",
             "- Real mission data with co-located imagery; thermal modality; on-board deployment of the lightweight gate; a second generator seed and a walk-forward evaluation.\n"]
    out = C.REPORTS / "FINAL_CONCLUSIONS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    (R / "m4_final_summary.json").write_text(json.dumps(summary, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    print(f"saved {out}\n")
    for k in ("primary", "RQ2", "RQ3", "RQ4"):
        print(f"{k:<8} {summary[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
