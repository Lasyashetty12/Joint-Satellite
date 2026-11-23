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

Fair comparison (version 2 report):
  * forecasters are SELECTED on validation and REPORTED on the test split
  * one table compares every fusion model on every input condition (mean ± std over seeds)
  * paired bootstrap (2000 resamples of the test passes, same passes for both models) gives a 95 %
    confidence interval for gated − early and gated − late fusion; descriptive only, the decision
    rules above are unchanged
  * if results/v1/ exists (evaluation/m4_archive_v1.py), version 1 and version 2 are shown side by
    side, both judged by the same rules

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
        "gated_no_multitask": "− multi-task loss", "gated_no_quality": "− quality check in gate"}
CLOUD = ["cloud_25", "cloud_50", "cloud_75", "cloud_100"]
DROPS = [f"tel_drop{k}_{f}" for k in (1, 2, 4) for f in ("zero", "recon")]
FC_NICE = {"naive": "Naive", "snaive": "Seasonal naive", "arimax": "ARIMAX", "xgboost": "XGBoost", "gru": "GRU", "lstm": "LSTM"}


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
        summary["forecast_best_model"] = best.idxmax(); summary["forecast_best_skill_h10_val"] = round(float(best.max()), 3)
    # final comparison on the TEST split (models were chosen on validation, never on test)
    t = fr[fr.split == "test"]
    if len(t):
        mae = t.pivot_table(index=["channel", "horizon"], columns="model", values="MAE")
        sk_t = (1 - mae.div(mae["naive"], axis=0)).groupby(level="horizon").mean()[order]
        tab = sk_t.T.reset_index(); tab.columns = ["model"] + [f"skill h={h}" for h in sk_t.index]
        tab["model"] = tab.model.map(FC_NICE)
        parts.append("**Mean skill vs naive on the TEST split (final comparison; higher is better)**\n\n" + md_table(tab))
        h10 = sk_t.loc[10] if 10 in sk_t.index else sk_t.iloc[-1]
        sel = summary.get("forecast_best_model", h10.idxmax())
        summary["forecast_best_skill_h10"] = round(float(h10[sel]), 3)
        summary["forecast_test_skill_h10"] = {m: round(float(v), 3) for m, v in h10.items()}
        for m in ("gru", "lstm"):
            if m in h10:
                summary[f"forecast_skill_h10_{m}"] = round(float(h10[m]), 3)
        rank = ", ".join(f"{FC_NICE[m]} {v:.3f}" for m, v in h10.sort_values(ascending=False).items())
        parts.append(f"Ranking at 5 min ahead on test: {rank}. `{sel}` was chosen on validation.")
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
    ab = [m for m in ("gated_fusion", "gated_no_quality", "gated_zero_fill", "gated_no_tel_in_gate", "gated_no_dropout",
                      "gated_no_residuals", "gated_no_multitask") if m in set(fr.model)]
    abt = pd.DataFrame([{"variant": NICE[m],
                         "flood F1 (all)": fr[(fr.model == m) & (fr.task == "eo") & (fr.scenario == "all_present")].F1.mean(),
                         "flood F1 (missing avg)": fr[(fr.model == m) & (fr.task == "eo") & fr.scenario.isin(MISSING)].F1.mean(),
                         "health F1 (all)": fr[(fr.model == m) & (fr.task == "health") & (fr.scenario == "all_present")].F1.mean(),
                         "health F1 (missing avg)": fr[(fr.model == m) & (fr.task == "health") & fr.scenario.isin(MISSING)].F1.mean()}
                        for m in ab])
    parts.append("**Ablations (test, mean over seeds)**\n\n" + md_table(abt))
    parts.append(all_models(fr))
    parts.append(bootstrap_section(summary))
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


def _ms(fr, m, task, scens) -> str:
    d = fr[(fr.model == m) & (fr.task == task) & fr.scenario.isin(scens)]
    if not len(d):
        return "–"
    per_seed = d.groupby("seed").F1.mean()
    return f"{per_seed.mean():.3f} ± {np.nan_to_num(per_seed.std()):.3f}"


def all_models_table(fr: pd.DataFrame, plain: bool = False) -> pd.DataFrame:
    """Every fusion model on every input condition, F1 mean ± std over seeds (each seed averaged over the
    scenarios of the group first). Unimodal models are scored only where their input exists."""
    order = [m for m in NICE if m in set(fr.model)]
    groups = {"all inputs": ["all_present"], "inputs missing (5)": MISSING, "cloud 25-100 %": CLOUD,
              "sensors lost (1-4)": DROPS}
    rows = []
    for m in order:
        r = {"model": NICE[m].replace("**", "") if plain else NICE[m]}
        for task, tname in (("eo", "flood"), ("health", "health")):
            for gname, sc in groups.items():
                r[f"{tname} F1, {gname}"] = _ms(fr, m, task, sc)
        rows.append(r)
    return pd.DataFrame(rows)


def all_models(fr: pd.DataFrame) -> str:
    return ("**Every model compared (test, F1 mean ± std over seeds)**\n\n" + md_table(all_models_table(fr))
            + "\n\n_Each cell averages the scenarios of its group per seed, then over seeds. A dash means the model "
              "has no input in that condition._")


def _f1_boot(y, p, thr, W) -> np.ndarray:
    pred = p >= thr
    tp = W @ (pred & (y == 1)).astype(float); fp = W @ (pred & (y == 0)).astype(float)
    fn = W @ (~pred & (y == 1)).astype(float)
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)


def paired_bootstrap(sp: pd.DataFrame, a: str, b: str, task: str, scens: list[str], B: int = 2000,
                     seed: int = 0) -> tuple[float, float, float] | None:
    """Difference in mean F1 (a - b) over seeds x scenarios, with a 95 % CI from resampling test passes.
    The same resampled passes are used for both models (paired)."""
    ids = np.sort(sp.pass_id.unique()); n = len(ids)
    W = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=B).astype(float)
    W = np.vstack([np.ones(n), W])                       # row 0 = the actual test set
    diffs, cnt = np.zeros(B + 1), 0
    for (sd, sc), d in sp[sp.scenario.isin(scens)].groupby(["seed", "scenario"]):
        da, db = d[d.model == a].set_index("pass_id").reindex(ids), d[d.model == b].set_index("pass_id").reindex(ids)
        if da.p_eo.isna().any() or db.p_eo.isna().any() or da[f"y_{task}"].isna().any():
            continue                                     # a model or the labels are missing for this run
        y = da[f"y_{task}"].values.astype(int)
        diffs += _f1_boot(y, da[f"p_{task}"].values, da[f"thr_{task}"].iloc[0], W) \
            - _f1_boot(y, db[f"p_{task}"].values, db[f"thr_{task}"].iloc[0], W)
        cnt += 1
    if not cnt:
        return None
    diffs /= cnt
    return float(diffs[0]), float(np.percentile(diffs[1:], 2.5)), float(np.percentile(diffs[1:], 97.5))


def bootstrap_table(summary: dict | None = None) -> pd.DataFrame | str:
    """Paired bootstrap table, or a sentence explaining why it is not available."""
    summary = {} if summary is None else summary
    f = R / "m4_fusion_scenario_predictions.csv.gz"
    if not f.exists():
        return "Paired bootstrap: per-pass predictions not available (run the version 2 fusion training)."
    sp = pd.read_csv(f)
    if "y_eo" not in sp.columns:
        return "Paired bootstrap: the prediction file has no labels; rerun the fusion training."
    rows = []
    for task, tname in (("eo", "flood"), ("health", "health")):
        for gname, sc in (("all inputs", ["all_present"]), ("inputs missing (5)", MISSING), ("cloud 25-100 %", CLOUD)):
            for other in ("early_fusion", "late_fusion"):
                r = paired_bootstrap(sp, "gated_fusion", other, task, sc) if other in set(sp.model) else None
                if r is None:
                    continue
                d, lo, hi = r
                verdict = "gated better" if lo > 0 else ("gated worse" if hi < 0 else "no clear difference")
                rows.append({"task": tname, "condition": gname, "gated minus": NICE[other], "difference in F1": round(d, 4),
                             "95 % CI": f"[{lo:+.3f}, {hi:+.3f}]", "reading": verdict})
                if task == "eo" and gname == "inputs missing (5)":
                    summary[f"boot_flood_missing_vs_{other}"] = {"diff": round(d, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
    if not rows:
        return "Paired bootstrap: no comparable predictions."
    return pd.DataFrame(rows)


def bootstrap_section(summary: dict) -> str:
    t = bootstrap_table(summary)
    if isinstance(t, str):
        return f"_{t}_"
    return ("**Is gated fusion really different from simple fusion? Paired bootstrap, 2000 resamples of the test "
            "passes**\n\n" + md_table(t) +
            "\n\n_If the 95 % interval contains 0, the two models cannot be told apart on this test set. "
            "Descriptive only: the decision rules are unchanged._")


def versions(summary: dict) -> str:
    v1f = R / "v1" / "m4_final_summary.json"
    if not v1f.exists():
        return ""
    v1 = json.loads(v1f.read_text(encoding="utf-8"))
    if str(v1.get("eo_dataset")) != str(summary.get("eo_dataset")):
        return ("## Version 1 and version 2\n\n_The archived version 1 results use a different image dataset "
                f"(`{v1.get('eo_dataset')}`), so they are not compared here._\n")
    w = lambda v: {True: "Supported", False: "Not supported", "partial": "Partly supported", None: "Withheld"}.get(v, str(v))  # noqa: E731
    keys = [("primary", "Primary"), ("RQ2", "RQ2"), ("RQ3", "RQ3"), ("RQ4", "RQ4")]
    t = pd.DataFrame([{"question": n, "version 1": w(v1.get(k)), "version 2": w(summary.get(k))} for k, n in keys])
    nums = [("fusion_flood_F1_all", "gated flood F1, all inputs"), ("fusion_flood_F1_missing_avg", "gated flood F1, inputs missing"),
            ("early_fusion_flood_F1_missing_avg", "early fusion flood F1, inputs missing"),
            ("late_fusion_flood_F1_missing_avg", "late fusion flood F1, inputs missing"),
            ("fusion_health_F1_all", "gated health F1, all inputs"),
            ("recon_health_F1_k4_zero", "health F1, 4 lost, zero-filled"), ("recon_health_F1_k4_recon", "health F1, 4 lost, rebuilt"),
            ("alpha_opt_clear", "optical trust, clear"), ("alpha_opt_cloudy", "optical trust, cloudy"),
            ("alpha_opt_100_pct_cloud", "optical trust, 100 % cloud"), ("alpha_opt_payload_fault", "optical trust, payload fault")]
    n = pd.DataFrame([{"measure": lab, "version 1": v1.get(k, np.nan), "version 2": summary.get(k, np.nan)} for k, lab in nums])
    return ("## Version 1 and version 2\n\n"
            "Version 2 changes the training and the model, not the rules: every model practises with lost telemetry "
            "sensors and with extra cloud, rebuilt channels stay flagged as not measured, and every model receives "
            "five image-quality numbers that the reliability gate also reads directly. Both versions are judged by the "
            "same decision rules on the same test passes.\n\n" + md_table(t) + "\n\n" + md_table(n) + "\n")


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


def rq3_sentence(s: dict) -> str:
    parts, worse = [], []
    for k in (1, 2, 4):
        z, r = s.get(f"recon_health_F1_k{k}_zero"), s.get(f"recon_health_F1_k{k}_recon")
        parts.append(f"{z} → {r} with {k} lost"); (worse.append(k) if r is not None and z is not None and r < z else None)
    head = "In the fused model, health F1 when lost channels are rebuilt instead of zero-filled: " + "; ".join(parts) + "."
    if not worse:
        return head + " Rebuilding is at least as good as zero-filling in every case."
    return head + f" Rebuilding is worse than zero-filling with {', '.join(map(str, worse))} channel(s) lost."


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
             + (" " + rq3_sentence(s) if "recon_health_F1_k1_zero" in s else "")
             + (" Image-dependent part provisional (synthetic EO)." if synthetic else ""),
             f"\n**RQ4 — do the learned trust weights shift sensibly when an input is degraded or removed?** {word(rq4)}"
             + ("" if rq4 is None else f". Mean α_opt for the flood decision: clear {s.get('alpha_opt_clear', float('nan')):.3f}, cloudy {s.get('alpha_opt_cloudy', float('nan')):.3f}, payload-fault-damaged {s.get('alpha_opt_payload_fault', float('nan')):.3f}, optical missing {s.get('alpha_opt_optical_missing', float('nan')):.3f}."),
             "\nDecision rules are stated in `evaluation/m4_final_report.py` and applied mechanically.\n",
             versions(s),
             "## 1. Forecasting ladder (MLTS)\n", f_txt,
             f"\nOn the test split at 5 min ahead: GRU skill {s.get('forecast_skill_h10_gru', float('nan')):.3f}, LSTM {s.get('forecast_skill_h10_lstm', float('nan')):.3f}; the validation choice `{s.get('forecast_best_model')}` scores {s.get('forecast_best_skill_h10', float('nan')):.3f}.\n",
             "## 2. Satellite-health anomaly detection\n", h_txt,
             "\n## 3. Earth-observation baselines (flood)\n", ("> Synthetic stand-in: pipeline check only.\n\n" if synthetic else "") + e_txt,
             "\n## 4. Fusion, missing inputs and ablations\n", ("> Synthetic EO stand-in: pipeline check only.\n\n" if synthetic else "") + fu_txt,
             "\n## 5. Telemetry channel reconstruction (test, 60-min gaps, MAE)\n", rc_txt,
             "\n## Limitations\n",
             "- Telemetry is **simulated** (generator v1); real-telemetry evidence comes from OPSSAT-AD, segment-level only.\n"
             "- Telemetry–image pairing is a **simulation assumption** (virtual passes): payload faults are coupled to damaged optical patches by design.\n"
             "- Small event counts in the simulated test split (9 events): one event ≈ ±0.1 F1.\n"
             "- Patches are 128×128 (2× downsampled) with 6 optical bands for CPU training; flood = water fraction ≥ 5 %, which includes permanent water.\n"
             "- Single generator seed; fusion results averaged over 3 training seeds.\n"
             "- The image-quality check is fitted on clean training patches and tested on SIMULATED damage (clouds, stripes, noise, saturation); real clouds in Sen1Floods11 are not labelled, so its behaviour on real clouds is not measured.\n",
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
