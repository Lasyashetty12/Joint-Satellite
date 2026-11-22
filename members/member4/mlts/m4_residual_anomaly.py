"""
m4_residual_anomaly.py - health-anomaly detection on Member 4's telemetry (Milestone 5).

Main method: FORECAST-RESIDUAL scoring (handbook section 4)
    r_c[t]    = |y_c[t] - y_hat_c[t]| / sd_c        y_hat = 1-step forecast made at t-1
                                                     sd_c  = std of residuals on NORMAL TRAIN points
    score[t]  = max over channels of r_c[t]          ("the most surprised channel")
    smooth[t] = EWMA(score, span = 10)               (reduces single-point noise)
    alarm[t]  = smooth[t] > tau                      (tau chosen on VALIDATION only, max event-F1)
    events    = alarms merged when separated by <= 5 min

Compared detectors (same smoothing, same threshold rule, same events):
    residual_<forecaster> for naive, snaive, arimax, xgboost, gru  (predictions from Milestone 4)
    residual_ensemble  max of XGBoost and ARIMAX scores, each scaled by its own val 99.9th percentile
    limit_check       fixed limits = train-normal 0.05 / 99.95 percentiles per channel (no tuning)
    isolation_forest  unsupervised baseline (m4_isolation_forest.py)

Scoring: event metrics with a 60-min tolerance after each labelled event (recovery tail),
AUPRC point-wise. Thresholds from VAL; TEST is reported once. Per-type recall is shown over
ALL splits as a diagnostic (detectors never trained on anomalies, thresholds from val only).

Outputs:
    results/health_metrics.csv           member, model, dataset, split, event_precision, event_recall,
                                         event_F1, AUPRC, mean_delay_min, false_alarms_per_day, threshold
    results/m4_health_per_type.csv       recall per anomaly type per detector (all splits)
    data/generated/m4_anomaly_scores.csv.gz   timestamp, model, score, threshold, pred, label (git-ignored)

Run from the repository root (after m4_forecast_eval.py):
    python members/member4/mlts/m4_residual_anomaly.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m4_ts_common as C  # noqa: E402
import m4_metrics as M  # noqa: E402

FORECASTERS = ["naive", "snaive", "arimax", "xgboost", "gru", "lstm"]
EWMA_SPAN = 10
MERGE_GAP = 10              # 5 min
TOL = 120                   # 60 min after a labelled event
LIMIT_Q = (0.05, 99.95)


# ------------------------------------------------------------------ scores
def residual_score(df: pd.DataFrame, pred: np.ndarray, horizons, masks, normal, channels=C.CHANNELS):
    """Normalised 1-step residuals -> (score [N], per-channel z [N, C], sd [C])."""
    k = list(horizons).index(1)
    y = df[channels].values
    yhat = np.full_like(y, np.nan, dtype=float)
    yhat[1:] = pred[:-1, k, :]                       # forecast for t made at t-1
    r = np.abs(y - yhat)
    ref = masks["train"] & normal
    sd = np.nanstd(r[ref], axis=0)
    sd[sd == 0] = 1.0
    z = r / sd
    with np.errstate(all="ignore"):
        score = np.where(np.isnan(z).all(axis=1), np.nan, np.nanmax(np.where(np.isnan(z), -np.inf, z), axis=1))
    return score, z, sd


def limit_score(df: pd.DataFrame, masks, normal, channels=C.CHANNELS, q=LIMIT_Q):
    ref = df.loc[masks["train"] & normal, channels]
    lo, hi = np.nanpercentile(ref.values, q[0], axis=0), np.nanpercentile(ref.values, q[1], axis=0)
    span = hi - lo
    x = df[channels].values
    exc = np.maximum((x - hi) / span, (lo - x) / span)
    with np.errstate(all="ignore"):
        s = np.nanmax(np.where(np.isnan(exc), -np.inf, exc), axis=1)
    s[np.isinf(s)] = np.nan
    return s, lo, hi


def smooth(score: np.ndarray, span: int = EWMA_SPAN) -> np.ndarray:
    return pd.Series(score).ewm(span=span, ignore_na=True).mean().values


# ------------------------------------------------------------------ thresholding
def true_events_in(events_typed, mask):
    return [(a, b, t) for a, b, t in events_typed if mask[a]]


def evaluate(s: np.ndarray, tau: float, mask: np.ndarray, events_typed, tol=TOL, gap=MERGE_GAP):
    flag = (s > tau) & mask & ~np.isnan(s)
    pe = M.events_from_binary(flag, max_gap=gap)
    te = true_events_in(events_typed, mask)
    return M.event_metrics(pe, [(a, b) for a, b, _ in te], int(mask.sum()), tol=tol), pe


def choose_threshold(s: np.ndarray, val_mask: np.ndarray, events_typed, n_grid: int = 80) -> float:
    v = s[val_mask & ~np.isnan(s)]
    cands = np.unique(np.quantile(v, np.concatenate([np.linspace(0.80, 0.99, 40),
                                                     np.linspace(0.99, 0.99995, n_grid - 40)])))
    best, best_key = cands[-1], (-1, 0)
    for tau in cands:
        m, _ = evaluate(s, tau, val_mask, events_typed)
        key = (round(m["event_F1"], 6), -m["false_alarms_per_day"])
        if key > best_key:
            best, best_key = tau, key
    return float(best)


def run_detector(name: str, raw: np.ndarray, df, masks, events_typed, label, do_smooth=True,
                 fixed_tau: float | None = None) -> tuple[list[dict], dict, np.ndarray, float, list]:
    s = smooth(raw) if do_smooth else raw
    tau = fixed_tau if fixed_tau is not None else choose_threshold(s, masks["val"], events_typed)
    rows = []
    for split in ("val", "test"):
        m, _ = evaluate(s, tau, masks[split], events_typed)
        ok = masks[split]
        rows.append({"member": "member4", "model": name, "dataset": "simulated_v1", "split": split,
                     **{k: m[k] for k in ("event_precision", "event_recall", "event_F1", "mean_delay_min",
                                          "false_alarms_per_day", "n_true_events", "n_pred_events")},
                     "AUPRC": M.auprc(s[ok], label[ok]), "threshold": tau})
    all_mask = masks["train"] | masks["val"] | masks["test"]
    _, pe_all = evaluate(s, tau, all_mask, events_typed)
    per_type = M.per_type_recall(pe_all, true_events_in(events_typed, all_mask), tol=TOL)
    return rows, per_type, s, tau, pe_all


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Member 4 health-anomaly detection + comparison.")
    ap.add_argument("--forecasters", default=",".join(FORECASTERS))
    ap.add_argument("--no-iforest", action="store_true")
    ap.add_argument("--save-scores", default="auto",
                    help="detector whose scores go to the contract file; 'auto' = best VALIDATION event-F1")
    args = ap.parse_args(argv)

    df = C.load_telemetry()
    masks = C.split_masks(df)
    labels = C.load_labels()
    label = C.label_mask(df, labels)
    events_typed = C.label_events(df, labels)
    normal = (label == 0) & ~df[C.CHANNELS].isna().any(axis=1).values

    rows, types, keep = [], {}, {}
    for f in [x.strip() for x in args.forecasters.split(",") if x.strip()]:
        path = C.GEN_DIR / "forecasts" / f"m4_pred_{f}.npz"
        if not path.exists():
            print(f"skip residual_{f}: {path} missing (run m4_forecast_eval.py)")
            continue
        z = np.load(path)
        raw, _, sd = residual_score(df, z["pred"], list(z["horizons"]), masks, normal)
        r, pt, s, tau, pe = run_detector(f"residual_{f}", raw, df, masks, events_typed, label)
        rows += r; types[f"residual_{f}"] = pt; keep[f"residual_{f}"] = (s, tau)
        print(f"residual_{f:<8} tau={tau:.3f}  val F1={r[0]['event_F1']:.3f}  test F1={r[1]['event_F1']:.3f}")

    # Ensemble: a strong short-term forecaster (XGBoost) "explains away" slow faults it can follow,
    # while ARIMAX (normal orbit pattern only) does not. Combine them: each smoothed score is scaled
    # by its own VALIDATION 99.9th percentile, then take the larger. Threshold again chosen on val.
    if {"residual_xgboost", "residual_arimax"} <= keep.keys():
        parts = []
        for nm in ("residual_xgboost", "residual_arimax"):
            sm = keep[nm][0]
            parts.append(sm / np.nanpercentile(sm[masks["val"]], 99.9))
        ens = np.fmax(*parts)
        r, pt, s, tau, _ = run_detector("residual_ensemble", ens, df, masks, events_typed, label, do_smooth=False)
        rows += r; types["residual_ensemble"] = pt; keep["residual_ensemble"] = (s, tau)
        print(f"residual_ensemble tau={tau:.3f}  val F1={r[0]['event_F1']:.3f}  test F1={r[1]['event_F1']:.3f}")

    raw, lo, hi = limit_score(df, masks, normal)
    r, pt, s, tau, _ = run_detector("limit_check", raw, df, masks, events_typed, label, do_smooth=False, fixed_tau=0.0)
    rows += r; types["limit_check"] = pt; keep["limit_check"] = (s, tau)
    print(f"limit_check       tau=0 (fixed)  val F1={r[0]['event_F1']:.3f}  test F1={r[1]['event_F1']:.3f}")

    if not args.no_iforest:
        import m4_isolation_forest as IF
        raw = IF.fit_score(df, masks, normal)
        r, pt, s, tau, _ = run_detector("isolation_forest", raw, df, masks, events_typed, label)
        rows += r; types["isolation_forest"] = pt; keep["isolation_forest"] = (s, tau)
        print(f"isolation_forest  tau={tau:.3f}  val F1={r[0]['event_F1']:.3f}  test F1={r[1]['event_F1']:.3f}")

    C.RESULTS.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    path = C.RESULTS / "health_metrics.csv"
    if path.exists():                                   # keep other datasets' rows (e.g. OPSSAT-AD)
        old = pd.read_csv(path)
        res = pd.concat([old[old.dataset != "simulated_v1"], res], ignore_index=True)
    res.round(4).to_csv(path, index=False)
    pt = pd.DataFrame(types).round(3)
    pt.index.name = "anomaly_type"
    pt.to_csv(C.RESULTS / "m4_health_per_type.csv")

    sim_val = res[(res.dataset == "simulated_v1") & (res.split == "val")]
    selected = sim_val.sort_values(["event_F1", "false_alarms_per_day"], ascending=[False, True]).iloc[0]["model"]
    name = selected if args.save_scores == "auto" else (
        args.save_scores if args.save_scores in keep else f"residual_{args.save_scores}")
    print(f"\nselected on validation: {selected}   (scores saved for: {name})")
    if name in keep:
        s, tau = keep[name]
        sel = masks["val"] | masks["test"]
        out = pd.DataFrame({"timestamp": df["timestamp"][sel], "model": name, "score": s[sel], "threshold": tau,
                            "pred": (s[sel] > tau).astype(int), "label": label[sel],
                            "split": df["split"][sel] if "split" in df else ""})
        out.to_csv(C.GEN_DIR / "m4_anomaly_scores.csv.gz", index=False, compression="gzip")

    pd.set_option("display.width", 170)
    show = res[res.dataset == "simulated_v1"][["model", "split", "event_precision", "event_recall", "event_F1",
                                                "AUPRC", "mean_delay_min", "false_alarms_per_day"]]
    print("\n" + show.round(3).to_string(index=False))
    print("\nRecall per anomaly type (all splits):\n" + pt.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
