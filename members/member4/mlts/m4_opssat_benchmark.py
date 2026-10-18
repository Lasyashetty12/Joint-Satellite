"""
m4_opssat_benchmark.py - validate Member 4's health detectors on REAL telemetry (OPSSAT-AD).

OPSSAT-AD is segment-level: each short segment of one channel is labelled normal or
anomalous. We turn each detector's per-sample score into ONE score per segment (its
maximum), choose the threshold on the official TRAIN segments (max F1), and report the
TEST segments once with the benchmark paper's metrics (precision, recall, F1, MCC,
AUC-ROC, AUC-PR).

Detectors (same ideas as on the simulated data):
  residual_naive  1-step naive forecast residual |x[t] - x[t-1]|, normalised per channel+sampling
                  by the std on normal train segments, EWMA(span 5), segment max
  residual_ar     AR(5) one-step forecast fitted by least squares on normal train segments of
                  the same channel+sampling (no fitting across segment boundaries)
  limit_check     fixed limits = normal-train 0.5 / 99.5 percentiles per channel+sampling
  isolation_forest  on the 18 per-segment features of dataset.csv (fitted on normal train)

Outputs: results/m4_opssat_results.csv, rows "OPSSAT-AD_segment" in results/health_metrics.csv

Run from the repository root:
    python members/member4/mlts/m4_opssat_benchmark.py
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
from m4_real_telemetry_loader import DEFAULT_ROOT, FEATURES, load_features, load_segments  # noqa: E402

AR_P = 5
EWMA = 5


def _ewma_max(z: np.ndarray, span: int = EWMA) -> float:
    if len(z) == 0:
        return 0.0
    return float(np.nanmax(pd.Series(z).ewm(span=span).mean().values))


def _key(s) -> tuple[str, int]:
    return (s.channel, s.sampling)


def score_residual_naive(segs) -> np.ndarray:
    ref: dict = {}
    for s in segs:
        if s.train and not s.anomaly and len(s.x) > 1:
            ref.setdefault(_key(s), []).append(np.diff(s.x))
    sd = {k: (np.std(np.concatenate(v)) or 1.0) for k, v in ref.items()}
    return np.array([_ewma_max(np.abs(np.diff(s.x)) / sd.get(_key(s), 1.0)) if len(s.x) > 1 else 0.0 for s in segs])


def _lagmat(x: np.ndarray, p: int):
    X = np.column_stack([x[p - i - 1:len(x) - i - 1] for i in range(p)] + [np.ones(len(x) - p)])
    return X, x[p:]


def score_residual_ar(segs, p: int = AR_P) -> np.ndarray:
    fits: dict = {}
    groups: dict = {}
    for s in segs:
        if s.train and not s.anomaly and len(s.x) > p + 1:
            groups.setdefault(_key(s), []).append(_lagmat(s.x, p))
    for k, parts in groups.items():
        X = np.vstack([a for a, _ in parts]); y = np.concatenate([b for _, b in parts])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        fits[k] = (coef, np.std(y - X @ coef) or 1.0)
    out = []
    for s in segs:
        k = _key(s)
        if k not in fits or len(s.x) <= p:
            out.append(0.0); continue
        coef, sd = fits[k]
        X, y = _lagmat(s.x, p)
        out.append(_ewma_max(np.abs(y - X @ coef) / sd))
    return np.array(out)


def score_limits(segs, q=(0.5, 99.5)) -> np.ndarray:
    ref: dict = {}
    for s in segs:
        if s.train and not s.anomaly:
            ref.setdefault(_key(s), []).append(s.x)
    lim = {k: np.percentile(np.concatenate(v), q) for k, v in ref.items()}
    out = []
    for s in segs:
        if _key(s) not in lim:
            out.append(0.0); continue
        lo, hi = lim[_key(s)]
        span = (hi - lo) or 1.0
        out.append(float(np.max(np.maximum((s.x - hi) / span, (lo - s.x) / span))))
    return np.array(out)


def score_iforest(feat: pd.DataFrame, seed: int = 42) -> np.ndarray:
    from sklearn.ensemble import IsolationForest
    cols = [c for c in FEATURES if c in feat.columns]
    X = feat[cols].astype(float).values
    ref = (feat["train"] == 1) & (feat["anomaly"] == 0)
    mu, sd = X[ref].mean(axis=0), X[ref].std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    m = IsolationForest(n_estimators=300, random_state=seed).fit(Z[ref.values])
    return -m.score_samples(Z)


def best_threshold(score: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    cands = np.unique(np.quantile(score, np.linspace(0.3, 0.995, 200)))
    f1s = [f1_score(y, score > c, zero_division=0) for c in cands]
    return float(cands[int(np.argmax(f1s))])


def evaluate(scores: dict, y: np.ndarray, train: np.ndarray) -> pd.DataFrame:
    rows = []
    for name, s in scores.items():
        tau = best_threshold(s[train], y[train])
        test = ~train
        m = M.segment_metrics(y[test], (s[test] > tau).astype(int), s[test])
        rows.append({"model": name, "threshold": tau, "n_test": int(test.sum()),
                     "n_test_anomalous": int(y[test].sum()), **m})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Member 4 detectors on OPSSAT-AD (real telemetry).")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    args = ap.parse_args(argv)
    segs = load_segments(Path(args.root))
    feat = load_features(Path(args.root)).set_index("segment")
    sid = np.array([s.segment for s in segs])
    y = np.array([s.anomaly for s in segs])
    train = np.array([s.train == 1 for s in segs])
    scores = {"residual_naive": score_residual_naive(segs), "residual_ar": score_residual_ar(segs),
              "limit_check": score_limits(segs)}
    if set(FEATURES) & set(feat.columns):
        f = feat.reindex(sid).reset_index()
        scores["isolation_forest"] = score_iforest(f)
    res = evaluate(scores, y, train)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(C.RESULTS / "m4_opssat_results.csv", index=False)

    hm_path = C.RESULTS / "health_metrics.csv"
    hm = pd.read_csv(hm_path) if hm_path.exists() else pd.DataFrame()
    if len(hm):
        hm = hm[hm.dataset != "OPSSAT-AD_segment"]
    add = pd.DataFrame({"member": "member4", "model": res.model, "dataset": "OPSSAT-AD_segment", "split": "test",
                        "event_precision": res.precision, "event_recall": res.recall, "event_F1": res.F1,
                        "AUPRC": res.AUC_PR, "mean_delay_min": np.nan, "false_alarms_per_day": np.nan,
                        "threshold": res.threshold})
    pd.concat([hm, add], ignore_index=True).round(4).to_csv(hm_path, index=False)
    print(f"segments {len(segs)} (train {train.sum()}, test {(~train).sum()}, anomalous {y.sum()})")
    print(res.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
