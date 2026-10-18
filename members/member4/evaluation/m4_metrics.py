"""
m4_metrics.py - anomaly-detection metrics used by every Member 4 detector.

Why EVENT metrics (handbook section 15, "inflated anomaly scores"):
a fault lasting 3 hours is ONE problem for an operator. Point-wise accuracy would reward
flagging 359 of its 360 points and ignore 20 separate false alarms. So we report:

  event recall     = labelled events that were detected / labelled events
  event precision  = predicted alarm events that overlap a labelled event / predicted events
  event F1         = harmonic mean of the two
  mean delay (min) = time from event start to the first alarm inside it (detected events)
  false alarms/day = predicted events overlapping no labelled event / days scored
  AUPRC            = point-wise area under precision-recall of the raw score (threshold-free)

Tolerance: an alarm up to `tol` samples AFTER a labelled event still counts as that event
(physical faults leave a recovery tail, see Milestone 3). Alarms before the start do not.
"""

from __future__ import annotations

import numpy as np

SAMPLE_MIN = 0.5          # 30 s cadence


def events_from_binary(flag: np.ndarray, max_gap: int = 0, min_len: int = 1) -> list[tuple[int, int]]:
    """Merge runs of True into (start, end) inclusive; runs separated by <= max_gap False are joined."""
    idx = np.flatnonzero(flag)
    if len(idx) == 0:
        return []
    events, s, e = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - e - 1 <= max_gap:
            e = i
        else:
            events.append((s, e)); s = e = i
    events.append((s, e))
    return [(a, b) for a, b in events if b - a + 1 >= min_len]


def event_metrics(pred_events: list[tuple[int, int]], true_events: list[tuple[int, int]],
                  n_samples: int, tol: int = 0, samples_per_day: int = 2880) -> dict:
    ext = [(a, b + tol) for a, b in true_events]
    detected, delays = 0, []
    for (a, b), (_, be) in zip(true_events, ext):
        hits = [max(pa, a) for pa, pb in pred_events if pa <= be and pb >= a]
        if hits:
            detected += 1
            delays.append((min(hits) - a) * SAMPLE_MIN)
    tp_pred = sum(1 for pa, pb in pred_events if any(pa <= be and pb >= a for a, be in ext))
    fp_pred = len(pred_events) - tp_pred
    rec = detected / len(true_events) if true_events else np.nan
    prec = tp_pred / len(pred_events) if pred_events else (np.nan if not true_events else 0.0)
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and prec + rec > 0) else 0.0
    return {"event_precision": prec, "event_recall": rec, "event_F1": f1,
            "mean_delay_min": float(np.mean(delays)) if delays else np.nan,
            "false_alarms_per_day": fp_pred / (n_samples / samples_per_day),
            "n_true_events": len(true_events), "n_pred_events": len(pred_events), "n_detected": detected}


def auprc(score: np.ndarray, label: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    ok = ~np.isnan(score)
    if ok.sum() == 0 or label[ok].sum() == 0:
        return np.nan
    return float(average_precision_score(label[ok], score[ok]))


def per_type_recall(pred_events, true_events_typed, tol: int = 0) -> dict:
    """{anomaly_type: detected / total} using the same overlap rule."""
    out: dict[str, list[int]] = {}
    for a, b, typ in true_events_typed:
        hit = any(pa <= b + tol and pb >= a for pa, pb in pred_events)
        d = out.setdefault(typ, [0, 0]); d[0] += hit; d[1] += 1
    return {k: v[0] / v[1] for k, v in out.items()}


def segment_metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> dict:
    """Segment-level (OPSSAT-AD) metrics recommended by the benchmark paper."""
    from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef,
                                 precision_score, recall_score, roc_auc_score)
    return {"precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "F1": float(f1_score(y_true, y_pred, zero_division=0)),
            "MCC": float(matthews_corrcoef(y_true, y_pred)),
            "AUC_ROC": float(roc_auc_score(y_true, score)),
            "AUC_PR": float(average_precision_score(y_true, score))}
