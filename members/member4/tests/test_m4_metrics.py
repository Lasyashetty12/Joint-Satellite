"""Milestone 5 tests: metrics give known values on toy cases; detectors and the OPSSAT-AD
pipeline behave correctly on small examples (handbook test ladder, 'Metrics' level)."""

import numpy as np
import pandas as pd
import pytest

import m4_metrics as M
import m4_opssat_benchmark as OB
import m4_real_telemetry_loader as RL
import m4_residual_anomaly as RA
import m4_ts_common as C


# ------------------------------------------------------------------ event metrics
def test_events_from_binary_merge_gap():
    f = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1], bool)
    assert M.events_from_binary(f, max_gap=0) == [(1, 2), (5, 5), (10, 10)]
    assert M.events_from_binary(f, max_gap=2) == [(1, 5), (10, 10)]


def test_event_metrics_known_case():
    true = [(10, 20), (50, 60)]
    pred = [(15, 16), (100, 101)]                  # one hit, one false alarm
    m = M.event_metrics(pred, true, n_samples=2880)
    assert m["event_recall"] == 0.5 and m["event_precision"] == 0.5 and m["event_F1"] == 0.5
    assert m["mean_delay_min"] == pytest.approx(5 * 0.5)
    assert m["false_alarms_per_day"] == pytest.approx(1.0)


def test_tolerance_counts_alarm_after_event():
    true = [(10, 20)]
    pred = [(25, 26)]
    assert M.event_metrics(pred, true, 2880, tol=0)["event_recall"] == 0
    assert M.event_metrics(pred, true, 2880, tol=10)["event_recall"] == 1


def test_alarm_before_event_not_counted_as_early():
    m = M.event_metrics([(5, 12)], [(10, 20)], 2880)
    assert m["mean_delay_min"] == 0.0              # overlapping alarm -> delay 0, not negative


def test_auprc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert M.auprc(np.array([0.1, 0.2, 0.8, 0.9]), y) == pytest.approx(1.0)
    assert np.isnan(M.auprc(np.array([0.1, 0.2]), np.array([0, 0])))


def test_per_type_recall():
    r = M.per_type_recall([(5, 6)], [(5, 8, "spike"), (50, 60, "drift")])
    assert r == {"spike": 1.0, "drift": 0.0}


def test_segment_metrics_known():
    m = M.segment_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), np.array([.1, .6, .7, .9]))
    assert m["recall"] == 1.0 and m["precision"] == pytest.approx(2 / 3)


# ------------------------------------------------------------------ detectors (simulated)
def _toy(n=400):
    ts = pd.date_range("2026-01-01", periods=n, freq="30s", tz="UTC")
    df = pd.DataFrame({"timestamp": ts})
    for c in C.CHANNELS:
        df[c] = np.sin(np.arange(n) / 10.0)
    return df


def test_residual_uses_forecast_made_one_step_earlier():
    df = _toy()
    pred = np.zeros((len(df), 1, len(C.CHANNELS)), np.float32)
    pred[:, 0, :] = df[C.CHANNELS].values           # "perfect" forecast of y[t] stored at t (wrong alignment)
    masks = {"train": np.ones(len(df), bool)}
    s, z, sd = RA.residual_score(df, pred, [1], masks, np.ones(len(df), bool))
    # the detector must compare y[t] with pred made at t-1, i.e. y[t-1] -> nonzero residual
    assert np.nanmax(z) > 0 and z.shape == (len(df), len(C.CHANNELS)) and np.isnan(s[0])


def test_limit_score_flags_out_of_range():
    df = _toy()
    df.loc[200, "bus_V"] = 50.0
    masks = {"train": np.arange(len(df)) < 150}
    s, lo, hi = RA.limit_score(df, masks, np.ones(len(df), bool))
    assert s[200] > 1.0 and (s[:150] > 0).mean() < 0.05      # percentile limits: only the extreme tail of train lies outside


def test_threshold_choice_on_toy():
    n = 2000
    s = np.zeros(n); s[500:510] = 5; s[1500:1505] = 5
    val = np.ones(n, bool)
    events = [(500, 509, "spike"), (1500, 1504, "spike")]
    tau = RA.choose_threshold(s, val, events)
    m, _ = RA.evaluate(s, tau, val, events)
    assert m["event_F1"] == 1.0


# ------------------------------------------------------------------ OPSSAT-AD pipeline on toy segments
def _segments():
    rng = np.random.default_rng(0)
    segs = []
    for i in range(80):
        x = np.sin(np.arange(60) / 6) + rng.normal(0, 0.02, 60)
        an = int(i % 5 == 0)
        if an:
            x[30] += 3
        segs.append(RL.Segment(i, "CH", 1, an, int(i < 60), np.arange(60.0), x))
    return segs


def test_opssat_residual_scores_separate_anomalies():
    segs = _segments()
    y = np.array([s.anomaly for s in segs])
    for f in (OB.score_residual_naive, OB.score_residual_ar):
        s = f(segs)
        assert s[y == 1].min() > s[y == 0].max()


def test_opssat_evaluate_runs():
    segs = _segments()
    y = np.array([s.anomaly for s in segs]); train = np.array([s.train == 1 for s in segs])
    res = OB.evaluate({"residual_ar": OB.score_residual_ar(segs)}, y, train)
    assert res.loc[0, "F1"] == 1.0 and res.loc[0, "n_test"] == 20


def test_loader_drops_duplicate_timestamps(tmp_path):
    t = pd.date_range("2022-01-01", periods=5, freq="1s", tz="UTC").astype(str).tolist()
    df = pd.DataFrame({"channel": "CH", "timestamp": t + [t[-1]], "value": [1, 2, 3, 4, 5, 5],
                       "label": "a2", "sampling": 1, "anomaly": 0, "segment": 7, "train": 1})
    df.to_csv(tmp_path / "segments.csv", index=False)
    segs = RL.load_segments(tmp_path)
    assert len(segs) == 1 and len(segs[0].x) == 5
