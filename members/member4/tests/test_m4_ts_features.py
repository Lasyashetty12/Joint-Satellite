"""Milestone 3 tests: temporal integrity (handbook test ladder).
- features at time t never use data after t (no look-ahead)
- chronological split: train < val < test, with gaps, no overlap
- scaler is fitted on train rows only
- helper statistics behave as expected on known toy series
"""

import numpy as np
import pandas as pd
import pytest

import m4_make_splits as SP
import m4_stationarity_decomposition as SD
import m4_telemetry_generator as gen
import m4_ts_common as C
import m4_ts_eda as EDA
import m4_ts_features as F


@pytest.fixture(scope="module")
def tele():
    return gen.generate(gen.load_config(days=3), with_dropouts=True)["telemetry"]


# ------------------------------------------------------------------ no look-ahead
def test_features_ignore_future(tele):
    t = 4000
    X1 = F.build_features(tele)
    future_changed = tele.copy()
    future_changed.loc[t + 1:, C.CHANNELS] = future_changed.loc[t + 1:, C.CHANNELS] * 3 + 100
    X2 = F.build_features(future_changed)
    pd.testing.assert_frame_equal(X1.iloc[:t + 1], X2.iloc[:t + 1])


def test_features_change_when_past_changes(tele):
    """Sanity: the test above is meaningful (features DO react to the past)."""
    X1 = F.build_features(tele)
    past_changed = tele.copy()
    past_changed.loc[3990, "SoC"] += 50
    X2 = F.build_features(past_changed)
    assert not np.allclose(X1.loc[4000, "SoC_rmean60"], X2.loc[4000, "SoC_rmean60"])


def test_lag_feature_is_exact_shift(tele):
    X = F.build_features(tele)
    assert X.loc[500, "bus_V_lag190"] == tele.loc[310, "bus_V"] or np.isnan(tele.loc[310, "bus_V"])


def test_fourier_phase_independent_of_subset(tele):
    full = F.build_features(tele)
    part = F.build_features(tele.iloc[1000:].reset_index(drop=True))
    assert np.allclose(full.loc[1000, ["orb_sin1", "orb_cos1"]].values,
                       part.loc[0, ["orb_sin1", "orb_cos1"]].values)


# ------------------------------------------------------------------ splits
def test_day_blocks_90_days():
    b = SP.day_blocks(90, gap_days=1)
    assert b == {"train": (0, 53), "val": (55, 71), "test": (73, 89)}


def test_assign_split_order_and_gaps(tele):
    per_day = 100
    df = pd.DataFrame({"x": np.arange(10 * per_day)})
    s = SP.assign_split(df, SP.day_blocks(10, gap_days=1), per_day)
    order = {"train": 0, "gap": 1, "val": 2, "test": 3}
    last_tr = np.where(s == "train")[0].max()
    first_val, last_val = np.where(s == "val")[0][[0, -1]]
    first_te = np.where(s == "test")[0].min()
    assert last_tr < first_val <= last_val < first_te
    assert (s == "gap").sum() == 2 * per_day
    assert set(s.unique()) <= set(order)


def test_bad_split_raises():
    with pytest.raises(ValueError):
        SP.day_blocks(3, gap_days=2)


# ------------------------------------------------------------------ scaler
def test_scaler_uses_train_only():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 100.0, 200.0]})
    train = frame.iloc[:3]
    sc = F.TrainScaler().fit(train)
    assert sc.mean_["a"] == pytest.approx(2.0)
    z = sc.transform(frame)
    assert z["a"].iloc[3] == pytest.approx((100 - 2.0) / np.std([1, 2, 3]))


def test_scaler_roundtrip(tmp_path):
    sc = F.TrainScaler().fit(pd.DataFrame({"a": [1.0, 3.0]}))
    sc.save(tmp_path / "s.json")
    sc2 = F.TrainScaler.load(tmp_path / "s.json")
    assert sc2.mean_ == sc.mean_ and sc2.std_ == sc.std_


# ------------------------------------------------------------------ helpers on toy series
def test_interval_indices_inclusive():
    ts = pd.Series(pd.date_range("2026-01-01", periods=10, freq="30s", tz="UTC"))
    a, b = C.interval_indices(ts, ts.iloc[2], ts.iloc[5])
    assert (a, b) == (2, 5)


def test_crosscorr_finds_known_lag():
    rng = np.random.default_rng(0)
    a = rng.normal(size=2000)
    b = np.roll(a, 7)                      # b responds 7 samples after a
    cc = EDA.crosscorr(a, b, 20)
    assert np.argmax(cc) - 20 == 7


def test_stationarity_white_noise_vs_random_walk():
    rng = np.random.default_rng(1)
    wn = rng.normal(size=3000)
    rw = np.cumsum(rng.normal(size=3000))
    assert SD.stationarity_tests(wn)["verdict"] == "stationary"
    assert SD.stationarity_tests(rw)["verdict"] == "non-stationary"


def test_seasonal_strength_high_for_sine():
    from statsmodels.tsa.seasonal import STL
    t = np.arange(190 * 20)
    x = np.sin(2 * np.pi * t / 190) + np.random.default_rng(2).normal(0, 0.05, len(t))
    fs, _ = SD.seasonal_strength(STL(x, period=190).fit())
    assert fs > 0.9


def test_suggest_p():
    pac = np.array([1.0, 0.8, 0.3, 0.001, 0.2])
    assert SD.suggest_p(pac, n=10000) == 2
