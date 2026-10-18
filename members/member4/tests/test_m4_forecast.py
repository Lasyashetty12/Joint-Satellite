"""Milestone 4 tests (handbook test ladder, 'Forecast' level):
output shape [N, H, C]; naive reproduces known toy values; models beat naive on a toy
seasonal series; metrics are correct on known numbers.
"""

import numpy as np
import pandas as pd
import pytest

import m4_forecast_arima as AR
import m4_forecast_eval as EV
import m4_forecast_lstm as LS
import m4_forecast_naive as NV
import m4_forecast_xgboost as XG
import m4_telemetry_generator as gen
import m4_ts_common as C


@pytest.fixture(scope="module")
def small():
    """4 clean days: train = days 0-2, val = day 3."""
    df = gen.generate(gen.load_config(days=4), with_anomalies=False, with_dropouts=False)["telemetry"]
    day = np.arange(len(df)) // C.samples_per_day()
    df["split"] = np.where(day < 3, "train", "val")
    masks = {"train": (df.split == "train").values, "val": (df.split == "val").values,
             "test": np.zeros(len(df), bool)}
    normal = np.ones(len(df), bool)
    return df, masks, normal


def toy_df(n=600):
    ts = pd.date_range("2026-01-01", periods=n, freq="30s", tz="UTC")
    d = pd.DataFrame({"timestamp": ts})
    for i, c in enumerate(C.CHANNELS):
        d[c] = np.arange(n, dtype=float) + 1000 * i
    return d


# ------------------------------------------------------------------ naive / seasonal-naive
def test_naive_known_values_and_shape():
    d = toy_df()
    p = NV.naive(d, horizons=(1, 5))
    assert p.shape == (600, 2, len(C.CHANNELS))
    assert p[10, 1, 0] == 10.0                      # forecast made at t=10 is y[10]


def test_seasonal_naive_known_values():
    d = toy_df()
    p = NV.seasonal_naive(d, horizons=(1, 10), period=190)
    assert p[200, 0, 0] == 200 + 1 - 190            # y[t + h - period]
    assert np.isnan(p[0, 0, 0])


def test_seasonal_naive_exact_on_periodic_series():
    d = toy_df(1000)
    for c in C.CHANNELS:
        d[c] = np.sin(2 * np.pi * np.arange(1000) / 190)
    p = NV.seasonal_naive(d, horizons=(10,), period=190)
    y = d[C.CHANNELS[0]].values
    assert np.allclose(p[200:-10, 0, 0], y[210:], atol=1e-6)


# ------------------------------------------------------------------ ARIMAX recursion
def test_arimax_recursion_matches_manual():
    rng = np.random.default_rng(0)
    n = 50
    X = np.ones((n, 1)); beta = np.array([2.0]); phi = np.array([0.5, 0.2])
    y = 2.0 + rng.normal(size=n)
    out = AR.forecast_all(y, X, beta, phi, [1, 2])
    u = y - 2.0
    t = 20
    u1 = 0.5 * u[t] + 0.2 * u[t - 1]
    u2 = 0.5 * u1 + 0.2 * u[t]
    assert out[t, 0] == pytest.approx(2.0 + u1)
    assert out[t, 1] == pytest.approx(2.0 + u2)


def test_arimax_beats_naive_on_toy(small):
    df, masks, normal = small
    pred, _ = AR.fit_predict(df, masks, normal, horizons=(10,), channels=["panel_temp"],
                             orders_path=C.RESULTS / "does_not_exist.json", verbose=False)
    rows = EV.score(pred, df, masks, normal, "arimax", horizons=(10,), channels=["panel_temp"], splits=("val",))
    base = EV.score(NV.naive(df, horizons=(10,), channels=["panel_temp"]), df, masks, normal, "naive",
                    horizons=(10,), channels=["panel_temp"], splits=("val",))
    assert rows[0]["MAE"] < base[0]["MAE"]


# ------------------------------------------------------------------ XGBoost / GRU
def test_xgboost_shape_and_beats_naive(small):
    df, masks, normal = small
    pred, _ = XG.fit_predict(df, masks, normal, horizons=(1, 10), channels=["SoC"], verbose=False)
    assert pred.shape == (len(df), 2, 1)
    r = EV.score(pred, df, masks, normal, "x", horizons=(1, 10), channels=["SoC"], splits=("val",))
    b = EV.score(NV.naive(df, horizons=(1, 10), channels=["SoC"]), df, masks, normal, "n",
                 horizons=(1, 10), channels=["SoC"], splits=("val",))
    assert r[1]["MAE"] < b[1]["MAE"]                 # horizon 10


def test_gru_shape_and_finite(small):
    df, masks, normal = small
    pred, info = LS.fit_predict(df, masks, normal, horizons=(1, 10), epochs=1, verbose=False,
                                save_checkpoint=False)
    assert pred.shape == (len(df), 2, len(C.CHANNELS))
    assert np.isfinite(pred[LS.W:-20]).mean() > 0.99
    assert len(info["history"]) == 1


def test_window_ok():
    normal = np.array([1, 1, 0, 1, 1, 1], bool)
    assert XG.window_ok(normal, 2).tolist() == [True, True, False, False, True, True]


def test_future_sunlit_matches_generator(small):
    df, _, _ = small
    F = LS.future_sunlit(df)
    assert (F[:-1, 0] == df["sunlit"].values[1:]).mean() > 0.999


# ------------------------------------------------------------------ metrics / scoring rule
def test_metrics_known_values():
    m = EV.metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert m["MAE"] == pytest.approx(2 / 3)
    assert m["RMSE"] == pytest.approx(np.sqrt(4 / 3))


def test_score_excludes_anomalous_targets():
    d = toy_df(100)
    masks = {"val": np.ones(100, bool)}
    normal = np.ones(100, bool); normal[50] = False
    pred = NV.naive(d, horizons=(1,))
    rows = EV.score(pred, d, masks, normal, "n", horizons=(1,), channels=[C.CHANNELS[0]], splits=("val",))
    assert rows[0]["n"] == 98                         # 99 origins minus the one whose target is t=50
