"""Generator tests (handbook test ladder, 'Generator' level):
same seed -> identical output; SoC within 0-100; eclipse fraction ~ config;
labels match injected intervals; plus physics sanity and dropout checks.

Run from the repository root:
    python -m pytest members/member4/tests -q
"""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import m4_telemetry_generator as gen

CH = gen.CHANNELS
CONTRACT_COLS = ["timestamp", "sunlit", "payload_on", *CH, "split"]


@pytest.fixture(scope="module")
def cfg():
    return gen.load_config(days=20)          # 20 days: enough events, still fast


@pytest.fixture(scope="module")
def full(cfg):
    return gen.generate(cfg)


@pytest.fixture(scope="module")
def nodrop(cfg):
    return gen.generate(cfg, with_dropouts=False)


@pytest.fixture(scope="module")
def clean(cfg):
    return gen.generate(cfg, with_anomalies=False, with_dropouts=False)


def _interval_mask(n, start, end):
    m = np.zeros(n, dtype=bool)
    m[start:end + 1] = True
    return m


# ---------------------------------------------------------------- reproducibility
def test_same_seed_identical():
    c = gen.load_config(days=2)
    a, b = gen.generate(c), gen.generate(c)
    pd.testing.assert_frame_equal(a["telemetry"], b["telemetry"])
    pd.testing.assert_frame_equal(a["labels"], b["labels"])


def test_different_seed_differs():
    a = gen.generate(gen.load_config(days=1, seed=1))["telemetry"]
    b = gen.generate(gen.load_config(days=1, seed=2))["telemetry"]
    assert not np.allclose(a["bus_V"].fillna(0), b["bus_V"].fillna(0))


# ---------------------------------------------------------------- schema / time
def test_contract_columns(full):
    assert list(full["telemetry"].columns) == CONTRACT_COLS
    assert list(full["labels"].columns) == ["event_id", "start", "end", "anomaly_type",
                                            "channels", "coupled_pass_id"]


def test_timestamps_monotonic_and_cadence(full, cfg):
    ts = full["telemetry"]["timestamp"]
    assert ts.is_monotonic_increasing
    assert (ts.diff().dropna() == pd.Timedelta(seconds=cfg["cadence_s"])).all()
    assert len(ts) == int(cfg["days"] * 86400 / cfg["cadence_s"])


# ---------------------------------------------------------------- physics sanity
def test_soc_within_0_100(full, clean):
    assert full["true"]["SoC"].min() >= 0 and full["true"]["SoC"].max() <= 100
    assert clean["true"]["SoC"].min() >= 0 and clean["true"]["SoC"].max() <= 100


def test_eclipse_fraction_matches_config(full, cfg):
    measured = 1 - full["telemetry"]["sunlit"].mean()
    assert abs(measured - cfg["eclipse_fraction"]) < 0.01


def test_solar_current_zero_in_eclipse(clean):
    df = clean["telemetry"]
    ecl, sun = df[df.sunlit == 0], df[df.sunlit == 1]
    assert abs(ecl["solar_I"].mean()) < 0.05
    assert sun["solar_I"].mean() > 2.0


def test_battery_discharges_in_eclipse(clean):
    df = clean["telemetry"]
    assert df.loc[df.sunlit == 0, "batt_I"].mean() < -0.5


def test_panel_hotter_in_sun(clean):
    df = clean["telemetry"]
    assert df.loc[df.sunlit == 1, "panel_temp"].mean() > df.loc[df.sunlit == 0, "panel_temp"].mean() + 20


def test_payload_only_in_sunlight_when_healthy(clean):
    df = clean["telemetry"]
    assert df.loc[df.sunlit == 0, "payload_on"].sum() == 0


# ---------------------------------------------------------------- labels vs injection
def test_some_events_generated(full):
    assert len(full["events"]) > 0


def test_events_do_not_overlap(full):
    ev = sorted(full["events"], key=lambda e: e.start)
    for a, b in zip(ev, ev[1:]):
        assert a.end < b.start


def test_before_first_event_identical_to_clean(full, clean):
    first = min(e.start for e in full["events"])
    a = full["telemetry"].iloc[:first][CH]
    b = clean["telemetry"].iloc[:first][CH]
    ok = a.isna() | np.isclose(a.fillna(0), b.fillna(0))
    assert ok.all().all()


def test_each_label_changes_its_channels(nodrop, clean):
    """Inside every labelled interval, the labelled channel differs from a run
    without that anomaly (dropouts off so NaNs do not hide changes)."""
    with_anom = nodrop
    for ev in with_anom["events"]:
        for c in ev.channels:
            a = with_anom["telemetry"][c].values[ev.start:ev.end + 1]
            b = clean["telemetry"][c].values[ev.start:ev.end + 1]
            assert np.abs(a - b).max() > 1e-6, (ev.anomaly_type, c)


def test_stuck_sensor_is_constant(nodrop):
    res = nodrop
    stuck = [e for e in res["events"] if e.anomaly_type == "stuck"]
    for ev in stuck:
        seg = res["telemetry"][ev.channels[0]].values[ev.start:ev.end + 1]
        assert np.all(seg == seg[0])


def test_label_timestamps_match_indices(full):
    ts = full["telemetry"]["timestamp"]
    for ev, (_, row) in zip(full["events"], full["labels"].iterrows()):
        assert pd.Timestamp(row["start"]) == ts.iloc[ev.start]
        assert pd.Timestamp(row["end"]) == ts.iloc[ev.end]


# ---------------------------------------------------------------- dropouts
def test_nans_only_inside_dropouts(full):
    df = full["telemetry"]
    n = len(df)
    for c in CH:
        allowed = np.zeros(n, dtype=bool)
        for d in full["drops"]:
            if c in d["channels"]:
                allowed |= _interval_mask(n, d["start"], d["end"])
        nan = df[c].isna().values
        assert not (nan & ~allowed).any(), c
        assert nan[allowed].all(), c


# ---------------------------------------------------------------- CLI smoke
def test_cli_help():
    out = subprocess.run([sys.executable, gen.__file__, "--help"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "generator" in out.stdout


def test_cli_writes_outputs(tmp_path):
    out = subprocess.run([sys.executable, gen.__file__, "--days", "0.5", "--out", str(tmp_path)],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    for f in ["m4_telemetry_v1.parquet", "m4_telemetry_labels.csv",
              "m4_telemetry_dropouts.csv", "m4_generator_summary.json"]:
        assert (tmp_path / f).exists(), f
    df = pd.read_parquet(tmp_path / "m4_telemetry_v1.parquet")
    assert len(df) == 1440
