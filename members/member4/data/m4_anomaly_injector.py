"""
m4_anomaly_injector.py - plans and applies labelled anomalies and dropouts.

Two families of anomalies (see GENERATOR_CARD.md):

  PHYSICS anomalies change the simulated satellite itself, so every related
  channel reacts consistently. They are converted into "override" arrays that
  the physics simulator in m4_telemetry_generator.py reads:
      v_drop          -> bus_V offset + extra load (current change)
      batt_fade       -> battery capacity scaled down (deeper eclipse discharge)
      thermal_runaway -> extra battery heat, ramping up
      payload_fault   -> payload power abnormal + erratic

  SENSOR anomalies only corrupt what one sensor reports (the physics is fine):
      spike, stuck, drift, noise_burst

Telemetry dropouts (NaN gaps) are NOT anomalies; they are stored separately.

This file is imported by m4_telemetry_generator.py; it has no CLI of its own
beyond --help.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PHYSICS_TYPES = {"v_drop", "batt_fade", "thermal_runaway", "payload_fault"}
SENSOR_TYPES = {"spike", "stuck", "drift", "noise_burst"}

# Duration range in MINUTES for each anomaly type (spike is in SAMPLES).
DURATION_MIN = {
    "spike": None,                 # 1-3 samples
    "stuck": (30, 180),
    "drift": (180, 720),
    "noise_burst": (20, 120),
    "v_drop": (30, 120),
    "batt_fade": (24 * 60, 72 * 60),
    "thermal_runaway": (60, 240),
    "payload_fault": (20, 90),
}

# Which channels each SENSOR anomaly may hit.
SENSOR_CHANNELS = ["batt_temp", "panel_temp", "bus_V", "batt_I", "solar_I", "SoC", "att_err", "payload_P"]

# Channels reported in the label for PHYSICS anomalies (the ones that visibly react).
PHYSICS_CHANNELS = {
    "v_drop": ["bus_V", "batt_I"],
    "batt_fade": ["SoC", "bus_V"],
    "thermal_runaway": ["batt_temp"],
    "payload_fault": ["payload_P", "batt_I"],
}


@dataclass
class Event:
    event_id: int
    anomaly_type: str
    start: int            # index of first affected sample
    end: int              # index of last affected sample (inclusive)
    channels: list[str]
    params: dict = field(default_factory=dict)


def _duration_samples(atype: str, rng: np.random.Generator, samples_per_min: float) -> int:
    if atype == "spike":
        return int(rng.integers(1, 4))
    lo, hi = DURATION_MIN[atype]
    return max(1, int(round(rng.uniform(lo, hi) * samples_per_min)))


def plan_events(n_samples: int, cfg: dict, rng: np.random.Generator) -> list[Event]:
    """Choose anomaly types, times and parameters. Events never overlap
    (a minimum gap is kept between them) so each label is unambiguous."""
    samples_per_min = 60.0 / cfg["cadence_s"]
    days = n_samples / (86400 / cfg["cadence_s"])
    n_events = int(rng.poisson(cfg["anomaly_rate_per_day"] * days))
    gap = int(cfg.get("anomaly_min_gap_min", 60) * samples_per_min)
    types = list(cfg["anomaly_types"])

    occupied: list[tuple[int, int]] = []
    events: list[Event] = []
    for _ in range(n_events):
        atype = types[int(rng.integers(len(types)))]
        dur = _duration_samples(atype, rng, samples_per_min)
        if dur >= n_samples - 2 * gap:
            continue
        for _try in range(200):
            s = int(rng.integers(gap, n_samples - dur - gap))
            e = s + dur - 1
            if all(e + gap < os_ or s - gap > oe for os_, oe in occupied):
                break
        else:
            continue  # could not place without overlap -> skip
        occupied.append((s, e))

        if atype in SENSOR_TYPES:
            ch = [SENSOR_CHANNELS[int(rng.integers(len(SENSOR_CHANNELS)))]]
        else:
            ch = list(PHYSICS_CHANNELS[atype])

        params: dict = {}
        sign = 1.0 if rng.random() < 0.5 else -1.0
        if atype == "spike":
            params = {"k_std": float(rng.uniform(6, 10)), "sign": sign}
        elif atype == "drift":
            params = {"k_std": float(rng.uniform(3, 6)), "sign": sign}
        elif atype == "noise_burst":
            params = {"k_noise": float(rng.uniform(5, 10))}
        elif atype == "v_drop":
            params = {"dV": float(rng.uniform(0.8, 2.0)), "extra_load_W": float(rng.uniform(5, 15))}
        elif atype == "batt_fade":
            params = {"capacity_scale": float(rng.uniform(0.5, 0.7))}
        elif atype == "thermal_runaway":
            params = {"max_extra_C": float(rng.uniform(15, 25))}
        elif atype == "payload_fault":
            params = {"power_factor": float(rng.uniform(1.6, 2.2)), "erratic_sd_W": float(rng.uniform(3, 6))}
        events.append(Event(0, atype, s, e, ch, params))

    events.sort(key=lambda ev: ev.start)
    for i, ev in enumerate(events, start=1):
        ev.event_id = i
    return events


def physics_overrides(n_samples: int, events: list[Event], rng: np.random.Generator,
                      payload_on_W: float = 30.0) -> dict:
    """Convert PHYSICS events into arrays read by the simulator."""
    ov = {
        "bus_V_offset": np.zeros(n_samples),
        "extra_load_W": np.zeros(n_samples),
        "capacity_scale": np.ones(n_samples),
        "extra_batt_heat_C": np.zeros(n_samples),
        "payload_force_W": np.full(n_samples, np.nan),  # NaN = use normal schedule
    }
    for ev in events:
        sl = slice(ev.start, ev.end + 1)
        n = ev.end - ev.start + 1
        p = ev.params
        if ev.anomaly_type == "v_drop":
            ov["bus_V_offset"][sl] -= p["dV"]
            ov["extra_load_W"][sl] += p["extra_load_W"]
        elif ev.anomaly_type == "batt_fade":
            ov["capacity_scale"][sl] *= p["capacity_scale"]
        elif ev.anomaly_type == "thermal_runaway":
            ov["extra_batt_heat_C"][sl] += np.linspace(0, p["max_extra_C"], n)
        elif ev.anomaly_type == "payload_fault":
            ov["payload_force_W"][sl] = payload_on_W * p["power_factor"] + rng.normal(0, p["erratic_sd_W"], n)
    return ov


def apply_sensor_anomalies(df: pd.DataFrame, events: list[Event], noise_sd: dict,
                           rng: np.random.Generator) -> pd.DataFrame:
    """Corrupt observed channels for SENSOR events. Scale uses the channel's
    standard deviation measured on the (already noisy) signal before corruption."""
    out = df.copy()
    ch_std = {c: float(df[c].std()) for c in SENSOR_CHANNELS}
    for ev in events:
        if ev.anomaly_type not in SENSOR_TYPES:
            continue
        c = ev.channels[0]
        col = out.columns.get_loc(c)
        s, e = ev.start, ev.end
        n = e - s + 1
        p = ev.params
        if ev.anomaly_type == "spike":
            out.iloc[s:e + 1, col] += p["sign"] * p["k_std"] * ch_std[c]
        elif ev.anomaly_type == "stuck":
            frozen = out.iloc[s - 1, col]          # last good value before the event
            out.iloc[s:e + 1, col] = frozen
        elif ev.anomaly_type == "drift":
            out.iloc[s:e + 1, col] += p["sign"] * np.linspace(0, p["k_std"] * ch_std[c], n)
        elif ev.anomaly_type == "noise_burst":
            out.iloc[s:e + 1, col] += rng.normal(0, p["k_noise"] * noise_sd[c], n)
    return out


def plan_dropouts(n_samples: int, cfg: dict, rng: np.random.Generator) -> list[dict]:
    """Downlink-loss gaps: 5-60 min; half the time all channels, otherwise 1-3 channels."""
    samples_per_min = 60.0 / cfg["cadence_s"]
    days = n_samples / (86400 / cfg["cadence_s"])
    n_drop = int(rng.poisson(cfg["dropout_rate_per_day"] * days))
    chans = list(cfg["channels"])
    drops = []
    for i in range(n_drop):
        dur = int(rng.uniform(5, 60) * samples_per_min)
        if dur >= n_samples:
            continue
        s = int(rng.integers(0, n_samples - dur))
        if rng.random() < 0.5:
            ch = chans
        else:
            k = int(rng.integers(1, 4))
            ch = sorted(rng.choice(chans, size=k, replace=False).tolist())
        drops.append({"dropout_id": i + 1, "start": s, "end": s + dur - 1, "channels": ch})
    return drops


def apply_dropouts(df: pd.DataFrame, drops: list[dict]) -> pd.DataFrame:
    out = df.copy()
    for d in drops:
        cols = [out.columns.get_loc(c) for c in d["channels"]]
        out.iloc[d["start"]:d["end"] + 1, cols] = np.nan
    return out


def events_to_labels(events: list[Event], timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Contract format (handbook section 13): event_id, start, end, anomaly_type,
    channels, coupled_pass_id. coupled_pass_id is filled later when passes are built."""
    rows = [{
        "event_id": ev.event_id,
        "start": timestamps[ev.start].isoformat(),
        "end": timestamps[ev.end].isoformat(),
        "anomaly_type": ev.anomaly_type,
        "channels": ";".join(ev.channels),
        "coupled_pass_id": "",
    } for ev in events]
    return pd.DataFrame(rows, columns=["event_id", "start", "end", "anomaly_type", "channels", "coupled_pass_id"])


if __name__ == "__main__":
    argparse.ArgumentParser(description="Member 4 anomaly injector (library module). "
                            "Run m4_telemetry_generator.py to generate data.").parse_args()
    print("This module is used by m4_telemetry_generator.py.")
