"""
m4_telemetry_generator.py - Virtual LEO satellite telemetry generator v1 (Member 4).

Produces physically linked telemetry (not independent random numbers):
    sunlight -> solar current -> battery current -> state of charge -> bus voltage
    sunlight -> panel temperature (lagged)     battery current -> battery temperature
plus labelled anomalies and (separately) telemetry dropouts.

Run from the repository root:
    python members/member4/data/m4_telemetry_generator.py --days 1 --plot
    python members/member4/data/m4_telemetry_generator.py            # full 90 days from config
    python members/member4/data/m4_telemetry_generator.py --help

Outputs (default folder members/member4/data/generated/):
    m4_telemetry_v1.parquet     timestamp, sunlit, payload_on, 8 channels, split
    m4_telemetry_labels.csv     event_id, start, end, anomaly_type, channels, coupled_pass_id
    m4_telemetry_dropouts.csv   dropout_id, start, end, channels   (NOT anomalies)
    m4_generator_summary.json   counts and sanity statistics
and, with --plot, reports/figures/m4_telemetry_day<N>.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m4_anomaly_injector as inj  # noqa: E402

MEMBER_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MEMBER_DIR / "configs" / "m4_generator_config.json"
DEFAULT_OUT = MEMBER_DIR / "data" / "generated"
CHANNELS = ["batt_temp", "panel_temp", "bus_V", "batt_I", "solar_I", "SoC", "att_err", "payload_P"]


def load_config(path: Path | str = DEFAULT_CONFIG, **overrides) -> dict:
    cfg = json.loads(Path(path).read_text())
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


# ----------------------------------------------------------------------------- time base
def time_base(cfg: dict) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Timestamps, orbit phase in [0,1), sunlit flag, orbit index."""
    dt = cfg["cadence_s"]
    n = int(cfg["days"] * 86400 // dt)
    ts = pd.date_range(cfg["start_time"], periods=n, freq=f"{dt}s")
    t_s = np.arange(n) * dt
    period_s = cfg["orbit_period_min"] * 60
    phase = (t_s % period_s) / period_s
    sunlit = (phase < 1.0 - cfg["eclipse_fraction"]).astype(np.int8)  # eclipse = last part of each orbit
    orbit_idx = (t_s // period_s).astype(int)
    return ts, phase, sunlit, orbit_idx


def payload_schedule(cfg: dict, phase: np.ndarray, orbit_idx: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    """Imaging passes happen only in sunlight (optical camera). Returns payload_on 0/1."""
    ph = cfg["physics"]
    orbits_per_day = 24 * 60 / cfg["orbit_period_min"]
    p_img = min(1.0, ph["imaging_passes_per_day"] / orbits_per_day)
    dur = ph["imaging_duration_min"] / cfg["orbit_period_min"]           # as a phase fraction
    sun_end = 1.0 - cfg["eclipse_fraction"]
    on = np.zeros(len(phase), dtype=np.int8)
    for k in np.unique(orbit_idx):
        if rng.random() < p_img:
            start = rng.uniform(0.05, max(0.06, sun_end - dur - 0.02))
            mask = (orbit_idx == k) & (phase >= start) & (phase < start + dur)
            on[mask] = 1
    return on


# ----------------------------------------------------------------------------- physics
def simulate_physics(cfg: dict, sunlit: np.ndarray, payload_on: np.ndarray,
                     overrides: dict | None, rng: np.random.Generator) -> dict:
    """True (noise-free) physical states. The loop is needed because SoC and
    temperatures depend on their own previous value."""
    ph = cfg["physics"]
    n = len(sunlit)
    dt_s = cfg["cadence_s"]
    dt_h = dt_s / 3600.0
    day = np.arange(n) * dt_s / 86400.0
    ov = overrides or inj.physics_overrides(n, [], rng, ph["payload_on_W"])

    # Sun angle (beta) changes slowly over weeks -> seasonal change in solar current.
    beta = np.deg2rad(ph["beta_amplitude_deg"]) * np.sin(2 * np.pi * day / ph["beta_period_days"])
    solar_I = (ph["solar_I_max_A"] * sunlit * np.cos(beta)
               * (1 - ph["panel_degradation_per_day"] * day))

    payload_P = np.where(payload_on == 1, ph["payload_on_W"], ph["payload_idle_W"]).astype(float)
    forced = ~np.isnan(ov["payload_force_W"])
    payload_P[forced] = np.clip(ov["payload_force_W"][forced], 0, None)

    load_I = (ph["base_load_W"] + payload_P + ov["extra_load_W"]) / ph["bus_V_nominal"]
    capacity = ph["battery_capacity_Ah"] * (1 - ph["capacity_fade_per_day"] * day) * ov["capacity_scale"]

    soc = np.empty(n); batt_I = np.empty(n); panel_T = np.empty(n); batt_T = np.empty(n)
    s = ph["initial_SoC"]
    pT = ph["panel_T_sun_C"] if sunlit[0] else ph["panel_T_eclipse_C"]
    bT = ph.get("initial_batt_T_C", ph["batt_T_base_C"])
    a_p = dt_s / (ph["panel_tau_min"] * 60)
    a_b = dt_s / (ph["batt_tau_min"] * 60)
    for i in range(n):
        bi = solar_I[i] - load_I[i]                  # + charging, - discharging
        if s >= 100.0 and bi > 0:                    # battery full: shunt dumps surplus
            bi = 0.0
        s = min(100.0, max(0.0, s + bi * dt_h / capacity[i] * 100.0))
        target_p = ph["panel_T_sun_C"] if sunlit[i] else ph["panel_T_eclipse_C"]
        pT += (target_p - pT) * a_p
        target_b = (ph["batt_T_base_C"] + ph["batt_T_sun_offset_C"] * sunlit[i]
                    + ph["batt_T_current_coeff"] * abs(bi) + ov["extra_batt_heat_C"][i])
        bT += (target_b - bT) * a_b
        soc[i], batt_I[i], panel_T[i], batt_T[i] = s, bi, pT, bT

    bus_V = ph["bus_V_min"] + ph["bus_V_k_soc"] * soc + ph["bus_V_r_batt"] * batt_I + ov["bus_V_offset"]

    att = np.empty(n); a = 0.0
    eps = rng.normal(0, ph["att_ar_sigma_deg"], n)
    for i in range(n):
        a = ph["att_ar_phi"] * a + eps[i]
        att[i] = a

    return {"batt_temp": batt_T, "panel_temp": panel_T, "bus_V": bus_V, "batt_I": batt_I,
            "solar_I": solar_I, "SoC": soc, "att_err": att, "payload_P": payload_P,
            "payload_on": (payload_P > ph["payload_idle_W"] + 1e-9).astype(np.int8)}


# ----------------------------------------------------------------------------- main pipeline
def generate(cfg: dict, with_anomalies: bool = True, with_dropouts: bool = True) -> dict:
    """Full pipeline. Separate random streams keep the base signal identical
    whether or not anomalies/dropouts are switched on."""
    ss = np.random.SeedSequence(cfg["seed"])
    rng_sched, rng_phys, rng_noise, rng_anom, rng_drop = [np.random.default_rng(s) for s in ss.spawn(5)]

    ts, phase, sunlit, orbit_idx = time_base(cfg)
    n = len(ts)
    payload_on = payload_schedule(cfg, phase, orbit_idx, rng_sched)

    events = inj.plan_events(n, cfg, rng_anom) if with_anomalies else []
    ov = inj.physics_overrides(n, events, rng_anom, cfg["physics"]["payload_on_W"])
    true = simulate_physics(cfg, sunlit, payload_on, ov, rng_phys)

    df = pd.DataFrame({"timestamp": ts, "sunlit": sunlit, "payload_on": true["payload_on"]})
    for c in CHANNELS:
        df[c] = true[c] + rng_noise.normal(0, cfg["measurement_noise_sd"][c], n)

    df = inj.apply_sensor_anomalies(df, events, cfg["measurement_noise_sd"], rng_anom)
    drops = inj.plan_dropouts(n, cfg, rng_drop) if with_dropouts else []
    df = inj.apply_dropouts(df, drops)
    df["split"] = "unassigned"   # filled in Milestone 3 (chronological split)

    labels = inj.events_to_labels(events, ts)
    drops_df = pd.DataFrame([{**d, "start": ts[d["start"]].isoformat(), "end": ts[d["end"]].isoformat(),
                              "channels": ";".join(d["channels"])} for d in drops],
                            columns=["dropout_id", "start", "end", "channels"])
    return {"telemetry": df, "labels": labels, "dropouts": drops_df, "events": events,
            "drops": drops, "true": true}


def summarize(out: dict, cfg: dict) -> dict:
    df, labels = out["telemetry"], out["labels"]
    true = out["true"]
    return {
        "version": cfg.get("version"),
        "seed": cfg["seed"],
        "rows": int(len(df)),
        "days": cfg["days"],
        "cadence_s": cfg["cadence_s"],
        "eclipse_fraction_measured": round(1 - float(df["sunlit"].mean()), 4),
        "true_SoC_min": round(float(true["SoC"].min()), 2),
        "true_SoC_max": round(float(true["SoC"].max()), 2),
        "payload_on_fraction": round(float(df["payload_on"].mean()), 4),
        "n_anomaly_events": int(len(labels)),
        "events_per_type": labels["anomaly_type"].value_counts().to_dict() if len(labels) else {},
        "n_dropouts": int(len(out["dropouts"])),
        "nan_fraction": round(float(df[CHANNELS].isna().mean().mean()), 5),
    }


def plot_day(out: dict, day: int, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = out["telemetry"]
    per_day = int(86400 / (df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds())
    d = df.iloc[day * per_day:(day + 1) * per_day]
    hours = (d["timestamp"] - d["timestamp"].iloc[0]).dt.total_seconds() / 3600
    units = {"batt_temp": "°C", "panel_temp": "°C", "bus_V": "V", "batt_I": "A",
             "solar_I": "A", "SoC": "%", "att_err": "deg", "payload_P": "W"}
    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(12, 14), sharex=True)
    ecl = (d["sunlit"].values == 0)
    for ax, c in zip(axes, CHANNELS):
        ax.fill_between(hours, 0, 1, where=ecl, transform=ax.get_xaxis_transform(),
                        color="0.85", step="mid", label="eclipse")
        ax.plot(hours, d[c], lw=0.8, color="#1f4e79")
        ax.set_ylabel(f"{c}\n[{units[c]}]", fontsize=8)
        ax.tick_params(labelsize=7)
    t0, t1 = d.index[0], d.index[-1]
    for ev in out["events"]:
        if ev.end >= t0 and ev.start <= t1:
            for ax, c in zip(axes, CHANNELS):
                if c in ev.channels:
                    ax.axvspan(hours.iloc[max(ev.start, t0) - t0], hours.iloc[min(ev.end, t1) - t0],
                               color="red", alpha=0.25)
                    ax.text(hours.iloc[max(ev.start, t0) - t0], ax.get_ylim()[1], ev.anomaly_type,
                            color="red", fontsize=7, va="top")
    axes[0].legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel(f"hours since start of day {day}")
    fig.suptitle(f"Member 4 virtual telemetry v1 - day {day} (grey = eclipse, red = injected anomaly)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def save(out: dict, summary: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "telemetry": out_dir / "m4_telemetry_v1.parquet",
        "labels": out_dir / "m4_telemetry_labels.csv",
        "dropouts": out_dir / "m4_telemetry_dropouts.csv",
        "summary": out_dir / "m4_generator_summary.json",
    }
    out["telemetry"].to_parquet(paths["telemetry"], index=False)
    out["labels"].to_csv(paths["labels"], index=False)
    out["dropouts"].to_csv(paths["dropouts"], index=False)
    paths["summary"].write_text(json.dumps(summary, indent=2))
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Member 4 virtual satellite telemetry generator v1.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="generator config JSON")
    ap.add_argument("--days", type=float, help="override number of days (e.g. 1 for a quick look)")
    ap.add_argument("--seed", type=int, help="override seed")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output folder")
    ap.add_argument("--no-anomalies", action="store_true", help="generate clean telemetry only")
    ap.add_argument("--no-dropouts", action="store_true", help="do not insert NaN gaps")
    ap.add_argument("--plot", action="store_true", help="save a one-day plot to reports/figures/")
    ap.add_argument("--plot-day", type=int, default=0, help="which day to plot (default 0)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, days=args.days, seed=args.seed)
    out = generate(cfg, with_anomalies=not args.no_anomalies, with_dropouts=not args.no_dropouts)
    summary = summarize(out, cfg)
    paths = save(out, summary, Path(args.out))

    print(json.dumps(summary, indent=2))
    for k, p in paths.items():
        print(f"saved {k:<10} -> {p}")
    if args.plot:
        fig = plot_day(out, args.plot_day, MEMBER_DIR / "reports" / "figures" / f"m4_telemetry_day{args.plot_day}.png")
        print(f"saved figure     -> {fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
