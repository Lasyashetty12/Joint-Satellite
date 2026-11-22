"""
m4_ts_common.py - shared helpers for Member 4's time-series (MLTS) scripts.

Keeps paths, channel lists, label masks and orbit (Fourier) terms in ONE place so
EDA, features, forecasting and anomaly detection all read the data the same way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MEMBER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MEMBER_DIR.parents[1]
GEN_DIR = MEMBER_DIR / "data" / "generated"
REPORTS = MEMBER_DIR / "reports"
FIGURES = REPORTS / "figures"
RESULTS = MEMBER_DIR / "results"          # small CSV/JSON results (committed)
CONFIG = MEMBER_DIR / "configs" / "m4_generator_config.json"

TELEMETRY = GEN_DIR / "m4_telemetry_v1.parquet"
LABELS = GEN_DIR / "m4_telemetry_labels.csv"
DROPOUTS = GEN_DIR / "m4_telemetry_dropouts.csv"
SPLITS = GEN_DIR / "m4_splits.json"

CHANNELS = ["batt_temp", "panel_temp", "bus_V", "batt_I", "solar_I", "SoC", "att_err", "payload_P"]
UNITS = {"batt_temp": "°C", "panel_temp": "°C", "bus_V": "V", "batt_I": "A",
         "solar_I": "A", "SoC": "%", "att_err": "deg", "payload_P": "W"}

for _sub in ("data", "mlts", "models", "training", "evaluation", "explainability"):
    _p = str(MEMBER_DIR / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_config(path: Path = CONFIG) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def samples_per_orbit(cfg: dict | None = None) -> int:
    cfg = cfg or load_config()
    return int(round(cfg["orbit_period_min"] * 60 / cfg["cadence_s"]))   # 95 min / 30 s = 190


def samples_per_day(cfg: dict | None = None) -> int:
    cfg = cfg or load_config()
    return int(86400 // cfg["cadence_s"])                                 # 2880


def load_telemetry(path: Path = TELEMETRY) -> pd.DataFrame:
    if not Path(path).exists():
        raise SystemExit(f"{path} not found. Run: python members/member4/data/m4_telemetry_generator.py")
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.reset_index(drop=True)


def load_labels(path: Path = LABELS) -> pd.DataFrame:
    lab = pd.read_csv(path)
    lab["start"] = pd.to_datetime(lab["start"], utc=True)
    lab["end"] = pd.to_datetime(lab["end"], utc=True)
    return lab


def _ns(ts: pd.Series) -> np.ndarray:
    """Timestamps as int64 nanoseconds since epoch (UTC), independent of pandas resolution."""
    return np.asarray(ts.values).astype("datetime64[ns]").astype("int64")


def interval_indices(ts: pd.Series, start, end) -> tuple[int, int]:
    """Inclusive index range of rows with start <= ts <= end (ts sorted, UTC)."""
    v = _ns(ts)
    a = int(np.searchsorted(v, pd.Timestamp(start).value, side="left"))
    b = int(np.searchsorted(v, pd.Timestamp(end).value, side="right")) - 1
    return a, b


def label_mask(df: pd.DataFrame, labels: pd.DataFrame) -> np.ndarray:
    """Point-wise ground truth: 1 inside any labelled anomaly interval."""
    m = np.zeros(len(df), dtype=np.int8)
    for _, r in labels.iterrows():
        a, b = interval_indices(df["timestamp"], r["start"], r["end"])
        m[a:b + 1] = 1
    return m


def label_events(df: pd.DataFrame, labels: pd.DataFrame) -> list[tuple[int, int, str]]:
    """[(start_idx, end_idx, anomaly_type), ...] in row indices of df."""
    out = []
    for _, r in labels.iterrows():
        a, b = interval_indices(df["timestamp"], r["start"], r["end"])
        if b >= a:
            out.append((a, b, r["anomaly_type"]))
    return out


def fourier_terms(ts: pd.Series, period_s: float, k: int = 3, start=None) -> pd.DataFrame:
    """sin/cos of the orbit phase (deterministic: known in advance, so not look-ahead)."""
    t0 = pd.Timestamp(start) if start is not None else ts.iloc[0]
    sec = (ts - t0).dt.total_seconds().values
    cols = {}
    for j in range(1, k + 1):
        cols[f"orb_sin{j}"] = np.sin(2 * np.pi * j * sec / period_s)
        cols[f"orb_cos{j}"] = np.cos(2 * np.pi * j * sec / period_s)
    return pd.DataFrame(cols, index=ts.index)


def load_splits(path: Path = SPLITS) -> dict:
    if not Path(path).exists():
        raise SystemExit(f"{path} not found. Run: python members/member4/data/m4_make_splits.py")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_masks(df: pd.DataFrame, splits: dict | None = None) -> dict[str, np.ndarray]:
    """Boolean masks for train/val/test from the 'split' column (preferred) or splits.json."""
    if "split" in df.columns and set(df["split"].unique()) & {"train", "val", "test"}:
        return {s: (df["split"] == s).values for s in ("train", "val", "test")}
    splits = splits or load_splits()
    out = {}
    for s in ("train", "val", "test"):
        a = pd.Timestamp(splits[s]["start"]); b = pd.Timestamp(splits[s]["end"])
        out[s] = ((df["timestamp"] >= a) & (df["timestamp"] <= b)).values
    return out
