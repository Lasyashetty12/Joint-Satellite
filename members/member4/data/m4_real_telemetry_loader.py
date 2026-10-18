"""
m4_real_telemetry_loader.py - load the OPSSAT-AD real telemetry benchmark (ESA OPS-SAT CubeSat).

Files (download once into <repo>/data_external/opssat_ad/, see DATASET_CARD.md):
    segments.csv : channel, timestamp, value, label, sampling, anomaly, segment, train
    dataset.csv  : segment, anomaly, train, channel, sampling + 18 hand-crafted features

Cleaning (decisions recorded in DATASET_CARD.md):
  * sort each segment by time and DROP duplicate timestamps (22,115 0-s gaps measured)
  * keep channels separate; keep the 1 s and 5 s sampling groups separate
  * use only the binary `anomaly` column (meaning of a2/a3/a4 is undocumented)
  * official split: train == 1 -> fit/threshold, train == 0 -> test once
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MEMBER_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = MEMBER_DIR.parents[1] / "data_external" / "opssat_ad"
FEATURES = ["duration", "len", "mean", "var", "std", "kurtosis", "skew", "n_peaks", "smooth10_n_peaks",
            "smooth20_n_peaks", "diff_peaks", "diff2_peaks", "diff_var", "diff2_var", "gaps_squared",
            "len_weighted", "var_div_duration", "var_div_len"]


@dataclass
class Segment:
    segment: int
    channel: str
    sampling: int
    anomaly: int
    train: int
    t: np.ndarray        # seconds since segment start
    x: np.ndarray        # values


def load_segments(root: Path = DEFAULT_ROOT) -> list[Segment]:
    path = Path(root) / "segments.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Download segments.csv and dataset.csv from "
                         f"https://zenodo.org/records/12588359 into {root}")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "value"])
    df = df.sort_values(["segment", "timestamp"]).drop_duplicates(["segment", "timestamp"], keep="first")
    segs = []
    for sid, g in df.groupby("segment", sort=True):
        t = (g["timestamp"] - g["timestamp"].iloc[0]).dt.total_seconds().values
        segs.append(Segment(int(sid), str(g["channel"].iloc[0]), int(g["sampling"].iloc[0]),
                            int(g["anomaly"].max()), int(g["train"].iloc[0]), t, g["value"].values.astype(float)))
    return segs


def load_features(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    path = Path(root) / "dataset.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found (see load_segments).")
    return pd.read_csv(path).sort_values("segment").reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Load OPSSAT-AD and print a short summary.")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    args = ap.parse_args(argv)
    segs = load_segments(Path(args.root))
    n = len(segs)
    print(f"segments: {n}  anomalous: {sum(s.anomaly for s in segs)}  train: {sum(s.train for s in segs)}")
    print(f"readings after de-duplication: {sum(len(s.x) for s in segs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
