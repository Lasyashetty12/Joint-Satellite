"""
m4_make_splits.py - chronological train / validation / test split of the telemetry.

Rule (handbook section 15): NEVER shuffle a time series before splitting.
We cut the 90 days into contiguous day blocks, ~60/20/20, with a 1-day GAP before
validation and before test. The gap stops lag/rolling features near a boundary
from "seeing" the next split.

    train : day 0  - day 53   (54 days)
    gap   : day 54
    val   : day 55 - day 71   (17 days)
    gap   : day 72
    test  : day 73 - day 89   (17 days)

Writes:
  - the `split` column into data/generated/m4_telemetry_v1.parquet  (train/val/test/gap)
  - data/generated/m4_splits.json  (exact boundaries, small; commit it)

Run from the repository root:
    python members/member4/data/m4_make_splits.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mlts"))
import m4_ts_common as C  # noqa: E402


def day_blocks(n_days: int, frac=(0.6, 0.2, 0.2), gap_days: int = 1) -> dict[str, tuple[int, int]]:
    """Inclusive day ranges. Gaps are taken out of the val and test blocks."""
    n_train = int(round(frac[0] * n_days))
    n_val = int(round(frac[1] * n_days))
    train = (0, n_train - 1)
    val = (n_train + gap_days, n_train + n_val - 1)
    test = (n_train + n_val + gap_days, n_days - 1)
    if not (train[1] < val[0] <= val[1] < test[0] <= test[1]):
        raise ValueError(f"bad split for {n_days} days: {train}, {val}, {test}")
    return {"train": train, "val": val, "test": test}


def assign_split(df: pd.DataFrame, blocks: dict, per_day: int) -> pd.Series:
    day = np.arange(len(df)) // per_day
    split = np.full(len(df), "gap", dtype=object)
    for name, (a, b) in blocks.items():
        split[(day >= a) & (day <= b)] = name
    return pd.Series(split, index=df.index, name="split")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Member 4 chronological split by day blocks with gaps.")
    ap.add_argument("--telemetry", default=str(C.TELEMETRY))
    ap.add_argument("--gap-days", type=int, default=1)
    args = ap.parse_args(argv)

    cfg = C.load_config()
    df = C.load_telemetry(Path(args.telemetry))
    per_day = C.samples_per_day(cfg)
    n_days = int(np.ceil(len(df) / per_day))
    blocks = day_blocks(n_days, gap_days=args.gap_days)
    df["split"] = assign_split(df, blocks, per_day)
    df.to_parquet(args.telemetry, index=False)

    labels = C.load_labels()
    info = {"method": "chronological day blocks, ~60/20/20, gap before val and test",
            "gap_days": args.gap_days, "samples_per_day": per_day, "n_days": n_days}
    for name, (a, b) in blocks.items():
        part = df[df["split"] == name]
        ev = labels[(labels["start"] >= part["timestamp"].iloc[0]) & (labels["end"] <= part["timestamp"].iloc[-1])]
        info[name] = {"days": [a, b], "n_days": b - a + 1, "rows": int(len(part)),
                      "start": part["timestamp"].iloc[0].isoformat(),
                      "end": part["timestamp"].iloc[-1].isoformat(),
                      "anomaly_events": int(len(ev)),
                      "events_per_type": ev["anomaly_type"].value_counts().to_dict()}
    info["gap_rows"] = int((df["split"] == "gap").sum())
    C.SPLITS.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    print(f"\nsaved split column -> {args.telemetry}\nsaved boundaries   -> {C.SPLITS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
