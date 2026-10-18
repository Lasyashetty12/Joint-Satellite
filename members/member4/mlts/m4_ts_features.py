"""
m4_ts_features.py - lag / rolling / difference / orbit-Fourier features with NO look-ahead.

Every feature at time t uses only values at times <= t (lags and rolling windows
look backwards; Fourier orbit terms are deterministic clock functions, known in
advance). A unit test perturbs the future and checks that past features do not change.

Also provides a train-only StandardScaler that is saved to JSON with the model.

Run from the repository root (quick inspection; models import the functions):
    python members/member4/mlts/m4_ts_features.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m4_ts_common as C  # noqa: E402

DEFAULT_LAGS = (1, 2, 3, 5, 10, 190)        # 190 = one orbit ago (seasonal lag)
DEFAULT_WINDOWS = (10, 60, 190)              # 5 min, 30 min, one orbit


def build_features(df: pd.DataFrame, channels=C.CHANNELS, lags=DEFAULT_LAGS,
                   windows=DEFAULT_WINDOWS, fourier_k: int = 3, period_s: float | None = None,
                   t0=None) -> pd.DataFrame:
    """Returns a feature frame aligned with df (row t = information available at time t)."""
    if period_s is None or t0 is None:
        cfg = C.load_config()
        period_s = period_s or cfg["orbit_period_min"] * 60
        t0 = t0 if t0 is not None else pd.Timestamp(cfg["start_time"])   # same phase origin for every split
    feats = {}
    for c in channels:
        x = df[c]
        feats[f"{c}"] = x                                         # current value (known at t)
        for L in lags:
            feats[f"{c}_lag{L}"] = x.shift(L)
        feats[f"{c}_diff1"] = x.diff(1)
        for w in windows:
            r = x.rolling(w, min_periods=max(2, w // 2))          # window ENDS at t (inclusive)
            feats[f"{c}_rmean{w}"] = r.mean()
            feats[f"{c}_rstd{w}"] = r.std()
            if w == windows[0]:
                feats[f"{c}_rmin{w}"] = r.min()
                feats[f"{c}_rmax{w}"] = r.max()
    out = pd.DataFrame(feats, index=df.index)
    for flag in ("sunlit", "payload_on"):
        if flag in df.columns:
            out[flag] = df[flag].astype(float)
    if fourier_k:
        out = pd.concat([out, C.fourier_terms(df["timestamp"], period_s, fourier_k, start=t0)], axis=1)
    return out


class TrainScaler:
    """Standardise with mean/std computed on TRAIN rows only (NaN-aware)."""

    def __init__(self):
        self.mean_: dict[str, float] = {}
        self.std_: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame) -> "TrainScaler":
        self.mean_ = {c: float(np.nanmean(frame[c].values)) for c in frame.columns}
        self.std_ = {c: float(np.nanstd(frame[c].values)) or 1.0 for c in frame.columns}
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        cols = list(frame.columns)
        mu = np.array([self.mean_[c] for c in cols]); sd = np.array([self.std_[c] for c in cols])
        return pd.DataFrame((frame.values - mu) / sd, index=frame.index, columns=cols)

    def inverse_column(self, values: np.ndarray, col: str) -> np.ndarray:
        return values * self.std_[col] + self.mean_[col]

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"mean": self.mean_, "std": self.std_}, indent=1))

    @classmethod
    def load(cls, path: Path) -> "TrainScaler":
        d = json.loads(Path(path).read_text())
        s = cls(); s.mean_, s.std_ = d["mean"], d["std"]
        return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Member 4: build no-look-ahead time-series features (inspection).")
    ap.add_argument("--rows", type=int, default=5, help="rows to print")
    args = ap.parse_args(argv)
    df = C.load_telemetry()
    X = build_features(df)
    masks = C.split_masks(df)
    print(f"feature matrix: {X.shape[0]} rows x {X.shape[1]} columns")
    print(f"train rows: {masks['train'].sum()}, val rows: {masks['val'].sum()}, test rows: {masks['test'].sum()}")
    print("NaN fraction (train):", round(float(X[masks['train']].isna().mean().mean()), 4))
    print(X.iloc[200:200 + args.rows, :8].to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
