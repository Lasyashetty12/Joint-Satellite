"""
m4_forecast_naive.py - the two reference forecasters every other model must beat.

    naive           y_hat[t+h] = y[t]              ("tomorrow = today")
    seasonal-naive  y_hat[t+h] = y[t+h-190]        ("same point one orbit ago")

Common output format for ALL Member 4 forecasters:
    pred : float32 array of shape [N, len(horizons), C]
    pred[t, k, c] = forecast of channel c at time t + horizons[k], made at origin t
                    (NaN where it cannot be made).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m4_ts_common as C  # noqa: E402

HORIZONS = (1, 5, 10)        # 30 s, 2.5 min, 5 min ahead


def naive(df: pd.DataFrame, horizons=HORIZONS, channels=C.CHANNELS, **_) -> np.ndarray:
    y = df[channels].values.astype(np.float32)
    return np.repeat(y[:, None, :], len(horizons), axis=1)


def seasonal_naive(df: pd.DataFrame, horizons=HORIZONS, channels=C.CHANNELS, period: int | None = None, **_) -> np.ndarray:
    period = period or C.samples_per_orbit()
    y = df[channels].values.astype(np.float32)
    n = len(y)
    out = np.full((n, len(horizons), len(channels)), np.nan, dtype=np.float32)
    for k, h in enumerate(horizons):
        if h > period:
            raise ValueError("seasonal-naive needs h <= period")
        src = np.arange(n) + h - period          # index of 'same orbit phase, one orbit earlier'
        ok = src >= 0
        out[ok, k, :] = y[src[ok]]
    return out


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description="Naive / seasonal-naive forecasters (library). "
                            "Run mlts/m4_forecast_eval.py to evaluate.").parse_args()
    print("Use m4_forecast_eval.py --models naive,snaive")
