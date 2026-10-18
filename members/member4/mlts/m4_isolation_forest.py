"""
m4_isolation_forest.py - Isolation Forest baseline for health anomalies (unsupervised).

Idea: anomalies are "few and different", so random splits isolate them in fewer steps.
Features at time t (no look-ahead): the 8 channel values, their 5-min rolling std and
first difference, plus sunlit and orbit sin/cos (so "cold panel" is normal in eclipse
but abnormal in sunlight). Standardised with TRAIN statistics; fitted on a random sample
of NORMAL train rows. Score = -score_samples (higher = more anomalous).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m4_ts_common as C  # noqa: E402
from m4_ts_features import TrainScaler, build_features  # noqa: E402

N_FIT = 60000


def if_features(df: pd.DataFrame) -> pd.DataFrame:
    X = build_features(df, lags=(), windows=(10,), fourier_k=1)
    keep = [c for c in X.columns if c in C.CHANNELS or c.endswith("_rstd10") or c.endswith("_diff1")
            or c in ("sunlit", "orb_sin1", "orb_cos1")]
    return X[keep]


def fit_score(df: pd.DataFrame, masks: dict, normal: np.ndarray, seed: int = 42) -> np.ndarray:
    from sklearn.ensemble import IsolationForest

    X = if_features(df)
    sc = TrainScaler().fit(X[masks["train"] & normal])
    Z = sc.transform(X).values
    finite = ~np.isnan(Z).any(axis=1)
    fit_rows = np.where(masks["train"] & normal & finite)[0]
    rng = np.random.default_rng(seed)
    fit_rows = rng.choice(fit_rows, size=min(N_FIT, len(fit_rows)), replace=False)
    model = IsolationForest(n_estimators=200, max_samples=4096, random_state=seed, n_jobs=-1).fit(Z[fit_rows])
    score = np.full(len(df), np.nan)
    score[finite] = -model.score_samples(Z[finite])
    return score


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description="Isolation Forest baseline (library). Run mlts/m4_residual_anomaly.py").parse_args()
    print("Use m4_residual_anomaly.py")
