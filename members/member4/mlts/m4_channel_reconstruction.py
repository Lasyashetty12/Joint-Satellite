"""
m4_channel_reconstruction.py - reconstruct a MISSING telemetry channel from the others (RQ3).

Physics links channels (sunlight -> solar current -> battery current -> SoC -> bus voltage),
so a lost channel can be estimated at time t from the OTHER channels at time t plus the orbit
phase / sunlit flag. One XGBoost regressor per channel, trained on NORMAL TRAIN rows only.

Compared on the TEST split (normal points, channel hidden for a whole 60-min gap):
    mean_fill            train mean of the channel
    ffill                last value observed before the gap (persists for the whole gap)
    cross_channel        XGBoost reconstruction from the other channels   (ours)
    cross_channel_4_lost the same, when 3 other channels are lost at the same time (hidden from the model)
Models are trained with other channels randomly hidden, saved to checkpoints/m4_recon_<channel>.json,
and `reconstruct()` never uses a lost channel to rebuild another one.

Outputs:
    results/m4_reconstruction_results.csv   channel, method, MAE, RMSE, n
    data/generated/m4_reconstruction.npz    reconstructed values for EVERY row (used by the
                                            missing-channel fusion experiments; git-ignored)

Run from the repository root:
    python members/member4/mlts/m4_channel_reconstruction.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m4_ts_common as C  # noqa: E402

GAP = 120          # 60-min gap for the forward-fill baseline
OUT_NPZ = C.GEN_DIR / "m4_reconstruction.npz"


def design(df: pd.DataFrame, target: str) -> np.ndarray:
    cfg = C.load_config()
    f = C.fourier_terms(df["timestamp"], cfg["orbit_period_min"] * 60, 2, start=pd.Timestamp(cfg["start_time"])).values
    others = [c for c in C.CHANNELS if c != target]
    return np.column_stack([df[others].values, df["sunlit"].values, f]).astype(np.float32)


P_OTHER_MISSING = 0.3          # training augmentation: each other channel hidden in 30 % of rows
MODEL_DIR = C.MEMBER_DIR / "checkpoints"


def fit_all(df, masks, normal, seed: int = 42, stride: int = 4):
    """One XGBoost model per channel. During training each OTHER channel is randomly hidden (NaN,
    which XGBoost handles natively), so the model also works when several channels are lost at once
    and is only ever given the channels that are really available. Models are saved for reuse."""
    import xgboost as xgb
    rng = np.random.default_rng(seed)
    recon = np.full((len(df), len(C.CHANNELS)), np.nan, np.float32)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for j, c in enumerate(C.CHANNELS):
        X = design(df, c)
        y = df[c].values
        ok = normal & ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        tr = np.where(masks["train"] & ok)[0][::stride]
        va = np.where(masks["val"] & ok)[0][::stride * 2]
        Xtr = X[tr].copy()
        Xtr[:, :7][rng.random((len(tr), 7)) < P_OTHER_MISSING] = np.nan
        m = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.8,
                             colsample_bytree=0.8, tree_method="hist", early_stopping_rounds=20,
                             random_state=seed, n_jobs=-1, verbosity=0)
        m.fit(Xtr, y[tr], eval_set=[(X[va], y[va])], verbose=False)
        m.save_model(MODEL_DIR / f"m4_recon_{c}.json")
        recon[:, j] = m.predict(X)
    return recon


_MODELS: dict = {}


def reconstruct(df: pd.DataFrame, rows: np.ndarray, missing: list[str]) -> np.ndarray:
    """Reconstruct the `missing` channels at `rows`, hiding ALL missing channels from every model
    (no channel is rebuilt from another lost channel). Returns [len(rows), len(missing)] raw units."""
    import xgboost as xgb
    out = np.empty((len(rows), len(missing)), np.float32)
    sub = df.iloc[rows].copy()
    sub[missing] = np.nan
    for k, c in enumerate(missing):
        if c not in _MODELS:
            m = xgb.XGBRegressor(); m.load_model(MODEL_DIR / f"m4_recon_{c}.json"); _MODELS[c] = m
        out[:, k] = _MODELS[c].predict(design(sub, c))
    return out


def evaluate(df, masks, normal, recon) -> pd.DataFrame:
    rows = []
    te = np.where(masks["test"] & normal)[0]
    for j, c in enumerate(C.CHANNELS):
        y = df[c].values
        mean = np.nanmean(y[masks["train"] & normal])
        # forward fill: value just before the 60-min gap that contains t
        gap_start = te - (te % GAP)
        prev = np.maximum(gap_start - 1, 0)
        ff = pd.Series(y).ffill().values[prev]
        # same channel, but 3 OTHER random channels are lost too (they are hidden from the model)
        rng = np.random.default_rng(100 + j)
        others = [o for o in C.CHANNELS if o != c]
        lost = [c] + list(rng.choice(others, size=3, replace=False))
        multi = reconstruct(df, te, lost)[:, 0]
        for name, est in [("mean_fill", np.full(len(te), mean)), ("ffill", ff), ("cross_channel", recon[te, j]),
                          ("cross_channel_4_lost", multi)]:
            e = y[te] - est
            ok = ~np.isnan(e)
            rows.append({"channel": c, "method": name, "MAE": float(np.mean(np.abs(e[ok]))),
                         "RMSE": float(np.sqrt(np.mean(e[ok] ** 2))), "n": int(ok.sum())})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    argparse.ArgumentParser(description="Member 4: cross-channel reconstruction of missing telemetry (RQ3).").parse_args(argv)
    df = C.load_telemetry()
    masks = C.split_masks(df)
    normal = (C.label_mask(df, C.load_labels()) == 0) & ~df[C.CHANNELS].isna().any(axis=1).values
    recon = fit_all(df, masks, normal)
    np.savez_compressed(OUT_NPZ, recon=recon, channels=np.array(C.CHANNELS))
    res = evaluate(df, masks, normal, recon)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    res.round(5).to_csv(C.RESULTS / "m4_reconstruction_results.csv", index=False)
    print(res.pivot(index="channel", columns="method", values="MAE").round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
