"""
m4_forecast_lstm.py - GRU sequence-to-horizon forecaster (deep MLTS model).

Input  : the last W = 60 samples (30 min) of 8 channels + sunlit + payload_on + orbit sin/cos,
         standardised with TRAIN-only statistics.
Output : the change y[t+h] - y[t] for h = 1..10 and all 8 channels at once (one network).
Model  : 1-layer GRU (hidden 64); its last state + the KNOWN-FUTURE sunlit flags for
         t+1..t+10 (eclipse schedule from the orbit, same covariate ARIMAX/XGBoost get)
         -> small MLP head. Small on purpose: a few minutes on a laptop CPU; GPU optional.
Training windows: train split, stride 4, window and targets fully normal (no labelled
anomaly, no NaN). Best epoch chosen on the validation loss; seeds fixed.

The file is named "lstm" as in the handbook; set CELL = "lstm" to swap the GRU for an LSTM.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m4_ts_common as C  # noqa: E402
from m4_forecast_naive import HORIZONS  # noqa: E402
from m4_forecast_xgboost import window_ok  # noqa: E402

W = 60
H_MAX = 10
HIDDEN = 64
CELL = "gru"
EPOCHS = 15
BATCH = 512
LR = 2e-3
TRAIN_STRIDE = 4
CHECKPOINT = C.MEMBER_DIR / "checkpoints" / "m4_gru_forecaster.pt"


def input_frame(df: pd.DataFrame) -> pd.DataFrame:
    cfg = C.load_config()
    f = C.fourier_terms(df["timestamp"], cfg["orbit_period_min"] * 60, 1, start=pd.Timestamp(cfg["start_time"]))
    return pd.concat([df[C.CHANNELS], df[["sunlit", "payload_on"]].astype(float), f], axis=1)


def build_model(n_in: int, n_out: int, cell: str = CELL, hidden: int = HIDDEN, n_future: int = H_MAX):
    import torch
    import torch.nn as nn

    class Seq2H(nn.Module):
        def __init__(self):
            super().__init__()
            rnn = nn.GRU if cell == "gru" else nn.LSTM
            self.rnn = rnn(n_in, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden + n_future, 128), nn.ReLU(), nn.Linear(128, n_out))

        def forward(self, x, fut):
            out, _ = self.rnn(x)
            return self.head(torch.cat([out[:, -1], fut], dim=1))

    return Seq2H()


def future_sunlit(df: pd.DataFrame) -> np.ndarray:
    """F[t, h-1] = sunlit at t+h from the orbit geometry (known in advance)."""
    cfg = C.load_config()
    period_s = cfg["orbit_period_min"] * 60
    sec = (df["timestamp"] - pd.Timestamp(cfg["start_time"])).dt.total_seconds().values
    out = np.empty((len(df), H_MAX), np.float32)
    for h in range(1, H_MAX + 1):
        phase = ((sec + h * cfg["cadence_s"]) % period_s) / period_s
        out[:, h - 1] = (phase < 1.0 - cfg["eclipse_fraction"]).astype(np.float32)
    return out


def make_windows(Z: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Z [N, F]; ends = indices t; returns [len(ends), W, F] for rows t-W+1..t."""
    offs = np.arange(-W + 1, 1)
    return Z[ends[:, None] + offs[None, :]]


def fit_predict(df: pd.DataFrame, masks: dict, normal: np.ndarray, horizons=HORIZONS,
                channels=C.CHANNELS, seed: int = 42, epochs: int = EPOCHS, verbose=True,
                save_checkpoint: bool = True, **_):
    import torch

    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    horizons = list(horizons)
    n = len(df)
    inp = input_frame(df)
    mu = inp[masks["train"]].mean(); sd = inp[masks["train"]].std().replace(0, 1)
    Z = ((inp - mu) / sd).values.astype(np.float32)
    y = df[channels].values.astype(np.float32)
    d1 = df[channels].diff().to_numpy(copy=True)
    d1[~(normal & np.roll(normal, 1))] = np.nan                     # step changes between NORMAL samples only
    ysd = (np.nanstd(d1[masks["train"]], axis=0) * np.sqrt(H_MAX)).astype(np.float32)   # target scale

    # targets: delta for h = 1..H_MAX, standardised
    T = np.full((n, H_MAX, len(channels)), np.nan, np.float32)
    for h in range(1, H_MAX + 1):
        T[:n - h, h - 1] = (y[h:] - y[:n - h]) / ysd
    ok_in = window_ok(normal, W)
    ok_tgt = window_ok(normal[::-1], H_MAX + 1)[::-1]                # normal[t .. t+H_MAX]
    ok = ok_in & ok_tgt & ~np.isnan(T).any(axis=(1, 2)) & ~np.isnan(Z).any(axis=1)
    ok[:W] = False
    tr = np.where(masks["train"] & ok)[0][::TRAIN_STRIDE]
    va = np.where(masks["val"] & ok)[0][::TRAIN_STRIDE * 2]

    FUT = future_sunlit(df)
    model = build_model(Z.shape[1], H_MAX * len(channels))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()
    Xva = torch.from_numpy(make_windows(Z, va)); Fva = torch.from_numpy(FUT[va]); Yva = torch.from_numpy(T[va].reshape(len(va), -1))
    best, best_state, history = np.inf, None, []
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        t0 = time.time(); model.train()
        perm = rng.permutation(tr)
        tot = 0.0
        for b in range(0, len(perm), BATCH):
            idx = perm[b:b + BATCH]
            xb = torch.from_numpy(make_windows(Z, idx)); yb = torch.from_numpy(T[idx].reshape(len(idx), -1))
            fb = torch.from_numpy(FUT[idx])
            opt.zero_grad(); loss = loss_fn(model(xb, fb), yb); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(Xva, Fva), Yva))
        history.append({"epoch": ep + 1, "train_loss": tot / len(perm), "val_loss": vl})
        if vl < best:
            best = vl; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  GRU epoch {ep + 1}/{epochs}  train {tot / len(perm):.4f}  val {vl:.4f}  ({time.time() - t0:.0f}s)")
    model.load_state_dict(best_state); model.eval()

    if save_checkpoint:
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": best_state, "mu": mu.to_dict(), "sd": sd.to_dict(), "ysd": ysd.tolist(),
                    "W": W, "H_MAX": H_MAX, "cell": CELL, "hidden": HIDDEN}, CHECKPOINT)

    pred = np.full((n, len(horizons), len(channels)), np.nan, np.float32)
    ends = np.where(~np.isnan(Z).any(axis=1))[0]
    ends = ends[ends >= W - 1]
    ok_win = window_ok(~np.isnan(Z).any(axis=1), W)
    ends = ends[ok_win[ends]]
    with torch.no_grad():
        for b in range(0, len(ends), 8192):
            e = ends[b:b + 8192]
            d = model(torch.from_numpy(make_windows(Z, e)), torch.from_numpy(FUT[e])).numpy().reshape(len(e), H_MAX, len(channels)) * ysd
            for k, h in enumerate(horizons):
                pred[e, k, :] = y[e] + d[:, h - 1, :]
    info = {"history": history, "best_val_loss": round(best, 5), "train_windows": int(len(tr)),
            "val_windows": int(len(va))}
    return pred, info


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description="GRU forecaster (library). Run mlts/m4_forecast_eval.py --models gru").parse_args()
    print("Use m4_forecast_eval.py --models gru")
