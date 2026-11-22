"""
m4_build_passes.py - build VIRTUAL SATELLITE PASSES (Milestone 7, handbook section 5).

A pass = one 60-minute telemetry window (120 samples x 8 channels) + one Sen1Floods11 patch
(optical + SAR) + labels. Pairing happens ONLY inside the same split (train with train, ...).
No public dataset pairs a satellite's health with its own images, so this pairing is a
SIMULATION ASSUMPTION and is stated as such in the paper.

Coupling (the key assumption): when the telemetry window contains a PAYLOAD FAULT, the paired
optical patch is degraded (stripes / noise / saturation). The ground truth flood label is NOT
changed: the flood is still there, the camera just cannot see it properly.
Independent nuisance: ~15 % of passes get clouds over the optical patch (nothing to do with health).

Per pass we store:
    telemetry values (train-standardised), forecast-residual z-scores from the validation-selected
    detector (Milestone 5), NaN mask; health_label (window overlaps a labelled anomaly),
    anomaly_type, eo_label, patch_id, optical degradation (kind, severity, seed), coupled flag.

Outputs (git-ignored): data/generated/passes/m4_passes.parquet, m4_pass_telemetry.npz
Small summary (committed): results/m4_pass_summary.csv

Run from the repository root (after Milestones 5 and 6):
    python members/member4/data/m4_build_passes.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MEMBER_DIR = Path(__file__).resolve().parents[1]
for _s in ("mlts", "data", "evaluation"):
    sys.path.insert(0, str(MEMBER_DIR / _s))
import m4_eo_patches as EP  # noqa: E402
import m4_ts_common as C  # noqa: E402

OUT = C.GEN_DIR / "passes"
WIN = 120                       # 60 min at 30 s
N_PASSES = {"train": 1600, "val": 400, "test": 400}
ANOM_SHARE = 0.4                # share of passes drawn to overlap a labelled event
CLOUD_P = 0.15
EO_POOL = {"train": ["train"], "val": ["val"], "test": ["test", "bolivia"]}


def selected_forecaster() -> str:
    """Forecaster behind the validation-selected residual detector (Milestone 5)."""
    f = C.RESULTS / "health_metrics.csv"
    if f.exists():
        h = pd.read_csv(f)
        h = h[(h.dataset == "simulated_v1") & (h.split == "val") & h.model.str.startswith("residual_")
              & ~h.model.str.contains("ensemble")]
        if len(h):
            best = h.sort_values(["event_F1", "false_alarms_per_day"], ascending=[False, True]).iloc[0].model
            name = best.replace("residual_", "")
            if (C.GEN_DIR / "forecasts" / f"m4_pred_{name}.npz").exists():
                return name
    for name in ("gru", "xgboost", "arimax", "naive"):
        if (C.GEN_DIR / "forecasts" / f"m4_pred_{name}.npz").exists():
            return name
    raise SystemExit("No forecasts found. Run mlts/m4_forecast_eval.py first.")


def residual_z(df, masks, normal, model: str) -> np.ndarray:
    import m4_residual_anomaly as RA
    z = np.load(C.GEN_DIR / "forecasts" / f"m4_pred_{model}.npz")
    _, zc, _ = RA.residual_score(df, z["pred"], list(z["horizons"]), masks, normal)
    return np.clip(zc, 0, 20).astype(np.float32)


def sample_starts(split_mask: np.ndarray, events, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.where(split_mask)[0]
    lo, hi = idx[0], idx[-1] - WIN + 1
    ev = [(a, b) for a, b, _ in events if split_mask[a]]
    n_anom = int(round(ANOM_SHARE * n)) if ev else 0
    starts = list(rng.integers(lo, hi + 1, size=n - n_anom))
    for _ in range(n_anom):
        a, b = ev[int(rng.integers(len(ev)))]
        s = int(rng.integers(max(lo, a - WIN + 1), min(hi, b) + 1))
        starts.append(s)
    return np.array(starts)


def build(seed: int = 42, n_passes: dict | None = None) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    n_passes = n_passes or N_PASSES
    df = C.load_telemetry()
    masks = C.split_masks(df)
    labels = C.load_labels()
    lab = C.label_mask(df, labels)
    events = C.label_events(df, labels)
    normal = (lab == 0) & ~df[C.CHANNELS].isna().any(axis=1).values
    fmodel = selected_forecaster()
    Z = residual_z(df, masks, normal, fmodel)
    mu = df.loc[masks["train"] & normal, C.CHANNELS].mean().values
    sd = df.loc[masks["train"] & normal, C.CHANNELS].std().values
    V = ((df[C.CHANNELS].values - mu) / sd).astype(np.float32)
    ev_type = np.full(len(df), "", dtype=object)
    for a, b, t in events:
        ev_type[a:b + 1] = t

    _, meta, _ = EP.load()
    rows, TEL, RES = [], [], []
    for split, n in n_passes.items():
        starts = sample_starts(masks[split], events, n, rng)
        pool = meta[meta.split.isin(EO_POOL[split]) & (meta.eo_label >= 0)].patch_id.values
        for s in starts:
            w = slice(s, s + WIN)
            types = [t for t in pd.unique(ev_type[w]) if t]
            coupled = "payload_fault" in types
            pid = int(rng.choice(pool))
            if coupled:
                kind, sev = str(rng.choice(["stripes", "noise", "saturation"])), float(rng.uniform(0.5, 1.0))
            elif rng.random() < CLOUD_P:
                kind, sev = "cloud", float(rng.uniform(0.3, 0.9))
            else:
                kind, sev = "none", 0.0
            rows.append({"pass_id": len(rows), "split": split, "t_start": df.timestamp.iloc[s].isoformat(),
                         "start_idx": int(s), "health_label": int(lab[w].max()), "anomaly_type": ";".join(types),
                         "patch_id": pid, "eo_label": int(meta.eo_label.iloc[pid]),
                         "opt_degrade": kind, "opt_severity": round(sev, 3), "degrade_seed": int(rng.integers(1 << 30)),
                         "coupled_payload_fault": int(coupled)})
            TEL.append(V[w]); RES.append(Z[w])
    passes = pd.DataFrame(rows)
    passes["eo_dataset"] = meta.dataset.iloc[0]
    passes["forecaster"] = fmodel
    tel = np.stack(TEL).astype(np.float32)
    arrays = {"tel": tel, "res": np.stack(RES).astype(np.float32), "tel_nan": np.isnan(tel)}
    return passes, arrays


def summary(passes: pd.DataFrame) -> pd.DataFrame:
    g = passes.groupby("split")
    return pd.DataFrame({"passes": g.size(), "health_pos": g.health_label.sum(), "eo_pos": g.eo_label.sum(),
                         "coupled_payload_fault": g.coupled_payload_fault.sum(),
                         "cloudy": g.opt_degrade.apply(lambda s: int((s == "cloud").sum())),
                         "unique_patches": g.patch_id.nunique()}).reindex(["train", "val", "test"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Member 4: build virtual satellite passes (telemetry + EO).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    passes, arrays = build(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    passes.to_parquet(OUT / "m4_passes.parquet", index=False)
    np.savez_compressed(OUT / "m4_pass_telemetry.npz", **arrays)
    s = summary(passes)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    s.to_csv(C.RESULTS / "m4_pass_summary.csv")
    print(f"forecaster for residual features: {passes.forecaster.iloc[0]}   EO data: {passes.eo_dataset.iloc[0]}")
    print(s.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
