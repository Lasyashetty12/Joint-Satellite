"""
m4_eo_patches.py - turn Sen1Floods11 hand-labelled chips into model-ready patches (Milestone 6).

Input  : <repo>/data_external/sen1floods11/HandLabeled/{S1Hand,S2Hand,LabelHand}/*.tif
         <repo>/data_external/sen1floods11/splits/flood_{train,valid,test,bolivia}_data.csv
         (download with: python members/member4/data/m4_download_eo_sample.py --event ALL)
Output : data/generated/eo/m4_eo_patches.npz   s1 [N,2,128,128], s2 [N,6,128,128] (float16, standardised),
                                                mask [N,128,128] (int8: -1 no data, 0 land, 1 water)
         data/generated/eo/m4_eo_meta.csv      patch_id, chip_id, event, split, flood_frac, valid_frac, eo_label, ...
         data/generated/eo/m4_eo_stats.json    clip ranges + mean/std (TRAIN chips only)

Decisions (DATASET_CARD.md section 1.4):
  * each 512x512 chip -> four 256x256 patches -> 2x average-pooled to 128x128 (laptop-CPU friendly)
  * S1 (VV, VH, dB): NaN kept as missing; clipped to the TRAIN 0.5/99.5 percentiles; standardised
  * S2: 6 water-sensitive bands B2, B3, B4 (blue/green/red), B8 (NIR), B11, B12 (SWIR); 0 = no data;
    reflectance = DN/10000; standardised with TRAIN statistics
  * label -1 = no data (declared by us; the files do not declare it)
  * eo_label = 1 if water fraction of VALID pixels >= tau_flood = 0.05 (computed at full resolution)
  * split = official hand-labelled split CSVs (by event); Bolivia = unseen-event test set
  * missing pixels -> 0 after standardisation (= the train mean)

--synthetic N : builds a SYNTHETIC stand-in (fake terrain, water, radar, optical, clouds) with the
same format, ONLY to test the pipeline where Sen1Floods11 cannot be downloaded. Every output
is tagged dataset = "synthetic_eo_devcheck" and must never be reported as a result.

Run from the repository root:
    python members/member4/data/m4_eo_patches.py                 # real data
    python members/member4/data/m4_eo_patches.py --synthetic 160 # pipeline check only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MEMBER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MEMBER_DIR.parents[1]
DEFAULT_ROOT = REPO_ROOT / "data_external" / "sen1floods11"
OUT_DIR = MEMBER_DIR / "data" / "generated" / "eo"
S2_BANDS = {"B2": 1, "B3": 2, "B4": 3, "B8": 7, "B11": 11, "B12": 12}   # index in the 13-band S2Hand file
S2_IDX = list(S2_BANDS.values())
TAU_FLOOD = 0.05
PATCH = 256
DOWN = 2
REAL_NAME = "sen1floods11_handlabeled"
SYN_NAME = "synthetic_eo_devcheck"


# ------------------------------------------------------------------ helpers
def chip_split_map(splits_dir: Path) -> dict[str, str]:
    """chip_id -> split from the official CSVs (valid -> val)."""
    out = {}
    for name, label in [("train", "train"), ("valid", "val"), ("test", "test"), ("bolivia", "bolivia")]:
        f = splits_dir / f"flood_{name}_data.csv"
        if f.exists():
            for v in pd.read_csv(f, header=None)[0]:
                cid = str(v).strip().rsplit("_", 1)[0]
                out.setdefault(cid, label)        # bolivia chips also appear nowhere else
    return out


def pool(a: np.ndarray, k: int = DOWN) -> np.ndarray:
    """NaN-aware k x k average pooling of [C, H, W]."""
    c, h, w = a.shape
    r = a.reshape(c, h // k, k, w // k, k)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(r, axis=(2, 4))


def pool_label(lab: np.ndarray, k: int = DOWN) -> np.ndarray:
    """[H, W] in {-1,0,1} -> majority water over valid pixels; -1 if a block has no valid pixel."""
    h, w = lab.shape
    r = lab.reshape(h // k, k, w // k, k).astype(np.float32)
    valid = (r >= 0).sum(axis=(1, 3))
    water = (r == 1).sum(axis=(1, 3))
    out = np.where(valid == 0, -1, (water * 2 >= valid).astype(np.int8))
    return out.astype(np.int8)


def split_patches(arr: np.ndarray, size: int = PATCH):
    """[C, 512, 512] -> list of ((row, col), [C, size, size])."""
    _, h, w = arr.shape
    return [((r, c), arr[:, r:r + size, c:c + size]) for r in range(0, h - size + 1, size)
            for c in range(0, w - size + 1, size)]


# ------------------------------------------------------------------ real data
def read_chip(root: Path, cid: str):
    import rasterio
    with rasterio.open(root / "S1Hand" / f"{cid}_S1Hand.tif") as ds:
        s1 = ds.read().astype(np.float32)
    with rasterio.open(root / "S2Hand" / f"{cid}_S2Hand.tif") as ds:
        s2 = ds.read().astype(np.float32)[S2_IDX]
    with rasterio.open(root / "LabelHand" / f"{cid}_LabelHand.tif") as ds:
        lab = ds.read(1).astype(np.int16)
    s1[~np.isfinite(s1)] = np.nan
    s2[s2 <= 0] = np.nan
    s2 = s2 / 10000.0
    return s1, s2, lab


def compute_stats(root: Path, train_chips: list[str], rng: np.random.Generator, per_chip: int = 20000) -> dict:
    s1v, s2v = [], []
    for cid in train_chips:
        s1, s2, _ = read_chip(root, cid)
        idx = rng.choice(s1.shape[1] * s1.shape[2], size=per_chip, replace=False)
        s1v.append(s1.reshape(2, -1)[:, idx]); s2v.append(s2.reshape(len(S2_IDX), -1)[:, idx])
    s1v = np.concatenate(s1v, axis=1); s2v = np.concatenate(s2v, axis=1)
    lo1, hi1 = np.nanpercentile(s1v, 0.5, axis=1), np.nanpercentile(s1v, 99.5, axis=1)
    lo2, hi2 = np.nanpercentile(s2v, 0.5, axis=1), np.nanpercentile(s2v, 99.5, axis=1)
    s1c = np.clip(s1v, lo1[:, None], hi1[:, None]); s2c = np.clip(s2v, lo2[:, None], hi2[:, None])
    return {"s1_clip": [lo1.tolist(), hi1.tolist()], "s2_clip": [lo2.tolist(), hi2.tolist()],
            "s1_mean": np.nanmean(s1c, axis=1).tolist(), "s1_std": np.nanstd(s1c, axis=1).tolist(),
            "s2_mean": np.nanmean(s2c, axis=1).tolist(), "s2_std": np.nanstd(s2c, axis=1).tolist(),
            "s1_bands": ["VV", "VH"], "s2_bands": list(S2_BANDS), "n_train_chips": len(train_chips)}


def standardise(x: np.ndarray, lo, hi, mu, sd) -> np.ndarray:
    lo, hi, mu, sd = (np.asarray(v, np.float32)[:, None, None] for v in (lo, hi, mu, sd))
    z = (np.clip(x, lo, hi) - mu) / sd
    return np.nan_to_num(z, nan=0.0)


def build_real(root: Path, max_chips: int | None = None, seed: int = 42) -> tuple[dict, pd.DataFrame, dict]:
    hand = root / "HandLabeled"
    cids = sorted(p.name.rsplit("_", 1)[0] for p in (hand / "LabelHand").glob("*_LabelHand.tif"))
    if not cids:
        raise SystemExit(f"No chips in {hand}. Run: python members/member4/data/m4_download_eo_sample.py --event ALL")
    smap = chip_split_map(root / "splits")
    cids = [c for c in cids if c in smap][:max_chips] if max_chips else [c for c in cids if c in smap]
    train = [c for c in cids if smap[c] == "train"]
    if not train:
        raise SystemExit("No train chips found (need the full download + splits CSVs).")
    rng = np.random.default_rng(seed)
    stats = compute_stats(hand, train, rng)
    S1, S2, MASK, rows = [], [], [], []
    for cid in cids:
        s1, s2, lab = read_chip(hand, cid)
        for (r, c), p1 in split_patches(s1):
            p2 = s2[:, r:r + PATCH, c:c + PATCH]
            pl = lab[r:r + PATCH, c:c + PATCH]
            valid = pl >= 0
            ff = float((pl[valid] == 1).mean()) if valid.any() else np.nan
            S1.append(standardise(pool(p1), *stats["s1_clip"], stats["s1_mean"], stats["s1_std"]).astype(np.float16))
            S2.append(standardise(pool(p2), *stats["s2_clip"], stats["s2_mean"], stats["s2_std"]).astype(np.float16))
            MASK.append(pool_label(pl))
            rows.append({"chip_id": cid, "event": cid.split("_", 1)[0], "split": smap[cid], "row": r, "col": c,
                         "flood_frac": ff, "valid_frac": float(valid.mean()),
                         "s2_missing_frac": float(np.isnan(p2).any(axis=0).mean()),
                         "eo_label": int(ff >= TAU_FLOOD) if np.isfinite(ff) else -1})
    arrays = {"s1": np.stack(S1), "s2": np.stack(S2), "mask": np.stack(MASK)}
    meta = pd.DataFrame(rows)
    meta.insert(0, "patch_id", np.arange(len(meta)))
    meta["dataset"] = REAL_NAME
    return arrays, meta, stats


# ------------------------------------------------------------------ synthetic stand-in (pipeline check only)
def _smooth_field(rng, size, scale):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(rng.normal(size=(size, size)), scale)


def build_synthetic(n_chips: int = 160, seed: int = 7) -> tuple[dict, pd.DataFrame, dict]:
    """Fake 512x512 'chips' with water bodies, radar and optical signatures and some clouds.
    Events are assigned to splits like the real data (by event, never by patch)."""
    rng = np.random.default_rng(seed)
    events = [f"Synth{chr(65 + i)}" for i in range(8)]
    ev_split = {e: s for e, s in zip(events, ["train"] * 4 + ["val", "test", "test", "bolivia"])}
    S1, S2, MASK, rows = [], [], [], []
    for k in range(n_chips):
        ev = events[k % len(events)]
        size = 512
        wet = rng.uniform(-0.3, 1.2)                                   # how flooded this chip is
        field = _smooth_field(rng, size, 18)
        water = field > np.quantile(field, 1 - np.clip(rng.beta(1.2, 3.0) * wet, 0, 0.8))
        lab = water.astype(np.int16)
        nod = _smooth_field(rng, size, 30) > 2.2 * _smooth_field(rng, size, 30).std()
        lab[nod] = -1
        speck = rng.gamma(4.0, 0.25, size=(2, size, size))
        vv = np.where(water, -19.0, -8.0 + 2 * _smooth_field(rng, size, 6)) + 10 * np.log10(speck[0])
        vh = np.where(water, -26.0, -15.0 + 2 * _smooth_field(rng, size, 6)) + 10 * np.log10(speck[1])
        s1 = np.stack([vv, vh]).astype(np.float32)
        veg = np.clip(0.5 + 0.3 * _smooth_field(rng, size, 10), 0, 1)
        land = np.stack([0.05 + 0.02 * veg, 0.07 + 0.03 * veg, 0.06 + 0.02 * (1 - veg),
                         0.25 + 0.2 * veg, 0.20 + 0.05 * (1 - veg), 0.12 + 0.04 * (1 - veg)])
        wat = np.array([0.07, 0.06, 0.04, 0.02, 0.01, 0.005])[:, None, None] * np.ones((1, size, size))
        s2 = np.where(water[None], wat, land) + rng.normal(0, 0.005, (6, size, size))
        if rng.random() < 0.25:                                        # clouds hide the ground in optical only
            cloud = _smooth_field(rng, size, 25) > 0.3 * _smooth_field(rng, size, 25).std()
            s2 = np.where(cloud[None], 0.6 + rng.normal(0, 0.02, (6, size, size)), s2)
        s2 = s2.astype(np.float32)
        for (r, c), p1 in split_patches(s1):
            p2 = s2[:, r:r + PATCH, c:c + PATCH]; pl = lab[r:r + PATCH, c:c + PATCH]
            valid = pl >= 0
            ff = float((pl[valid] == 1).mean()) if valid.any() else np.nan
            S1.append(pool(p1)); S2.append(pool(p2)); MASK.append(pool_label(pl))
            rows.append({"chip_id": f"{ev}_{k:04d}", "event": ev, "split": ev_split[ev], "row": r, "col": c,
                         "flood_frac": ff, "valid_frac": float(valid.mean()), "s2_missing_frac": 0.0,
                         "eo_label": int(ff >= TAU_FLOOD) if np.isfinite(ff) else -1})
    meta = pd.DataFrame(rows)
    meta.insert(0, "patch_id", np.arange(len(meta)))
    meta["dataset"] = SYN_NAME
    S1, S2 = np.stack(S1), np.stack(S2)
    tr = (meta.split == "train").values
    stats = {"s1_clip": [np.percentile(S1[tr], 0.5, axis=(0, 2, 3)).tolist(), np.percentile(S1[tr], 99.5, axis=(0, 2, 3)).tolist()],
             "s2_clip": [np.percentile(S2[tr], 0.5, axis=(0, 2, 3)).tolist(), np.percentile(S2[tr], 99.5, axis=(0, 2, 3)).tolist()]}
    stats["s1_mean"] = S1[tr].mean(axis=(0, 2, 3)).tolist(); stats["s1_std"] = S1[tr].std(axis=(0, 2, 3)).tolist()
    stats["s2_mean"] = S2[tr].mean(axis=(0, 2, 3)).tolist(); stats["s2_std"] = S2[tr].std(axis=(0, 2, 3)).tolist()
    stats.update({"s1_bands": ["VV", "VH"], "s2_bands": list(S2_BANDS), "synthetic": True})
    S1 = np.stack([standardise(x, *stats["s1_clip"], stats["s1_mean"], stats["s1_std"]) for x in S1]).astype(np.float16)
    S2 = np.stack([standardise(x, *stats["s2_clip"], stats["s2_mean"], stats["s2_std"]) for x in S2]).astype(np.float16)
    return {"s1": S1, "s2": S2, "mask": np.stack(MASK)}, meta, stats


# ------------------------------------------------------------------ IO
def save(arrays: dict, meta: pd.DataFrame, stats: dict, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "m4_eo_patches.npz", **arrays)
    meta.to_csv(out_dir / "m4_eo_meta.csv", index=False)
    (out_dir / "m4_eo_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")


def load(out_dir: Path = OUT_DIR) -> tuple[dict, pd.DataFrame, dict]:
    f = out_dir / "m4_eo_patches.npz"
    if not f.exists():
        raise SystemExit(f"{f} missing. Run: python members/member4/data/m4_eo_patches.py")
    z = np.load(f)
    return ({k: z[k] for k in z.files}, pd.read_csv(out_dir / "m4_eo_meta.csv"),
            json.loads((out_dir / "m4_eo_stats.json").read_text(encoding="utf-8")))


def to_rgb(s2_std: np.ndarray, stats: dict) -> np.ndarray:
    """Standardised [6,H,W] -> displayable uint8 RGB (B4, B3, B2) with a 2-98 % stretch."""
    mu = np.asarray(stats["s2_mean"])[:, None, None]; sd = np.asarray(stats["s2_std"])[:, None, None]
    refl = s2_std.astype(np.float32) * sd + mu
    rgb = refl[[2, 1, 0]]
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return (np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Member 4: Sen1Floods11 chips -> 128x128 model-ready patches.")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--max-chips", type=int, default=None, help="limit chips (debugging)")
    ap.add_argument("--synthetic", type=int, default=0, help="N fake chips for a PIPELINE CHECK ONLY")
    args = ap.parse_args(argv)
    if args.synthetic:
        arrays, meta, stats = build_synthetic(args.synthetic)
        print("WARNING: synthetic stand-in built for pipeline checking. Not a result.")
    else:
        arrays, meta, stats = build_real(Path(args.root), args.max_chips)
    save(arrays, meta, stats)
    print(f"dataset: {meta.dataset.iloc[0]}   patches: {len(meta)}   shapes: "
          f"s1 {arrays['s1'].shape}, s2 {arrays['s2'].shape}")
    print(meta.groupby("split").agg(patches=("patch_id", "size"), flood=("eo_label", lambda s: int((s == 1).sum())),
                                     mean_flood_frac=("flood_frac", "mean")).round(3).to_string())
    print(f"saved -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
