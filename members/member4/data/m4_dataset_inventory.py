"""
m4_dataset_inventory.py - inspect downloaded datasets BEFORE using them (dataset selection gate).

Nothing is assumed: every property (shape, bands, dtype, no-data, label values,
alignment, flood fraction, CSV columns, label counts) is measured from the files.

Run from the repository root:
    python members/member4/data/m4_dataset_inventory.py sen1floods11 --plot
    python members/member4/data/m4_dataset_inventory.py opssat
    python members/member4/data/m4_dataset_inventory.py --help

Default input folders:  <repo>/data_external/sen1floods11/HandLabeled   and   <repo>/data_external/opssat_ad
Outputs (small, safe to commit) go to members/member4/reports/:
    m4_eo_inventory.csv, m4_eo_inventory_summary.json, figures/m4_eo_sample_<chip>.png
    m4_opssat_inventory.json
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
REPORTS = MEMBER_DIR / "reports"
DEFAULT_EO = REPO_ROOT / "data_external" / "sen1floods11" / "HandLabeled"
DEFAULT_OPSSAT = REPO_ROOT / "data_external" / "opssat_ad"
TAU_FLOOD = 0.05
LAYERS = ["S1Hand", "S2Hand", "LabelHand"]


# ============================================================================ EO (Sen1Floods11)
def flood_fraction(label: np.ndarray, nodata_value: int = -1) -> tuple[float, float]:
    """Returns (flood_frac over VALID pixels, valid_frac). Label: -1 no data, 0 not water, 1 water."""
    valid = label != nodata_value
    n_valid = int(valid.sum())
    if n_valid == 0:
        return float("nan"), 0.0
    return float((label[valid] == 1).sum() / n_valid), float(n_valid / label.size)


def find_chips(root: Path) -> list[str]:
    ids = [p.name.rsplit("_", 1)[0] for p in (root / "LabelHand").glob("*_LabelHand.tif")]
    return sorted(ids)


def inspect_chip(root: Path, cid: str, tau: float = TAU_FLOOD) -> dict:
    import rasterio

    row: dict = {"chip_id": cid, "event": cid.split("_", 1)[0]}
    meta = {}
    arrays = {}
    for layer in LAYERS:
        path = root / layer / f"{cid}_{layer}.tif"
        if not path.exists():
            row[f"{layer}_missing"] = True
            continue
        with rasterio.open(path) as ds:
            a = ds.read()
            meta[layer] = (ds.crs.to_string() if ds.crs else None, tuple(ds.transform)[:6], ds.width, ds.height)
            arrays[layer] = a
            row[f"{layer}_bands"] = ds.count
            row[f"{layer}_shape"] = f"{ds.height}x{ds.width}"
            row[f"{layer}_dtype"] = str(a.dtype)
            row[f"{layer}_nodata"] = ds.nodata
            finite = a[np.isfinite(a)] if np.issubdtype(a.dtype, np.floating) else a.ravel()
            row[f"{layer}_min"] = float(finite.min()) if finite.size else float("nan")
            row[f"{layer}_max"] = float(finite.max()) if finite.size else float("nan")
            row[f"{layer}_nan_frac"] = float(np.isnan(a).mean()) if np.issubdtype(a.dtype, np.floating) else 0.0
    row["all_layers_present"] = all(l in arrays for l in LAYERS)
    row["aligned"] = row["all_layers_present"] and len(set(meta.values())) == 1
    if "LabelHand" in arrays:
        lab = arrays["LabelHand"][0]
        vals, cnts = np.unique(lab, return_counts=True)
        row["label_values"] = ";".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, cnts))
        ff, vf = flood_fraction(lab)
        row["flood_frac"] = round(ff, 4) if not np.isnan(ff) else np.nan
        row["valid_frac"] = round(vf, 4)
        row["eo_label"] = int(ff >= tau) if not np.isnan(ff) else -1
    return row


def attach_official_split(df: pd.DataFrame, splits_dir: Path) -> pd.DataFrame:
    """Uses flood_{train,valid,test,bolivia}_data.csv if they were downloaded."""
    df = df.copy()
    df["official_split"] = ""
    for split in ["train", "valid", "test", "bolivia"]:
        f = splits_dir / f"flood_{split}_data.csv"
        if not f.exists():
            continue
        ids = {chip_id_from_split(v) for v in pd.read_csv(f, header=None)[0]}
        mask = df["chip_id"].isin(ids) & (df["official_split"] == "")
        df.loc[mask, "official_split"] = split
    return df


def chip_id_from_split(value: str) -> str:
    """'Bolivia_103757_S1Hand.tif' -> 'Bolivia_103757'."""
    return str(value).strip().rsplit("_", 1)[0]


def plot_chip(root: Path, cid: str, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio

    with rasterio.open(root / "S2Hand" / f"{cid}_S2Hand.tif") as ds:
        s2 = ds.read().astype(float)
    with rasterio.open(root / "S1Hand" / f"{cid}_S1Hand.tif") as ds:
        s1 = ds.read().astype(float)
    with rasterio.open(root / "LabelHand" / f"{cid}_LabelHand.tif") as ds:
        lab = ds.read(1)

    def stretch(x):
        lo, hi = np.nanpercentile(x, [2, 98])
        return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)

    # Sentinel-2 true colour = B4, B3, B2 -> band indices 3, 2, 1 (verify band order in DATASET_CARD)
    rgb = np.dstack([stretch(s2[i]) for i in (3, 2, 1)]) if s2.shape[0] >= 4 else stretch(s2[0])
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    ax[0].imshow(rgb); ax[0].set_title("S2 optical (B4,B3,B2)")
    ax[1].imshow(stretch(s1[0]), cmap="gray"); ax[1].set_title("S1 VV")
    ax[2].imshow(stretch(s1[1]) if s1.shape[0] > 1 else stretch(s1[0]), cmap="gray"); ax[2].set_title("S1 VH")
    cmap = matplotlib.colors.ListedColormap(["#bbbbbb", "#f2e6c9", "#1f77b4"])
    ax[3].imshow(lab, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    ax[3].set_title("Label: grey=-1 no data, beige=0, blue=1 water")
    for a in ax:
        a.axis("off")
    fig.suptitle(cid)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def run_eo(root: Path, tau: float, plot: bool) -> dict:
    chips = find_chips(root)
    if not chips:
        raise SystemExit(f"No *_LabelHand.tif files under {root / 'LabelHand'}. "
                         f"Run m4_download_eo_sample.py first.")
    df = pd.DataFrame([inspect_chip(root, c, tau) for c in chips])
    df = attach_official_split(df, root.parent / "splits")
    REPORTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS / "m4_eo_inventory.csv", index=False)

    summary = {
        "n_chips": int(len(df)),
        "events": df["event"].value_counts().to_dict(),
        "all_layers_present": int(df["all_layers_present"].sum()),
        "aligned": int(df["aligned"].sum()),
        "S1_bands": sorted(df["S1Hand_bands"].dropna().unique().tolist()),
        "S2_bands": sorted(df["S2Hand_bands"].dropna().unique().tolist()),
        "shapes": sorted(set(df["S1Hand_shape"].dropna()) | set(df["S2Hand_shape"].dropna())),
        "dtypes": {l: sorted(df[f"{l}_dtype"].dropna().unique().tolist()) for l in LAYERS},
        "nodata": {l: sorted(map(str, df[f"{l}_nodata"].unique().tolist())) for l in LAYERS},
        "S1_range": [float(df["S1Hand_min"].min()), float(df["S1Hand_max"].max())],
        "S2_range": [float(df["S2Hand_min"].min()), float(df["S2Hand_max"].max())],
        "label_values_seen": sorted({int(p.split(":")[0]) for s in df["label_values"].dropna() for p in s.split(";")}),
        "tau_flood": tau,
        "eo_label_counts": df["eo_label"].value_counts().to_dict(),
        "flood_frac_mean": round(float(df["flood_frac"].mean()), 4),
        "official_split_counts": df["official_split"].value_counts().to_dict(),
    }
    (REPORTS / "m4_eo_inventory_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if plot:
        # plot the chip with the most flood pixels, a good visual check
        best = df.sort_values("flood_frac", ascending=False).iloc[0]["chip_id"]
        summary["figure"] = str(plot_chip(root, best, REPORTS / "figures" / f"m4_eo_sample_{best}.png"))
    return summary


# ============================================================================ OPSSAT-AD (or any CSV benchmark)
LABEL_HINTS = ("anomal", "label", "train", "channel", "segment", "sampling")


def inspect_csv(path: Path, max_unique: int = 20) -> dict:
    df = pd.read_csv(path)
    info = {
        "file": path.name,
        "size_MB": round(path.stat().st_size / 1e6, 2),
        "rows": int(len(df)),
        "columns": {c: str(t) for c, t in df.dtypes.items()},
        "nan_counts": {c: int(v) for c, v in df.isna().sum().items() if v},
        "head": df.head(3).astype(str).to_dict(orient="records"),
        "value_counts": {},
    }
    for c in df.columns:
        if any(h in c.lower() for h in LABEL_HINTS) and df[c].nunique() <= max_unique:
            info["value_counts"][c] = {str(k): int(v) for k, v in df[c].value_counts().items()}
    time_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    for c in time_cols[:1]:
        ts = pd.to_datetime(df[c], errors="coerce", utc=True)
        if ts.notna().any():
            info["time_range"] = [str(ts.min()), str(ts.max())]
            d = ts.dropna().sort_values().diff().dropna()
            if len(d):
                info["median_cadence_s"] = float(d.dt.total_seconds().median())
    return info


def run_opssat(root: Path) -> dict:
    files = sorted(root.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files in {root}. Download dataset.csv and segments.csv from "
                         f"https://zenodo.org/records/12588359 into that folder.")
    result = {"source": "https://zenodo.org/records/12588359", "files": [inspect_csv(f) for f in files]}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "m4_opssat_inventory.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Member 4 dataset inventory (measure, never assume).")
    sub = ap.add_subparsers(dest="dataset", required=True)
    e = sub.add_parser("sen1floods11", help="inspect Sen1Floods11 HandLabeled chips")
    e.add_argument("--root", default=str(DEFAULT_EO))
    e.add_argument("--tau", type=float, default=TAU_FLOOD, help="flood fraction threshold for eo_label")
    e.add_argument("--plot", action="store_true")
    o = sub.add_parser("opssat", help="inspect OPSSAT-AD CSV files")
    o.add_argument("--root", default=str(DEFAULT_OPSSAT))
    args = ap.parse_args(argv)

    if args.dataset == "sen1floods11":
        res = run_eo(Path(args.root), args.tau, args.plot)
        print(json.dumps(res, indent=2, default=str))
        print(f"\nPer-chip table: {REPORTS / 'm4_eo_inventory.csv'}")
    else:
        res = run_opssat(Path(args.root))
        for f in res["files"]:
            print(f"\n=== {f['file']} ({f['size_MB']} MB, {f['rows']} rows) ===")
            print("columns:", f["columns"])
            print("label-like value counts:", json.dumps(f["value_counts"], indent=1))
            for k in ("time_range", "median_cadence_s", "nan_counts"):
                if k in f:
                    print(f"{k}: {f[k]}")
        print(f"\nSaved: {REPORTS / 'm4_opssat_inventory.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
