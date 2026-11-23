"""
m4_missing_modality.py - simulate missing and degraded inputs (Milestone 7, handbook section 5).

Image degradations (applied to STANDARDISED optical patches [6, H, W]):
    cloud       smooth bright blob(s) covering `severity` of the patch (all bands bright, ground hidden)
    stripes     detector-line failure: random columns stuck high/low      (payload fault signature)
    noise       strong sensor noise                                       (payload fault signature)
    saturation  part of the image clipped to the sensor maximum           (payload fault signature)

Modality masks m_tel, m_opt, m_sar in {0, 1}: 0 = the input is missing for that pass.
At least one modality is always present.

Telemetry channel dropout: k channels of a pass window set to NaN (sensor or downlink loss).

Everything is driven by an explicit numpy Generator, so every scenario is reproducible from its seed.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

PAYLOAD_KINDS = ("stripes", "noise", "saturation")
MODALITIES = ("tel", "opt", "sar")


def _blob(rng: np.random.Generator, h: int, w: int, frac: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    f = gaussian_filter(rng.normal(size=(h, w)), sigma=max(h, w) / 10)
    return f >= np.quantile(f, 1 - frac)


def degrade_optical(x: np.ndarray, kind: str, severity: float, rng: np.random.Generator) -> np.ndarray:
    """x: standardised optical patch [C, H, W]; severity in [0, 1]. Returns a new array."""
    y = np.array(x, dtype=np.float32, copy=True)
    if kind == "none" or severity <= 0:
        return y
    c, h, w = y.shape
    if kind == "cloud":
        m = _blob(rng, h, w, float(np.clip(severity, 0, 1)))
        y[:, m] = 3.0 + rng.normal(0, 0.15, (c, int(m.sum())))           # bright in every band
    elif kind == "stripes":
        n = max(1, int(severity * w * 0.5))
        cols = rng.choice(w, size=n, replace=False)
        y[:, :, cols] = rng.choice([-3.0, 4.0], size=(1, 1, n))
    elif kind == "noise":
        y += rng.normal(0, 2.5 * severity, y.shape).astype(np.float32)
    elif kind == "saturation":
        m = _blob(rng, h, w, float(np.clip(0.6 * severity, 0, 1)))
        y[:, m] = 4.0
    else:
        raise ValueError(kind)
    return y


def sample_masks(n: int, p_missing: dict[str, float], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Independent missingness per modality, then repair rows where all three are missing."""
    m = {k: (rng.random(n) >= p_missing.get(k, 0.0)).astype(np.int8) for k in MODALITIES}
    none = (m["tel"] + m["opt"] + m["sar"]) == 0
    if none.any():
        keep = rng.choice(MODALITIES, size=int(none.sum()))
        for i, k in zip(np.where(none)[0], keep):
            m[k][i] = 1
    return m


def drop_channels(window: np.ndarray, k: int, rng: np.random.Generator) -> tuple[np.ndarray, list[int]]:
    """window [T, C] -> copy with k random channels set to NaN; returns (window, dropped indices)."""
    w = np.array(window, dtype=np.float32, copy=True)
    idx = sorted(rng.choice(w.shape[1], size=min(k, w.shape[1]), replace=False).tolist()) if k > 0 else []
    w[:, idx] = np.nan
    return w, idx


# ------------------------------------------------------------------ image quality check (version 2)
# A simple screen in the spirit of standard Sentinel-2 cloud / defect masking: it does not know which
# damage was simulated; every threshold is fitted on CLEAN training patches only.
QUALITY_NAMES = ("white_cover", "bright_cover", "dark_cover", "flat_columns", "noise")
NQ = len(QUALITY_NAMES)


def _hf(x: np.ndarray) -> float:
    """Mean high-frequency energy: |x - 3x3 box blur|, averaged over bands and pixels."""
    from scipy.ndimage import uniform_filter
    return float(np.mean(np.abs(x - uniform_filter(x, size=(1, 3, 3), mode="nearest"))))


def fit_quality_reference(clean: np.ndarray) -> dict:
    """clean: [N, C, H, W] undamaged TRAIN patches -> thresholds for the quality check (all from clean data)."""
    x = np.nan_to_num(np.asarray(clean, np.float32))
    bmean = x.mean(axis=1).ravel()                                      # per pixel, mean over bands
    bsd = x.std(axis=1).ravel()                                         # per pixel, spread over bands
    col_sd = x.std(axis=2).mean(axis=1).ravel()                         # per patch and column, mean over bands
    return {"white_sd": float(np.percentile(bsd, 2.5)), "white_mean": float(np.percentile(bmean, 75)),
            "hi": float(np.percentile(bmean, 97.5)), "lo": float(np.percentile(bmean, 2.5)),
            "flat_col_sd": float(0.5 * np.percentile(col_sd, 1)),
            "hf_median": float(np.median([_hf(p) for p in x])), "n_patches": int(len(x))}


def quality_features(x: np.ndarray, ref: dict) -> np.ndarray:
    """x: optical patch [C, H, W] (standardised) -> 5 numbers in [0, 1]:
    white_cover   pixels that look the same in every band and are brighter than most ground (cloud-like)
    bright_cover  pixels brighter, averaged over bands, than 97.5 % of clean training pixels
    dark_cover    pixels darker than 97.5 % of clean training pixels
    flat_columns  image columns with almost no variation (stuck detector lines)
    noise         excess pixel-to-pixel noise relative to clean training patches (log scale)"""
    x = np.nan_to_num(np.asarray(x, np.float32))
    bmean, bsd = x.mean(axis=0), x.std(axis=0)
    white = float(((bsd < ref["white_sd"]) & (bmean > ref["white_mean"])).mean())
    bright = float((bmean > ref["hi"]).mean())
    dark = float((bmean < ref["lo"]).mean())
    flat = float((x.std(axis=1).mean(axis=0) < ref["flat_col_sd"]).mean())
    noise = float(np.clip(np.log(max(_hf(x), 1e-6) / ref["hf_median"]) / 3.0, 0.0, 1.0))
    return np.array([white, bright, dark, flat, noise], np.float32)


def quality_reference(s2: np.ndarray, passes, path) -> dict:
    """Fit once on the clean TRAIN patches used by the passes (no damage simulated), cache as JSON."""
    import json
    from pathlib import Path
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    ids = np.unique(passes.patch_id.values[(passes.split.values == "train") & (passes.opt_degrade.values == "none")])
    ids = np.random.default_rng(0).choice(ids, size=min(400, len(ids)), replace=False)
    ref = fit_quality_reference(s2[np.sort(ids)].astype(np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ref, indent=1), encoding="utf-8")
    return ref


# Evaluation scenarios used by m4_experiments.py (Milestone 10)
SCENARIOS = {
    "all_present": {"masks": (1, 1, 1)},
    "missing_optical": {"masks": (1, 0, 1)},
    "missing_sar": {"masks": (1, 1, 0)},
    "missing_telemetry": {"masks": (0, 1, 1)},
    "only_telemetry": {"masks": (1, 0, 0)},
    "only_optical": {"masks": (0, 1, 0)},
    "only_sar": {"masks": (0, 0, 1)},
    "cloud_25": {"masks": (1, 1, 1), "cloud": 0.25},
    "cloud_50": {"masks": (1, 1, 1), "cloud": 0.50},
    "cloud_75": {"masks": (1, 1, 1), "cloud": 0.75},
    "cloud_100": {"masks": (1, 1, 1), "cloud": 1.00},
}


if __name__ == "__main__":
    argparse.ArgumentParser(description="Missing/degraded-modality simulator (library).").parse_args()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 128, 128)).astype(np.float32)
    for k in ("cloud", "stripes", "noise", "saturation"):
        y = degrade_optical(x, k, 0.5, rng)
        print(f"{k:<11} changed {np.mean(np.abs(y - x) > 1e-6):.2%} of values")
    sys.exit(0)
