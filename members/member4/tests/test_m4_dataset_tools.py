"""Offline tests for the dataset download + inventory tools (no internet needed).
They build tiny fake GeoTIFFs / CSVs that mimic the real formats.

Run from the repository root:
    python -m pytest members/member4/tests -q
"""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

import m4_dataset_inventory as inv  # noqa: E402
import m4_download_eo_sample as dl  # noqa: E402


# ------------------------------------------------------------------ helpers
def write_tif(path, arr, transform, nodata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                       count=arr.shape[0], dtype=arr.dtype, crs="EPSG:4326",
                       transform=transform, nodata=nodata) as ds:
        ds.write(arr)


def make_chip(root, cid, flood_frac, size=32, shift=False):
    tf = from_origin(10.0, 20.0, 0.0001, 0.0001)
    tf_lab = from_origin(10.5, 20.0, 0.0001, 0.0001) if shift else tf
    rng = np.random.default_rng(0)
    s1 = rng.normal(-15, 3, (2, size, size)).astype("float32")
    s1[0, 0, 0] = np.nan
    s2 = rng.integers(0, 4000, (13, size, size)).astype("float32")
    lab = np.zeros((1, size, size), dtype="int16")
    lab[0, :2, :] = -1                                     # 2 no-data rows
    n_valid = (size - 2) * size
    n_flood = int(round(flood_frac * n_valid))
    flat = lab[0, 2:, :].ravel()
    flat[:n_flood] = 1
    lab[0, 2:, :] = flat.reshape(size - 2, size)
    write_tif(root / "S1Hand" / f"{cid}_S1Hand.tif", s1, tf)
    write_tif(root / "S2Hand" / f"{cid}_S2Hand.tif", s2, tf)
    write_tif(root / "LabelHand" / f"{cid}_LabelHand.tif", lab, tf_lab, nodata=-1)


@pytest.fixture
def eo_root(tmp_path, monkeypatch):
    root = tmp_path / "sen1floods11" / "HandLabeled"
    make_chip(root, "Bolivia_1", 0.30)
    make_chip(root, "Bolivia_2", 0.01)
    make_chip(root, "Ghana_3", 0.10, shift=True)          # misaligned on purpose
    splits = tmp_path / "sen1floods11" / "splits"
    splits.mkdir(parents=True)
    pd.DataFrame([["Bolivia_1_S1Hand.tif", "Bolivia_1_LabelHand.tif"],
                  ["Bolivia_2_S1Hand.tif", "Bolivia_2_LabelHand.tif"]]).to_csv(
        splits / "flood_bolivia_data.csv", header=False, index=False)
    monkeypatch.setattr(inv, "REPORTS", tmp_path / "reports")
    return root


# ------------------------------------------------------------------ flood fraction
def test_flood_fraction_ignores_nodata():
    lab = np.array([[-1, -1], [1, 0]])
    ff, vf = inv.flood_fraction(lab)
    assert ff == 0.5 and vf == 0.5


def test_flood_fraction_all_nodata():
    ff, vf = inv.flood_fraction(np.full((3, 3), -1))
    assert np.isnan(ff) and vf == 0.0


# ------------------------------------------------------------------ EO inventory
def test_eo_inventory_measures_properties(eo_root):
    s = inv.run_eo(eo_root, tau=0.05, plot=True)
    assert s["n_chips"] == 3
    assert s["S1_bands"] == [2] and s["S2_bands"] == [13]
    assert s["label_values_seen"] == [-1, 0, 1]
    assert s["aligned"] == 2                               # Ghana_3 is shifted
    assert s["eo_label_counts"] == {1: 2, 0: 1}            # 0.30, 0.10 >= 0.05 ; 0.01 < 0.05
    assert s["official_split_counts"].get("bolivia") == 2
    assert (inv.REPORTS / "m4_eo_inventory.csv").exists()
    assert (inv.REPORTS / "figures").exists()


def test_eo_inventory_nan_reported(eo_root):
    inv.run_eo(eo_root, tau=0.05, plot=False)
    df = pd.read_csv(inv.REPORTS / "m4_eo_inventory.csv")
    assert (df["S1Hand_nan_frac"] > 0).all()


def test_split_chip_id_parsing():
    assert inv.chip_id_from_split("Bolivia_103757_S1Hand.tif") == "Bolivia_103757"


# ------------------------------------------------------------------ OPSSAT-like CSV inventory
def test_opssat_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(inv, "REPORTS", tmp_path / "reports")
    root = tmp_path / "opssat"
    root.mkdir()
    pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=6, freq="5s").astype(str),
                  "channel": ["CADC0872"] * 6, "value": [1, 2, 3, 4, np.nan, 6],
                  "anomaly": [0, 0, 1, 0, 0, 0], "train": [1, 1, 1, 0, 0, 0]}).to_csv(root / "segments.csv", index=False)
    res = inv.run_opssat(root)
    f = res["files"][0]
    assert f["rows"] == 6
    assert f["value_counts"]["anomaly"] == {"0": 5, "1": 1}
    assert f["median_cadence_s"] == 5.0
    assert f["nan_counts"] == {"value": 1}


def test_cadence_measured_per_channel(tmp_path, monkeypatch):
    """Two channels sampled at the SAME instants every 5 s: cadence must be 5, not 0."""
    monkeypatch.setattr(inv, "REPORTS", tmp_path / "reports")
    root = tmp_path / "opssat2"
    root.mkdir()
    t = pd.date_range("2022-01-01", periods=4, freq="5s").astype(str).tolist()
    pd.DataFrame({"channel": ["A"] * 4 + ["B"] * 4, "timestamp": t + t, "value": range(8),
                  "segment": [1] * 4 + [2] * 4, "sampling": [5] * 8}).to_csv(root / "segments.csv", index=False)
    f = inv.run_opssat(root)["files"][0]
    assert f["median_cadence_s"] == 5.0
    assert f["median_cadence_s_by_sampling"] == {"5": 5.0}


# ------------------------------------------------------------------ downloader (mocked network)
FAKE = {
    "S1Hand": ["Bolivia_1_S1Hand.tif", "Bolivia_2_S1Hand.tif", "Ghana_3_S1Hand.tif"],
    "S2Hand": ["Bolivia_1_S2Hand.tif", "Bolivia_2_S2Hand.tif", "Ghana_3_S2Hand.tif"],
    "LabelHand": ["Bolivia_1_LabelHand.tif", "Ghana_3_LabelHand.tif"],   # Bolivia_2 label missing
}


def fake_fetch(url):
    if "/storage/v1/" in url:
        for layer, names in FAKE.items():
            if f"%2F{layer}%2F" in url or f"/{layer}/" in url:
                return json.dumps({"items": [{"name": dl.HAND + f"{layer}/{n}", "size": "1000000"}
                                             for n in names]}).encode()
        return json.dumps({"items": [{"name": dl.SPLITS_PREFIX + "flood_bolivia_data.csv", "size": "834"}]}).encode()
    return b"fake-bytes"


def test_listing_and_selection():
    listing = {l: dl.list_objects(dl.HAND + l + "/", fetch=fake_fetch) for l in dl.LAYERS}
    s = dl.summarize(listing)
    assert s["S1Hand"]["files"] == 3 and s["S1Hand"]["total_MB"] == 3.0
    assert s["LabelHand"]["per_event"] == {"Bolivia": 1, "Ghana": 1}
    assert dl.select_chips(listing, "Bolivia", None) == ["Bolivia_1"]     # only chips with all 3 layers
    assert dl.select_chips(listing, "ALL", None) == ["Bolivia_1", "Ghana_3"]


def test_download_skips_existing(tmp_path):
    dest = tmp_path / "a.tif"
    assert dl.download("x/a.tif", dest, fetch=fake_fetch) is True
    assert dl.download("x/a.tif", dest, fetch=fake_fetch) is False
    assert dest.read_bytes() == b"fake-bytes"


def test_default_output_is_outside_member_folder():
    assert "members" not in dl.DEFAULT_OUT.relative_to(dl.REPO_ROOT).parts


# ------------------------------------------------------------------ CLI smoke
@pytest.mark.parametrize("script", [inv.__file__, dl.__file__])
def test_cli_help(script):
    out = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0
