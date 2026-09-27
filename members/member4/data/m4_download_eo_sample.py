"""
m4_download_eo_sample.py - list and download a SMALL Sen1Floods11 hand-labelled sample.

No Google Cloud CLI needed: the bucket is public, so plain HTTPS works.

Run from the repository root:
    # 1) Look first: counts and total size of each hand-labelled folder (no download)
    python members/member4/data/m4_download_eo_sample.py --list-only

    # 2) Tiny sample: one event (Bolivia = 15 chips) + the official split CSVs
    python members/member4/data/m4_download_eo_sample.py --event Bolivia

    # 3) Later (benchmark v1): all hand-labelled chips
    python members/member4/data/m4_download_eo_sample.py --event ALL

Files are saved OUTSIDE your member folder, in <repo>/data_external/sen1floods11/
so they can never be committed by `git add members/member4/`.
Already-downloaded files are skipped, so you can safely re-run after an interruption.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "sen1floods11"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
MEDIA = f"https://storage.googleapis.com/{BUCKET}/"
HAND = "v1.1/data/flood_events/HandLabeled/"
LAYERS = {"S1Hand": "S1Hand", "S2Hand": "S2Hand", "LabelHand": "LabelHand"}
SPLITS_PREFIX = "v1.1/splits/flood_handlabeled/"

MEMBER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MEMBER_DIR.parents[1]
DEFAULT_OUT = REPO_ROOT / "data_external" / "sen1floods11"


def list_url(prefix: str, page_token: str | None = None) -> str:
    q = {"prefix": prefix, "fields": "items(name,size),nextPageToken", "maxResults": "1000"}
    if page_token:
        q["pageToken"] = page_token
    return API + "?" + urllib.parse.urlencode(q)


def _get(url: str, retries: int = 3, timeout: int = 60) -> bytes:
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                raise RuntimeError(f"GET failed after {retries} tries: {url}\n{exc}") from exc
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def list_objects(prefix: str, fetch=_get) -> list[dict]:
    """All objects under a prefix: [{'name':..., 'size': int}, ...]."""
    items, token = [], None
    while True:
        data = json.loads(fetch(list_url(prefix, token)))
        items += [{"name": it["name"], "size": int(it["size"])} for it in data.get("items", [])]
        token = data.get("nextPageToken")
        if not token:
            return items


def chip_id(filename: str) -> str:
    """'Bolivia_103757_S2Hand.tif' -> 'Bolivia_103757'."""
    base = filename.rsplit("/", 1)[-1]
    return base.rsplit("_", 1)[0]


def event_of(filename: str) -> str:
    return filename.rsplit("/", 1)[-1].split("_", 1)[0]


def summarize(listing: dict[str, list[dict]]) -> dict:
    out = {}
    for layer, items in listing.items():
        events: dict[str, int] = {}
        for it in items:
            events[event_of(it["name"])] = events.get(event_of(it["name"]), 0) + 1
        out[layer] = {"files": len(items), "total_MB": round(sum(i["size"] for i in items) / 1e6, 1),
                      "per_event": dict(sorted(events.items()))}
    return out


def select_chips(listing: dict[str, list[dict]], event: str, max_chips: int | None) -> list[str]:
    """Chip IDs present in ALL three layers (so optical, SAR and label are aligned)."""
    sets = [{chip_id(i["name"]) for i in items} for items in listing.values()]
    common = sorted(set.intersection(*sets))
    if event.upper() != "ALL":
        common = [c for c in common if c.split("_", 1)[0].lower() == event.lower()]
    return common[:max_chips] if max_chips else common


def download(name: str, dest: Path, fetch=_get) -> bool:
    """Returns True if downloaded, False if it already existed."""
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(fetch(MEDIA + urllib.parse.quote(name)))
    tmp.replace(dest)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Member 4: list/download a small Sen1Floods11 hand-labelled sample.")
    ap.add_argument("--list-only", action="store_true", help="only print counts and sizes, download nothing")
    ap.add_argument("--event", default="Bolivia", help="flood event name (e.g. Bolivia, Ghana) or ALL")
    ap.add_argument("--max-chips", type=int, default=None, help="limit number of chips")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output folder (keep it outside members/)")
    args = ap.parse_args(argv)

    print("Listing hand-labelled folders ...")
    listing = {layer: list_objects(HAND + folder + "/") for layer, folder in LAYERS.items()}
    summary = summarize(listing)
    print(json.dumps(summary, indent=2))
    if args.list_only:
        return 0

    out = Path(args.out)
    chips = select_chips(listing, args.event, args.max_chips)
    if not chips:
        print(f"No chips found for event '{args.event}'. Events available: "
              f"{list(summary['LabelHand']['per_event'])}")
        return 1
    sizes = {i["name"]: i["size"] for items in listing.values() for i in items}
    need = [(HAND + f"{layer}/{c}_{layer}.tif", out / "HandLabeled" / layer / f"{c}_{layer}.tif")
            for c in chips for layer in LAYERS]
    total_mb = sum(sizes.get(n, 0) for n, _ in need) / 1e6
    print(f"\nDownloading {len(chips)} chips x 3 layers = {len(need)} files (~{total_mb:.1f} MB) to {out}")

    new = 0
    for k, (name, dest) in enumerate(need, 1):
        new += download(name, dest)
        if k % 15 == 0 or k == len(need):
            print(f"  {k}/{len(need)} files")
    for it in list_objects(SPLITS_PREFIX):
        new += download(it["name"], out / "splits" / it["name"].rsplit("/", 1)[-1])
    (out / "m4_download_manifest.json").write_text(json.dumps(
        {"event": args.event, "chips": chips, "n_files": len(need), "approx_MB": round(total_mb, 1),
         "source": f"gs://{BUCKET}/{HAND}", "summary": summary}, indent=2))
    print(f"Done. {new} new files downloaded (existing files skipped). Manifest: {out / 'm4_download_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
