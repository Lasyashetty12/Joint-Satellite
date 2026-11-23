"""
m4_archive_v1.py - keep the version 1 fusion results before a version 2 run overwrites them.

Copies (only if they are version 1 results and no archive exists yet):
    results/fusion_results.csv, m4_alpha_summary.csv, m4_fusion_test_predictions.csv,
    m4_fusion_train_info.json, m4_final_summary.json          -> results/v1/
    reports/FINAL_CONCLUSIONS.md                               -> results/v1/FINAL_CONCLUSIONS_v1.md
    checkpoints/m4_fusion_*.pt                                 -> checkpoints/v1/   (git-ignored)

The final report then shows version 1 and version 2 side by side, judged by the same decision rules.
Safe to run more than once: an existing archive is never overwritten.

Run from the repository root:
    python members/member4/evaluation/m4_archive_v1.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

MEMBER_DIR = Path(__file__).resolve().parents[1]
RESULTS, REPORTS, CKPT = MEMBER_DIR / "results", MEMBER_DIR / "reports", MEMBER_DIR / "checkpoints"
ARCHIVE = RESULTS / "v1"
FILES = ["fusion_results.csv", "m4_alpha_summary.csv", "m4_fusion_test_predictions.csv",
         "m4_fusion_train_info.json", "m4_final_summary.json"]


def is_v1(path: Path) -> bool:
    if not path.exists():
        return False
    cols = pd.read_csv(path, nrows=5)
    return "protocol" not in cols.columns or (cols["protocol"] != "v2").all()


def archive() -> str:
    fr = RESULTS / "fusion_results.csv"
    if (ARCHIVE / "fusion_results.csv").exists():
        return f"archive already exists: {ARCHIVE} (left unchanged)"
    if not is_v1(fr):
        return "nothing to archive: no version 1 fusion results found"
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    copied = []
    for f in FILES:
        if (RESULTS / f).exists():
            shutil.copy2(RESULTS / f, ARCHIVE / f); copied.append(f)
    if (REPORTS / "FINAL_CONCLUSIONS.md").exists():
        shutil.copy2(REPORTS / "FINAL_CONCLUSIONS.md", ARCHIVE / "FINAL_CONCLUSIONS_v1.md"); copied.append("FINAL_CONCLUSIONS.md")
    ck = sorted(CKPT.glob("m4_fusion_*.pt"))
    if ck:
        (CKPT / "v1").mkdir(parents=True, exist_ok=True)
        for f in ck:
            shutil.copy2(f, CKPT / "v1" / f.name)
    return f"archived {len(copied)} result files and {len(ck)} checkpoints -> {ARCHIVE}"


def main() -> int:
    print(archive())
    return 0


if __name__ == "__main__":
    sys.exit(main())
