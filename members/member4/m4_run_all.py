"""
m4_run_all.py - run the WHOLE Member 4 pipeline, Milestone 2 to 15, in order (fresh-clone reproduction).

Run from the repository root, with the virtual environment active:
    python members/member4/m4_run_all.py                     # everything, real Sen1Floods11 (downloads ~1.75 GB once)
    python members/member4/m4_run_all.py --from eo_patches   # resume from a stage
    python members/member4/m4_run_all.py --only fusion,report
    python members/member4/m4_run_all.py --only archive_v1,fusion,fusion_plots,report    # version 2 fusion
    python members/member4/m4_run_all.py --fast              # fewer epochs/seeds (quick check, ~25 min CPU)
    python members/member4/m4_run_all.py --synthetic-eo      # NO image data available: pipeline check only
Then:
    streamlit run members/member4/dashboard/m4_app.py

Every stage is an ordinary script you can also run on its own. The run stops at the first failure.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PY = sys.executable
EO_ROOT = REPO / "data_external" / "sen1floods11" / "HandLabeled" / "LabelHand"
OPSSAT = REPO / "data_external" / "opssat_ad" / "segments.csv"


def stages(a) -> list[tuple[str, list[str], str]]:
    m = "members/member4/"
    eo_epochs = ["--epochs", "5"] if a.fast else []
    fu = (["--seeds", "42", "--epochs", "20"] if a.fast else [])
    eo_cmd = [m + "data/m4_eo_patches.py"] + (["--synthetic", "160"] if a.synthetic_eo else [])
    return [
        ("generator", [m + "data/m4_telemetry_generator.py", "--plot"], "M2 telemetry generator v1 (90 days)"),
        ("splits", [m + "data/m4_make_splits.py"], "M3 chronological split"),
        ("eda", [m + "mlts/m4_ts_eda.py"], "M3 time-series EDA"),
        ("stationarity", [m + "mlts/m4_stationarity_decomposition.py"], "M3 STL / ADF / KPSS / ACF-PACF"),
        ("forecast", [m + "mlts/m4_forecast_eval.py"], "M4 forecasting ladder: naive -> LSTM"),
        ("forecast_plots", [m + "explainability/m4_plot_forecasts.py"], "M4 figures"),
        ("health", [m + "mlts/m4_residual_anomaly.py"], "M5 health anomaly detection"),
        ("opssat", [m + "mlts/m4_opssat_benchmark.py"], "M5 real telemetry (OPSSAT-AD)"),
        ("health_plots", [m + "explainability/m4_plot_health.py"], "M5 figures"),
        ("eo_download", [m + "data/m4_download_eo_sample.py", "--event", "ALL"], "M6 Sen1Floods11 download (once)"),
        ("eo_patches", eo_cmd, "M6 image patches"),
        ("eo_train", [m + "training/m4_train_eo.py"] + eo_epochs, "M6 optical-only / SAR-only / optical+SAR CNN"),
        ("reconstruction", [m + "mlts/m4_channel_reconstruction.py"], "M10 telemetry channel reconstruction"),
        ("passes", [m + "data/m4_build_passes.py"], "M7 virtual passes + payload-fault coupling"),
        ("archive_v1", [m + "evaluation/m4_archive_v1.py"], "keep version 1 fusion results before version 2"),
        ("fusion", [m + "training/m4_train_fusion.py"] + fu, "M8-M11 fusion, missing inputs, ablations, seeds"),
        ("fusion_plots", [m + "explainability/m4_plot_fusion.py"], "M11 figures"),
        ("report", [m + "evaluation/m4_final_report.py"], "M14 final conclusions from measured results"),
        ("tests", ["-m", "pytest", m + "tests", "-q"], "all tests"),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the full Member 4 pipeline.")
    ap.add_argument("--from", dest="start", default=None, help="first stage to run")
    ap.add_argument("--only", default=None, help="comma list of stages")
    ap.add_argument("--fast", action="store_true", help="fewer epochs / one seed")
    ap.add_argument("--synthetic-eo", action="store_true", help="no image data: synthetic stand-in (NOT results)")
    a = ap.parse_args(argv)
    st = stages(a)
    names = [s[0] for s in st]
    if a.only:
        keep = set(x.strip() for x in a.only.split(","))
        st = [s for s in st if s[0] in keep]
    elif a.start:
        if a.start not in names:
            raise SystemExit(f"unknown stage {a.start}; stages: {', '.join(names)}")
        st = st[names.index(a.start):]
    t_all = time.time()
    for name, cmd, desc in st:
        if name == "opssat" and not OPSSAT.exists():
            print(f"[skip] {name:<15} {desc}: {OPSSAT} not found (optional)"); continue
        if name == "eo_download" and (a.synthetic_eo or EO_ROOT.exists() and len(list(EO_ROOT.glob('*.tif'))) >= 446):
            print(f"[skip] {name:<15} {desc}: {'synthetic mode' if a.synthetic_eo else 'already downloaded'}"); continue
        print(f"\n[run ] {name:<15} {desc}", flush=True)
        t0 = time.time()
        r = subprocess.run([PY] + cmd, cwd=REPO)
        if r.returncode != 0:
            print(f"\n[FAIL] {name} (exit {r.returncode}). Fix it, then resume with:  python members/member4/m4_run_all.py --from {name}")
            return r.returncode
        print(f"[ ok ] {name:<15} {time.time() - t0:.0f}s", flush=True)
    print(f"\nAll stages done in {(time.time() - t_all) / 60:.1f} min. Dashboard:  streamlit run members/member4/dashboard/m4_app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
