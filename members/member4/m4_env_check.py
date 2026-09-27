"""
m4_env_check.py - Phase 1 / Milestone 1 smoke test for Member 4.

What it does (in about 30 seconds, CPU only):
  1. Prints Python, OS and Git versions (evidence for the milestone).
  2. Checks that every required library can be imported and prints its version.
  3. Runs a tiny PyTorch forward pass (the kind of code our CNN/GRU encoders use).
  4. Fits a tiny ARIMA model with statsmodels on a synthetic "orbit" sine wave
     (the kind of code our MLTS forecasters use).
  5. Fits a tiny XGBoost regressor on lag features.
  6. Optionally writes a JSON report to members/member4/reports/m4_env_report.json

Run from the repository root:
    python members/member4/m4_env_check.py
    python members/member4/m4_env_check.py --save-report
    python members/member4/m4_env_check.py --help

Exit code 0 = all REQUIRED checks passed. Optional packages only warn.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path

MEMBER_DIR = Path(__file__).resolve().parent
MIN_PYTHON = (3, 10)

# import name -> pip name. Required ones are needed in Phase 1.
REQUIRED = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "torch": "torch",
    "matplotlib": "matplotlib",
    "pytest": "pytest",
}
# Needed later (EO imagery, dashboard, explainability). Missing = warning only.
OPTIONAL = {
    "torchvision": "torchvision",
    "rasterio": "rasterio",
    "plotly": "plotly",
    "streamlit": "streamlit",
    "shap": "shap",
}


def check_python() -> dict:
    ok = sys.version_info[:2] >= MIN_PYTHON
    return {"name": "python", "ok": ok, "version": platform.python_version(),
            "detail": f"need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}"}


def check_git() -> dict:
    try:
        out = subprocess.run(["git", "--version"], capture_output=True, text=True, timeout=10)
        return {"name": "git", "ok": out.returncode == 0, "version": out.stdout.strip()}
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"name": "git", "ok": False, "version": None, "detail": str(exc)}


def check_imports(packages: dict) -> list[dict]:
    results = []
    for mod, pip_name in packages.items():
        try:
            m = importlib.import_module(mod)
            results.append({"name": pip_name, "ok": True,
                            "version": getattr(m, "__version__", "unknown")})
        except Exception as exc:  # noqa: BLE001 - we want to report any import failure
            results.append({"name": pip_name, "ok": False, "version": None,
                            "detail": f"{type(exc).__name__}: {exc}"})
    return results


def torch_forward_test() -> dict:
    """Tiny GRU + linear head: input [batch=4, time=20, channels=8] -> [4, 1]."""
    import torch

    torch.manual_seed(42)
    gru = torch.nn.GRU(input_size=8, hidden_size=16, batch_first=True)
    head = torch.nn.Linear(16, 1)
    x = torch.randn(4, 20, 8)
    _, h = gru(x)
    y = head(h[-1])
    ok = tuple(y.shape) == (4, 1) and torch.isfinite(y).all().item()
    return {"name": "torch_forward", "ok": bool(ok), "output_shape": list(y.shape),
            "cuda_available": torch.cuda.is_available()}


def make_orbit_series(n: int = 400, period: int = 19, seed: int = 42):
    """Synthetic 'orbit' signal: sine with period `period` samples + small noise."""
    import numpy as np

    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 20 + 5 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.3, n)


def statsmodels_arima_test(horizon: int = 10) -> dict:
    """ARIMAX with Fourier terms of the orbit period (the handbook's recommended trick)."""
    import warnings

    import numpy as np
    from statsmodels.tsa.arima.model import ARIMA

    period = 19
    y = make_orbit_series(period=period)
    n = len(y)
    t_all = np.arange(n + horizon)
    exog_all = np.column_stack([np.sin(2 * np.pi * t_all / period),
                                np.cos(2 * np.pi * t_all / period)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(y, exog=exog_all[:n], order=(1, 0, 0)).fit()
        fc = fit.forecast(steps=horizon, exog=exog_all[n:])
    # Compare against the true noise-free signal for the next `horizon` steps.
    truth = 20 + 5 * np.sin(2 * np.pi * t_all[n:] / period)
    mae = float(np.mean(np.abs(fc - truth)))
    return {"name": "statsmodels_arimax", "ok": len(fc) == horizon and mae < 1.5,
            "forecast_len": int(len(fc)), "mae_vs_clean_signal": round(mae, 4)}


def xgboost_lag_test() -> dict:
    """XGBoost on 3 lag features, chronological 80/20 split (never shuffled)."""
    import numpy as np
    import xgboost as xgb

    y = make_orbit_series()
    lags = 3
    X = np.column_stack([y[i:len(y) - lags + i] for i in range(lags)])
    target = y[lags:]
    cut = int(0.8 * len(target))
    model = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42, verbosity=0)
    model.fit(X[:cut], target[:cut])
    pred = model.predict(X[cut:])
    mae = float(np.mean(np.abs(pred - target[cut:])))
    return {"name": "xgboost_lags", "ok": mae < 2.0, "test_mae": round(mae, 4)}


def run_checks(include_optional: bool = True) -> dict:
    report = {"member": "member4", "prefix": "m4_", "platform": platform.platform()}
    report["python"] = check_python()
    report["git"] = check_git()
    report["required_packages"] = check_imports(REQUIRED)
    report["optional_packages"] = check_imports(OPTIONAL) if include_optional else []

    imports_ok = all(r["ok"] for r in report["required_packages"])
    functional = []
    if imports_ok:
        for fn in (torch_forward_test, statsmodels_arima_test, xgboost_lag_test):
            try:
                functional.append(fn())
            except Exception as exc:  # noqa: BLE001
                functional.append({"name": fn.__name__, "ok": False,
                                   "detail": f"{type(exc).__name__}: {exc}"})
    report["functional_tests"] = functional

    report["all_required_ok"] = bool(
        report["python"]["ok"] and report["git"]["ok"] and imports_ok
        and functional and all(f["ok"] for f in functional)
    )
    return report


def print_report(report: dict) -> None:
    def line(r):
        mark = "PASS" if r["ok"] else "FAIL"
        extra = {k: v for k, v in r.items() if k not in ("name", "ok")}
        print(f"  [{mark}] {r['name']:<20} {extra}")

    print("\n=== Member 4 environment check ===")
    print(f"Platform: {report['platform']}")
    print("\nCore tools:")
    line(report["python"])
    line(report["git"])
    print("\nRequired packages:")
    for r in report["required_packages"]:
        line(r)
    if report["optional_packages"]:
        print("\nOptional packages (needed later; missing = OK for now):")
        for r in report["optional_packages"]:
            mark = "PASS" if r["ok"] else "WARN"
            print(f"  [{mark}] {r['name']:<20} {r.get('version') or r.get('detail')}")
    print("\nFunctional tests:")
    for r in report["functional_tests"]:
        line(r)
    print("\nRESULT:", "ALL REQUIRED CHECKS PASSED" if report["all_required_ok"]
          else "SOME REQUIRED CHECKS FAILED - see FAIL lines above")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Member 4 Phase-1 environment smoke test (Python, Git, "
                    "PyTorch, statsmodels, XGBoost).")
    parser.add_argument("--save-report", action="store_true",
                        help="write reports/m4_env_report.json inside members/member4/")
    parser.add_argument("--skip-optional", action="store_true",
                        help="do not check optional packages")
    args = parser.parse_args(argv)

    report = run_checks(include_optional=not args.skip_optional)
    print_report(report)
    if args.save_report:
        out = MEMBER_DIR / "reports" / "m4_env_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nReport saved to {out}")
    return 0 if report["all_required_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
