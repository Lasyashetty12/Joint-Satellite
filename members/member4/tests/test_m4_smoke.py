"""Smoke tests for Member 4, Phase 1 / Milestone 1.

Run from the repository root:
    python -m pytest members/member4/tests -q
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

MEMBER_DIR = Path(__file__).resolve().parents[1]
SCRIPT = MEMBER_DIR / "m4_env_check.py"


def load_env_check():
    spec = importlib.util.spec_from_file_location("m4_env_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_exists_with_prefix():
    assert SCRIPT.exists()
    assert SCRIPT.name.startswith("m4_")


def test_help_runs():
    out = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0
    assert "Member 4" in out.stdout


def test_folder_skeleton_exists():
    for sub in ["data", "mlts", "models", "training", "evaluation",
                "explainability", "dashboard", "tests", "reports"]:
        assert (MEMBER_DIR / sub).is_dir(), f"missing folder: {sub}"


def test_executable_files_use_m4_prefix():
    """Handbook rule: every executable .py in the member workspace starts with m4_
    (test files are named test_m4_*)."""
    for py in MEMBER_DIR.rglob("*.py"):
        name = py.name
        if name in ("__init__.py", "conftest.py"):
            continue
        assert name.startswith("m4_") or name.startswith("test_m4_"), name


def test_torch_forward_shape():
    r = load_env_check().torch_forward_test()
    assert r["ok"], r
    assert r["output_shape"] == [4, 1]


def test_arimax_forecast():
    r = load_env_check().statsmodels_arima_test(horizon=10)
    assert r["ok"], r
    assert r["forecast_len"] == 10


def test_xgboost_lags():
    r = load_env_check().xgboost_lag_test()
    assert r["ok"], r


def test_synthetic_series_is_reproducible():
    m = load_env_check()
    a = m.make_orbit_series(seed=42)
    b = m.make_orbit_series(seed=42)
    assert (a == b).all()
