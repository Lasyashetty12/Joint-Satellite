"""Makes Member 4 modules importable in tests (e.g. `import m4_telemetry_generator`)."""
import sys
from pathlib import Path

MEMBER_DIR = Path(__file__).resolve().parents[1]
for sub in ["data", "mlts", "models", "training", "evaluation", "explainability"]:
    p = str(MEMBER_DIR / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
