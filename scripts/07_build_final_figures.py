"""Run all configured core stages and build the final run summary."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cso_spatial_drivers.cli import main
if __name__ == "__main__":
    raise SystemExit(main([*sys.argv[1:], "--through", "report"]))
