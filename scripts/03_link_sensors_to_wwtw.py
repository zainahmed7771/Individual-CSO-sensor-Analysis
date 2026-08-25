"""Validate the accepted outcome-blind sensor-to-WWTW assignment contract."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cso_spatial_drivers.config import load_config, validate_input_paths

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/demo.yaml")
    args = parser.parse_args()
    config = load_config(args.config, repository_root=ROOT)
    problems = validate_input_paths(config)
    assignments = pd.read_csv(config.paths["assignments"])
    required = {"sensor_uid", "company", "uwwCode", "assignment_evidence"}
    missing = required.difference(assignments.columns)
    if missing: problems.append(f"assignments missing columns: {sorted(missing)}")
    if not missing and assignments["sensor_uid"].duplicated().any(): problems.append("assignments contain duplicate sensor_uid values")
    if not missing and assignments["assignment_evidence"].astype("string").fillna("").str.strip().eq("").any(): problems.append("assignments contain blank evidence")
    if problems:
        print("ASSIGNMENT CHECK: FAILED")
        for problem in problems: print(f"- {problem}")
        return 1
    print(f"ASSIGNMENT CHECK: PASSED ({len(assignments):,} accepted rows)")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
