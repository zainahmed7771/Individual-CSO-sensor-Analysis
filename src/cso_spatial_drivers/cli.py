"""Command-line interface for the reproducible CSO pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, validate_input_contracts
from .workflow import STAGES, run_pipeline


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--config", default="config/demo.yaml", help="YAML pipeline configuration")
    command.add_argument("--check-inputs", action="store_true", help="Validate inputs and exit without running stages")
    command.add_argument("--through", choices=STAGES, default="report", help="Stop after this stage")
    command.add_argument("--force", action="store_true", help="Replace declared generated files when output_root already contains a run")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config = load_config(args.config, repository_root=root)
    problems = validate_input_contracts(config)
    if args.check_inputs:
        print(f"Configuration: {config.source}")
        print(f"Profile: {config.profile}")
        for name, path in config.paths.items():
            print(f"{name:20} {path}")
        if problems:
            print("INPUT CHECK: FAILED")
            for problem in problems:
                print(f"- {problem}")
            return 1
        print("INPUT CHECK: PASSED")
        return 0
    manifest = run_pipeline(config, through=args.through, force=args.force)
    print(json.dumps(manifest, indent=2))
    print(f"PIPELINE COMPLETED: {config.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
