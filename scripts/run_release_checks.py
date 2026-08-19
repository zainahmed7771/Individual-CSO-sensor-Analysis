"""Run the public test/demo checks and write a machine-neutral test summary."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="pytest_", dir=ROOT / "audit") as temp:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", temp],
            cwd=ROOT, capture_output=True, text=True, check=False, env=env,
        )
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    match = re.search(r"(\d+) passed", result.stdout)
    summary = {"passed": int(match.group(1)) if match else 0, "returncode": result.returncode, "scope": "public scientific-invariant tests"}
    (ROOT / "audit/test_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if result.returncode:
        raise SystemExit(result.returncode)

    demo = subprocess.run([sys.executable, "scripts/run_reproducible_demo.py"], cwd=ROOT, capture_output=True, text=True, check=False, env=env)
    print(demo.stdout, end="")
    if demo.returncode or "DEMONSTRATION ONLY - NOT THE SCIENTIFIC RESULTS" not in demo.stdout:
        print(demo.stderr, file=sys.stderr, end="")
        raise SystemExit(demo.returncode or 1)

    audit = subprocess.run([sys.executable, "scripts/validate_release.py"], cwd=ROOT, check=False, env=env)
    raise SystemExit(audit.returncode)


if __name__ == "__main__":
    main()
