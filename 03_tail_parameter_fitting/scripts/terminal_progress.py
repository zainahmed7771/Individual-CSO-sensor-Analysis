"""Small dependency-free terminal progress helper used by historical scripts."""
from __future__ import annotations

import sys
import time


class TerminalProgress:
    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = max(int(total), 0)
        self.current = 0
        self.started = time.monotonic()
        self._print("starting")

    def _print(self, suffix: str) -> None:
        elapsed = time.monotonic() - self.started
        print(f"[{self.current:>5}/{self.total:<5}] {self.label}: {suffix} ({elapsed:.1f}s)", file=sys.stderr)

    def update(self, step: int = 1, suffix: str = "") -> None:
        self.current = min(self.total, self.current + int(step))
        self._print(suffix or "running")

    def close(self, suffix: str = "complete") -> None:
        self.current = self.total
        self._print(suffix)
