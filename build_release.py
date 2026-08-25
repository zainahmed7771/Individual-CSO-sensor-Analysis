"""Compatibility shim for the retired one-off release assembler."""

raise SystemExit(
    "The public repository is already assembled. Run "
    "'python scripts/run_release_checks.py' to validate it. "
    "The original builder is preserved under archive/legacy_release_builder/."
)
