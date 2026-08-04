#!/usr/bin/env python3
"""CI check: every invokable skill script must declare a valid '# ---'
metadata header. This is a standalone reimplementation of the parsing rules
in physical-ai-platform-demo's features/pai-mcp-server/src/pai_mcp_server/
script_meta.py -- duplicated rather than imported, since that's a different
repo's package and this one has no dependency on it. Keep the two in sync
by hand if the header schema changes.

Scripts under a skill's scripts/lib/ (or any lib/ directory) are helper
modules imported by other scripts, not directly invokable via run_script,
so they're excluded from this check.
"""
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_ROOT = REPO_ROOT / "skills"

_HEADER_RE = re.compile(r"^# ---\n((?:#.*\n)*?)# ---\n", re.MULTILINE)
_COMMENT_PREFIX_RE = re.compile(r"^#\s?", re.MULTILINE)
_VALID_TYPES = {"string", "integer", "number", "boolean", "array"}


def find_scripts() -> list[Path]:
    return sorted(
        p for p in SKILLS_ROOT.glob("*/scripts/**/*")
        if p.suffix in (".py", ".sh") and "lib" not in p.relative_to(SKILLS_ROOT).parts
    )


def validate_header(path: Path) -> list[str]:
    """Returns a list of error strings, empty if the header is valid."""
    text = path.read_text(encoding="utf-8")
    match = _HEADER_RE.search(text)
    if not match:
        return ["missing '# ---' metadata header block"]

    try:
        raw_yaml = _COMMENT_PREFIX_RE.sub("", match.group(1))
        meta = yaml.safe_load(raw_yaml) or {}
    except yaml.YAMLError as e:
        return [f"header is not valid YAML: {e}"]

    errors = []
    if not meta.get("description"):
        errors.append("header missing required 'description' field")

    for i, raw_param in enumerate(meta.get("parameters", []) or []):
        label = raw_param.get("name", f"#{i}")
        for required_field in ("name", "type"):
            if not raw_param.get(required_field):
                errors.append(f"parameter '{label}' missing required '{required_field}' field")
        param_type = raw_param.get("type")
        if param_type and param_type not in _VALID_TYPES:
            errors.append(f"parameter '{label}' has unknown type '{param_type}' (valid: {sorted(_VALID_TYPES)})")

    return errors


def main() -> None:
    scripts = find_scripts()
    if not scripts:
        print("No scripts found under skills/*/scripts/ -- nothing to check.")
        sys.exit(0)

    failures = 0
    for path in scripts:
        rel = path.relative_to(REPO_ROOT)
        errors = validate_header(path)
        if errors:
            failures += 1
            print(f"FAIL {rel}")
            for e in errors:
                print(f"  - {e}")
        else:
            print(f"OK   {rel}")

    print(f"\n{len(scripts)} script(s) checked, {failures} failure(s).")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
