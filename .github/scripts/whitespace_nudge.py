#!/usr/bin/env python3
"""Toggles the trailing newline on a few tracked files, safely.

Run by `.github/workflows/whitespace-nudge.yml`. The whole point is
that the repository still works afterwards, so this is built around
one rule: **verify before committing, revert on any doubt.**

Why this particular edit. Whitespace is not uniformly harmless in
Python - a stray space inside a line's leading indentation is a
syntax error, and trailing spaces on a code line are a lint finding
someone eventually has to clean up. The number of newlines at end of
file is different: the parser does not care, linters do not flag one
versus two, runtime behaviour is identical, and it is still a genuine
one-line diff that git will record.

It toggles rather than appends. Appending would add a blank line per
run and a file would accumulate hundreds of them over a year; toggling
means every file only ever oscillates between one and two trailing
newlines.

Safety, in order:
  1. Only files git already tracks, and only text.
  2. Never itself, the workflow, or anything under .github.
  3. After editing, every tracked *.py is re-parsed with `ast.parse`.
     One failure reverts the entire working tree and exits non-zero.
"""
from __future__ import annotations

import ast
import os
import random
import subprocess
import sys
from pathlib import Path

# Extensions safe to end-pad. Deliberately conservative: no JSON (a
# trailing newline is fine but some strict readers are not), no binary,
# nothing whose meaning could depend on exact bytes.
_SAFE_SUFFIXES = {".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".jinja2"}

# Never touched. The workflow and this script must stay byte-stable, or
# a bad edit could disable the very check that catches bad edits.
_EXCLUDED_DIRS = {".github", ".git", "node_modules", ".venv"}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def tracked_files() -> list[Path]:
    """Every tracked file eligible for a nudge."""
    found: list[Path] = []
    for line in _git("ls-files", "-z").split("\0"):
        if not line:
            continue
        path = Path(line)
        if path.suffix not in _SAFE_SUFFIXES:
            continue
        if any(part in _EXCLUDED_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        found.append(path)
    return found


def toggle_trailing_newline(path: Path) -> bool:
    """One trailing newline becomes two, two become one.

    Returns False when the file is empty or unreadable as UTF-8 text -
    a file this cannot read confidently is a file it does not edit.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False
    if not text.strip():
        return False

    if text.endswith("\n\n"):
        updated = text[:-1]
    elif text.endswith("\n"):
        updated = text + "\n"
    else:
        # No trailing newline at all: adding one is an improvement and
        # POSIX-correct, so take it.
        updated = text + "\n"

    if updated == text:
        return False
    # newline="" so Python does not translate \n into \r\n on Windows
    # runners and rewrite every line ending in the file.
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(updated)
    return True


def every_python_file_still_parses() -> tuple[bool, str]:
    """Re-parses every tracked .py file. This is the safety net."""
    for line in _git("ls-files", "-z", "*.py").split("\0"):
        if not line:
            continue
        path = Path(line)
        if not path.is_file():
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            return False, f"{path}: {exc}"
        except (UnicodeDecodeError, OSError) as exc:
            return False, f"{path}: unreadable ({exc})"
    return True, ""


def emit(name: str, value: str) -> None:
    """Writes a workflow output, when running inside Actions."""
    target = os.environ.get("GITHUB_OUTPUT")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")
    print(f"{name}={value}")


def main() -> int:
    try:
        wanted = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    except ValueError:
        wanted = 5
    wanted = max(1, min(wanted, 25))

    candidates = tracked_files()
    if not candidates:
        emit("changed", "false")
        emit("files", "0")
        print("nothing eligible to touch")
        return 0

    touched: list[Path] = []
    for path in random.sample(candidates, min(wanted, len(candidates))):
        if toggle_trailing_newline(path):
            touched.append(path)

    if not touched:
        emit("changed", "false")
        emit("files", "0")
        print("no file actually changed")
        return 0

    ok, problem = every_python_file_still_parses()
    if not ok:
        # Revert everything. Better to push nothing than to push a
        # repository that does not import.
        subprocess.run(["git", "checkout", "--", "."], check=False)
        print(f"VERIFICATION FAILED, reverted: {problem}", file=sys.stderr)
        emit("changed", "false")
        emit("files", "0")
        return 1

    for path in touched:
        print(f"  touched {path}")
    emit("changed", "true")
    emit("files", str(len(touched)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
