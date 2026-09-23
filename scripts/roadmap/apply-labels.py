#!/usr/bin/env python3
"""Create the labels declared in docs/contributing/roadmap/labels.toml.

Labels live in repository settings, not in git, so a fork does not inherit them.
That is why the issue templates and the stale bot reference labels that do not
exist, and GitHub silently skips a label it cannot find.

This script only creates labels that are MISSING. It never touches the colour or
description of a label that already exists unless --update-existing is passed.

Dry run by default: nothing is written to GitHub unless --apply is passed.

Usage:
    apply-labels.py                    # show what would be created
    apply-labels.py --apply            # create the missing labels
    apply-labels.py --apply --update-existing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path

LABELS_FILE = (
    Path(__file__).resolve().parents[2]
    / "docs" / "contributing" / "roadmap" / "labels.toml"
)


def say(msg: str = "") -> None:
    """Write a line to stdout.

    The print builtin is not used because this repository lints scripts/ with
    flake8-no-print, which rejects it (NP100).
    """
    sys.stdout.write(f"{msg}\n")


def warn(msg: str = "") -> None:
    """Write a line to stderr, for the same reason as say()."""
    sys.stderr.write(f"{msg}\n")


def gh(*args: str, check: bool = True) -> str:
    try:
        result = subprocess.run(
            ["gh", *args], capture_output=True, text=True, check=check
        )
    except FileNotFoundError:
        sys.exit("error: the gh CLI was not found on PATH")
    except subprocess.CalledProcessError as exc:
        sys.exit(f"error: gh {' '.join(args)} failed:\n{exc.stderr.strip()}")
    return result.stdout


def load_declared() -> list[dict]:
    if not LABELS_FILE.is_file():
        sys.exit(f"error: missing {LABELS_FILE}")
    with LABELS_FILE.open("rb") as fh:
        try:
            doc = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            sys.exit(f"error: {LABELS_FILE.name} is not valid TOML: {exc}")

    labels = doc.get("labels")
    if not labels:
        sys.exit(f"error: {LABELS_FILE.name} declares no [[labels]] entries")

    for entry in labels:
        for field in ("name", "color", "description", "group", "why"):
            if field not in entry or not str(entry[field]).strip():
                sys.exit(f"error: label entry missing '{field}': {entry!r}")
        color = entry["color"]
        if len(color) != 6 or any(c not in "0123456789abcdefABCDEF" for c in color):
            sys.exit(f"error: label {entry['name']!r} has an invalid colour {color!r} "
                     "(expected 6 hex digits, no '#')")
    return labels


def fetch_existing(repo: str) -> dict[str, dict]:
    raw = gh("api", f"repos/{repo}/labels?per_page=100", "--paginate")
    # --paginate concatenates JSON arrays, so parse defensively.
    existing: dict[str, dict] = {}
    decoder = json.JSONDecoder()
    text = raw.strip()
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos] in " \t\r\n,":
            pos += 1
        if pos >= len(text):
            break
        obj, end = decoder.raw_decode(text, pos)
        pos = end
        for item in obj if isinstance(obj, list) else [obj]:
            existing[item["name"]] = item
    return existing


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", default="tc-mb/llama.cpp-omni")
    parser.add_argument("--apply", action="store_true",
                        help="actually create labels (default is a dry run)")
    parser.add_argument("--update-existing", action="store_true",
                        help="also overwrite colour and description of existing labels")
    args = parser.parse_args()

    declared = load_declared()
    existing = fetch_existing(args.repo)

    to_create, to_update, unchanged = [], [], []
    for entry in declared:
        current = existing.get(entry["name"])
        if current is None:
            to_create.append(entry)
        elif args.update_existing and (
            current.get("color", "").lower() != entry["color"].lower()
            or (current.get("description") or "") != entry["description"]
        ):
            to_update.append(entry)
        else:
            unchanged.append(entry)

    say(f"repository : {args.repo}")
    say(f"declared   : {len(declared)} labels")
    say(f"existing   : {len(existing)} labels in the repo")
    say(f"mode       : {'APPLY' if args.apply else 'DRY RUN (nothing is written)'}")
    say()

    if to_create:
        say(f"Would create {len(to_create)} label(s):")
        for entry in to_create:
            say(f"  + {entry['name']:<28} #{entry['color']}  [{entry['group']}]")
            say(f"      {entry['description']}")
        say()
    else:
        say("No missing labels - nothing to create.")
        say()

    if to_update:
        say(f"Would update {len(to_update)} existing label(s):")
        for entry in to_update:
            say(f"  ~ {entry['name']}")
        say()

    say(f"Already present, left untouched: {len(unchanged)}")
    for entry in unchanged:
        say(f"  = {entry['name']}")

    if not args.apply:
        say()
        say("Dry run only. Re-run with --apply to write these changes to GitHub.")
        return 0

    if to_update and not args.update_existing:
        say("\n(error) refusing to update existing labels without --update-existing")
        return 1

    for entry in to_create:
        gh("label", "create", entry["name"],
           "--color", entry["color"],
           "--description", entry["description"],
           "--repo", args.repo)
        say(f"created: {entry['name']}")

    for entry in to_update:
        gh("label", "edit", entry["name"],
           "--color", entry["color"],
           "--description", entry["description"],
           "--repo", args.repo)
        say(f"updated: {entry['name']}")

    say("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
