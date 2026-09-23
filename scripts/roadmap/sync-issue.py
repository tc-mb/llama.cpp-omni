#!/usr/bin/env python3
"""Create or update the pinned roadmap issue from docs/contributing/roadmap/tasks.toml.

The issue body is a generated artifact. This script renders it with render.py and
then upserts it: if an issue with the exact expected title already exists it is
edited, otherwise a new one is created.

The issue is always labelled `roadmap`, which is on the stale bot's exemption
list. Without that label it would be marked stale after 30 days and closed 14
days later.

Dry run by default: nothing is written to GitHub unless --apply is passed.

Usage:
    sync-issue.py                    # show what would change
    sync-issue.py --apply            # create or update the issue
    sync-issue.py --apply --pin      # also pin it to the top of the issue list
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# GitHub has no `gh issue pin` subcommand, so pinning goes through GraphQL.
PIN_MUTATION = (
    "query=mutation($id: ID!) { pinIssue(input: {issueId: $id}) "
    "{ issue { number } } }"
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


def load_render_module():
    spec = importlib.util.spec_from_file_location("roadmap_render", HERE / "render.py")
    if spec is None or spec.loader is None:
        sys.exit("error: could not load render.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gh(*args: str) -> str:
    try:
        result = subprocess.run(
            ["gh", *args], capture_output=True, text=True, check=True
        )
    except FileNotFoundError:
        sys.exit("error: the gh CLI was not found on PATH")
    except subprocess.CalledProcessError as exc:
        sys.exit(f"error: gh {' '.join(args)} failed:\n{exc.stderr.strip()}")
    return result.stdout


def find_existing_issue(repo: str, title: str) -> dict | None:
    raw = gh("issue", "list", "--repo", repo, "--state", "all", "--limit", "500",
             "--json", "number,title,state")
    for item in json.loads(raw):
        if item["title"] == title:
            return item
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", default="tc-mb/llama.cpp-omni")
    parser.add_argument("--apply", action="store_true",
                        help="actually write to GitHub (default is a dry run)")
    parser.add_argument("--pin", action="store_true",
                        help="pin the issue after creating or updating it")
    args = parser.parse_args()

    render = load_render_module()

    try:
        tasks_doc = render.load_toml(render.TASKS_FILE)
        tasks = render.validate(tasks_doc)
    except render.ValidationError as exc:
        warn(f"error: {exc}")
        return 1

    states: dict[int, dict] = {}
    numbers = sorted({t["issue"] for t in tasks if t.get("issue")})
    if numbers:
        try:
            states = render.fetch_issue_states(args.repo, numbers)
        except render.ValidationError as exc:
            warn(f"error: {exc}")
            return 1

    body = render.render(tasks, args.repo, states)
    title = render.ISSUE_TITLE

    existing = find_existing_issue(args.repo, title)

    say(f"repository : {args.repo}")
    say(f"title      : {title}")
    say(f"tasks      : {len(tasks)}")
    say(f"body       : {len(body)} bytes, {body.count(chr(10))} lines")
    say(f"mode       : {'APPLY' if args.apply else 'DRY RUN (nothing is written)'}")
    say()
    if existing:
        say(f"Existing issue found: #{existing['number']} (state: {existing['state']})")
        say("Action: edit its description in place.")
    else:
        say("No issue with that exact title exists.")
        say("Action: create a new issue.")
    say("Label to add: roadmap (required - it is on the stale bot exemption list)")
    say()

    if not args.apply:
        say("--- rendered body ---")
        say(body)
        say("--- end of body ---")
        say()
        say("Dry run only. Re-run with --apply to write to GitHub.")
        return 0

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(body)
        body_file = fh.name

    try:
        if existing:
            gh("issue", "edit", str(existing["number"]), "--repo", args.repo,
               "--body-file", body_file, "--add-label", "roadmap")
            issue_number = existing["number"]
            say(f"updated: #{issue_number}")
        else:
            out = gh("issue", "create", "--repo", args.repo, "--title", title,
                     "--body-file", body_file, "--label", "roadmap")
            say(out.strip())
            issue_number = None
            for token in out.replace("/", " ").split():
                if token.isdigit():
                    issue_number = int(token)
                    break
            if issue_number:
                say(f"created: #{issue_number}")
    finally:
        Path(body_file).unlink(missing_ok=True)

    if args.pin:
        if not issue_number:
            say("warning: could not determine the issue number, skipping --pin")
        else:
            node_id = json.loads(
                gh("issue", "view", str(issue_number), "--repo", args.repo, "--json", "id")
            )["id"]
            gh("api", "graphql",
               "-f", PIN_MUTATION,
               "-f", f"id={node_id}")
            say(f"pinned: #{issue_number}")

    say("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
