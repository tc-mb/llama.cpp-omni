#!/usr/bin/env python3
"""Validate the roadmap task list and render it into a GitHub issue body.

The task list lives in docs/contributing/roadmap/tasks.toml and is the single
source of truth. The public issue body is a generated artifact and must never be
edited by hand -- the next sync would overwrite it.

Requires Python 3.11+ (uses tomllib from the standard library). No third-party
dependencies, no network access unless --github is passed.

Usage:
    render.py                       # validate, then print the issue body
    render.py --check               # validate only, no output
    render.py --out FILE            # write the issue body to FILE
    render.py --github              # also read live issue state via the gh CLI
    render.py --repo OWNER/NAME     # repository to query and link to
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path


def say(msg: str = "") -> None:
    """Write a line to stdout.

    The print builtin is not used because this repository lints scripts/ with
    flake8-no-print, which rejects it (NP100).
    """
    sys.stdout.write(f"{msg}\n")


def warn(msg: str = "") -> None:
    """Write a line to stderr, for the same reason as say()."""
    sys.stderr.write(f"{msg}\n")


REPO_ROOT = Path(__file__).resolve().parents[2]
ROADMAP_DIR = REPO_ROOT / "docs" / "contributing" / "roadmap"
TASKS_FILE = ROADMAP_DIR / "tasks.toml"
LABELS_FILE = ROADMAP_DIR / "labels.toml"

ISSUE_TITLE = "[Roadmap] Community task list"

# The community chats on the OpenBMB Discord server. Kept here rather than
# inline so the channel can be changed in one place.
DISCORD_URL = "https://discord.com/invite/7q3ry8Ny8K"

DIFFICULTIES = ("easy", "medium", "hard", "expert")

DIFFICULTY_HEADING = {
    "easy": "Easy - good for newcomers",
    "medium": "Medium - one submodule to understand",
    "hard": "Hard - large, but the path is known",
    "expert": "Expert - original development, the path is not known yet",
}

DIFFICULTY_BADGE = {
    "easy": "`difficulty: easy`",
    "medium": "`difficulty: medium`",
    "hard": "`difficulty: hard`",
    "expert": "`difficulty: expert`",
}

# Rendered into the issue body, in the same order as DIFFICULTIES.
DIFFICULTY_DESCRIPTION = {
    "easy": "One file, no architecture knowledge needed, you can verify it locally.",
    "medium": "You need to understand one submodule. Touches 2-3 files. "
              "May need a model run to verify.",
    "hard": "Large but understood work: a port, a restructure, or offloading to an "
            "existing backend. Weeks of effort, but no invention required.",
    "expert": "Original development. The approach is not known yet, so it needs design "
              "work and probably a written proposal before any code. Real risk of a "
              "dead end.",
}

# domain -> label that must exist in labels.toml
DOMAIN_LABELS = {
    "vision": "vision",
    "audio": "audio",
    "tts": "tts",
    "duplex": "duplex",
    "server": "server",
    "model": "model",
    "ggml": "ggml",
    "conversion": "conversion",
    "python": "python",
    "build": "build",
    "docs": "documentation",
    "devops": "devops",
    "demo": "demo",
    "app": "app",
}

REQUIRED_FIELDS = ("id", "title", "difficulty", "domain")
OPTIONAL_FIELDS = ("issue", "owner", "note", "parent", "umbrella", "kind")
ALLOWED_FIELDS = frozenset(REQUIRED_FIELDS + OPTIONAL_FIELDS)

KINDS = ("task", "research")


class ValidationError(Exception):
    """Raised for any problem that must block publishing."""


def fail(msg: str) -> None:
    raise ValidationError(msg)


def load_toml(path: Path) -> dict:
    if not path.is_file():
        fail(f"missing file: {path}")
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        fail(f"{path.name} is not valid TOML: {exc}")


def load_known_domains() -> set[str] | None:
    """Read labels.toml to learn which domain labels are declared.

    Returns None when labels.toml is absent, in which case the cross-check is
    skipped rather than failing the build.
    """
    if not LABELS_FILE.is_file():
        return None
    data = load_toml(LABELS_FILE)
    return {entry["name"] for entry in data.get("labels", []) if "name" in entry}


def validate_relationships(tasks: list[dict]) -> None:
    """Check umbrella / parent consistency.

    Rules:
      - a parent must reference an existing task that declares umbrella = true
      - the parent must be defined earlier in the file, which also rules out cycles
      - a task that has a parent cannot itself be an umbrella (no nested umbrellas)
      - a parent must be referenced by at least one child, otherwise it is just a
        normal task and the umbrella flag is misleading
    """
    by_id = {t["id"]: t for t in tasks}
    order = {t["id"]: i for i, t in enumerate(tasks)}
    children: dict[str, list[str]] = {}

    for task in tasks:
        parent = task.get("parent")
        if parent is None:
            continue

        where = f"tasks.toml: {task['id']}"

        if parent not in by_id:
            fail(f"{where}: 'parent' points at {parent}, which does not exist")
        if order[parent] > order[task["id"]]:
            fail(f"{where}: 'parent' {parent} is defined below it. "
                 "An umbrella must come before its subtasks.")
        if not by_id[parent].get("umbrella"):
            fail(f"{where}: 'parent' {parent} does not declare 'umbrella = true'")
        if task.get("umbrella"):
            fail(f"{where}: a subtask cannot also be an umbrella (no nesting)")
        if task.get("issue") and not by_id[parent].get("issue"):
            # Not fatal, but a subtask with its own issue under an umbrella with
            # none is usually a sign the umbrella should own the tracking issue.
            warn(f"warning: {task['id']} has an issue but its umbrella {parent} "
                 "has none; consider giving the umbrella the tracking issue")

        children.setdefault(parent, []).append(task["id"])

    for task in tasks:
        if task.get("umbrella") and task["id"] not in children:
            fail(
                f"tasks.toml: {task['id']} declares 'umbrella = true' but has no "
                "subtasks. Drop the flag or add children with 'parent = "
                f"\"{task['id']}\"'."
            )


def validate(tasks_doc: dict) -> list[dict]:
    tasks = tasks_doc.get("tasks", [])
    if not isinstance(tasks, list):
        fail("tasks.toml: 'tasks' must be a list of [[tasks]] tables")

    # Check the difficulty labels exist even when the list is empty, so a
    # mismatch between DIFFICULTIES and labels.toml surfaces immediately rather
    # than as a silently missing label later.
    known_labels = load_known_domains()
    if known_labels is not None:
        for level in DIFFICULTIES:
            if f"difficulty: {level}" not in known_labels:
                fail(
                    f"difficulty level {level!r} has no matching "
                    f"'difficulty: {level}' label in labels.toml. "
                    "Add it there or the label will be missing on GitHub."
                )

    # An empty list is a valid state: the roadmap starts empty and grows.
    if not tasks:
        return []

    seen_ids: dict[str, int] = {}
    seen_issues: dict[int, str] = {}
    ids: list[int] = []

    for index, task in enumerate(tasks):
        where = f"tasks.toml: [[tasks]] block #{index + 1}"
        if not isinstance(task, dict):
            fail(f"{where}: expected a table")

        unknown = set(task) - ALLOWED_FIELDS
        if unknown:
            fail(
                f"{where}: unknown field(s) {sorted(unknown)}. "
                f"Allowed: {sorted(ALLOWED_FIELDS)}"
            )

        for field in REQUIRED_FIELDS:
            if field not in task:
                fail(f"{where}: missing required field '{field}'")
            if not isinstance(task[field], str) or not task[field].strip():
                fail(f"{where}: '{field}' must be a non-empty string")

        task_id = task["id"]
        where = f"tasks.toml: {task_id}"

        if seen_ids.get(task_id) is not None:
            fail(f"{where}: duplicate id, already used by block #{seen_ids[task_id] + 1}")
        seen_ids[task_id] = index

        if not task_id.startswith("T-"):
            fail(f"{where}: id must start with 'T-' (for example T-001)")
        digits = task_id[2:]
        if not digits.isdigit() or len(digits) < 3:
            fail(f"{where}: id must be T- followed by at least 3 digits")
        if len(digits) > 3 and digits[0] == "0":
            fail(f"{where}: id must not have leading zeros beyond the 3-digit width")
        ids.append(int(digits))

        if task_id != f"T-{int(digits):03d}":
            fail(f"{where}: id must be zero-padded to at least 3 digits "
                 f"(expected T-{int(digits):03d})")

        if "\n" in task["title"]:
            fail(f"{where}: 'title' must be a single line")

        if task["difficulty"] not in DIFFICULTIES:
            fail(f"{where}: 'difficulty' must be one of {list(DIFFICULTIES)}, "
                 f"got {task['difficulty']!r}")

        domain = task["domain"]
        if domain not in DOMAIN_LABELS:
            fail(f"{where}: unknown 'domain' {domain!r}. "
                 f"Allowed: {sorted(DOMAIN_LABELS)}")
        if known_labels is not None:
            label = DOMAIN_LABELS[domain]
            if label not in known_labels:
                fail(f"{where}: domain {domain!r} maps to label {label!r}, "
                     f"which is not declared in labels.toml")

        issue = task.get("issue")
        if issue is not None:
            if not isinstance(issue, int) or isinstance(issue, bool) or issue <= 0:
                fail(f"{where}: 'issue' must be a positive integer, got {issue!r}")
            if issue in seen_issues:
                fail(f"{where}: 'issue' {issue} is already used by {seen_issues[issue]}")
            seen_issues[issue] = task_id

        owner = task.get("owner")
        if owner is not None:
            if not isinstance(owner, str) or not owner.strip():
                fail(f"{where}: 'owner' must be a non-empty string when present")
            if owner.startswith("@"):
                fail(f"{where}: 'owner' must not include a leading '@'")

        note = task.get("note")
        if note is not None:
            if not isinstance(note, str) or not note.strip():
                fail(f"{where}: 'note' must be a non-empty string when present")
            if "\n" in note:
                fail(f"{where}: 'note' must be a single line")

        umbrella = task.get("umbrella")
        if umbrella is not None and not isinstance(umbrella, bool):
            fail(f"{where}: 'umbrella' must be a boolean (true/false)")

        kind = task.get("kind", "task")
        if kind not in KINDS:
            fail(f"{where}: 'kind' must be one of {list(KINDS)}, got {kind!r}")

        parent = task.get("parent")
        if parent is not None:
            if not isinstance(parent, str) or not parent.strip():
                fail(f"{where}: 'parent' must be a non-empty string when present")
            if parent == task_id:
                fail(f"{where}: 'parent' must not point at itself")

    validate_relationships(tasks)

    if ids != sorted(ids):
        fail("tasks.toml: task ids are not in ascending order. "
             "The list is append-only, so new tasks go at the end with the next id.")

    expected = list(range(ids[0], ids[0] + len(ids)))
    if ids != expected:
        missing = sorted(set(expected) - set(ids))
        warn(f"warning: task ids are not contiguous, missing "
             f"{', '.join(f'T-{n:03d}' for n in missing)}. "
             "This is allowed but usually means an id was reused or a task was deleted.")

    return tasks


def fetch_issue_states(repo: str, numbers: list[int]) -> dict[int, dict]:
    """Return per-issue live state from GitHub, keyed by issue number.

    Each value is {"state": "OPEN"/"CLOSED", "assignees": [login, ...]}.

    Reading assignees here is what makes claiming work without editing
    tasks.toml: the assignment on GitHub is the source of truth, and the
    optional 'owner' field in the file is only a fallback.
    """
    if not numbers:
        return {}
    info: dict[int, dict] = {}
    for number in numbers:
        try:
            out = subprocess.run(
                ["gh", "issue", "view", str(number), "--repo", repo,
                 "--json", "state,assignees"],
                capture_output=True, text=True, check=True,
            ).stdout
        except FileNotFoundError:
            fail("--github requires the gh CLI, which was not found on PATH")
        except subprocess.CalledProcessError as exc:
            fail(f"gh failed for issue #{number}: {exc.stderr.strip()}")
        data = json.loads(out)
        info[number] = {
            "state": data.get("state", "OPEN"),
            "assignees": [a["login"] for a in data.get("assignees", [])],
        }
    return info


def resolve_owner(task: dict, info: dict[int, dict]) -> str | None:
    """Pick the owner to display: GitHub assignee first, file value second.

    A discrepancy means someone was assigned on GitHub after the file was last
    written, which is the normal case. The GitHub side wins, and the difference
    is reported so the file can be brought in line.
    """
    number = task.get("issue")
    github_owner = None
    if number is not None and number in info:
        assignees = info[number]["assignees"]
        if assignees:
            github_owner = assignees[0]

    file_owner = task.get("owner")
    if github_owner and file_owner and github_owner != file_owner:
        warn(f"warning: {task.get('id', '?')} is assigned to {github_owner} on GitHub "
             f"but task file says {file_owner}; using the GitHub value")
    return github_owner or file_owner


def render(tasks: list[dict], repo: str, states: dict[int, str]) -> str:
    top_level = [t for t in tasks if not t.get("parent")]
    children: dict[str, list[dict]] = {}
    for task in tasks:
        if task.get("parent"):
            children.setdefault(task["parent"], []).append(task)

    counts = {level: 0 for level in DIFFICULTIES}
    for task in top_level:
        counts[task["difficulty"]] += 1

    out: list[str] = []
    out.append(f"# {ISSUE_TITLE}")
    out.append("")
    out.append(
        "> **Generated file - do not edit this description.** "
        f"It is rendered from [`docs/contributing/roadmap/tasks.toml`]"
        f"(https://github.com/{repo}/blob/master/docs/contributing/roadmap/tasks.toml) "
        "and any manual edit here is overwritten on the next sync."
    )
    out.append("")
    out.append(
        "A list of things we want to get done, sorted by difficulty. "
        "If you want to work on something, claim it in the comments."
    )
    out.append("")

    out.append("## How to claim a task")
    out.append("")
    out.append("1. Comment on **the issue linked next to the task** saying how you plan to approach it.")
    out.append("2. A maintainer assigns it to you and the task moves to `owner` in the list below.")
    out.append("3. Open a pull request and link it in that issue.")
    out.append("")
    out.append(
        "**Please talk to us before writing code.** For anything above `easy`, we would rather "
        "agree on the approach first than review a large pull request that goes the wrong way."
    )
    out.append("")
    out.append(
        "**Tasks are recycled after 14 days of no progress.** If there is no pull request or "
        "draft pull request within 14 days of claiming, we release the assignment and the task "
        "goes back to the list. This is not a penalty - it just keeps tasks from being parked "
        "indefinitely. Tell us if you are stuck and we will help break it down."
    )
    out.append("")
    out.append(
        "If a task has no issue linked yet, comment on **this** issue and we will open one "
        "for it."
    )
    out.append("")
    out.append(
        f"Questions, or just want to chat? Join us on [Discord]({DISCORD_URL}). "
        "Note that decisions belong in the issue, not in chat - see the "
        f"[contributing guide](https://github.com/{repo}/blob/master/"
        "docs/contributing/README.md#where-to-talk)."
    )
    out.append("")

    out.append("## Difficulty")
    out.append("")
    out.append("| Level | What it means |")
    out.append("|-------|---------------|")
    for level in DIFFICULTIES:
        out.append(f"| {level.capitalize()} | {DIFFICULTY_DESCRIPTION[level]} |")
    out.append("")

    if not top_level:
        out.append(
            "_The list is empty right now. Tasks will appear here as they are triaged._"
        )
        out.append("")
    else:
        summary = ", ".join(f"**{counts[level]} {level}**" for level in DIFFICULTIES)
        out.append(
            f"Currently {summary} top-level items"
            + (f", plus **{len(tasks) - len(top_level)}** subtasks." if children else ".")
        )
        out.append("")

    for level in DIFFICULTIES:
        grouped = [t for t in top_level if t["difficulty"] == level]
        if not grouped and not top_level:
            continue
        out.append(f"## {DIFFICULTY_HEADING[level]} ({len(grouped)})")
        out.append("")
        if not grouped:
            out.append("_Nothing here right now - check back after the next triage pass._")
            out.append("")
            continue
        for task in grouped:
            out.append(render_task_line(task, repo, states, children))
        out.append("")

    if children:
        out.append("## Subtasks by difficulty")
        out.append("")
        out.append(
            "_The same subtasks listed above under their umbrella, regrouped by "
            "difficulty so the small ones are easy to find._"
        )
        out.append("")
        for level in DIFFICULTIES:
            in_level = [t for t in tasks if t.get("parent")
                        and t["difficulty"] == level]
            if not in_level:
                continue
            out.append(f"**{level.capitalize()}**")
            out.append("")
            for task in in_level:
                parent = task["parent"]
                number = task.get("issue")
                link = (f" [#{number}](https://github.com/{repo}/issues/{number})"
                        if number is not None else "")
                out.append(f"- **{task['id']}** {task['title']}{link} - part of `{parent}`")
            out.append("")

    out.append("<details>")
    out.append("<summary>Task list format for maintainers</summary>")
    out.append("")
    out.append(
        "Tasks live in `docs/contributing/roadmap/tasks.toml`. To add one, append a "
        "`[[tasks]]` block, then run:"
    )
    out.append("")
    out.append("```bash")
    out.append("python3 scripts/roadmap/render.py --check")
    out.append("python3 scripts/roadmap/sync-issue.py          # dry run")
    out.append("python3 scripts/roadmap/sync-issue.py --apply  # publish")
    out.append("```")
    out.append("")
    out.append("See `docs/contributing/roadmap/README.md` for the full spec.")
    out.append("")
    out.append("</details>")

    return "\n".join(out) + "\n"


def render_task_line(task: dict, repo: str, states: dict[int, dict],
                     children: dict[str, list[dict]] | None = None,
                     depth: int = 0) -> str:
    number = task.get("issue")
    if number is not None and states.get(number, {}).get("state") == "CLOSED":
        box = "x"
    else:
        box = " "

    indent = "  " * depth
    parts = [f"{indent}- [{box}] **{task['id']}** {task['title']}"]

    if task.get("kind") == "research":
        parts.append("`research`")

    if number is not None:
        parts.append(f"[#{number}](https://github.com/{repo}/issues/{number})")
    elif depth == 0:
        parts.append("_no issue yet_")

    parts.append(DIFFICULTY_BADGE[task["difficulty"]])
    parts.append(f"`{task['domain']}`")

    owner = resolve_owner(task, states)
    if owner:
        parts.append(f"assigned to @{owner}")

    kids = (children or {}).get(task["id"], [])
    if kids:
        parts.append(f"_(umbrella, {len(kids)} subtask{'s' if len(kids) != 1 else ''})_")

    line = " - ".join(parts)

    note = task.get("note")
    if note:
        line += f"\n{indent}  <br/>_{note}_"

    if kids:
        for child in kids:
            line += "\n" + render_task_line(child, repo, states, None, depth + 1)

    return line


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="validate only, do not render")
    parser.add_argument("--out", metavar="FILE",
                        help="write the rendered body to FILE instead of stdout")
    parser.add_argument("--github", action="store_true",
                        help="read live issue state via the gh CLI so closed tasks render as done")
    parser.add_argument("--repo", default="tc-mb/llama.cpp-omni",
                        help="repository used for issue links and queries")
    parser.add_argument("--tasks", metavar="FILE",
                        help="override the tasks file (default: docs/contributing/roadmap/tasks.toml)")
    args = parser.parse_args()

    tasks_path = Path(args.tasks) if args.tasks else TASKS_FILE

    try:
        tasks_doc = load_toml(tasks_path)
        tasks = validate(tasks_doc)
    except ValidationError as exc:
        warn(f"error: {exc}")
        return 1

    if args.check:
        warn(f"ok: {len(tasks)} tasks validated")
        return 0

    states: dict[int, dict] = {}
    if args.github:
        numbers = sorted({t["issue"] for t in tasks if t.get("issue")})
        try:
            states = fetch_issue_states(args.repo, numbers)
        except ValidationError as exc:
            warn(f"error: {exc}")
            return 1

    body = render(tasks, args.repo, states)

    if args.out:
        Path(args.out).write_text(body, encoding="utf-8")
        warn(f"ok: {len(tasks)} tasks rendered to {args.out}")
    else:
        sys.stdout.write(body)

    return 0


if __name__ == "__main__":
    sys.exit(main())
