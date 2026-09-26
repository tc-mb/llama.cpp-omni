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

ISSUE_TITLE = "[Roadmap] Community task list"  # the English title; see STRINGS for other languages

# The community chats on the project's own Discord server. Kept here rather
# than inline so the invite can be changed in one place.
DISCORD_URL = "https://discord.com/invite/eYBZhN9SG"

DIFFICULTIES = ("easy", "medium", "hard", "expert")

# Labels are language neutral: they are what shows up on GitHub.
DIFFICULTY_BADGE = {
    "easy": "`difficulty: easy`",
    "medium": "`difficulty: medium`",
    "hard": "`difficulty: hard`",
    "expert": "`difficulty: expert`",
}

DIFFICULTY_HEADING = {
    "en": {
        "easy": "Easy - good for newcomers",
        "medium": "Medium - one submodule to understand",
        "hard": "Hard - large, but the path is known",
        "expert": "Expert - original development, the path is not known yet",
    },
    "zh": {
        "easy": "简单 - 适合新手上手",
        "medium": "中等 - 需要理解一个子模块",
        "hard": "困难 - 工作量大，但路径已知",
        "expert": "专家 - 原创开发，路径尚未确定",
    },
}

DIFFICULTY_DESCRIPTION = {
    "en": {
        "easy": "One file, no architecture knowledge needed, you can verify it locally.",
        "medium": "You need to understand one submodule. Touches 2-3 files. "
                  "May need a model run to verify.",
        "hard": "Large but understood work: a port, a restructure, or offloading to an "
                "existing backend. Weeks of effort, but no invention required.",
        "expert": "Original development. The approach is not known yet, so it needs design "
                  "work and probably a written proposal before any code. Real risk of a "
                  "dead end.",
    },
    "zh": {
        "easy": "只碰一个文件；不需要架构知识；本地就能自测。",
        "medium": "需要理解一个子模块；改动跨 2-3 个文件；可能要跑一次模型验证。",
        "hard": "工作量大但路径已知：移植、重构、或搬到已有后端。需要数周投入，但不需要发明东西。",
        "expert": "原创开发：方法尚未确定，需要先做设计、大概率要先写提案，而且有走不通的风险。",
    },
}

# UI strings for the rendered issue body.
STRINGS = {
    "en": {
        "issue_title": "[Roadmap] Community task list",
        "generated": (
            "> **Generated file - do not edit this description.** "
            "It is rendered from [`docs/contributing/roadmap/tasks.toml`]"
            "(https://github.com/{repo}/blob/master/docs/contributing/roadmap/tasks.toml) "
            "and any manual edit here is overwritten on the next sync."
        ),
        "intro": "A list of things we want to get done, sorted by difficulty. "
                 "If you want to work on something, claim it in the comments.",
        "claim_heading": "## How to claim a task",
        "claim_steps": [
            "Comment on **the issue linked next to the task** saying how you plan to approach it.",
            "A maintainer assigns it to you and the task moves to `owner` in the list below.",
            "Open a pull request and link it in that issue.",
        ],
        "claim_warning": "**Please talk to us before writing code.** For anything above `easy`, "
                         "we would rather agree on the approach first than review a large pull "
                         "request that goes the wrong way.",
        "claim_recycle": "**Tasks are recycled after 14 days of no progress.** If there is no "
                         "pull request or draft pull request within 14 days of claiming, we "
                         "release the assignment and the task goes back to the list. This is not "
                         "a penalty - it just keeps tasks from being parked indefinitely. Tell us "
                         "if you are stuck and we will help break it down.",
        "no_issue_yet": "If a task has no issue linked yet, comment on **this** issue and we "
                        "will open one for it.",
        "discord": "Questions, or just want to chat? Join us on [Discord]({discord}). Note that "
                   "decisions belong in the issue, not in chat - see the "
                   "[contributing guide](https://github.com/{repo}/blob/master/"
                   "docs/contributing/README.md#where-to-talk).",
        "difficulty_heading": "## Difficulty",
        "difficulty_table_head": "| Level | What it means |",
        "level_name": {"easy": "Easy", "medium": "Medium", "hard": "Hard", "expert": "Expert"},
        "empty": "_The list is empty right now. Tasks will appear here as they are triaged._",
        "summary": "Currently {summary} top-level items{extra}",
        "level_count": "**{n} {word}**",
        "joiner": ", ",
        "summary_subtasks": ", plus **{n}** subtasks.",
        "summary_end": ".",
        "level_word": {"easy": "easy", "medium": "medium", "hard": "hard", "expert": "expert"},
        "nothing_here": "_Nothing here right now - check back after the next triage pass._",
        "subtasks_heading": "## Subtasks by difficulty",
        "subtasks_intro": "_The same subtasks listed above under their umbrella, regrouped by "
                          "difficulty so the small ones are easy to find._",
        "part_of": "part of",
        "umbrella": "umbrella, {n} subtask(s)",
        "no_issue": "_no issue yet_",
        "assigned": "assigned to @{owner}",
        "maint_summary": "Task list format for maintainers",
        "maint_body": "Tasks live in `docs/contributing/roadmap/tasks.toml`. To add one, append "
                      "a `[[tasks]]` block, then run:",
        "maint_footer": "See `docs/contributing/roadmap/README.md` for the full spec.",
    },
    "zh": {
        "issue_title": "[路线图] 社区任务清单",
        "generated": (
            "> **这是生成文件，请勿直接编辑正文。** "
            "内容由 [`docs/contributing/roadmap/tasks.toml`]"
            "(https://github.com/{repo}/blob/master/docs/contributing/roadmap/tasks.toml) "
            "渲染而来，任何手工修改都会在下次同步时被覆盖。"
        ),
        "intro": "这里列出我们想做的事情，按难度分组。想认领某一条，在评论里说一声。",
        "claim_heading": "## 怎么认领",
        "claim_steps": [
            "在任务旁边链接的 **那个 issue** 下留言，说明你打算怎么做。",
            "维护者会指派给你，任务会带上认领人。",
            "提交 pull request，并把链接贴到那个 issue 里。",
        ],
        "claim_warning": "**动手写代码前请先和我们聊一下。** 对 `easy` 以上的任务，我们更希望先对齐方案，"
                         "而不是 review 一个方向跑偏的大 PR。",
        "claim_recycle": "**14 天没有进展的任务会被回收。** 认领后 14 天内没有 PR 或 draft PR，"
                         "我们会解除指派，任务回到清单里。这不是惩罚，只是避免任务被占着不动。"
                         "卡住了随时说，我们帮你拆小。",
        "no_issue_yet": "如果某个任务还没有对应的 issue，在 **本条** issue 下留言，我们会开一个。",
        "discord": "有问题，或者只是想聊聊？来 [Discord]({discord})。注意：结论要写回 issue，"
                   "不要只留在聊天里 —— 见"
                   "[贡献指南](https://github.com/{repo}/blob/master/"
                   "docs/contributing/README.md#where-to-talk)。",
        "difficulty_heading": "## 难度",
        "difficulty_table_head": "| 等级 | 含义 |",
        "level_name": {"easy": "简单", "medium": "中等", "hard": "困难", "expert": "专家"},
        "empty": "_清单目前是空的。任务会在梳理后出现在这里。_",
        "summary": "当前顶层任务：{summary}{extra}",
        "level_count": "**{n} 个{word}**",
        "joiner": "、",
        "summary_subtasks": "，另有 **{n}** 个子任务。",
        "summary_end": "。",
        "level_word": {"easy": "简单", "medium": "中等", "hard": "困难", "expert": "专家"},
        "nothing_here": "_这里暂时没有内容，等下一次梳理后再看。_",
        "subtasks_heading": "## 按难度重排的子任务",
        "subtasks_intro": "_上面挂在各自伞形条目下的子任务，这里按难度重新分组，方便找到小任务。_",
        "part_of": "属于",
        "umbrella": "伞形条目，{n} 个子任务",
        "no_issue": "_暂无 issue_",
        "assigned": "已指派给 @{owner}",
        "maint_summary": "任务清单格式（维护者用）",
        "maint_body": "任务数据在 `docs/contributing/roadmap/tasks.toml`。新增一条就追加一个 "
                      "`[[tasks]]` 块，然后运行：",
        "maint_footer": "完整规范见 `docs/contributing/roadmap/README.md`。",
    },
}

LANGS = tuple(STRINGS)

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

REQUIRED_FIELDS = ("id", "title", "title_zh", "difficulty", "domain")
OPTIONAL_FIELDS = ("issue", "owner", "note", "note_zh", "parent", "umbrella", "kind")
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
        if "\n" in task["title_zh"]:
            fail(f"{where}: 'title_zh' must be a single line")
        if not any("\u4e00" <= ch <= "\u9fff" for ch in task["title_zh"]):
            warn(f"warning: {task_id}: 'title_zh' contains no Chinese characters. "
                 "Did the English title get pasted into the Chinese field?")

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

        note_zh = task.get("note_zh")
        if note is not None and note_zh is None:
            fail(f"{where}: 'note' is set, so 'note_zh' is required too. "
                 "The two must be kept in sync.")
        if note is None and note_zh is not None:
            fail(f"{where}: 'note_zh' is set but 'note' is not. "
                 "Provide both or neither.")
        if note_zh is not None:
            if not isinstance(note_zh, str) or not note_zh.strip():
                fail(f"{where}: 'note_zh' must be a non-empty string when present")
            if "\n" in note_zh:
                fail(f"{where}: 'note_zh' must be a single line")

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


def render(tasks: list[dict], repo: str, states: dict[int, dict],
           lang: str = "en") -> str:
    s = STRINGS[lang]
    top_level = [t for t in tasks if not t.get("parent")]
    children: dict[str, list[dict]] = {}
    for task in tasks:
        if task.get("parent"):
            children.setdefault(task["parent"], []).append(task)

    counts = {level: 0 for level in DIFFICULTIES}
    for task in top_level:
        counts[task["difficulty"]] += 1

    out: list[str] = []
    out.append(f"# {s['issue_title']}")
    out.append("")
    out.append(s["generated"].format(repo=repo))
    out.append("")
    out.append(s["intro"])
    out.append("")

    out.append(s["claim_heading"])
    out.append("")
    for index, step in enumerate(s["claim_steps"], start=1):
        out.append(f"{index}. {step}")
    out.append("")
    out.append(s["claim_warning"])
    out.append("")
    out.append(s["claim_recycle"])
    out.append("")
    out.append(s["no_issue_yet"])
    out.append("")
    out.append(s["discord"].format(discord=DISCORD_URL, repo=repo))
    out.append("")

    out.append(s["difficulty_heading"])
    out.append("")
    out.append(s["difficulty_table_head"])
    out.append("|-------|---------------|")
    for level in DIFFICULTIES:
        out.append(f"| {s['level_name'][level]} | {DIFFICULTY_DESCRIPTION[lang][level]} |")
    out.append("")

    if not top_level:
        out.append(s["empty"])
        out.append("")
    else:
        summary = s["joiner"].join(
            s["level_count"].format(n=counts[level], word=s["level_word"][level])
            for level in DIFFICULTIES
        )
        extra = (s["summary_subtasks"].format(n=len(tasks) - len(top_level))
                 if children else s["summary_end"])
        out.append(s["summary"].format(summary=summary, extra=extra))
        out.append("")

    for level in DIFFICULTIES:
        grouped = [t for t in top_level if t["difficulty"] == level]
        if not grouped and not top_level:
            continue
        out.append(f"## {DIFFICULTY_HEADING[lang][level]} ({len(grouped)})")
        out.append("")
        if not grouped:
            out.append(s["nothing_here"])
            out.append("")
            continue
        for task in grouped:
            out.append(render_task_line(task, repo, states, children, lang=lang))
        out.append("")

    if children:
        out.append(s["subtasks_heading"])
        out.append("")
        out.append(s["subtasks_intro"])
        out.append("")
        for level in DIFFICULTIES:
            in_level = [t for t in tasks if t.get("parent")
                        and t["difficulty"] == level]
            if not in_level:
                continue
            out.append(f"**{s['level_name'][level]}**")
            out.append("")
            for task in in_level:
                parent = task["parent"]
                number = task.get("issue")
                link = (f" [#{number}](https://github.com/{repo}/issues/{number})"
                        if number is not None else "")
                out.append(f"- **{task['id']}** {localized(task, 'title', lang)}{link}"
                           f" - {s['part_of']} `{parent}`")
            out.append("")

    out.append("<details>")
    out.append(f"<summary>{s['maint_summary']}</summary>")
    out.append("")
    out.append(s["maint_body"])
    out.append("")
    out.append("```bash")
    out.append("python3 scripts/roadmap/render.py --check")
    out.append("python3 scripts/roadmap/sync-issue.py          # dry run")
    out.append("python3 scripts/roadmap/sync-issue.py --apply  # publish")
    out.append("```")
    out.append("")
    out.append(s["maint_footer"])
    out.append("")
    out.append("</details>")

    return "\n".join(out) + "\n"


def localized(task: dict, field: str, lang: str) -> str:
    """Return the field in the requested language, falling back to English.

    The validator guarantees the _zh variant exists for every task, so the
    fallback only matters for tasks written before the field was introduced.
    """
    if lang == "en":
        return task[field]
    return task.get(f"{field}_zh") or task[field]


def render_task_line(task: dict, repo: str, states: dict[int, dict],
                     children: dict[str, list[dict]] | None = None,
                     depth: int = 0, lang: str = "en") -> str:
    s = STRINGS[lang]
    number = task.get("issue")
    if number is not None and states.get(number, {}).get("state") == "CLOSED":
        box = "x"
    else:
        box = " "

    indent = "  " * depth
    parts = [f"{indent}- [{box}] **{task['id']}** {localized(task, 'title', lang)}"]

    if task.get("kind") == "research":
        parts.append("`research`")

    if number is not None:
        parts.append(f"[#{number}](https://github.com/{repo}/issues/{number})")
    elif depth == 0:
        parts.append(s["no_issue"])

    parts.append(DIFFICULTY_BADGE[task["difficulty"]])
    parts.append(f"`{task['domain']}`")

    owner = resolve_owner(task, states)
    if owner:
        parts.append(s["assigned"].format(owner=owner))

    kids = (children or {}).get(task["id"], [])
    if kids:
        parts.append(f"_({s['umbrella'].format(n=len(kids))})_")

    line = " - ".join(parts)

    note = localized(task, "note", lang) if task.get("note") else None
    if note:
        line += f"\n{indent}  <br/>_{note}_"

    if kids:
        for child in kids:
            line += "\n" + render_task_line(child, repo, states, None, depth + 1, lang)

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
    parser.add_argument("--lang", default="en", choices=LANGS,
                        help="language of the rendered output (the public issue is always en)")
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

    body = render(tasks, args.repo, states, lang=args.lang)

    if args.out:
        Path(args.out).write_text(body, encoding="utf-8")
        warn(f"ok: {len(tasks)} tasks rendered ({args.lang}) to {args.out}")
    else:
        sys.stdout.write(body)

    return 0


if __name__ == "__main__":
    sys.exit(main())
