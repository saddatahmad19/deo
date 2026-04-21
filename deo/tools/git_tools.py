"""
deo/tools/git_tools.py

Git-aware tools for the Forge agent system.

These give agents situational awareness of:
  - What files have changed (git status)
  - What the changes look like (git diff)
  - Recent commit history
  - Current branch
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from langchain_core.tools import tool


def _run_git(args: list[str], cwd: str | Path) -> tuple[str, str, int]:
    """Run a git command and return (stdout, stderr, returncode)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.stdout, result.stderr, result.returncode
    except FileNotFoundError:
        return "", "git not found in PATH", 1
    except subprocess.TimeoutExpired:
        return "", "git command timed out", 1


_root_dir: Path = Path.cwd()


def configure_git_tools(root_dir: Path) -> None:
    """Set the working directory for git tools."""
    global _root_dir
    _root_dir = root_dir


@tool
def git_status() -> str:
    """
    Show the working tree status (like `git status`).

    Use this to understand what files have been modified, added, or deleted.
    Helpful before making changes to understand the current state.

    Returns:
        Git status output showing changed, staged, and untracked files.
    """
    stdout, stderr, code = _run_git(["status", "--short", "--branch"], _root_dir)

    if code != 0:
        if "not a git repository" in stderr:
            return "⚠️  Not a git repository."
        return f"❌ git status failed: {stderr}"

    if not stdout.strip():
        return "✅ Working tree clean — no changes."

    return f"📋 Git Status:\n```\n{stdout.rstrip()}\n```"


@tool
def git_diff(
    filepath: str = "",
    staged: bool = False,
    stat_only: bool = False,
) -> str:
    """
    Show git diff for understanding recent changes.

    Args:
        filepath: Optional — limit diff to a specific file.
        staged: If True, show staged (cached) changes instead of unstaged.
        stat_only: If True, show only the stat summary (no line-level diff).
                   Use this for a quick overview before looking at full diffs.

    Returns:
        Git diff output.
    """
    args = ["diff"]
    if staged:
        args.append("--cached")
    if stat_only:
        args.append("--stat")
    if filepath:
        args.extend(["--", filepath])

    stdout, stderr, code = _run_git(args, _root_dir)

    if code != 0:
        return f"❌ git diff failed: {stderr}"

    if not stdout.strip():
        label = "staged" if staged else "unstaged"
        target = f" for {filepath}" if filepath else ""
        return f"No {label} changes{target}."

    # Truncate very large diffs
    if len(stdout) > 8000:
        stdout = stdout[:8000] + "\n... [diff truncated — use filepath= to narrow scope]"

    return f"```diff\n{stdout.rstrip()}\n```"


@tool
def git_log(n: int = 10) -> str:
    """
    Show recent commit history.

    Args:
        n: Number of commits to show (default: 10, max: 50).

    Returns:
        Recent commit log with hashes, dates, and messages.
    """
    n = min(n, 50)
    stdout, stderr, code = _run_git(
        ["log", f"-{n}", "--oneline", "--decorate", "--graph"],
        _root_dir
    )

    if code != 0:
        if "not a git repository" in stderr:
            return "⚠️  Not a git repository."
        return f"❌ git log failed: {stderr}"

    if not stdout.strip():
        return "No commits yet."

    return f"📜 Recent commits:\n```\n{stdout.rstrip()}\n```"


@tool
def git_branch() -> str:
    """
    Show current branch and list local branches.

    Returns:
        Current branch name and list of all local branches.
    """
    stdout, stderr, code = _run_git(["branch", "--show-current"], _root_dir)

    if code != 0:
        return f"❌ Failed to get branch: {stderr}"

    current = stdout.strip()

    all_stdout, _, _ = _run_git(["branch"], _root_dir)
    branches = all_stdout.strip() if all_stdout else "(none)"

    return f"🌿 Current branch: {current}\n\nAll branches:\n```\n{branches}\n```"


GIT_TOOLS = [git_status, git_diff, git_log, git_branch]
