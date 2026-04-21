"""
deo/init/agents_md.py

Analyse a repository and produce:
  - AGENTS.md content (a structured project memory document)
  - The list of file paths that were important enough to include
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------

_PRIORITY_NAMES = {
    "pyproject.toml", "package.json", "Cargo.toml", "go.mod",
    "pom.xml", "build.gradle", "setup.py", "setup.cfg",
    "Makefile", "Dockerfile", "docker-compose.yml",
    "README.md", "AGENTS.md", ".deo.toml",
}

_SKIP_DIRS = {
    ".git", ".deo", ".venv", "venv", "node_modules", "__pycache__",
    ".mypy_cache", ".ruff_cache", "dist", "build", ".egg-info",
    "site-packages",
}

_USEFUL_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx",
    ".go", ".rs", ".java", ".kt",
    ".md", ".toml", ".yaml", ".yml",
    ".sql", ".graphql", ".proto",
}

_MAX_TOTAL_CHARS = 60_000
_MAX_FILE_CHARS = 4_000


def _is_skip_dir(path: Path) -> bool:
    return any(part in _SKIP_DIRS for part in path.parts)


def _score_file(path: Path, root: Path) -> int:
    """Higher score = more architecturally important."""
    rel = path.relative_to(root)
    score = 0
    name = path.name.lower()

    if name in _PRIORITY_NAMES:
        score += 100

    if name in ("main.py", "app.py", "cli.py", "index.ts", "index.js", "server.py"):
        score += 80

    depth = len(rel.parts)
    score += max(0, 20 - depth * 4)

    try:
        size = path.stat().st_size
        if size < 5_000:
            score += 10
        elif size > 50_000:
            score -= 20
    except OSError:
        pass

    return score


def select_important_files(root: Path) -> list[Path]:
    """
    Walk the repo and return a ranked list of architecturally important files.
    Budget: _MAX_TOTAL_CHARS total across all returned files.
    """
    candidates: list[tuple[int, Path]] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _is_skip_dir(path.relative_to(root)):
            continue
        if path.suffix not in _USEFUL_EXTENSIONS and path.name not in _PRIORITY_NAMES:
            continue
        score = _score_file(path, root)
        candidates.append((score, path))

    candidates.sort(key=lambda x: x[0], reverse=True)

    selected: list[Path] = []
    total_chars = 0
    for _, path in candidates:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if total_chars + min(size, _MAX_FILE_CHARS) > _MAX_TOTAL_CHARS:
            break
        selected.append(path)
        total_chars += min(size, _MAX_FILE_CHARS)

    return selected


# ---------------------------------------------------------------------------
# AGENTS.md generation
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a senior software engineer producing a concise project memory document
called AGENTS.md. This file is read by AI coding agents at the start of every
session to orient them quickly.

Your output must follow this exact structure:

# Project: <name>

## Purpose
One or two sentences describing what the project does.

## Architecture
Bullet list of the key components and how they interact.
Be concrete — name actual files and classes where relevant.

## Key Files
A table: | File | Responsibility |
List the 8-15 most important files only.

## Conventions
Bullet list of coding conventions, naming patterns, and project-specific rules
an agent must know to contribute correctly.

## External Dependencies
Bullet list of the most important external services, APIs, or libraries.

## Agent Guidance
3-5 bullet points telling a coding agent what to be careful about in this repo.
Include gotchas, footguns, and non-obvious constraints.

Be concise. Total length: 400-700 words. No filler.
"""


def _build_user_message(root: Path, files: list[Path]) -> str:
    repo_name = root.name
    lines = [f"Repository: `{repo_name}`\n\nFile contents:\n"]
    for path in files:
        rel = path.relative_to(root)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        truncated = content[:_MAX_FILE_CHARS]
        if len(content) > _MAX_FILE_CHARS:
            truncated += "\n... (truncated)"
        lines.append(f"### {rel}\n```\n{truncated}\n```\n")
    return "\n".join(lines)


def generate_agents_md(root: Path, model_string: str = "anthropic:claude-haiku-4-5-20251001") -> tuple[str, list[Path]]:
    """
    Analyse the repository at `root` and return:
      - agents_md_text: full AGENTS.md content
      - important_files: files that were fed to the LLM

    Uses a fast/cheap model by default — override with `model_string`.
    """
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage, SystemMessage

    important_files = select_important_files(root)
    user_message = _build_user_message(root, important_files)

    provider, _, name = model_string.partition(":")
    llm = init_chat_model(model=name, model_provider=provider, temperature=0.2)

    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    agents_md_text = response.content if isinstance(response.content, str) else str(response.content)
    return agents_md_text, important_files
