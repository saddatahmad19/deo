"""
deo/agents/subagents.py

Subagent definitions for Forge's multi-agent coding pipeline.

Architecture:
  orchestrator
    ├── planner-agent     — task breakdown, architecture planning
    ├── implementer-agent — code writing, refactoring
    ├── reviewer-agent    — code review, bug finding
    ├── tester-agent      — test writing
    └── documenter-agent  — documentation (optional)

Each subagent is a dict matching DeepAgents' SubAgent schema:
  {
    "name": str,
    "description": str,     # Used by orchestrator to decide when to delegate
    "system_prompt": str,
    "tools": list,
    "model": str (optional)
  }

Subagents inherit the orchestrator's filesystem backend and get access
to the `task` tool to spawn their own sub-subagents if needed.
"""

from __future__ import annotations

from typing import Any

from deo.prompts.templates import (
    DOCUMENTER_PROMPT,
    IMPLEMENTER_PROMPT,
    PLANNER_PROMPT,
    REVIEWER_PROMPT,
    TESTER_PROMPT,
)
from deo.tools.chroma_tools import CHROMA_TOOLS
from deo.tools.git_tools import GIT_TOOLS


def build_planner_subagent(model: str | None = None) -> dict[str, Any]:
    """
    Planner subagent: decomposes complex tasks into step-by-step implementation plans.

    Best for:
      - Multi-file changes
      - Architecture decisions
      - Understanding codebase before making changes
    """
    config: dict[str, Any] = {
        "name": "planner-agent",
        "description": (
            "Technical planning agent. Use when a task requires architecture decisions, "
            "touches 3+ files, or needs careful decomposition before implementation. "
            "Produces a detailed implementation plan the implementer can follow."
        ),
        "system_prompt": PLANNER_PROMPT,
        "tools": [
            *CHROMA_TOOLS,      # semantic_search, add_to_index, index_status, find_file
            *GIT_TOOLS,         # git_status, git_diff, git_log, git_branch
        ],
    }
    if model:
        config["model"] = model
    return config


def build_implementer_subagent(model: str | None = None) -> dict[str, Any]:
    """
    Implementer subagent: writes, edits, and refactors code.

    Best for:
      - New feature implementation
      - Refactoring existing code
      - Following an implementation plan from the planner
    """
    config: dict[str, Any] = {
        "name": "implementer-agent",
        "description": (
            "Code implementation agent. Use when you need to write substantial new code, "
            "implement a feature from a plan, or do a non-trivial refactor. "
            "Writes clean, idiomatic, fully-functional code."
        ),
        "system_prompt": IMPLEMENTER_PROMPT,
        "tools": [
            *CHROMA_TOOLS,
            *GIT_TOOLS,
        ],
    }
    if model:
        config["model"] = model
    return config


def build_reviewer_subagent(model: str | None = None) -> dict[str, Any]:
    """
    Reviewer subagent: reviews code for correctness, security, performance.

    Best for:
      - Post-implementation code review
      - Security audits
      - Performance analysis
      - Finding bugs in existing code
    """
    config: dict[str, Any] = {
        "name": "reviewer-agent",
        "description": (
            "Code review agent. Use after implementation to check for bugs, security issues, "
            "performance problems, and style violations. Produces a structured review with "
            "severity labels (critical/warning/suggestion) and concrete fixes."
        ),
        "system_prompt": REVIEWER_PROMPT,
        "tools": [
            *CHROMA_TOOLS,
            *GIT_TOOLS,
        ],
    }
    if model:
        config["model"] = model
    return config


def build_tester_subagent(model: str | None = None) -> dict[str, Any]:
    """
    Tester subagent: writes unit and integration tests.

    Best for:
      - Writing tests for new code
      - Improving test coverage
      - Writing test strategies
    """
    config: dict[str, Any] = {
        "name": "tester-agent",
        "description": (
            "Test engineering agent. Use when you need to write tests for new or existing code, "
            "improve test coverage, or design a testing strategy. Detects existing test frameworks "
            "and follows project conventions. Runs tests after writing them."
        ),
        "system_prompt": TESTER_PROMPT,
        "tools": [
            *CHROMA_TOOLS,
            *GIT_TOOLS,
        ],
    }
    if model:
        config["model"] = model
    return config


def build_documenter_subagent(model: str | None = None) -> dict[str, Any]:
    """
    Documenter subagent: writes docs, docstrings, and README content.

    Best for:
      - Adding/updating docstrings
      - Writing README sections
      - Documenting APIs
    """
    config: dict[str, Any] = {
        "name": "documenter-agent",
        "description": (
            "Technical documentation agent. Use when you need to write or update docstrings, "
            "README files, API documentation, or inline comments. "
            "Follows project documentation style."
        ),
        "system_prompt": DOCUMENTER_PROMPT,
        "tools": [
            *CHROMA_TOOLS,
        ],
    }
    if model:
        config["model"] = model
    return config


def build_subagents(
    model_string: str | None = None,
    enable_planner: bool = True,
    enable_implementer: bool = True,
    enable_reviewer: bool = True,
    enable_tester: bool = True,
    enable_documenter: bool = False,
) -> list[dict[str, Any]]:
    """
    Build the active subagent list based on config toggles.

    Args:
        model_string: Model override for subagents (e.g. "anthropic:claude-haiku-4-5-20251001"
                      for cheaper/faster subagents). If None, inherits orchestrator's model.
        enable_*: Toggle individual subagents on/off.

    Returns:
        List of subagent dicts ready to pass to create_deep_agent(subagents=...).
    """
    subagents = []

    if enable_planner:
        subagents.append(build_planner_subagent(model_string))

    if enable_implementer:
        subagents.append(build_implementer_subagent(model_string))

    if enable_reviewer:
        subagents.append(build_reviewer_subagent(model_string))

    if enable_tester:
        subagents.append(build_tester_subagent(model_string))

    if enable_documenter:
        subagents.append(build_documenter_subagent(model_string))

    return subagents
