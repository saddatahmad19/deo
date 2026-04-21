"""Subagent factory functions for deo."""
from __future__ import annotations
from typing import Any

from deo.agents.schemas import (
    PlanResult, ImplementResult, ReviewResult, RunResult, DocResult,
)
from deo.prompts.templates import (
    PLANNER_PROMPT,
    IMPLEMENTER_PROMPT,
    REVIEWER_PROMPT,
    TESTER_PROMPT,
    DOCUMENTER_PROMPT,
)
from deo.tools.chroma_tools import CHROMA_TOOLS
from deo.tools.git_tools import GIT_TOOLS


def build_planner_subagent(model: str | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "name": "planner-agent",
        "description": (
            "Use for ANY task that touches 2+ files, requires architecture decisions, "
            "or needs a sequenced plan before code is written. "
            "Call this FIRST before implementer. Returns a structured PlanResult."
        ),
        "system_prompt": PLANNER_PROMPT,
        "response_format": PlanResult,
        "tools": [*CHROMA_TOOLS, *GIT_TOOLS],
    }
    if model:
        config["model"] = model
    return config


def build_implementer_subagent(model: str | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "name": "implementer-agent",
        "description": (
            "Use to write or refactor code, following a plan from planner-agent. "
            "Always call planner-agent first for multi-file tasks. "
            "Returns a structured ImplementResult with files changed."
        ),
        "system_prompt": IMPLEMENTER_PROMPT,
        "response_format": ImplementResult,
        "tools": [*CHROMA_TOOLS, *GIT_TOOLS],
    }
    if model:
        config["model"] = model
    return config


def build_reviewer_subagent(model: str | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "name": "reviewer-agent",
        "description": (
            "Use AFTER implementer-agent completes. Reviews changed files for "
            "correctness, security, and regressions. Returns ReviewResult with "
            "approved=True/False and a list of issues by severity."
        ),
        "system_prompt": REVIEWER_PROMPT,
        "response_format": ReviewResult,
        "tools": [*CHROMA_TOOLS, *GIT_TOOLS],
    }
    if model:
        config["model"] = model
    return config


def build_tester_subagent(model: str | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "name": "tester-agent",
        "description": (
            "Use AFTER implementer-agent completes. Writes and runs tests for changed "
            "code. Run in parallel with reviewer-agent. Returns RunResult with "
            "all_pass=True/False and list of failures."
        ),
        "system_prompt": TESTER_PROMPT,
        "response_format": RunResult,
        "tools": [*CHROMA_TOOLS, *GIT_TOOLS],
    }
    if model:
        config["model"] = model
    return config


def build_documenter_subagent(model: str | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "name": "documenter-agent",
        "description": (
            "Use ONLY after reviewer-agent returns approved=True. Updates docstrings, "
            "README, and AGENTS.md. Returns DocResult with list of files updated."
        ),
        "system_prompt": DOCUMENTER_PROMPT,
        "response_format": DocResult,
        "tools": [*CHROMA_TOOLS],
    }
    if model:
        config["model"] = model
    return config


def build_subagents(
    enable_planner: bool = True,
    enable_implementer: bool = True,
    enable_reviewer: bool = True,
    enable_tester: bool = True,
    enable_documenter: bool = False,
    planner_model: str | None = None,
    implementer_model: str | None = None,
    reviewer_model: str | None = None,
    tester_model: str | None = None,
    documenter_model: str | None = None,
) -> list[dict[str, Any]]:
    """Build active subagent list based on config toggles and per-role models."""
    subagents = []
    if enable_planner:
        subagents.append(build_planner_subagent(planner_model))
    if enable_implementer:
        subagents.append(build_implementer_subagent(implementer_model))
    if enable_reviewer:
        subagents.append(build_reviewer_subagent(reviewer_model))
    if enable_tester:
        subagents.append(build_tester_subagent(tester_model))
    if enable_documenter:
        subagents.append(build_documenter_subagent(documenter_model))
    return subagents
