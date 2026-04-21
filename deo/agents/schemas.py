"""Pydantic response schemas for deo subagents."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


class PlanStep(BaseModel):
    step: str
    file: str
    action: Literal["create", "modify", "delete"]


class PlanResult(BaseModel):
    steps: list[PlanStep]
    files_affected: list[str]
    risks: list[str]
    estimated_complexity: Literal["low", "medium", "high"]


class FileChange(BaseModel):
    filepath: str
    action: Literal["created", "modified", "deleted"]
    summary: str


class ImplementResult(BaseModel):
    files_changed: list[FileChange]
    summary: str
    caveats: list[str]


class ReviewIssue(BaseModel):
    severity: Literal["critical", "warning", "suggestion"]
    file: str
    line: int | None = None
    message: str
    suggestion: str


class ReviewResult(BaseModel):
    issues: list[ReviewIssue]
    approved: bool
    verdict: str


class RunResult(BaseModel):
    test_files: list[str]
    coverage_estimate: str
    all_pass: bool
    failures: list[str]


class DocResult(BaseModel):
    files_updated: list[str]
    summary: str
