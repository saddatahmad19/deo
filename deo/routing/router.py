"""
deo/routing/router.py

Per-turn decision router.

For every user turn Deo decides:
  - category  — what kind of engineering work this is (question, quick_edit,
                feature, refactor, debug, review, tests, docs, research)
  - preset    — FreeLLMAPI routing preset (smartest/fastest/balanced/reliable),
                which maps to a model id such as auto:smart
  - profile   — which Deo agent profile (subagent set) fits the category
  - harnesses — which external harnesses are best suited, as hints

Signals, in order:
  1. Local heuristics (free, deterministic, always available).
  2. FreeLLMAPI decision route (POST /v1/decisions) — one call per turn that
     answers both the category and preset questions with probabilities. Used
     when its top probability clears `min_confidence`.
  3. Learned outcomes — a Thompson-sampling bandit over presets per category
     (deo.learning) nudges the preset toward what has actually worked here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from deo.routing.freellmapi import PRESET_CRITERIA, PRESETS, DecisionError, ask_decisions


@dataclass(frozen=True)
class CategorySpec:
    description: str
    preset: str
    profile: str
    task_type: str           # FreeLLMAPI X-FreeLLM-Task-Type
    harnesses: tuple[str, ...]


CATEGORIES: dict[str, CategorySpec] = {
    "question": CategorySpec(
        "Explain or answer a question about code or concepts; no files change.",
        "fastest", "fast", "chat", ()),
    "quick_edit": CategorySpec(
        "A small, well-specified change to one or two files (rename, tweak, add a flag).",
        "balanced", "fast", "code", ("aider", "claude")),
    "feature": CategorySpec(
        "Build new functionality that spans several files and needs a plan.",
        "smartest", "default", "code", ("claude", "codex", "omh")),
    "refactor": CategorySpec(
        "Restructure, migrate or rename code across many files without changing behavior.",
        "smartest", "default", "code", ("claude", "omh", "codex")),
    "debug": CategorySpec(
        "Diagnose and fix a bug, failing test, crash or error message.",
        "smartest", "default", "code", ("codex", "omh", "claude")),
    "review": CategorySpec(
        "Review or audit code for correctness, security or quality.",
        "smartest", "review", "code", ("claude", "codex")),
    "tests": CategorySpec(
        "Write or repair automated tests and raise coverage.",
        "reliable", "write-tests", "code", ("aider", "codex")),
    "docs": CategorySpec(
        "Write or update documentation, READMEs, docstrings or comments.",
        "fastest", "docs", "chat", ("aider",)),
    "research": CategorySpec(
        "Explore and understand a large codebase or long context before acting.",
        "smartest", "fast", "code", ("reasonix", "dsh", "hermes")),
}


@dataclass
class RouteDecision:
    category: str
    preset: str
    profile: str
    task_type: str
    harnesses: tuple[str, ...]
    confidence: float
    source: str                       # heuristic | decision | decision+learned | ...
    notes: list[str] = field(default_factory=list)

    def summary(self, presets: dict[str, str]) -> str:
        model = presets.get(self.preset, self.preset)
        return (
            f"route: {self.category} · {self.preset} ({model}) · profile {self.profile} "
            f"· {self.source} {self.confidence:.2f}"
        )


class PresetLearner(Protocol):
    def choose_preset(self, category: str, proposed: str, candidates: tuple[str, ...]) -> tuple[str, str]:
        """Return (preset, reason)."""


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

_RULES: list[tuple[str, re.Pattern[str], float]] = [
    ("review", re.compile(r"\b(review|audit|security|vulnerab|code smell|critique)\w*", re.I), 3.0),
    ("tests", re.compile(r"\b(unit tests?|write tests?|add tests?|test coverage|coverage|pytest|jest|vitest|spec files?)\b", re.I), 3.0),
    ("debug", re.compile(r"\b(bug|error|exception|traceback|stack ?trace|crash\w*|fails?|failing|broken|regression|doesn'?t work|not working|fix)\b", re.I), 2.0),
    ("refactor", re.compile(r"\b(refactor\w*|restructur\w*|migrat\w*|clean ?up|extract|decouple|modulari[sz]e|rename .* across)\b", re.I), 2.5),
    ("docs", re.compile(r"\b(document\w*|readme|docstrings?|changelog|comments?|docs)\b", re.I), 2.0),
    ("research", re.compile(r"\b(investigate|explore|understand|map out|walk me through|architecture overview|how is .* structured|where (is|are)|find all)\b", re.I), 2.0),
    ("feature", re.compile(r"\b(implement|add|build|create|support|introduce|integrate|new (endpoint|feature|command|page))\b", re.I), 1.5),
    ("quick_edit", re.compile(r"\b(rename|typo|tweak|bump|change .* to|set .* to|one[- ]liner|small change)\b", re.I), 2.0),
    ("question", re.compile(r"^\s*(what|why|how|when|which|who|can you explain|explain|is it|does|do)\b|\?\s*$", re.I), 1.5),
]

_CODE_MARKERS = re.compile(r"```|^\s*\$ \w|\b(def|class|function|const|let|import)\s+\w|\w+\(\)|\w+\.\w+\(|[\w/]+\.\w{1,4}\b", re.M)


def heuristic_route(prompt: str) -> RouteDecision:
    text = prompt.strip()
    scores: dict[str, float] = {}
    for category, pattern, weight in _RULES:
        hits = len(pattern.findall(text))
        if hits:
            scores[category] = scores.get(category, 0.0) + weight * min(hits, 3)

    words = len(text.split())
    if words <= 25 and scores.get("feature") and not scores.get("refactor"):
        scores["quick_edit"] = scores.get("quick_edit", 0.0) + 1.0
    if words >= 120:
        scores["feature"] = scores.get("feature", 0.0) + 1.0
    # Questions that also ask to change code are not plain questions.
    if "question" in scores and any(k in scores for k in ("feature", "debug", "refactor", "quick_edit")):
        scores["question"] *= 0.4

    if not scores:
        category, confidence = ("question" if words <= 12 else "feature"), 0.3
    else:
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        category = ranked[0][0]
        total = sum(scores.values())
        confidence = round(ranked[0][1] / total, 2) if total else 0.3

    spec = CATEGORIES[category]
    task_type = spec.task_type
    if task_type == "chat" and _CODE_MARKERS.search(text):
        task_type = "code"
    return RouteDecision(category, spec.preset, spec.profile, task_type, spec.harnesses,
                         confidence, "heuristic")


# ---------------------------------------------------------------------------
# Decision router
# ---------------------------------------------------------------------------

class DecisionRouter:
    """Combines heuristics, FreeLLMAPI decisions and learned outcomes."""

    def __init__(self, routing_config: Any, learner: PresetLearner | None = None) -> None:
        self.cfg = routing_config
        self.learner = learner
        self._cache: dict[str, RouteDecision] = {}
        self.decisions_ok: bool | None = None   # None until first attempt

    def _ask_gateway(self, prompt: str) -> dict[str, Any] | None:
        questions = {
            "category": {
                "type": "choice",
                "instructions": (
                    "You route requests for a multi-agent coding harness. "
                    "Which kind of software-engineering work is this request?"
                ),
                "criteria": {k: v.description for k, v in CATEGORIES.items()},
            },
            "preset": {
                "type": "choice",
                "instructions": (
                    "Pick the routing strategy that serves this request best. Judge by how hard "
                    "the request is and what the person needs from the answer, not by its topic."
                ),
                "criteria": dict(PRESET_CRITERIA),
            },
        }
        try:
            answers = ask_decisions(
                f"Request: {prompt[:1600]}",
                questions,
                model=self.cfg.decision_model,
                timeout=self.cfg.decision_timeout,
            )
            self.decisions_ok = True
            return answers
        except DecisionError:
            self.decisions_ok = False
            return None

    def decide(self, prompt: str) -> RouteDecision:
        key = prompt.strip()[:2000]
        if key in self._cache:
            cached = self._cache[key]
            return RouteDecision(**{**cached.__dict__, "source": cached.source + "+cache"})

        decision = heuristic_route(prompt)

        if self.cfg.enabled and self.cfg.use_decisions and self.decisions_ok is not False:
            answers = self._ask_gateway(prompt)
            if answers:
                cat = answers.get("category")
                pre = answers.get("preset")
                used = []
                if cat and cat.choice in CATEGORIES and cat.top_probability >= self.cfg.min_confidence:
                    spec = CATEGORIES[cat.choice]
                    if cat.choice != decision.category:
                        decision.notes.append(f"heuristic said {decision.category}")
                    decision.category = cat.choice
                    decision.preset = spec.preset
                    decision.profile = spec.profile
                    decision.harnesses = spec.harnesses
                    if spec.task_type == "code":
                        decision.task_type = "code"
                    decision.confidence = round(cat.top_probability, 2)
                    used.append("category")
                if pre and pre.choice in PRESETS and pre.top_probability >= self.cfg.min_confidence:
                    decision.preset = pre.choice
                    decision.confidence = round(max(decision.confidence, pre.top_probability), 2)
                    used.append("preset")
                if used:
                    decision.source = "decision"

        if self.cfg.adaptive and self.learner is not None:
            preset, reason = self.learner.choose_preset(decision.category, decision.preset, PRESETS)
            if preset != decision.preset:
                decision.notes.append(f"learned: {reason}")
                decision.preset = preset
                decision.source += "+learned"

        self._cache[key] = decision
        if len(self._cache) > 200:
            self._cache.pop(next(iter(self._cache)))
        return decision
