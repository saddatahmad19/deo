"""
deo/tools/harness_tools.py

Tools that let Deo's agents delegate to external harnesses (Aider, Claude Code,
Codex, Hermes / Oh My Hermes, Reasonix, DeepSeek Harness, ...) and feed the
self-improvement loop (lessons, skills).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from deo.harnesses.registry import HARNESSES, all_statuses, harness_status, run_harness

_ctx: dict[str, Any] = {"root": Path.cwd(), "config": None, "route_state": None, "learner": None}


def configure_harness_tools(root: Path, harness_config: Any, route_state: Any = None,
                            learner: Any = None) -> None:
    _ctx.update(root=root, config=harness_config, route_state=route_state, learner=learner)


def _cfg(attr: str, default: Any) -> Any:
    return getattr(_ctx["config"], attr, default) if _ctx["config"] is not None else default


@tool
def list_harnesses() -> str:
    """
    List external coding harnesses available for delegation, what each is best at,
    whether it is ready, and its past success rate in this project.
    """
    learner = _ctx["learner"]
    lines = []
    for st in all_statuses(_cfg("commands", {}), _cfg("disabled", [])):
        mark = "✅ ready" if st.ready else f"⛔ {st.reason}"
        rate = ""
        if learner is not None and st.spec.id in learner.harness_stats:
            ok, n = learner.harness_stats[st.spec.id]
            rate = f"  [{ok}/{n} ok]"
        lines.append(f"- {st.spec.id} ({st.spec.name}): {mark}{rate}\n    best for: "
                     f"{', '.join(st.spec.best_for)} — {st.spec.strengths}")
    return "\n".join(lines)


@tool
def delegate_to_harness(harness: str, task: str, files: str = "") -> str:
    """
    Delegate a self-contained coding task to an external harness running on the same
    FreeLLMAPI gateway. Use the harness whose strengths match the task:
    aider for one-shot targeted edits, claude/codex for agentic multi-file work and
    debugging, omh (Oh My Hermes ulw-loop) for parallel build-until-tests-pass,
    reasonix/dsh for long-context exploration, hermes for research.
    Always verify the result yourself afterwards (git_diff, run tests).

    Args:
        harness: Harness id from list_harnesses (e.g. "aider", "claude", "codex", "omh").
        task: Complete, standalone instructions including acceptance criteria.
        files: Optional comma-separated file paths to focus on (used by aider).
    """
    if not _cfg("enabled", True):
        return "Harness delegation is disabled in [harnesses]."
    spec = HARNESSES.get(harness.strip().lower())
    if spec is None:
        return f"Unknown harness '{harness}'. Options: {', '.join(HARNESSES)}"
    status = harness_status(spec, _cfg("commands", {}), _cfg("disabled", []))
    if not status.ready:
        return f"Harness '{spec.id}' is not ready: {status.reason}. {spec.note}".strip()
    rs = _ctx["route_state"]
    model = rs.model_for("implementer") if rs is not None else "auto"
    file_list = [f.strip() for f in files.split(",") if f.strip()]
    run = run_harness(status, task, model=model, cwd=_ctx["root"],
                      timeout=int(_cfg("timeout", 900)), files=file_list)
    if _ctx["learner"] is not None:
        _ctx["learner"].observe_harness(spec.id, run.ok, run.duration)
    tail = run.output[-6000:]
    verdict = "timed out" if run.timed_out else f"exit {run.returncode}"
    return f"{spec.name} finished ({verdict}, {run.duration:.0f}s).\n--- output (tail) ---\n{tail}"


@tool
def record_lesson(lesson: str, tags: str = "") -> str:
    """
    Save a short, specific, reusable lesson about THIS repository (a command that
    works, a pitfall, a convention) so future sessions recall it automatically.
    """
    learner = _ctx["learner"]
    if learner is None:
        return "Learning is disabled."
    saved = learner.lessons.add(lesson, tags=[t.strip() for t in tags.split(",") if t.strip()])
    return "Lesson saved." if saved else "Lesson too short; not saved."


@tool
def recall_lessons(query: str) -> str:
    """Recall lessons learned in earlier sessions relevant to a query."""
    learner = _ctx["learner"]
    if learner is None:
        return "Learning is disabled."
    hits = learner.lessons.recall(query, k=6)
    return "\n".join(f"- {h.text}" for h in hits) or "No relevant lessons yet."


@tool
def save_skill(name: str, description: str, instructions: str) -> str:
    """
    After completing a complex, repeatable workflow successfully, save it as a skill
    (Deep Agents SKILL.md) that future sessions load. name: lowercase-hyphenated.
    """
    from deo.learning.experience import save_skill as _save

    try:
        path = _save(Path(_ctx["root"]) / ".deo" / "skills", name, description, instructions)
    except (ValueError, FileExistsError) as exc:
        return f"Skill not saved: {exc}"
    return f"Skill saved to {path.relative_to(_ctx['root'])}; it loads in the next session."


HARNESS_TOOLS = [list_harnesses, delegate_to_harness]
LEARNING_TOOLS = [record_lesson, recall_lessons, save_skill]
