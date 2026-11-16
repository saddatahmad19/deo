"""Custom middleware composition for Deo Deep Agents."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ContextT, ModelRequest, ResponseT
from langchain_core.messages import ContentBlock, SystemMessage

from deepagents.middleware import create_summarization_tool_middleware


class PromptAppendMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Append high-signal operational guidance to each model request."""

    def __init__(self, prompt_block: str) -> None:
        self._prompt_block = prompt_block.strip()

    def modify_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        system_message = _append_to_system_message(request.system_message, self._prompt_block)
        return request.override(system_message=system_message)


def _append_to_system_message(system_message: SystemMessage | None, text: str) -> SystemMessage:
    """Append text as a new content block on the system message."""
    new_content: list[ContentBlock] = list(system_message.content_blocks) if system_message else []
    if new_content:
        text = f"\n\n{text}"
    new_content.append({"type": "text", "text": text})
    return SystemMessage(content_blocks=new_content)


_ORCHESTRATOR_RUNTIME_GUIDANCE = """\
## Deo Runtime Guidance
- Workspace root: `{root_dir}`
- This is a terminal coding environment. Prefer concrete repository operations over abstract advice.
- Use the `task` tool for complex isolated work; run independent tasks in parallel.
- Before final output, ensure verification commands were attempted and their outcomes are reported.
- Keep user-facing summaries concise and operational: changes, verification, residual risk.
"""


_SUBAGENT_RUNTIME_GUIDANCE = {
    "planner": """\
## Planner Runtime Guidance
- Produce plans that are directly executable with filesystem and shell tools.
- Include verification checkpoints and command-level validation, not only conceptual steps.
""",
    "implementer": """\
## Implementer Runtime Guidance
- Favor minimal, high-confidence edits over large rewrites.
- After edits, run targeted verification commands and include concrete results.
""",
    "reviewer": """\
## Reviewer Runtime Guidance
- Prioritize concrete, reproducible findings with file/line references.
- Focus on behavior and risk; avoid stylistic nitpicks unless they affect reliability.
""",
    "tester": """\
## Tester Runtime Guidance
- Add tests that prove behavior changed as intended and guard against regressions.
- Report both what is covered and what remains uncovered.
""",
    "documenter": """\
## Documenter Runtime Guidance
- Keep docs synchronized with actual commands, flags, and runtime behavior.
- Prefer concise, example-driven updates over broad prose.
""",
}


def build_orchestrator_middleware(
    model: str | Any,
    backend: Any,
    root_dir: str,
) -> list[AgentMiddleware[Any, Any, Any]]:
    """Compose middleware stack additions for the top-level orchestrator."""
    return [
        create_summarization_tool_middleware(model, backend),
        PromptAppendMiddleware(_ORCHESTRATOR_RUNTIME_GUIDANCE.format(root_dir=root_dir)),
    ]


def build_subagent_middleware(role: str, root_dir: str) -> list[AgentMiddleware[Any, Any, Any]]:
    """Compose middleware stack additions for a specific subagent role."""
    base = _SUBAGENT_RUNTIME_GUIDANCE.get(role, "")
    runtime = f"Workspace root for this task: `{root_dir}`"
    guidance = f"{base}\n\n{runtime}".strip()
    return [PromptAppendMiddleware(guidance)] if guidance else []
