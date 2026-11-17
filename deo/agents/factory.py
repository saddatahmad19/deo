"""
deo/agents/factory.py

The central factory for creating the Deo orchestrator agent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend

from deo.agents.middleware import build_orchestrator_middleware
from deo.agents.subagents import build_subagents
from deo.config import DeoConfig
from deo.memory.chroma_manager import ChromaManager
from deo.memory.indexer import CodebaseIndexer
from deo.prompts.templates import ORCHESTRATOR_PROMPT
from deo.tools.chroma_tools import CHROMA_TOOLS, configure_chroma_tools
from deo.tools.git_tools import GIT_TOOLS, configure_git_tools
from deo.tools.harness_tools import HARNESS_TOOLS, LEARNING_TOOLS, configure_harness_tools

_ROLE_BY_AGENT = {
    "planner-agent": "planner", "implementer-agent": "implementer", "reviewer-agent": "reviewer",
    "tester-agent": "tester", "documenter-agent": "documenter",
}


def make_route_state(config: DeoConfig, session_id: str = "") -> Any:
    """Shared per-session routing state, or None when not using FreeLLMAPI routing."""
    if config.model.provider.lower().replace("-", "_") not in ("freellmapi", "freellmapi_openai"):
        return None
    if not config.routing.enabled:
        return None
    from deo.routing.freellmapi import RouteState

    return RouteState(presets=dict(config.routing.presets), role_presets=dict(config.routing.role_presets),
                      default_model=config.model.name, session_id=session_id,
                      compress=config.routing.compress)


def _resolve_subagent_models(config: DeoConfig, subagents: list[dict[str, Any]], route_state: Any) -> None:
    """Turn per-role model strings into model objects.

    Deep Agents resolves string models with init_chat_model, which cannot build
    a FreeLLMAPI model, so FreeLLMAPI ids become RoutedChatOpenAI instances and
    other providers are built through _build_model.
    """
    from copy import deepcopy

    for spec in subagents:
        role = _ROLE_BY_AGENT.get(spec.get("name", ""), spec.get("name", ""))
        raw = spec.get("model")
        if isinstance(raw, str):
            sub = deepcopy(config)
            sub.model.override(raw)
            is_free = sub.model.provider.lower() in ("freellmapi", "freellmapi_openai")
            if is_free and route_state is not None:
                route_state.role_models[role] = sub.model.name
                spec["model"] = _build_model(sub, role=role, route_state=route_state)
            else:
                spec["model"] = _build_model(sub)
        elif raw is None and route_state is not None:
            spec["model"] = _build_model(config, role=role, route_state=route_state)


def _normalize_azure_endpoint_and_version(raw_endpoint: str) -> tuple[str, str | None]:
    """
    Normalize Azure endpoint input for AzureChatOpenAI.

    Users often paste the Foundry "Target URI" (for example:
    https://<resource>.cognitiveservices.azure.com/openai/responses?api-version=...).
    AzureChatOpenAI expects only the resource base endpoint; it appends the path.
    """
    endpoint = raw_endpoint.strip()
    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT must be a valid URL like "
            "https://<resource>.openai.azure.com/ or "
            "https://<resource>.cognitiveservices.azure.com/."
        )

    inferred_api_version = parse_qs(parsed.query).get("api-version", [None])[0]
    normalized_endpoint = f"{parsed.scheme}://{parsed.netloc}/"
    return normalized_endpoint, inferred_api_version


def _build_model(config: DeoConfig, role: str = "orchestrator", route_state: Any = None) -> Any:
    """Instantiate the LLM based on config.model.provider.

    For FreeLLMAPI with a route_state, returns a RoutedChatOpenAI that picks
    its model id (auto:smart, auto:fast, ...) and routing headers per request.
    """
    provider = config.model.provider.lower().replace("-", "_")
    name = config.model.name

    # Normalize aliases
    if provider in ("google", "gemini", "google_genai"):
        provider = "google_genai"
    elif provider in ("azure", "azure_openai"):
        provider = "azure_openai"

    if provider in ("freellmapi", "freellmapi_openai"):
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("FREELLMAPI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "FreeLLMAPI requires FREELLMAPI_API_KEY (your unified gateway key). "
                "Set FREELLMAPI_BASE_URL if your gateway is not at http://localhost:3001/v1."
            )
        kwargs = dict(
            model=name,
            api_key=api_key,
            base_url=os.environ.get(
                "FREELLMAPI_BASE_URL",
                os.environ.get("OPENAI_BASE_URL", "http://localhost:3001/v1"),
            ),
            temperature=config.model.temperature,
            max_retries=config.model.max_retries,
            timeout=config.model.timeout,
        )
        if route_state is None:
            return ChatOpenAI(**kwargs)
        from deo.routing.freellmapi import RoutedChatOpenAI
        return RoutedChatOpenAI(**kwargs, include_response_headers=True,
                                route_state=route_state, route_role=role)

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=name,
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            temperature=config.model.temperature,
            max_retries=config.model.max_retries,
            timeout=config.model.timeout,
        )

    if provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI
        normalized_endpoint, inferred_api_version = _normalize_azure_endpoint_and_version(
            os.environ["AZURE_OPENAI_ENDPOINT"]
        )
        return AzureChatOpenAI(
            azure_deployment=name,
            azure_endpoint=normalized_endpoint,
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=(
                inferred_api_version
                or os.environ.get("AZURE_OPENAI_API_VERSION")
                or os.environ.get("OPENAI_API_VERSION")
                or "2024-02-01"
            ),
            temperature=config.model.temperature,
            max_retries=config.model.max_retries,
            timeout=config.model.timeout,
            # Models like gpt-5.3-codex only support the Responses API on Azure;
            # use_responses_api=True enables it unconditionally for all Azure deployments.
            use_responses_api=True,
        )

    # anthropic / openai / google_genai — all read their key from standard env vars
    return init_chat_model(
        model=name,
        model_provider=provider,
        temperature=config.model.temperature,
        max_retries=config.model.max_retries,
        timeout=config.model.timeout,
    )


def _to_backend_path(root_dir: Path, path: Path) -> str | None:
    """
    Convert a local directory path into a Deep Agents backend path.

    Backend paths must be absolute POSIX-style paths rooted at the backend root.
    """
    try:
        rel = path.resolve().relative_to(root_dir.resolve())
    except ValueError:
        return None
    return f"/{rel.as_posix().rstrip('/')}/"


def _install_builtin_skills(root_dir: Path) -> None:
    """Copy packaged skills into the project-rooted Deep Agents backend.

    Deep Agents resolves skill paths through its project backend, so package
    paths outside root_dir cannot be passed directly as skill sources.
    Existing project copies are never overwritten.
    """
    packaged_skills = Path(__file__).resolve().parents[1] / "skills"
    if not packaged_skills.is_dir():
        return
    destination = root_dir / ".deo" / "skills"
    for skill_file in packaged_skills.glob("*/SKILL.md"):
        target = destination / skill_file.parent.name / "SKILL.md"
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(skill_file.read_text(encoding="utf-8"), encoding="utf-8")


def _resolve_skill_sources(root_dir: Path) -> list[str]:
    """
    Discover project-rooted Deep Agents skill directories in priority order.

    Built-in skills are copied into .deo/skills first; repository-local
    deo/skills is applied afterward and can override the packaged defaults.
    """
    candidates = [
        root_dir / ".deo" / "skills",
        root_dir / "deo" / "skills",
    ]
    discovered: list[str] = []
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if not list(candidate.glob("*/SKILL.md")):
            continue
        backend_path = _to_backend_path(root_dir, candidate)
        if backend_path:
            discovered.append(backend_path)
    return discovered


def create_deo_agent(
    config: DeoConfig,
    manager: ChromaManager | None,
    indexer: CodebaseIndexer | None,
    checkpointer: MemorySaver | None = None,
    extra_tools: list[Any] | None = None,
    extra_system_prompt: str = "",
    route_state: Any = None,
    learner: Any = None,
) -> Any:  # CompiledStateGraph
    """Build and return the Deo orchestrator agent."""

    # ------------------------------------------------------------------
    # 1. Configure shared tool state
    # ------------------------------------------------------------------
    configure_git_tools(config.root_dir)
    if manager is not None and indexer is not None:
        configure_chroma_tools(manager, indexer, n_results=config.chroma.n_results,
                               dense_weight=config.chroma.dense_weight,
                               max_per_file=config.chroma.max_chunks_per_file)
        if learner is not None and config.learning.file_priors:
            manager.file_prior = learner.files.prior
        chroma_tools = list(CHROMA_TOOLS)
    else:
        chroma_tools = []

    # ------------------------------------------------------------------
    # 2. Assemble orchestrator tools
    # ------------------------------------------------------------------
    orchestrator_tools = [
        *chroma_tools,
        *GIT_TOOLS,
        *(HARNESS_TOOLS if config.harnesses.enabled else []),
        *(LEARNING_TOOLS if learner is not None else []),
        *(extra_tools or []),
    ]
    configure_harness_tools(config.root_dir, config.harnesses, route_state, learner)

    # ------------------------------------------------------------------
    # 3. Build system prompt
    # ------------------------------------------------------------------
    system_prompt = ORCHESTRATOR_PROMPT

    # Append AGENTS.md content if present
    agents_md_path = config.root_dir / config.agents_md_path
    if agents_md_path.exists():
        agents_md_content = agents_md_path.read_text(encoding="utf-8", errors="replace")
        system_prompt = (
            system_prompt
            + "\n\n## Project Memory (AGENTS.md)\n\n"
            + agents_md_content
        )

    # Append any extra instructions
    if extra_system_prompt:
        system_prompt = system_prompt + "\n\n" + extra_system_prompt

    # ------------------------------------------------------------------
    # 4. Build subagents
    # ------------------------------------------------------------------
    _install_builtin_skills(config.root_dir)
    skill_sources = _resolve_skill_sources(config.root_dir)
    subagents = build_subagents(
        enable_planner=config.subagents.planner,
        enable_implementer=config.subagents.implementer,
        enable_reviewer=config.subagents.reviewer,
        enable_tester=config.subagents.tester,
        enable_documenter=config.subagents.documenter,
        planner_model=config.subagents.planner_model,
        implementer_model=config.subagents.implementer_model,
        reviewer_model=config.subagents.reviewer_model,
        tester_model=config.subagents.tester_model,
        documenter_model=config.subagents.documenter_model,
    )

    # ------------------------------------------------------------------
    # 5. Initialize model
    # ------------------------------------------------------------------
    model = _build_model(config, role="orchestrator", route_state=route_state)
    _resolve_subagent_models(config, subagents, route_state)

    # ------------------------------------------------------------------
    # 6. Create agent
    # ------------------------------------------------------------------
    checkpointer = checkpointer or MemorySaver()
    backend = LocalShellBackend(root_dir=str(config.root_dir), virtual_mode=True)
    middleware = build_orchestrator_middleware(model, backend, str(config.root_dir))

    agent = create_deep_agent(
        model=model,
        tools=orchestrator_tools,
        system_prompt=system_prompt,
        subagents=subagents,
        skills=skill_sources or None,
        middleware=middleware,
        backend=backend,
        checkpointer=checkpointer,
    )

    return agent
