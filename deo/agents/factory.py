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

from deo.agents.subagents import build_subagents
from deo.config import DeoConfig
from deo.memory.chroma_manager import ChromaManager
from deo.memory.indexer import CodebaseIndexer
from deo.prompts.templates import ORCHESTRATOR_PROMPT
from deo.tools.chroma_tools import CHROMA_TOOLS, configure_chroma_tools
from deo.tools.git_tools import GIT_TOOLS, configure_git_tools


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


def _build_model(config: DeoConfig) -> Any:
    """Instantiate the LLM based on config.model.provider."""
    provider = config.model.provider.lower().replace("-", "_")
    name = config.model.name

    # Normalize aliases
    if provider in ("google", "gemini", "google_genai"):
        provider = "google_genai"
    elif provider in ("azure", "azure_openai"):
        provider = "azure_openai"

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
        )

    # anthropic / openai / google_genai — all read their key from standard env vars
    return init_chat_model(
        model=name,
        model_provider=provider,
        temperature=config.model.temperature,
        max_retries=config.model.max_retries,
        timeout=config.model.timeout,
    )


def create_deo_agent(
    config: DeoConfig,
    manager: ChromaManager | None,
    indexer: CodebaseIndexer | None,
    checkpointer: MemorySaver | None = None,
    extra_tools: list[Any] | None = None,
    extra_system_prompt: str = "",
) -> Any:  # CompiledStateGraph
    """Build and return the Deo orchestrator agent."""

    # ------------------------------------------------------------------
    # 1. Configure shared tool state
    # ------------------------------------------------------------------
    configure_git_tools(config.root_dir)
    if manager is not None and indexer is not None:
        configure_chroma_tools(manager, indexer, n_results=config.chroma.n_results)
        chroma_tools = list(CHROMA_TOOLS)
    else:
        chroma_tools = []

    # ------------------------------------------------------------------
    # 2. Assemble orchestrator tools
    # ------------------------------------------------------------------
    orchestrator_tools = [
        *chroma_tools,
        *GIT_TOOLS,
        *(extra_tools or []),
    ]

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
    subagents = build_subagents(
        model_string=None,  # Inherit orchestrator model
        enable_planner=config.subagents.planner,
        enable_implementer=config.subagents.implementer,
        enable_reviewer=config.subagents.reviewer,
        enable_tester=config.subagents.tester,
        enable_documenter=config.subagents.documenter,
    )

    # ------------------------------------------------------------------
    # 5. Initialize model
    # ------------------------------------------------------------------
    model = _build_model(config)

    # ------------------------------------------------------------------
    # 6. Create agent
    # ------------------------------------------------------------------
    checkpointer = checkpointer or MemorySaver()

    agent = create_deep_agent(
        model=model,
        tools=orchestrator_tools,
        system_prompt=system_prompt,
        subagents=subagents,
        backend=LocalShellBackend(root_dir=str(config.root_dir)),
        checkpointer=checkpointer,
    )

    return agent
