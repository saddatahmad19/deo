# Project: deo

## Purpose
Deo is a terminal-native AI coding agent powered by Deep Agents and ChromaDB. It orchestrates a multi-agent pipeline (planner, implementer, reviewer, tester, documenter) to handle complex coding tasks with semantic codebase search, real shell access, and persistent project context.

## Architecture
- **Orchestrator Agent** (`deo/agents/factory.py`) — Senior engineer dispatcher coordinating specialized subagents via DeepAgents' `task` tool
- **Subagents** (`deo/agents/subagents.py`) — Five specialized roles (planner, implementer, reviewer, tester, documenter), each with focused system prompts and Pydantic response schemas
- **ChromaDB Integration** (`deo/memory/`) — Incremental semantic indexing of codebase; agents search before coding via `semantic_search` tool
- **CLI/TUI** (`deo/cli.py`, `deo/tui/`) — Typer-based commands with prompt-toolkit REPL, Rich formatting, slash commands (`/help`, `/index`, `/profile`, etc.)
- **Tool Registry** (`deo/tools/registry.py`) — Centralized tool management; git tools, ChromaDB search, extensible custom tools
- **Config System** (`deo/config.py`) — Hierarchical TOML loading: defaults → `~/.deo/config.toml` → `.deo.toml` → CLI flags
- **Agent Profiles** (`deo/agents/registry.py`) — Named configurations (default, fast, review, write-tests, full) for quick mode switching

## Key Files
| File | Responsibility |
|------|-----------------|
| `deo/cli.py` | Main entry point; REPL loop, slash commands, project root detection, initialization |
| `deo/agents/factory.py` | Creates orchestrator and subagent instances; model initialization and tool binding |
| `deo/agents/subagents.py` | Factory functions for each subagent with system prompts and response schemas |
| `deo/agents/schemas.py` | Pydantic models for structured outputs (PlanResult, ImplementResult, ReviewResult, RunResult, DocResult) |
| `deo/agents/middleware.py` | Custom middleware for appending runtime guidance to model requests |
| `deo/config.py` | Configuration dataclasses and hierarchical loading logic |
| `deo/memory/chroma_manager.py` | ChromaDB client initialization and collection management |
| `deo/memory/indexer.py` | Incremental codebase indexing with language detection and chunking |
| `deo/tools/registry.py` | Tool registration and filtering by tags/names |
| `deo/tools/git_tools.py` | Git status, diff, log, branch tools for agent situational awareness |
| `deo/tui/display.py` | Rich-based terminal output (banners, panels, formatted messages) |
| `deo/tui/input.py` | prompt-toolkit REPL with slash command completion and status toolbar |
| `deo/init/repo_indexer.py` | Chroma Cloud collection creation and file indexing for `deo init` command |

## Conventions
- **Model strings**: `"provider:name"` format (e.g., `"anthropic:claude-sonnet-4-6"`, `"openai:gpt-4o"`); split on `:` to extract provider and name
- **Subagent response schemas**: All subagents return Pydantic models with structured fields; orchestrator parses and synthesizes into user-facing output
- **Tool naming**: LangChain `@tool` decorators; names auto-derived from function names; git tools prefixed `git_`, chroma tools prefixed `semantic_`
- **Config keys**: Snake_case in TOML (e.g., `implementer_model`, `chunk_size`); dataclass fields match exactly
- **Project root detection**: Walk up from cwd looking for `.git`, `pyproject.toml`, `package.json`, `.deo.toml`, etc.; fall back to cwd
- **Paths**: Use `Path` objects internally; convert to strings only for subprocess/API calls
- **Error handling**: ChromaDB failures caught and logged; agent continues with graceful degradation (no index available)
- **Skills**: Documented in `deo/skills/*.md` (safe-shell-execution, subagent-handoff, coding-workloop); agents reference these patterns

## External Dependencies
- **deepagents** ≥0.4.0 — Multi-agent orchestration with `task` tool and filesystem backend
- **langchain** ≥0.3.0, **langgraph** ≥0.4.0 — LLM chains, graph workflows, checkpoint memory
- **langchain-anthropic** ≥0.3.0 — Claude integration; also supports langchain-openai, langchain-google-genai
- **chromadb** ≥0.6.0 — Vector database for semantic search; supports local and Cloud modes
- **typer** ≥0.12.0, **rich** ≥13.0.0, **prompt-toolkit** ≥3.0.0 — CLI, TUI, and terminal formatting
- **tiktoken** ≥0.7.0 — Token counting for prompt sizing
- **pathspec** ≥0.12.0 — Gitignore-style pattern matching for exclusions

## Agent Guidance
- **Per-role model overrides**: Config allows `implementer_model`, `reviewer_model`, etc. to differ from orchestrator model; pass these to `build_subagents()` in factory. Tests verify this flow (`test_factory_models.py`).
- **Subagent handoff contract**: Always include objective, scope, constraints, and deliverable format when delegating via `task` tool. Parallelize independent tasks; enforce sequencing for dependent ones.
- **ChromaDB graceful degradation**: If Chroma init fails (missing API key, network error), log warning and continue; agents work without semantic search but lose codebase context.
- **Slash command parsing**: Commands start with `/`; check for exact match first, then aliases. `/clear` and `/help` are special (no session context). `/profile <name>` requires argument parsing.
- **Startup UI redesign**: `print_banner()` now renders a single framed Panel with DEO ASCII (cyan, centered), tagline (magenta), and footer stats (session, model, path, chunks). Remove separate `print_session_info()` call; merge all info into banner.