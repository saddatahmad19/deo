# Project: Deo

## Purpose
Deo is a full-screen, terminal-native multi-agent software engineering harness built on Deep Agents, LangGraph, and ChromaDB. It plans, implements, reviews, tests, and documents code while maintaining an incremental semantic index and shared project context.

## Architecture
- **CLI and session lifecycle** — deo/cli.py wires project discovery, configuration, indexing, agent construction, slash commands, and one-shot execution.
- **Full-screen terminal UI** — deo/tui/fullscreen.py owns the alternate-screen transcript, pinned composer, live status HUD, slash completion, and asynchronous turn execution. deo/tui/display.py streams model/tool events; deo/tui/input.py is the non-TTY fallback.
- **Model factory** — deo/agents/factory.py builds the orchestrator and subagents. FreeLLMAPI is the default OpenAI-compatible provider.
- **Agent roles** — deo/agents/subagents.py, deo/agents/registry.py, and deo/prompts/ define the planner, implementer, reviewer, tester, documenter, profiles, and prompts.
- **Project context/bootstrap** — deo/init/project_setup.py safely creates AGENTS.md, a CLAUDE.md import, and greenfield starter files. deo/skills/greenfield-bootstrap/SKILL.md describes the bootstrap workflow.
- **Semantic memory** — deo/memory/indexer.py performs hash-based incremental indexing and deleted-file cleanup; deo/memory/chroma_manager.py persists chunk metadata and accesses Chroma Cloud.
- **Tools** — deo/tools/ provides semantic search, indexing, and read-only Git context.

## Key Files
| File | Responsibility |
|------|-----------------|
| deo/cli.py | Typer CLI, startup indexing, interactive session, init and sync/watch commands |
| deo/tui/fullscreen.py | Full-screen prompt_toolkit application |
| deo/tui/display.py | Rich streaming and non-full-screen output |
| deo/config.py | Hierarchical configuration and FreeLLMAPI defaults |
| deo/agents/factory.py | Provider construction, skills, tools, and orchestrator assembly |
| deo/agents/subagents.py | Specialized agent definitions |
| deo/memory/indexer.py | Incremental project index and deletion reconciliation |
| deo/memory/chroma_manager.py | Chroma collection operations and file hash cache |
| deo/init/project_setup.py | Safe existing-project and greenfield initialization |
| deo/init/agents_md.py | Evidence-based AGENTS.md generation |
| deo/skills/greenfield-bootstrap/SKILL.md | New-repository bootstrap contract |
| tests/ | Configuration, factory, schema, subagent, and project-setup tests |

## Conventions
- **Default model**: provider freellmapi, model ID auto:default; full model string freellmapi:auto:default.
- **Model overrides**: recognize known provider prefixes only. IDs such as auto:default are valid model IDs and must not be split into provider auto and model default.
- **FreeLLMAPI**: OpenAI-compatible base URL defaults to http://localhost:3001/v1; read the key from FREELLMAPI_API_KEY (fall back to OPENAI_API_KEY) and allow FREELLMAPI_BASE_URL override.
- **Subagent models**: None inherits the orchestrator model; explicit role overrides are passed through build_subagents().
- **Configuration**: snake_case TOML keys; precedence is defaults → ~/.deo/config.toml → project .deo.toml → CLI flags.
- **Project docs**: preserve user-authored AGENTS.md and CLAUDE.md by default. CLAUDE.md may import shared rules using @AGENTS.md.
- **Indexing**: use CodebaseIndexer.index(force=False) for incremental refreshes; it detects changed files and removes deleted paths. Avoid the legacy separate-collection initializer.
- **TUI**: keep model execution off the prompt_toolkit event loop. Send streamed output back through the UI loop; never print directly to the terminal while the alternate screen is active.
- **Tests**: add focused pytest coverage for model/config parsing, project setup, and CLI/index lifecycle changes.

## External Dependencies
- deepagents, langchain, langgraph — agent orchestration and model integration
- langchain-openai — FreeLLMAPI and OpenAI-compatible providers (core dependency)
- langchain-anthropic, langchain-google-genai — alternate model providers
- chromadb — Chroma Cloud vector database
- prompt-toolkit, rich, typer — full-screen TUI, formatting, and CLI
- pathspec, python-dotenv, tiktoken — indexing exclusions, environment loading, and token utilities

## Verification
- python -m compileall -q deo tests
- python -m pytest -q

## Agent Guidance
- Do not commit API keys, .env, private keys, or certificates.
- Do not choose a license for a greenfield project; ask the owner.
- Do not fabricate architecture, test commands, or CI success in generated project context.
- deo init preserves existing context files; use --force only when regeneration is requested.
- deo sync --watch continuously refreshes the incremental index; Ctrl+C stops the watcher.
- Verify terminal behavior in an actual TTY after changing prompt_toolkit layout or key bindings.
