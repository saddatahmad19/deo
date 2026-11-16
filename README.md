# ⚡ Deo

**A full-screen, terminal-native multi-agent software engineering harness** built on [Deep Agents](https://github.com/langchain-ai/deepagents) + [ChromaDB](https://github.com/chroma-core/chroma).

Think Claude Code, Codex, or OpenCode — but fully open source, model-agnostic, and with semantic codebase search built in.

```
  ██████╗ ███████╗ ██████╗
  ██╔══██╗██╔════╝██╔═══██╗
  ██║  ██║█████╗  ██║   ██║
  ██║  ██║██╔══╝  ██║   ██║
  ██████╔╝███████╗╚██████╔╝
  ╚═════╝ ╚══════╝ ╚═════╝
```

---

## Features

- 🤖 **Multi-agent pipeline** — Specialized agents for planning, implementation, review, and testing
- 🔍 **Semantic codebase search** — ChromaDB-powered vector search so agents understand your project before touching it
- 🛠️ **Real shell access** — Runs tests, formatters, builds — anything you can run in a terminal
- 📁 **Full filesystem access** — Read, write, edit files with DeepAgents' battle-tested filesystem backend
- 🔌 **Model agnostic** — Works with Claude, GPT-4, Gemini, Ollama, and any LangChain-compatible model
- 🖥️ **Full-screen TUI** — Persistent transcript, bottom-pinned composer, live model/index HUD, slash-command completion, and terminal resize support
- 🎭 **Agent profiles** — Switch between `fast`, `review`, `write-tests`, `full` modes instantly
- 📝 **AGENTS.md memory** — Project-specific context that persists across sessions
- ⚙️ **Extensible** — Add custom tools, agents, and prompts via the registry system

---

## Installation

```bash
pip install deo-agent
# or
uv add deo-agent
```

Set your FreeLLMAPI gateway credentials (the default provider):

```bash
export FREELLMAPI_BASE_URL="http://localhost:3001/v1"
export FREELLMAPI_API_KEY="your-unified-key"
```

FreeLLMAPI uses the OpenAI-compatible API and the auto:default routing profile by default. Set these in your local .env file if preferred. Other providers remain configurable.

---

## Quick Start

```bash
# Start interactive session in your project
cd /path/to/my/project
deo

# One-shot command
deo "Add input validation to the user registration endpoint"

# Use a specific model
deo --model openai:gpt-4o

# Code review mode only
deo --profile review "Review the auth module for security issues"

# Force full re-index
deo --force-index
```

---

## Multi-Agent Architecture

Deo uses a **multi-prompt** orchestration system where a senior orchestrator agent coordinates specialized subagents:

```
User prompt
    │
    ▼
┌─────────────────────────────────┐
│      Orchestrator Agent         │
│  (senior engineer + dispatcher) │
│                                 │
│  Tools: semantic_search,        │
│         git_status/diff,        │
│         add_to_index,           │
│         find_file               │
└──────┬──────────────────────────┘
       │ task tool (DeepAgents)
       │
       ├──▶ planner-agent
       │       Decomposes multi-file tasks
       │       into step-by-step plans
       │
       ├──▶ implementer-agent
       │       Writes production-quality code
       │       following project conventions
       │
       ├──▶ reviewer-agent
       │       Checks correctness, security,
       │       performance, style
       │
       ├──▶ tester-agent
       │       Writes tests, runs them,
       │       fixes failures
       │
       └──▶ documenter-agent (optional)
               Docstrings, README, AGENTS.md
```

Each subagent has:

- Its own focused system prompt
- Access to ChromaDB semantic search
- Access to git tools
- The full DeepAgents filesystem backend (read/write/edit/execute)

---

## ChromaDB Semantic Search

Deo automatically indexes your codebase into ChromaDB on startup. This gives all agents the ability to find relevant code _before_ writing anything new.

```
.deo/
└── chroma/               ← Persistent ChromaDB index
    ├── chroma.sqlite3
    └── file_hashes.json  ← Tracks which files need re-indexing
```

The index is **incremental** — only changed files are re-indexed. For a 10k-file project, re-indexing takes seconds after the initial build.

### Available semantic tools (usable by all agents):

| Tool                     | Description                                        |
| ------------------------ | -------------------------------------------------- |
| `semantic_search(query)` | Find relevant code by natural language description |
| `add_to_index(filepath)` | Index a newly created file immediately             |
| `find_file(pattern)`     | Find indexed files by path pattern                 |
| `index_status()`         | Show index statistics                              |

---

## Agent Profiles

Switch modes with `--profile` or the `/profile` command in the REPL:

| Profile       | Subagents                                 | Best For                      |
| ------------- | ----------------------------------------- | ----------------------------- |
| `default`     | planner + implementer + reviewer + tester | Most tasks                    |
| `fast`        | none                                      | Quick questions, simple edits |
| `review`      | reviewer only                             | Security/code audits          |
| `write-tests` | tester only                               | Adding test coverage          |
| `full`        | all (incl. documenter)                    | New feature development       |

---

## Configuration

Deo uses hierarchical TOML config:

1. `~/.deo/config.toml` — global user config
2. `.deo.toml` — project-local config (overrides global)

Initialize a project config:

```bash
deo config --init
```

Example `.deo.toml`:

```toml
[model]
provider = "freellmapi"
name = "auto:default"
temperature = 0.0

[chroma]
auto_index = true
chunk_size = 1500
n_results = 8

[subagents]
planner = true
implementer = true
reviewer = true
tester = true
documenter = false

[display]
show_tool_calls = true
```

---

## Project setup and continuous indexing

Run `deo init` in any project to create shared agent context, connect `CLAUDE.md` to `AGENTS.md`, and build or update the project index. Existing context files are preserved by default. Use `deo init --force` to regenerate AGENTS.md. Run `deo sync` for an incremental refresh, or `deo sync --watch` to keep the index current while you work. Startup indexing is incremental and checks changed and deleted files.

For an empty directory, Deo applies a greenfield bootstrap: safe `.gitignore`, honest README and AGENTS.md starter, and a `CLAUDE.md` import. It deliberately does not choose a license, invent build commands, or create speculative CI/docs scaffolding.

## AGENTS.md

Create an `AGENTS.md` at your project root to give Deo persistent context about your project:

```markdown
# Project: MyApp

## Architecture

- Backend: FastAPI + PostgreSQL
- Frontend: Next.js 15 + TypeScript
- Auth: Clerk (JWT)
- Testing: pytest + vitest

## Conventions

- Use snake_case for Python, camelCase for TypeScript
- All DB queries go through repository classes in `src/repositories/`
- API routes follow REST conventions defined in `docs/api.md`
- Error handling: raise `AppError` subclasses, never bare exceptions

## Key Files

- `src/config.py` — All configuration and env vars
- `src/database.py` — DB connection and session management
- `src/middleware/auth.py` — Authentication logic
```

Deo reads this file and includes it in every agent's context.

---

## Interactive REPL Commands

| Command           | Description                 |
| ----------------- | --------------------------- |
| `/help`           | Show help                   |
| `/status`         | Show index and session info |
| `/index`          | Re-index the codebase       |
| `/profile <name>` | Switch agent profile        |
| `/profiles`       | List all profiles           |
| `/clear`          | Clear the screen            |
| `/exit`           | Exit Deo                    |

---

## Extending Deo

### Adding custom tools

```python
from langchain_core.tools import tool
from deo.tools.registry import get_registry

@tool
def query_database(sql: str) -> str:
    """Run a read-only SQL query against the development database."""
    # your implementation
    ...

# Register it
get_registry().register(query_database, tags=["database", "custom"])
```

### Adding custom agent profiles

```python
from deo.agents.registry import AgentProfile, register_profile

register_profile(AgentProfile(
    name="security-audit",
    description="Deep security review with no code changes.",
    enable_planner=False,
    enable_implementer=False,
    enable_reviewer=True,
    enable_tester=False,
    system_prompt_addon=(
        "Focus exclusively on security vulnerabilities. "
        "Check for OWASP Top 10, injection attacks, auth bypass, and data exposure."
    ),
))
```

### Adding custom prompts

```python
from deo.prompts.templates import register_prompt

register_prompt("my-agent", """
You are a specialized database migration agent...
""")
```

### Using the Python API

```python
from pathlib import Path
from deo.config import DeoConfig
from deo.agents.factory import create_deo_agent
from deo.memory.chroma_manager import ChromaManager
from deo.memory.indexer import CodebaseIndexer
from langgraph.checkpoint.memory import MemorySaver

config = DeoConfig.load(Path("/my/project"))
config.ensure_deo_dir()

manager = ChromaManager(config.chroma_persist_path)
indexer = CodebaseIndexer(config.root_dir, manager, config.chroma)
indexer.index()  # Build index

agent = create_deo_agent(config, manager, indexer)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Add rate limiting to the API"}]
}, config={"configurable": {"thread_id": "my-session"}})
```

---

## Project Layout

```
deo/
├── pyproject.toml
├── deo/
│   ├── cli.py                  ← typer CLI entry point + REPL
│   ├── config.py               ← DeoConfig (hierarchical TOML)
│   ├── agents/
│   │   ├── factory.py          ← create_deo_agent() — wires everything together
│   │   ├── subagents.py        ← All subagent definitions
│   │   └── registry.py         ← AgentProfile registry
│   ├── tools/
│   │   ├── chroma_tools.py     ← semantic_search, add_to_index, find_file, index_status
│   │   ├── git_tools.py        ← git_status, git_diff, git_log, git_branch
│   │   └── registry.py         ← ToolRegistry
│   ├── memory/
│   │   ├── chroma_manager.py   ← ChromaDB collection management
│   │   └── indexer.py          ← File chunking + codebase walking
│   ├── prompts/
│   │   └── templates.py        ← All system prompts + PromptRegistry
│   └── tui/
│       └── display.py          ← Rich-based streaming terminal UI
```

---

## Requirements

- Python 3.11+
- An API key for your chosen model provider
- ~500MB disk for ChromaDB's default embedding model (downloads automatically on first run)

---

## License

MIT
