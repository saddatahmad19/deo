"""
deo/cli.py

Main CLI entry point for the Deo terminal AI coding tool.

Commands:
  deo             — Start interactive REPL session
  deo "prompt"    — Single one-shot prompt
  deo index       — Index/re-index the codebase
  deo status      — Show index and session status
  deo config      — Print/init config
  deo profiles    — List agent profiles
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any, Optional

import typer

from deo.config import DeoConfig
from deo.tui.display import (
    StreamDisplay,
    console,
    print_banner,
    print_error,
    print_info,
    print_rule,
    print_success,
    print_warning,
)
from deo.tui.input import SlashCommand, build_prompt_session, make_prompt_message

# ---------------------------------------------------------------------------
# Slash command registry
# ---------------------------------------------------------------------------

SLASH_COMMANDS: list[SlashCommand] = [
    SlashCommand("help",     "Show this help message"),
    SlashCommand("clear",    "Clear the terminal screen"),
    SlashCommand("status",   "Show index status and session info", aliases=["index-status"]),
    SlashCommand("index",    "Re-index the codebase"),
    SlashCommand("init",     "Generate AGENTS.md and index important files into a named Chroma collection"),
    SlashCommand("profile",  "Switch agent profile", usage="profile <name>"),
    SlashCommand("profiles", "List available agent profiles"),
    SlashCommand("exit",     "Exit Deo", aliases=["quit", "q"]),
]


def _print_help(commands: list[SlashCommand]) -> None:
    from rich.markdown import Markdown

    rows = []
    for cmd in commands:
        label = f"`/{cmd.usage if cmd.usage else cmd.name}`"
        if cmd.aliases:
            label += " / " + " / ".join(f"`/{a}`" for a in cmd.aliases)
        rows.append(f"| {label} | {cmd.description} |")

    table = "\n".join(rows)
    text = f"""\
## Deo Commands

| Command | Description |
|---------|-------------|
{table}

## Tips

- Use **semantic_search** — Deo automatically searches your codebase before coding
- Start complex tasks with a clear description; the planner will break it down
- Add `AGENTS.md` to your project root with architecture notes for persistent context
- Use `.deo.toml` in your project for project-specific config
"""
    console.print(Markdown(text))


app = typer.Typer(
    name="deo",
    help="Deo — Terminal AI coding agent powered by Deep Agents + ChromaDB",
    add_completion=False,
    rich_markup_mode="rich",
)


# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------

_ROOT_MARKERS = [
    ".git", "pyproject.toml", "package.json", "Cargo.toml",
    "go.mod", "pom.xml", "build.gradle", ".deo.toml",
]


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from start path looking for a project root marker."""
    current = (start or Path.cwd()).resolve()
    for path in [current, *current.parents]:
        if any((path / marker).exists() for marker in _ROOT_MARKERS):
            return path
    return current  # Fall back to cwd


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def _initialize(
    config: DeoConfig,
    force_reindex: bool = False,
    quiet: bool = False,
) -> tuple[Any, Any, Any]:  # (manager, indexer, agent)
    """Initialize ChromaDB, indexer, and the orchestrator agent."""
    from deo.agents.factory import create_deo_agent
    from deo.memory.chroma_manager import ChromaManager
    from deo.memory.indexer import CodebaseIndexer
    from langgraph.checkpoint.memory import MemorySaver

    config.ensure_deo_dir()

    # Init ChromaDB (optional; continue without it if unavailable)
    manager = None
    indexer = None
    try:
        manager = ChromaManager(
            persist_path=config.chroma_persist_path,
            collection_name=config.chroma.collection_name,
        )

        indexer = CodebaseIndexer(
            root=config.root_dir,
            manager=manager,
            config=config.chroma,
        )

        # Index codebase
        if config.chroma.auto_index:
            n_existing = manager.count()
            if n_existing == 0 or force_reindex:
                if not quiet:
                    print_info("Indexing codebase for semantic search...")
                _run_index(indexer, quiet=quiet)
            else:
                if not quiet:
                    print_info(
                        f"Using existing index: {n_existing} chunks across "
                        f"{len(manager.get_indexed_files())} files"
                    )
    except Exception as e:
        if not quiet:
            print_warning(
                "Chroma unavailable; continuing without semantic index/tools "
                f"({e})."
            )

    # Build agent
    checkpointer = MemorySaver()
    agent = create_deo_agent(
        config=config,
        manager=manager,
        indexer=indexer,
        checkpointer=checkpointer,
    )

    return manager, indexer, agent


def _run_index(indexer: Any, quiet: bool = False) -> None:
    """Run codebase indexing with progress display."""
    from deo.tui.display import make_index_progress

    with make_index_progress() as progress:
        task = progress.add_task("Indexing...", total=None)

        def on_progress(filepath: str, current: int, total: int) -> None:
            progress.update(
                task,
                total=total,
                completed=current,
                description=f"[dim]{filepath[:50]}[/]",
            )

        result = indexer.index(
            force=False,
            progress_callback=on_progress if not quiet else None,
        )

    if not quiet:
        print_success(str(result))


def _run_init(config: DeoConfig, quiet: bool = False) -> None:
    """Generate AGENTS.md and index important files into a named Chroma collection."""
    from deo.init.agents_md import generate_agents_md
    from deo.init.repo_indexer import index_important_files

    root = config.root_dir
    agents_md_path = root / config.agents_md_path

    if not quiet:
        print_info(f"Analysing repository: {root.name} ...")

    agents_md_text, important_files = generate_agents_md(
        root=root,
        model_string=config.model.model_string,
    )

    agents_md_path.write_text(agents_md_text, encoding="utf-8")
    if not quiet:
        print_success(f"Wrote {agents_md_path} ({len(agents_md_text)} chars)")

    if not quiet:
        print_info(f"Indexing {len(important_files)} important files into Chroma ...")

    try:
        collection_name, n_chunks = index_important_files(
            root=root,
            important_files=important_files,
        )
        if not quiet:
            print_success(
                f"Indexed {n_chunks} chunks into collection [bold]{collection_name}[/]"
            )
    except Exception as e:
        if not quiet:
            print_warning(f"Chroma indexing failed: {e}")


# ---------------------------------------------------------------------------
# Streaming run loop
# ---------------------------------------------------------------------------

def _stream_agent_response(
    agent: Any,
    messages: list[dict],
    thread_id: str,
    show_tool_calls: bool = True,
    show_thinking: bool = False,
) -> str:
    """
    Stream the agent response and display it using Rich.

    Returns the final text content of the response.
    """
    display = StreamDisplay(
        show_tool_calls=show_tool_calls,
        show_thinking=show_thinking,
    )

    config = {"configurable": {"thread_id": thread_id}}
    final_content = ""

    try:
        with display.streaming_context():
            for chunk in agent.stream(
                {"messages": messages},
                config=config,
                stream_mode="messages",
            ):
                display.handle_stream_chunk(chunk)

                # Accumulate final content for return value
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    msg, _ = chunk
                    content = getattr(msg, "content", None)
                    if isinstance(content, str):
                        final_content += content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                final_content += block.get("text", "")

    except KeyboardInterrupt:
        console.print("\n[dim]⚡ Interrupted[/]")
    except Exception as e:
        print_error(f"Agent error: {e}")
        raise

    return final_content


# ---------------------------------------------------------------------------
# Slash command handler
# ---------------------------------------------------------------------------

def _handle_slash_command(
    cmd: str,
    config: DeoConfig,
    manager: Any,
    indexer: Any,
    agent: Any,
    session_id: str,
) -> tuple[bool, Any, Any, Any]:
    """
    Handle REPL slash commands.

    Returns (should_continue, manager, indexer, agent) —
    agent may be rebuilt on profile switch.
    """
    from deo.agents.registry import list_profiles

    parts = cmd.strip().split()
    command = parts[0].lower()

    if command in ("/exit", "/quit", "/q"):
        console.print("[dim]Goodbye.[/]")
        sys.exit(0)

    elif command == "/help":
        _print_help(SLASH_COMMANDS)

    elif command == "/clear":
        console.clear()
        print_banner()

    elif command in ("/status", "/index-status"):
        from deo.tools.chroma_tools import index_status
        result = index_status.invoke({})
        console.print(result)

    elif command == "/index":
        if indexer is None:
            print_warning("Chroma is unavailable. Skipping index.")
        else:
            print_info("Re-indexing codebase...")
            _run_index(indexer, quiet=False)

    elif command == "/init":
        _run_init(config, quiet=False)

    elif command == "/profiles":
        profiles = list_profiles()
        console.print("\n[bold]Agent Profiles:[/]\n")
        for p in profiles:
            console.print(f"  [bold cyan]{p.name}[/]  —  {p.description}")
        console.print()

    elif command == "/profile":
        if len(parts) < 2:
            print_warning("Usage: /profile <name>")
        else:
            profile_name = parts[1]
            try:
                from deo.agents.factory import create_deo_agent
                from deo.agents.registry import get_profile
                from langgraph.checkpoint.memory import MemorySaver
                profile = get_profile(profile_name)

                # Apply profile overrides to config
                config.subagents.planner = profile.enable_planner
                config.subagents.implementer = profile.enable_implementer
                config.subagents.reviewer = profile.enable_reviewer
                config.subagents.tester = profile.enable_tester
                config.subagents.documenter = profile.enable_documenter

                checkpointer = MemorySaver()
                agent = create_deo_agent(
                    config=config,
                    manager=manager,
                    indexer=indexer,
                    checkpointer=checkpointer,
                    extra_system_prompt=profile.system_prompt_addon,
                )
                print_success(f"Switched to profile: [bold]{profile_name}[/] — {profile.description}")
            except KeyError:
                print_error(f"Profile '{profile_name}' not found. Use /profiles to list.")

    else:
        print_warning(f"Unknown command: {command}. Type /help for available commands.")

    print_rule()
    return True, manager, indexer, agent


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    prompt: Optional[str] = typer.Option(None, "--prompt", "-q", help="One-shot prompt (omit for interactive REPL)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model string (e.g. anthropic:claude-sonnet-4-6)"),
    profile: str = typer.Option("default", "--profile", "-p", help="Agent profile to use"),
    no_index: bool = typer.Option(False, "--no-index", help="Skip auto-indexing on startup"),
    force_index: bool = typer.Option(False, "--force-index", help="Force full re-index on startup"),
    show_tools: bool = typer.Option(True, "--show-tools/--hide-tools", help="Show tool calls in output"),
    root: Optional[str] = typer.Option(None, "--root", "-r", help="Project root directory"),
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit"),
) -> None:
    """
    Deo — Terminal AI coding agent powered by Deep Agents + ChromaDB.

    Run without arguments to start an interactive session.
    Pass a prompt to run a one-shot command.
    """
    if ctx.invoked_subcommand is not None:
        return

    if version:
        console.print("deo-agent 0.1.0")
        raise typer.Exit()

    # Detect project root
    project_root = Path(root).resolve() if root else find_project_root()

    # Load config
    config = DeoConfig.load(project_root)

    # Apply CLI overrides
    if model:
        config.model.override(model)
    if no_index:
        config.chroma.auto_index = False

    # Initialize
    try:
        manager, indexer, agent = _initialize(
            config,
            force_reindex=force_index,
            quiet=bool(prompt),  # Quiet if one-shot
        )
    except ImportError as e:
        print_error(f"Missing dependency: {e}")
        print_error("Run: pip install deo-agent[all]")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Initialization failed: {e}")
        raise typer.Exit(1)

    # Apply profile
    from deo.agents.registry import get_profile, list_profiles
    try:
        p = get_profile(profile)
        config.subagents.planner = p.enable_planner
        config.subagents.implementer = p.enable_implementer
        config.subagents.reviewer = p.enable_reviewer
        config.subagents.tester = p.enable_tester
        config.subagents.documenter = p.enable_documenter
    except KeyError:
        print_warning(f"Profile '{profile}' not found, using 'default'.")

    session_id = str(uuid.uuid4())[:8]

    # -----------------------------------------------------------------------
    # One-shot mode
    # -----------------------------------------------------------------------
    if prompt:
        _stream_agent_response(
            agent,
            messages=[{"role": "user", "content": prompt}],
            thread_id=session_id,
            show_tool_calls=show_tools,
        )
        raise typer.Exit()

    # -----------------------------------------------------------------------
    # Interactive REPL
    # -----------------------------------------------------------------------
    print_banner(
        session_id=session_id,
        model=config.model.model_string,
        root_dir=str(project_root),
        n_indexed=manager.count() if manager is not None else 0,
    )

    thread_id = session_id
    prompt_session = build_prompt_session(
        commands=SLASH_COMMANDS,
        session_id=session_id,
        model=config.model.model_string,
        cwd=str(project_root),
        get_n_indexed=lambda: manager.count() if manager is not None else 0,
    )

    while True:
        try:
            user_input = prompt_session.prompt(make_prompt_message).strip()
        except KeyboardInterrupt:
            console.print("[dim]Interrupted. Use /exit to quit.[/]")
            continue
        except EOFError:
            console.print("[dim]Goodbye.[/]")
            break

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            should_continue, manager, indexer, agent = _handle_slash_command(
                user_input, config, manager, indexer, agent, session_id
            )
            if not should_continue:
                break
            continue

        # Stream agent response — only pass the new message; MemorySaver
        # holds prior history in the checkpointer under thread_id.
        print_rule()

        _stream_agent_response(
            agent,
            messages=[{"role": "user", "content": user_input}],
            thread_id=thread_id,
            show_tool_calls=show_tools,
            show_thinking=config.display.show_thinking,
        )

        print_rule()


@app.command("index")
def index_command(
    force: bool = typer.Option(False, "--force", "-f", help="Force full re-index"),
    root: Optional[str] = typer.Option(None, "--root", "-r", help="Project root directory"),
) -> None:
    """Index (or re-index) the project codebase for semantic search."""
    from deo.memory.chroma_manager import ChromaManager
    from deo.memory.indexer import CodebaseIndexer

    project_root = Path(root).resolve() if root else find_project_root()
    config = DeoConfig.load(project_root)
    config.ensure_deo_dir()

    try:
        manager = ChromaManager(
            persist_path=config.chroma_persist_path,
            collection_name=config.chroma.collection_name,
        )
    except Exception as e:
        print_warning(f"Chroma is unavailable. Skipping index ({e}).")
        return

    indexer = CodebaseIndexer(root=project_root, manager=manager, config=config.chroma)

    print_info(f"Indexing: {project_root}")
    _run_index(indexer, quiet=False)


@app.command("status")
def status_command(
    root: Optional[str] = typer.Option(None, "--root", "-r"),
) -> None:
    """Show index status, config, and available profiles."""
    from deo.agents.registry import list_profiles
    from deo.memory.chroma_manager import ChromaManager

    project_root = Path(root).resolve() if root else find_project_root()
    config = DeoConfig.load(project_root)

    n_chunks = 0
    n_files = 0
    chroma_status = f"{config.chroma_persist_path}"
    try:
        manager = ChromaManager(
            persist_path=config.chroma_persist_path,
            collection_name=config.chroma.collection_name,
        )
        n_chunks = manager.count()
        n_files = len(manager.get_indexed_files())
    except Exception as e:
        chroma_status = f"{config.chroma_persist_path} (unavailable: {e})"

    console.print(f"\n[bold]Deo Status[/]")
    console.print(f"  Project:  {project_root}")
    console.print(f"  Model:    {config.model.model_string}")
    console.print(f"  Index:    {n_chunks} chunks / {n_files} files")
    console.print(f"  ChromaDB: {chroma_status}")
    console.print()

    console.print("[bold]Available Profiles:[/]")
    for p in list_profiles():
        console.print(f"  [cyan]{p.name:15}[/] {p.description}")
    console.print()


@app.command("init")
def init_command(
    root: Optional[str] = typer.Option(None, "--root", "-r", help="Project root directory"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model override for analysis"),
) -> None:
    """Generate AGENTS.md and index important files into a named Chroma collection."""
    project_root = Path(root).resolve() if root else find_project_root()
    config = DeoConfig.load(project_root)
    if model:
        config.model.override(model)
    config.ensure_deo_dir()
    _run_init(config, quiet=False)


@app.command("config")
def config_command(
    init: bool = typer.Option(False, "--init", help="Create a default .deo.toml in current directory"),
    root: Optional[str] = typer.Option(None, "--root", "-r"),
) -> None:
    """Print current config or initialize a new .deo.toml."""
    project_root = Path(root).resolve() if root else find_project_root()
    config = DeoConfig.load(project_root)

    if init:
        config_path = project_root / ".deo.toml"
        if config_path.exists():
            print_warning(f".deo.toml already exists at {config_path}")
        else:
            config_path.write_text(config.to_example_toml())
            print_success(f"Created {config_path}")
        return

    console.print(f"\n[bold]Deo Configuration[/]")
    console.print(f"  Project root: {project_root}")
    console.print(f"  Model:        {config.model.model_string}")
    console.print(f"  Temperature:  {config.model.temperature}")
    console.print(f"  ChromaDB:     {config.chroma_persist_path}")
    console.print(f"  Auto-index:   {config.chroma.auto_index}")
    console.print(f"  Subagents:    planner={config.subagents.planner} "
                  f"implementer={config.subagents.implementer} "
                  f"reviewer={config.subagents.reviewer} "
                  f"tester={config.subagents.tester}")
    console.print()


@app.command("profiles")
def profiles_command() -> None:
    """List all available agent profiles."""
    from deo.agents.registry import list_profiles

    console.print("\n[bold]Agent Profiles:[/]\n")
    for p in list_profiles():
        console.print(f"  [bold cyan]{p.name}[/]")
        console.print(f"    {p.description}")
        agents_on = [
            name for name, enabled in [
                ("planner", p.enable_planner),
                ("implementer", p.enable_implementer),
                ("reviewer", p.enable_reviewer),
                ("tester", p.enable_tester),
                ("documenter", p.enable_documenter),
            ] if enabled
        ]
        if agents_on:
            console.print(f"    Subagents: {', '.join(agents_on)}")
        else:
            console.print(f"    Subagents: [dim]none (orchestrator only)[/]")
        console.print()
