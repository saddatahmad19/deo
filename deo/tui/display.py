"""
deo/tui/display.py

Terminal UI for Deo using Rich.

Provides:
  - Streaming token output with live updating
  - Styled tool call display
  - Status/progress messages
  - Welcome banner
  - Error formatting
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, Iterator

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.style import Style
from rich.text import Text
from rich.theme import Theme


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

DEO_THEME = Theme({
    "deo.banner": "bold cyan",
    "deo.prompt": "bold green",
    "deo.tool": "yellow",
    "deo.tool.name": "bold yellow",
    "deo.tool.result": "dim white",
    "deo.subagent": "bold magenta",
    "deo.error": "bold red",
    "deo.warning": "yellow",
    "deo.success": "bold green",
    "deo.info": "dim cyan",
    "deo.thinking": "dim italic white",
    "deo.code": "green",
    "deo.filepath": "cyan underline",
    "deo.dim": "dim white",
})

console = Console(theme=DEO_THEME, highlight=True)
err_console = Console(theme=DEO_THEME, stderr=True)


# ---------------------------------------------------------------------------
# Banner & greeting
# ---------------------------------------------------------------------------

BANNER = r"""
  ██████╗ ███████╗ ██████╗
  ██╔══██╗██╔════╝██╔═══██╗
  ██║  ██║█████╗  ██║   ██║
  ██║  ██║██╔══╝  ██║   ██║
  ██████╔╝███████╗╚██████╔╝
  ╚═════╝ ╚══════╝ ╚═════╝
"""


def print_banner(version: str = "0.1.0") -> None:
    console.print(Text(BANNER, style="deo.banner"))
    console.print(
        f"  Terminal AI Coding Agent  [dim]v{version}[/]  "
        f"[dim]—  type [bold]/help[/] for commands[/]\n"
    )


def print_rule(title: str = "") -> None:
    console.print(Rule(title, style="dim"))


def print_info(msg: str) -> None:
    console.print(f"[deo.info]ℹ  {msg}[/]")


def print_success(msg: str) -> None:
    console.print(f"[deo.success]✅ {msg}[/]")


def print_warning(msg: str) -> None:
    console.print(f"[deo.warning]⚠  {msg}[/]")


def print_error(msg: str) -> None:
    err_console.print(f"[deo.error]✗  {msg}[/]")


# ---------------------------------------------------------------------------
# Indexing progress
# ---------------------------------------------------------------------------

def make_index_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


# ---------------------------------------------------------------------------
# Streaming response display
# ---------------------------------------------------------------------------

class StreamDisplay:
    """
    Manages the streaming display of agent responses.

    Usage:
        display = StreamDisplay()
        with display.streaming_context():
            for chunk in agent.stream(...):
                display.handle_chunk(chunk)
    """

    def __init__(self, show_tool_calls: bool = True, show_thinking: bool = False) -> None:
        self.show_tool_calls = show_tool_calls
        self.show_thinking = show_thinking
        self._current_content = ""
        self._in_tool = False
        self._current_tool_name = ""

    @contextmanager
    def streaming_context(self) -> Iterator[None]:
        """Context manager for a single streaming response."""
        self._current_content = ""
        self._in_tool = False
        try:
            yield
        finally:
            # Print any remaining buffered content
            if self._current_content:
                # Render final content as Markdown if non-trivial
                self._flush_content()

    def handle_stream_chunk(self, chunk: Any) -> None:
        """
        Process a single chunk from agent.stream(stream_mode='messages').
        chunk is a (message_chunk, metadata) tuple.
        """
        if not isinstance(chunk, tuple) or len(chunk) != 2:
            return

        message, metadata = chunk
        content = getattr(message, "content", None)

        if content is None:
            return

        # Handle tool calls
        tool_calls = getattr(message, "tool_calls", None) or []
        for tc in tool_calls:
            if self.show_tool_calls:
                self._print_tool_call(tc)

        # Handle text content
        if isinstance(content, str) and content:
            self._stream_text(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text" and block.get("text"):
                        self._stream_text(block["text"])
                    elif block.get("type") == "tool_use" and self.show_tool_calls:
                        self._print_tool_call_block(block)
                    elif block.get("type") == "thinking" and self.show_thinking:
                        thinking_text = block.get("thinking", "")
                        if thinking_text:
                            console.print(f"[deo.thinking]💭 {thinking_text[:200]}...[/]")

    def _stream_text(self, text: str) -> None:
        """Buffer and print streaming text tokens."""
        self._current_content += text
        # Print token-by-token for live feel
        console.print(text, end="", markup=False, highlight=False)

    def _flush_content(self) -> None:
        """Flush any remaining buffered content."""
        console.print("")  # Ensure newline after streaming

    def _print_tool_call(self, tool_call: dict) -> None:
        name = tool_call.get("name", "unknown")
        args = tool_call.get("args", {})
        self._print_tool_header(name, args)

    def _print_tool_call_block(self, block: dict) -> None:
        name = block.get("name", "unknown")
        inp = block.get("input", {})
        self._print_tool_header(name, inp)

    def _print_tool_header(self, name: str, args: dict) -> None:
        # Map tool names to icons
        icons = {
            "semantic_search": "🔍",
            "add_to_index": "📥",
            "index_status": "📊",
            "find_file": "🗂",
            "git_status": "📋",
            "git_diff": "±",
            "git_log": "📜",
            "git_branch": "🌿",
            "write_file": "✍",
            "read_file": "📖",
            "edit_file": "✏",
            "execute": "⚡",
            "write_todos": "📝",
            "task": "🤖",
            "ls": "📁",
            "glob": "🔎",
            "grep": "🔬",
        }
        icon = icons.get(name, "🔧")

        # Summarize args
        arg_summary = ""
        if args:
            first_key = next(iter(args))
            first_val = str(args[first_key])
            if len(first_val) > 60:
                first_val = first_val[:60] + "..."
            arg_summary = f"[dim]{first_key}=[/][deo.dim]{first_val!r}[/]"
            if len(args) > 1:
                arg_summary += f"[dim] +{len(args)-1} more[/]"

        console.print(
            f"\n[deo.tool]{icon} [bold]{name}[/][/] {arg_summary}",
            highlight=False,
        )

    def handle_subagent_spawn(self, agent_name: str, task: str) -> None:
        """Display when the orchestrator spawns a subagent."""
        console.print(
            Panel(
                f"[dim]{task[:120]}{'...' if len(task) > 120 else ''}[/]",
                title=f"[deo.subagent]🤖 Spawning {agent_name}[/]",
                border_style="magenta",
                padding=(0, 1),
            )
        )


def print_tool_result(tool_name: str, result: str, truncate: int = 400) -> None:
    """Print a tool result inline."""
    if len(result) > truncate:
        result = result[:truncate] + f"\n[dim]... ({len(result) - truncate} chars truncated)[/]"
    console.print(f"[deo.tool.result]  → {result}[/]", highlight=False)


def print_final_response(content: str) -> None:
    """Render the final agent response as Markdown."""
    console.print()
    console.print(Markdown(content))


def prompt_user(session_id: str) -> str:
    """Get user input with a styled prompt."""
    try:
        return console.input(f"[deo.prompt]❯[/] ").strip()
    except (EOFError, KeyboardInterrupt):
        return "/exit"


def print_session_info(session_id: str, model: str, root_dir: str, n_indexed: int) -> None:
    """Print session metadata at startup."""
    console.print(Panel(
        f"[dim]session:[/] [bold]{session_id}[/]\n"
        f"[dim]model:[/]   [bold]{model}[/]\n"
        f"[dim]root:[/]    [deo.filepath]{root_dir}[/]\n"
        f"[dim]index:[/]   [bold]{n_indexed}[/] chunks in semantic search",
        title="[deo.banner]⚡ Deo Session[/]",
        border_style="cyan",
        padding=(0, 2),
    ))


# ---------------------------------------------------------------------------
# Slash command help
# ---------------------------------------------------------------------------

HELP_TEXT = """\
## Deo Commands

| Command | Description |
|---------|-------------|
| `/help` | Show this help message |
| `/clear` | Clear the terminal screen |
| `/status` | Show index status and session info |
| `/index` | Re-index the codebase |
| `/profile <name>` | Switch agent profile |
| `/profiles` | List available agent profiles |
| `/exit` or `/quit` | Exit Deo |

## Tips

- Use **semantic_search** — Deo automatically searches your codebase before coding
- Start complex tasks with a clear description; the planner will break it down
- Add `AGENTS.md` to your project root with architecture notes for persistent context
- Use `.deo.toml` in your project for project-specific config
"""


def print_help() -> None:
    console.print(Markdown(HELP_TEXT))
