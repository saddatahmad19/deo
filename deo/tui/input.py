"""
deo/tui/input.py

prompt_toolkit-based REPL input with slash command autocomplete
and a Claude Code-style two-line status toolbar.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style


@dataclass
class SlashCommand:
    name: str
    description: str
    usage: str = ""
    aliases: list[str] = field(default_factory=list)


_STYLE = Style.from_dict({
    "prompt":                                    "bold #00d4ff",
    "completion-menu.completion":                "bg:#0a0a1e #7070a0",
    "completion-menu.completion.current":        "bg:#00bcd4 #000000 bold",
    "completion-menu.meta.completion":           "bg:#060616 #404055",
    "completion-menu.meta.completion.current":   "bg:#007080 #ccccdd",
    # toolbar rows
    "bottom-toolbar":                            "bg:#070714 #3a3a55",
    "bottom-toolbar.text":                       "bg:#070714 #3a3a55",
    # named classes used inside the HTML toolbar
    "tb.sep":                                    "bg:#070714 #1e1e35",
    "tb.label":                                  "bg:#070714 #3d3d5c",
    "tb.value":                                  "bg:#070714 #c8c8e8 bold",
    "tb.accent":                                 "bg:#070714 #00bcd4 bold",
    "tb.dim":                                    "bg:#070714 #28283d",
    "tb.pill-l":                                 "bg:#0e0e28 #00d4ff bold",
    "tb.pill-r":                                 "bg:#0e0e28 #7b7ba8",
    "b":                                         "bold #aaaacc",
})

_branch_cache: tuple[float, str] = (0.0, "")


def _git_branch() -> str:
    """Return the current git branch name, cached for 5 s to avoid repeated forks."""
    global _branch_cache
    now = time.monotonic()
    if now - _branch_cache[0] < 5.0:
        return _branch_cache[1]
    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            stderr=subprocess.DEVNULL,
            timeout=0.5,
        ).decode().strip()
    except Exception:
        branch = ""
    _branch_cache = (now, branch)
    return branch


class SlashCommandCompleter(Completer):
    def __init__(self, commands: list[SlashCommand]) -> None:
        self._commands = commands

    def get_completions(
        self, document: Document, complete_event
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        typed = text[1:].lower()
        seen: set[str] = set()
        for cmd in self._commands:
            for name in [cmd.name, *cmd.aliases]:
                if name not in seen and name.startswith(typed):
                    seen.add(name)
                    yield Completion(
                        f"/{name}",
                        start_position=-len(text),
                        display=f"/{name}",
                        display_meta=cmd.description,
                    )
                    break


def make_prompt_message() -> list[tuple[str, str]]:
    """Full-width separator line + prompt glyph, recomputed on every render."""
    import shutil
    cols = shutil.get_terminal_size().columns
    return [
        ("fg:#12122a", "▔" * cols + "\n"),
        ("fg:#00d4ff bold", "❯ "),
    ]


def build_prompt_session(
    commands: list[SlashCommand],
    session_id: str,
    model: str,
    cwd: str,
    get_n_indexed: Callable[[], int],
) -> PromptSession:
    completer = SlashCommandCompleter(commands)
    project = Path(cwd).name
    model_short = model.split(":")[-1] if ":" in model else model

    def toolbar() -> HTML:
        import shutil
        cols = shutil.get_terminal_size().columns

        branch = _git_branch()
        n = get_n_indexed()

        # ── Left cluster ─────────────────────────────────────────────────────
        # deo pill  ·  project (branch)  ·  session id
        branch_tag = (
            f'  <tb.sep>⌥</tb.sep>  <tb.label>{branch}</tb.label>'
            if branch else ""
        )
        left_plain = (
            f" deo   {project}"
            + (f"  ⌥  {branch}" if branch else "")
            + f"  ·  {session_id} "
        )
        left_html = (
            f' <tb.pill-l> deo </tb.pill-l>'
            f'  <tb.value>{project}</tb.value>'
            f'{branch_tag}'
            f'  <tb.dim>·</tb.dim>'
            f'  <tb.label>session</tb.label> <tb.value>{session_id}</tb.value>'
        )

        # ── Right cluster ─────────────────────────────────────────────────────
        # model  ·  N chunks  ·  /help
        right_plain = (
            f" {model_short}  ·  {n} chunks  ·  /help "
        )
        right_html = (
            f'<tb.label>model</tb.label> <tb.value>{model_short}</tb.value>'
            f'  <tb.dim>·</tb.dim>'
            f'  <tb.value>{n}</tb.value> <tb.label>chunks</tb.label>'
            f'  <tb.dim>·</tb.dim>'
            f'  <tb.accent>/help</tb.accent> '
        )

        # Measure plain-text widths to compute padding
        pad = cols - len(left_plain) - len(right_plain)
        spacer = " " * max(pad, 2)

        return HTML(left_html + spacer + right_html)

    return PromptSession(
        completer=completer,
        style=_STYLE,
        bottom_toolbar=toolbar,
        complete_while_typing=True,
        complete_in_thread=True,
        mouse_support=False,
    )
