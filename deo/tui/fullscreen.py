"""Full-screen terminal interface for Deo.

The screen is an alternate-screen application with a persistent transcript,
a bottom-pinned composer, a live status line, and asynchronous agent turns.
Agent execution runs off the UI event loop so streaming never freezes input
rendering or terminal resize handling.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea
from prompt_toolkit.document import Document


OutputCallback = Callable[[str], None]
SubmitCallback = Callable[[str, OutputCallback], None]


class FullScreenTUI:
    """A responsive full-screen Deo chat surface."""

    def __init__(
        self,
        *,
        project: str,
        model: str,
        session_id: str,
        indexed_chunks: Callable[[], int],
        on_submit: SubmitCallback,
    ) -> None:
        self.project = project
        self.model = model
        self.session_id = session_id
        self.indexed_chunks = indexed_chunks
        self.on_submit = on_submit
        self.busy = False
        self.exit_requested = False
        self.status = "Ready"
        self._output = Buffer(multiline=True)
        self._output.insert_text(
            "  DEO  /  THE AI THAT CODES WHILE YOU THINK\n"
            f"  {project}  ·  {model}  ·  session {session_id}\n"
            "  Type a task, /help for commands, Ctrl+C to exit.\n\n"
        )

        self.input = TextArea(
            height=1,
            prompt="❯ ",
            multiline=False,
            wrap_lines=False,
            style="class:composer",
            history=InMemoryHistory(),
            completer=WordCompleter(["/help", "/status", "/index", "/init", "/sync", "/profiles", "/profile", "/clear", "/exit"], sentence=True),
            complete_while_typing=True,
            accept_handler=self._accept,
        )
        self.transcript = Window(
            content=BufferControl(buffer=self._output, focusable=False),
            wrap_lines=True,
            always_hide_cursor=True,
            style="class:transcript",
        )
        self.header = Window(
            content=FormattedTextControl(self._header_text),
            height=3,
            style="class:header",
        )
        self.status_bar = Window(
            content=FormattedTextControl(self._status_text),
            height=1,
            style="class:status",
        )
        self.hint_bar = Window(
            content=FormattedTextControl(
                [("class:hint", " Enter send  ·  / commands  ·  Ctrl+C quit  ·  Ctrl+L clear ")]
            ),
            height=1,
            style="class:hint",
        )
        self.key_bindings = KeyBindings()
        self.key_bindings.add("c-c")(self._exit)
        self.key_bindings.add("c-l")(self._clear)
        self.application = Application(
            layout=Layout(
                HSplit(
                    [
                        self.header,
                        Window(height=1, char="─", style="class:rule"),
                        self.transcript,
                        Window(height=1, char="─", style="class:rule"),
                        self.input,
                        self.status_bar,
                        self.hint_bar,
                    ]
                ),
                focused_element=self.input,
            ),
            key_bindings=self.key_bindings,
            style=Style.from_dict(
                {
                    "header": "bg:#111827 #67e8f9",
                    "transcript": "bg:#080d18 #e5e7eb",
                    "composer": "bg:#111827 #f8fafc",
                    "status": "bg:#111827 #a5b4fc",
                    "hint": "bg:#080d18 #64748b",
                    "rule": "#334155",
                    "completion-menu": "bg:#1e293b #e2e8f0",
                    "completion-menu.completion.current": "bg:#0891b2 #ffffff",
                }
            ),
            full_screen=True,
            mouse_support=True,
            refresh_interval=0.5,
        )

    def _header_text(self):
        chunks = self.indexed_chunks()
        state = "● WORKING" if self.busy else "● READY"
        return [
            ("bold #67e8f9", "  DEO  "),
            ("#94a3b8", "│  "),
            ("bold #f8fafc", self.project),
            ("#64748b", "  │  "),
            ("#c4b5fd", self.model),
            ("#64748b", "  │  "),
            ("#94a3b8", f"{chunks:,} indexed chunks"),
            ("#64748b", "  │  "),
            ("bold #34d399" if not self.busy else "bold #fbbf24", state),
            ("", "\n"),
            ("#64748b", "  THE AI THAT CODES WHILE YOU THINK"),
            ("", "\n"),
        ]

    def _status_text(self):
        state = "Working…" if self.busy else self.status
        return [("bg:#111827 #a5b4fc", f"  {state}  ·  {self.model}  ·  {self.indexed_chunks():,} chunks")]

    def _append(self, text: str) -> None:
        if not text:
            return
        current = self._output.text
        if len(current) + len(text) > 250_000:
            current = current[-200_000:]
            self._output.set_document(Document(current, len(current)), bypass_readonly=True)
        self._output.insert_text(text, bypass_readonly=True)
        self._output.cursor_position = len(self._output.text)
        self.application.invalidate()

    async def _run_turn(self, prompt: str) -> None:
        loop = asyncio.get_running_loop()

        def emit(text: str) -> None:
            loop.call_soon_threadsafe(self._append, text)

        try:
            await asyncio.to_thread(self.on_submit, prompt, emit)
            self.status = "Turn complete"
        except Exception as exc:
            self._append(f"\n[Deo error] {exc}\n")
            self.status = "Turn failed"
        finally:
            self.busy = False
            self.application.invalidate()
            if self.exit_requested:
                self.application.exit()

    def _accept(self, buffer: Buffer) -> bool:
        prompt = buffer.text.strip()
        if not prompt:
            return True
        if self.busy:
            self.status = "A turn is already running"
            self.application.invalidate()
            return True
        self._append(f"\n❯ {prompt}\n\n")
        self.busy = True
        self.status = "Thinking"
        self.application.invalidate()
        self.application.create_background_task(self._run_turn(prompt))
        return True

    def request_exit(self) -> None:
        """Exit after the current callback unwinds."""
        self.exit_requested = True

    def clear_transcript(self) -> None:
        """Clear the conversation surface without leaving the alternate screen."""
        self._output.set_document(Document("", 0), bypass_readonly=True)
        self.application.invalidate()

    def _clear(self, event) -> None:
        self.clear_transcript()

    def _exit(self, event) -> None:
        if self.busy:
            self.status = "A turn is running; wait for it to finish before exiting"
            self.application.invalidate()
            return
        event.app.exit()

    def run(self) -> None:
        self.application.run()
