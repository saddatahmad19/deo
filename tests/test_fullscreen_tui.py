from deo.tui.fullscreen import FullScreenTUI
from deo.tui.display import StreamDisplay


def test_fullscreen_tui_builds_persistent_terminal_surface():
    tui = FullScreenTUI(
        project="demo",
        model="freellmapi:auto:default",
        session_id="session1",
        indexed_chunks=lambda: 12,
        on_submit=lambda prompt, write: write("response"),
    )
    assert tui.application is not None
    assert tui.input is not None
    assert tui.transcript is not None
    assert tui.indexed_chunks() == 12


def test_stream_display_routes_output_to_fullscreen_callback():
    chunks = []
    display = StreamDisplay(output_callback=chunks.append)
    display._stream_text("hello")
    display._flush_content()
    assert "".join(chunks) == "hello\n"
