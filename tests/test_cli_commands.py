from typer.main import get_command

from deo.cli import app


def test_init_and_sync_register_watch_mode():
    root = get_command(app)
    for name in ("init", "sync"):
        assert name in root.commands
        parameters = {parameter.name for parameter in root.commands[name].params}
        assert "watch" in parameters
        assert "interval" in parameters
