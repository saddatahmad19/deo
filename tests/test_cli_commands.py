from typer.testing import CliRunner

from deo.cli import app


def test_init_and_sync_expose_watch_mode():
    runner = CliRunner()
    for command in ("init", "sync"):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, result.exception
        assert "--watch" in result.output
