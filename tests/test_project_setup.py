from pathlib import Path

from deo.init.project_setup import is_greenfield, setup_project


def test_empty_directory_is_greenfield(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    assert is_greenfield(tmp_path)


def test_project_with_source_is_not_greenfield(tmp_path: Path):
    (tmp_path / "main.py").write_text("print('hello')", encoding="utf-8")
    assert not is_greenfield(tmp_path)


def test_greenfield_setup_creates_shared_agent_context(tmp_path: Path):
    result = setup_project(tmp_path)
    assert result.greenfield
    assert (tmp_path / ".gitignore").exists()
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "AGENTS.md").exists()
    assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8") == "@AGENTS.md\n"
    assert not (tmp_path / "LICENSE").exists()


def test_setup_preserves_custom_claude_and_agents_files(tmp_path: Path):
    (tmp_path / "main.py").write_text("print('hello')", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# Custom rules\n", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("# Custom Claude rules\n", encoding="utf-8")
    result = setup_project(tmp_path)
    assert (tmp_path / "AGENTS.md").read_text(encoding="utf-8") == "# Custom rules\n"
    assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8") == "# Custom Claude rules\n"
    assert "AGENTS.md" in result.preserved
    assert "CLAUDE.md" in result.preserved


def test_existing_project_uses_observed_fallback_without_model_credentials(tmp_path: Path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='sample'\n", encoding="utf-8")

    def unavailable(*args, **kwargs):
        raise RuntimeError("missing gateway key")

    monkeypatch.setattr("deo.init.project_setup.generate_agents_md", unavailable)
    result = setup_project(tmp_path)
    generated = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    assert "Observed Repository Surface" in generated
    assert any("missing gateway key" in note for note in result.notes)


def test_fresh_git_repository_with_only_metadata_is_greenfield(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text(".env\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# New project\n", encoding="utf-8")
    assert is_greenfield(tmp_path)
