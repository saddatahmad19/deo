from pathlib import Path

from deo.agents.factory import _install_builtin_skills


def test_builtin_greenfield_skill_is_installed_inside_project_backend(tmp_path: Path):
    _install_builtin_skills(tmp_path)
    skill = tmp_path / ".deo" / "skills" / "greenfield-bootstrap" / "SKILL.md"
    assert skill.is_file()
    assert "omh-idea-to-deploy" in skill.read_text(encoding="utf-8")


def test_builtin_skill_install_does_not_overwrite_project_customization(tmp_path: Path):
    skill = tmp_path / ".deo" / "skills" / "greenfield-bootstrap" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# Project-customized skill\n", encoding="utf-8")
    _install_builtin_skills(tmp_path)
    assert skill.read_text(encoding="utf-8") == "# Project-customized skill\n"
