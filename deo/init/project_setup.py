"""Safe project bootstrap for existing and greenfield repositories."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from deo.init.agents_md import generate_agents_md, select_important_files


@dataclass
class ProjectSetupResult:
    greenfield: bool
    created: list[str] = field(default_factory=list)
    preserved: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


_IGNORED_FOR_EMPTY_CHECK = {
    ".git", ".deo", ".DS_Store", ".idea", ".vscode", ".github",
    ".gitignore", ".env", ".env.example", ".editorconfig",
    "README.md", "LICENSE", "NOTICE", "pyproject.toml", "package.json",
    "Cargo.toml", "go.mod", "pom.xml", "build.gradle", "Makefile",
}


def is_greenfield(root: Path) -> bool:
    """True when no meaningful project files exist yet."""
    for item in root.iterdir():
        if item.name in _IGNORED_FOR_EMPTY_CHECK or item.name == ".gitkeep":
            continue
        return False
    return True


def _write_if_missing(root: Path, relative: str, content: str, result: ProjectSetupResult) -> None:
    path = root / relative
    if path.exists():
        result.preserved.append(relative)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    result.created.append(relative)


def _ensure_gitignore(root: Path, result: ProjectSetupResult) -> None:
    path = root / ".gitignore"
    patterns = [
        "# Deo local state",
        ".deo/",
        "",
        "# Local secrets and credentials", ".env", ".env.*", "!.env.example",
        "*.pem", "*.key", "*.p12", "*.pfx", "",
        "# Operating system and editor files", ".DS_Store", "Thumbs.db", ".idea/", ".vscode/",
    ]
    if (root / "pyproject.toml").exists() or (root / "requirements.txt").exists():
        patterns += ["", "# Python", "__pycache__/", "*.py[cod]", ".venv/", "venv/", ".pytest_cache/", ".ruff_cache/"]
    if (root / "package.json").exists():
        patterns += ["", "# Node", "node_modules/", "dist/", "coverage/"]
    if (root / "Cargo.toml").exists():
        patterns += ["", "# Rust", "target/"]
    if not path.exists():
        path.write_text("\n".join(patterns) + "\n", encoding="utf-8")
        result.created.append(".gitignore")
        return
    existing = path.read_text(encoding="utf-8", errors="replace")
    required = [".deo/", ".env", ".env.*", "!.env.example", "*.pem", "*.key"]
    missing = [line for line in required if line not in existing.splitlines()]
    if missing:
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n# Deo local-secret safeguards\n" + "\n".join(missing) + "\n")
        result.created.append(".gitignore (secret safeguards appended)")
    else:
        result.preserved.append(".gitignore")


def _greenfield_agents_md(name: str) -> str:
    return f"""# Project: {name}

## Purpose
Describe the product, its users, and the problem it solves. Replace this placeholder before implementation.

## Working Contract
- Inspect the repository and its existing conventions before changing files.
- Prefer small, reviewable changes with focused tests.
- Never invent build, test, or deployment commands; discover them from the project toolchain.
- Run the relevant formatter, linter, tests, and build before claiming a change is verified.
- Keep secrets in local environment variables; never commit credentials or private keys.
- Update this file when durable architecture, commands, or contribution rules change.

## Architecture
Not established yet. Record observed components and their relationships as the project takes shape.

## Commands
No build, test, or lint commands have been verified yet. Add exact commands here after the toolchain is selected and the commands have been run.

## Boundaries
- Do not overwrite user data, credentials, or existing project configuration during bootstrap.
- Ask the project owner before choosing a software license, production environment, or external service.
"""


def _observed_agents_md(root: Path) -> str:
    """Produce a conservative context file when no model credential is available."""
    observed = select_important_files(root)
    paths = [f"- {path.relative_to(root).as_posix()}" for path in observed[:30]]
    listing = "\n".join(paths) if paths else "- No source files were detected."
    return f"""# Project: {root.name}

## Purpose
Not yet established. Confirm the product purpose and primary users with the project owner.

## Observed Repository Surface
The following files were found by a bounded repository scan. Their presence does not by itself establish runtime behavior:
{listing}

## Architecture
Not yet verified. Read the entrypoints and manifests before describing component boundaries.

## Commands
No build, test, lint, or deployment command has been executed by Deo during this bootstrap.

## Agent Guidance
- Treat the observed file list as navigation hints, not proof of behavior.
- Inspect the nearest package manifest and CI configuration before choosing commands.
- Preserve existing conventions and user-authored configuration.
- Run focused tests and the project's full verification suite before claiming completion.
- Keep credentials and local environment files out of version control.
"""


def _install_packaged_skills(root: Path, result: ProjectSetupResult) -> None:
    """Make bundled Deep Agents skills visible inside this project's backend."""
    packaged = Path(__file__).resolve().parents[1] / "skills"
    for skill_file in packaged.glob("*/SKILL.md"):
        relative = Path(".deo") / "skills" / skill_file.parent.name / "SKILL.md"
        target = root / relative
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(skill_file.read_text(encoding="utf-8"), encoding="utf-8")
        result.created.append(relative.as_posix())


def setup_project(
    root: Path,
    model_string: str = "freellmapi:auto:default",
    *,
    force: bool = False,
) -> ProjectSetupResult:
    """Create shared agent context and bootstrap files without clobbering user work."""
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    greenfield = is_greenfield(root)
    result = ProjectSetupResult(greenfield=greenfield)
    _ensure_gitignore(root, result)
    _install_packaged_skills(root, result)

    agents_path = root / "AGENTS.md"
    if not agents_path.exists() or force:
        if greenfield:
            content = _greenfield_agents_md(root.name)
        else:
            try:
                content, _ = generate_agents_md(root, model_string=model_string)
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("The model returned an empty AGENTS.md.")
            except Exception as exc:
                content = _observed_agents_md(root)
                result.notes.append(f"Model-based AGENTS.md generation was unavailable ({exc}); wrote a conservative evidence-based starter instead.")
        agents_path.write_text(content.rstrip() + "\n", encoding="utf-8")
        result.created.append("AGENTS.md")
    else:
        result.preserved.append("AGENTS.md")

    claude_path = root / "CLAUDE.md"
    pointer = "@AGENTS.md\n"
    if not claude_path.exists():
        claude_path.write_text(pointer, encoding="utf-8")
        result.created.append("CLAUDE.md")
    elif claude_path.read_text(encoding="utf-8", errors="replace").strip() == "@AGENTS.md":
        result.preserved.append("CLAUDE.md")
    else:
        result.preserved.append("CLAUDE.md")
        result.notes.append("Existing CLAUDE.md was preserved; add @AGENTS.md to it if you want Claude Code to import shared instructions.")

    if greenfield:
        _write_if_missing(
            root,
            "README.md",
            f"""# {root.name}

Project purpose and audience: TODO.

## Status

Greenfield scaffold. Select the language, runtime, package manager, and license with the project owner before documenting setup or release commands.

## Development

No build or test command is documented yet because no toolchain has been selected.
""",
            result,
        )
        result.notes.extend(
            [
                "Greenfield bootstrap applied: .gitignore, README.md, AGENTS.md, and CLAUDE.md.",
                "No LICENSE was created; the project owner must choose the license.",
                "CI and docs/ were deferred until a toolchain and verified commands exist and documentation actually overflows.",
                "Git initialization and the first commit are intentionally not performed by Deo.",
            ]
        )
    return result
