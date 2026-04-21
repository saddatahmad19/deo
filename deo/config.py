"""
deo/config.py

Hierarchical configuration system for Deo.
Loading order (later overrides earlier):
  1. Hardcoded defaults
  2. ~/.deo/config.toml  (global user config)
  3. .deo.toml           (project-local config)
  4. CLI flags           (runtime overrides)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """LLM model configuration."""
    provider: str = "anthropic"
    name: str = "claude-sonnet-4-6"
    max_retries: int = 6
    timeout: int = 120
    temperature: float = 0.0

    @property
    def model_string(self) -> str:
        return f"{self.provider}:{self.name}"

    def override(self, model_string: str | None) -> None:
        """Override model from a 'provider:name' string or just 'name'."""
        if not model_string:
            return
        if ":" in model_string:
            provider, name = model_string.split(":", 1)
            self.provider = provider
            self.name = name
        else:
            self.name = model_string


@dataclass
class ChromaConfig:
    """ChromaDB vector store configuration."""
    persist_dir: str = ".deo/chroma"
    collection_name: str = "codebase"
    chunk_size: int = 1500          # characters per chunk
    chunk_overlap: int = 200        # overlap between adjacent chunks
    auto_index: bool = True         # auto-index on startup
    n_results: int = 8              # default search result count
    # Glob patterns to exclude from indexing
    exclude_patterns: list[str] = field(default_factory=lambda: [
        "**/__pycache__/**",
        "**/*.pyc",
        "**/.git/**",
        "**/node_modules/**",
        "**/.deo/**",
        "**/dist/**",
        "**/build/**",
        "**/.venv/**",
        "**/venv/**",
        "**/*.min.js",
        "**/*.min.css",
        "**/*.lock",
        "**/package-lock.json",
        "**/*.map",
        "**/*.egg-info/**",
        "**/.mypy_cache/**",
        "**/.ruff_cache/**",
    ])
    # File extensions to index
    include_extensions: list[str] = field(default_factory=lambda: [
        ".py", ".ts", ".tsx", ".js", ".jsx",
        ".go", ".rs", ".java", ".kt",
        ".c", ".cpp", ".h", ".hpp",
        ".md", ".mdx", ".txt", ".rst",
        ".toml", ".yaml", ".yml", ".json",
        ".sql", ".graphql", ".proto",
        ".sh", ".bash", ".zsh",
        ".css", ".scss", ".html",
        ".env.example", "Dockerfile", "Makefile",
    ])


@dataclass
class SubagentToggles:
    """Enable/disable individual subagents."""
    planner: bool = True
    implementer: bool = True
    reviewer: bool = True
    tester: bool = True
    documenter: bool = False


@dataclass
class DisplayConfig:
    """Terminal display settings."""
    theme: str = "dark"         # dark | light
    show_tool_calls: bool = True
    show_thinking: bool = False  # show agent reasoning steps
    stream: bool = True


@dataclass
class DeoConfig:
    """Root configuration object."""
    model: ModelConfig = field(default_factory=ModelConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    subagents: SubagentToggles = field(default_factory=SubagentToggles)
    display: DisplayConfig = field(default_factory=DisplayConfig)

    # Runtime-populated
    root_dir: Path = field(default_factory=Path.cwd)
    deo_dir: Path = field(default_factory=lambda: Path.cwd() / ".deo")
    agents_md_path: str = "AGENTS.md"

    # ---------------------------------------------------------------------------
    # Loaders
    # ---------------------------------------------------------------------------

    @classmethod
    def load(cls, project_root: Path | None = None) -> "DeoConfig":
        """
        Load and merge config from global + project sources.

        Args:
            project_root: Path to project root. Defaults to cwd.
        """
        project_root = project_root or Path.cwd()
        config = cls()
        config.root_dir = project_root
        config.deo_dir = project_root / ".deo"

        # 1. Global user config
        global_path = Path.home() / ".deo" / "config.toml"
        if global_path.exists():
            with open(global_path, "rb") as f:
                config._apply(tomllib.load(f))

        # 2. Project-local config
        local_path = project_root / ".deo.toml"
        if local_path.exists():
            with open(local_path, "rb") as f:
                config._apply(tomllib.load(f))

        return config

    def _apply(self, data: dict[str, Any]) -> None:
        """Merge a raw TOML dict into this config."""
        if "model" in data:
            m = data["model"]
            for attr in ("provider", "name", "max_retries", "timeout", "temperature"):
                if attr in m:
                    setattr(self.model, attr, m[attr])

        if "chroma" in data:
            c = data["chroma"]
            for attr in ("persist_dir", "collection_name", "chunk_size",
                         "chunk_overlap", "auto_index", "n_results"):
                if attr in c:
                    setattr(self.chroma, attr, c[attr])
            if "exclude_patterns" in c:
                self.chroma.exclude_patterns.extend(c["exclude_patterns"])
            if "include_extensions" in c:
                self.chroma.include_extensions = c["include_extensions"]

        if "subagents" in data:
            s = data["subagents"]
            for attr in ("planner", "implementer", "reviewer", "tester", "documenter"):
                if attr in s:
                    setattr(self.subagents, attr, s[attr])

        if "display" in data:
            d = data["display"]
            for attr in ("theme", "show_tool_calls", "show_thinking", "stream"):
                if attr in d:
                    setattr(self.display, attr, d[attr])

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    @property
    def chroma_persist_path(self) -> Path:
        p = Path(self.chroma.persist_dir)
        if not p.is_absolute():
            p = self.root_dir / p
        return p

    def ensure_deo_dir(self) -> None:
        """Create .deo/ directory structure if it doesn't exist."""
        self.deo_dir.mkdir(parents=True, exist_ok=True)
        (self.deo_dir / "chroma").mkdir(exist_ok=True)

    def to_example_toml(self) -> str:
        """Generate an example .deo.toml with all defaults documented."""
        return """\
# Deo project configuration
# Place this at the root of your project as .deo.toml

[model]
# provider options:
#   anthropic   → ANTHROPIC_API_KEY       (claude-sonnet-4-6, claude-opus-4-7)
#   openai      → OPENAI_API_KEY          (gpt-4o, gpt-4o-mini, o3-mini)
#   gemini      → GOOGLE_API_KEY          (gemini-2.5-pro, gemini-2.0-flash)
#   azure       → AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT  (deployment name)
#   openrouter  → OPENROUTER_API_KEY      (openai/gpt-4o, anthropic/claude-3-5-sonnet)
provider = "anthropic"
name = "claude-sonnet-4-6"   # model name (or deployment name for azure)
max_retries = 6
timeout = 120
temperature = 0.0

[chroma]
persist_dir = ".deo/chroma"  # Where to store the ChromaDB index
collection_name = "codebase"
chunk_size = 1500              # Characters per chunk
chunk_overlap = 200
auto_index = true              # Auto-index on startup
n_results = 8                  # Semantic search result count

[subagents]
planner = true
implementer = true
reviewer = true
tester = true
documenter = false

[display]
theme = "dark"
show_tool_calls = true
show_thinking = false
stream = true
"""
