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
    provider: str = "freellmapi"
    name: str = "auto:default"
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
            known_providers = {
                "anthropic", "openai", "google", "gemini", "google_genai",
                "azure", "azure_openai", "openrouter", "freellmapi",
            }
            if provider.lower().replace("-", "_") in known_providers:
                self.provider = provider
                self.name = name
            else:
                # FreeLLMAPI routing IDs such as auto:default are model IDs.
                self.name = model_string
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
    # Backend: "auto" uses Chroma Cloud when CHROMA_API_KEY/TENANT/DATABASE are
    # set and a local persistent store under persist_dir otherwise.
    mode: str = "auto"              # auto | cloud | local
    # Hybrid retrieval: dense vectors fused (RRF) with lexical BM25. On Chroma
    # Cloud this uses a sparse BM25 index + the Search API; locally it uses a
    # SQLite FTS5 index kept next to the vector store.
    hybrid: bool = True
    dense_weight: float = 0.6       # RRF weight of dense results (lexical = 1 - dense)
    max_chunks_per_file: int = 2    # diversity cap applied after fusion
    # Embedding function: "default" (Chroma's built-in) or "qwen-code"
    # (Chroma Cloud Qwen3 embeddings with the nl_to_code retrieval task).
    embedding: str = "default"
    contextual_headers: bool = True # prefix chunks with path/symbol context before embedding
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
    """Enable/disable individual subagents and optionally override their model."""
    planner: bool = True
    implementer: bool = True
    reviewer: bool = True
    tester: bool = True
    documenter: bool = False

    # Per-role model overrides. None = inherit orchestrator model.
    # Example: "freellmapi:auto:default" or "openai:gpt-4o"
    planner_model: str | None = None
    implementer_model: str | None = None
    reviewer_model: str | None = None
    tester_model: str | None = None
    documenter_model: str | None = None


@dataclass
class RoutingConfig:
    """FreeLLMAPI routing: decision routes, per-role routes and task hints."""
    enabled: bool = True
    # Ask FreeLLMAPI's POST /v1/decisions which preset/category fits each turn.
    # Falls back to local heuristics on timeout or error.
    use_decisions: bool = True
    decision_model: str = "auto"
    decision_timeout: float = 2.5
    min_confidence: float = 0.5
    # Pick the agent profile for each turn from the task category, unless the
    # user pinned one with --profile or /profile.
    auto_profile: bool = True
    # Learn which preset works best per task category (Thompson sampling).
    adaptive: bool = True
    # Routing preset -> FreeLLMAPI model id.
    presets: dict[str, str] = field(default_factory=lambda: {
        "smartest": "auto:smart",
        "fastest": "auto:fast",
        "balanced": "auto:balanced",
        "reliable": "auto:reliable",
    })
    # Per-role preset used when a role has no explicit *_model override.
    # "turn" means: follow the preset decided for the current turn.
    role_presets: dict[str, str] = field(default_factory=lambda: {
        "orchestrator": "turn",
        "planner": "smartest",
        "implementer": "turn",
        "reviewer": "smartest",
        "tester": "reliable",
        "documenter": "fastest",
    })
    # X-FreeLLM-Compress header value ("" leaves the gateway default).
    compress: str = ""


@dataclass
class HarnessConfig:
    """External coding harnesses Deo can delegate to (aider, claude, codex, ...)."""
    enabled: bool = True
    timeout: int = 900              # seconds per delegated run
    # Harness ids to never use, even when installed.
    disabled: list[str] = field(default_factory=list)
    # Per-harness argv overrides, e.g. {"reasonix": ["reasonix", "run", "{task}"]}.
    commands: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class LearningConfig:
    """Continuous self-improvement from recorded outcomes."""
    enabled: bool = True
    inject_lessons: int = 3         # lessons recalled into each turn (0 disables)
    file_priors: bool = True        # boost files that helped in successful episodes


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
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    harnesses: HarnessConfig = field(default_factory=HarnessConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)

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
                         "chunk_overlap", "auto_index", "n_results", "mode", "hybrid",
                         "dense_weight", "max_chunks_per_file", "embedding",
                         "contextual_headers"):
                if attr in c:
                    setattr(self.chroma, attr, c[attr])
            if "exclude_patterns" in c:
                self.chroma.exclude_patterns.extend(c["exclude_patterns"])
            if "include_extensions" in c:
                self.chroma.include_extensions = c["include_extensions"]

        if "subagents" in data:
            s = data["subagents"]
            for attr in (
                "planner", "implementer", "reviewer", "tester", "documenter",
                "planner_model", "implementer_model", "reviewer_model",
                "tester_model", "documenter_model",
            ):
                if attr in s:
                    setattr(self.subagents, attr, s[attr])

        if "routing" in data:
            r = data["routing"]
            for attr in ("enabled", "use_decisions", "decision_model", "decision_timeout",
                         "min_confidence", "auto_profile", "adaptive", "compress"):
                if attr in r:
                    setattr(self.routing, attr, r[attr])
            if isinstance(r.get("presets"), dict):
                self.routing.presets.update(r["presets"])
            if isinstance(r.get("role_presets"), dict):
                self.routing.role_presets.update(r["role_presets"])

        if "harnesses" in data:
            h = data["harnesses"]
            for attr in ("enabled", "timeout", "disabled"):
                if attr in h:
                    setattr(self.harnesses, attr, h[attr])
            if isinstance(h.get("commands"), dict):
                self.harnesses.commands.update(h["commands"])

        if "learning" in data:
            lc = data["learning"]
            for attr in ("enabled", "inject_lessons", "file_priors"):
                if attr in lc:
                    setattr(self.learning, attr, lc[attr])

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
        (self.deo_dir / "experience").mkdir(exist_ok=True)

    @property
    def experience_dir(self) -> Path:
        return self.deo_dir / "experience"

    def to_example_toml(self) -> str:
        """Generate an example .deo.toml with all defaults documented."""
        return """\
# Deo project configuration
# Place this at the root of your project as .deo.toml

[model]
# provider options:
#   freellmapi  → FREELLMAPI_API_KEY + FREELLMAPI_BASE_URL (OpenAI-compatible gateway)
#   anthropic   → ANTHROPIC_API_KEY
#   openai      → OPENAI_API_KEY
#   gemini      → GOOGLE_API_KEY
#   azure       → AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT
#   openrouter  → OPENROUTER_API_KEY
provider = "freellmapi"
name = "auto:default"        # FreeLLMAPI automatic default routing profile
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
mode = "auto"                  # auto (cloud if CHROMA_* env set) | cloud | local
hybrid = true                  # dense + BM25 fusion (Cloud sparse index / local FTS5)
dense_weight = 0.6             # RRF weight for dense results
max_chunks_per_file = 2        # diversity cap per file
embedding = "default"          # default | qwen-code (Chroma Cloud, nl_to_code task)
contextual_headers = true      # embed chunks with path + symbol context

[subagents]
planner = true
implementer = true
reviewer = true
tester = true
documenter = false

# Optional: override the model for individual subagents (inherits orchestrator model if unset)
# planner_model = "freellmapi:auto:default"
# implementer_model = "freellmapi:auto:default"
# reviewer_model = "freellmapi:auto:default"
# tester_model = "freellmapi:auto:default"
# documenter_model = "freellmapi:auto:default"

[routing]
enabled = true
use_decisions = true           # ask FreeLLMAPI POST /v1/decisions per turn
decision_model = "auto"
decision_timeout = 2.5
min_confidence = 0.5
auto_profile = true            # choose the agent profile from the task category
adaptive = true                # learn the best preset per category from outcomes
compress = ""                  # X-FreeLLM-Compress: off|lossless|standard|aggressive

[routing.presets]
smartest = "auto:smart"
fastest = "auto:fast"
balanced = "auto:balanced"
reliable = "auto:reliable"

[routing.role_presets]         # "turn" follows the per-turn decision
orchestrator = "turn"
planner = "smartest"
implementer = "turn"
reviewer = "smartest"
tester = "reliable"
documenter = "fastest"

[harnesses]
enabled = true
timeout = 900
disabled = []
# commands.reasonix = ["reasonix", "run", "{task}"]

[learning]
enabled = true
inject_lessons = 3
file_priors = true

[display]
theme = "dark"
show_tool_calls = true
show_thinking = false
stream = true
"""
