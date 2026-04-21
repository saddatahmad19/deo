"""
deo/memory/indexer.py

Codebase indexer for Forge's ChromaDB semantic search.

Responsibilities:
  - Walk the project directory tree respecting .gitignore + custom excludes
  - Chunk source files intelligently (by function/class for code, paragraph for prose)
  - Compute file hashes for incremental re-indexing
  - Report progress during initial indexing
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

try:
    import pathspec
    PATHSPEC_AVAILABLE = True
except ImportError:
    PATHSPEC_AVAILABLE = False

from deo.config import ChromaConfig
from deo.memory.chroma_manager import ChromaManager


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class IndexResult:
    files_indexed: int = 0
    files_skipped: int = 0
    chunks_added: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def __str__(self) -> str:
        return (
            f"Indexed {self.files_indexed} files "
            f"({self.chunks_added} chunks) in {self.elapsed_seconds:.1f}s. "
            f"{self.files_skipped} unchanged, {len(self.errors)} errors."
        )


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".cs": "csharp",
    ".md": "markdown", ".mdx": "markdown",
    ".txt": "text", ".rst": "text",
    ".sql": "sql",
    ".graphql": "graphql", ".gql": "graphql",
    ".sh": "bash", ".bash": "bash", ".zsh": "zsh",
    ".toml": "toml",
    ".yaml": "yaml", ".yml": "yaml",
    ".json": "json",
    ".html": "html", ".htm": "html",
    ".css": "css", ".scss": "scss",
    ".proto": "protobuf",
    "Dockerfile": "dockerfile",
    "Makefile": "makefile",
}


def detect_language(filepath: Path) -> str:
    """Return a language identifier for a given file path."""
    name = filepath.name
    if name in _EXT_TO_LANGUAGE:
        return _EXT_TO_LANGUAGE[name]
    suffix = filepath.suffix.lower()
    return _EXT_TO_LANGUAGE.get(suffix, "text")


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

def _chunk_by_size(
    content: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[str], list[int]]:
    """
    Simple character-based chunking with overlap.
    Returns (chunks, start_lines) where start_lines[i] is the 1-indexed
    line number where chunk i begins (approximate).
    """
    if len(content) <= chunk_size:
        return [content], [1]

    chunks = []
    start_positions = []
    pos = 0

    while pos < len(content):
        end = min(pos + chunk_size, len(content))

        # Try to break at a newline near the chunk boundary
        if end < len(content):
            newline_pos = content.rfind("\n", pos, end)
            if newline_pos > pos + chunk_size // 2:
                end = newline_pos + 1

        chunk = content[pos:end]
        if chunk.strip():
            chunks.append(chunk)
            start_positions.append(content[:pos].count("\n") + 1)

        pos = end - chunk_overlap
        if pos <= 0:
            pos = end

    return chunks, start_positions


def _chunk_python(
    content: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[str], list[int]]:
    """
    Python-aware chunking: split at class/function boundaries.
    Falls back to size-based chunking for large blocks.
    """
    # Split at top-level class/function definitions
    pattern = re.compile(r'^(?:class |def |async def )', re.MULTILINE)
    lines = content.splitlines(keepends=True)
    split_lines: list[int] = [0]

    for match in pattern.finditer(content):
        line_no = content[:match.start()].count("\n")
        if line_no > 0 and line_no not in split_lines:
            split_lines.append(line_no)

    split_lines.append(len(lines))
    split_lines = sorted(set(split_lines))

    chunks = []
    start_lines = []

    for i in range(len(split_lines) - 1):
        start = split_lines[i]
        end = split_lines[i + 1]
        block = "".join(lines[start:end])

        if len(block) > chunk_size:
            # Block too large — fall back to size-based
            sub_chunks, sub_starts = _chunk_by_size(block, chunk_size, chunk_overlap)
            for sc, ss in zip(sub_chunks, sub_starts):
                chunks.append(sc)
                start_lines.append(start + ss)
        elif block.strip():
            chunks.append(block)
            start_lines.append(start + 1)

    return chunks or [content], start_lines or [1]


def chunk_file(
    content: str,
    language: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[str], list[int]]:
    """
    Chunk a file's content using the best strategy for its language.

    Returns:
        (chunks, start_lines) where start_lines[i] is the 1-indexed
        line number where chunk i begins (approximate).
    """
    if not content.strip():
        return [], []

    if language == "python":
        return _chunk_python(content, chunk_size, chunk_overlap)
    else:
        return _chunk_by_size(content, chunk_size, chunk_overlap)


# ---------------------------------------------------------------------------
# File hash
# ---------------------------------------------------------------------------

def file_hash(content: str) -> str:
    return hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------

def iter_indexable_files(
    root: Path,
    config: ChromaConfig,
) -> Iterator[Path]:
    """
    Yield all indexable files under `root`, respecting:
      - config.include_extensions
      - config.exclude_patterns (pathspec gitignore-style)
      - .gitignore at root (if pathspec is available)
    """
    # Build a spec from config exclude patterns
    spec_patterns = list(config.exclude_patterns)

    # Also read .gitignore if it exists
    gitignore_path = root / ".gitignore"
    if gitignore_path.exists() and PATHSPEC_AVAILABLE:
        try:
            spec_patterns.extend(gitignore_path.read_text().splitlines())
        except OSError:
            pass

    if PATHSPEC_AVAILABLE and spec_patterns:
        exclude_spec = pathspec.PathSpec.from_lines("gitwildmatch", spec_patterns)
    else:
        exclude_spec = None

    include_exts = set(config.include_extensions)
    include_names = {e for e in include_exts if not e.startswith(".")}  # e.g. "Dockerfile"

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(root)
        rel_str = str(rel)

        # Check against exclude spec
        if exclude_spec and exclude_spec.match_file(rel_str):
            continue

        # Check extension or filename
        if path.suffix.lower() not in include_exts and path.name not in include_names:
            continue

        # Skip files that are too large (>500KB — likely generated or binary)
        try:
            if path.stat().st_size > 500_000:
                continue
        except OSError:
            continue

        yield path


# ---------------------------------------------------------------------------
# Main indexer
# ---------------------------------------------------------------------------

class CodebaseIndexer:
    """
    High-level indexer that walks a project and populates ChromaDB.
    """

    def __init__(
        self,
        root: Path,
        manager: ChromaManager,
        config: ChromaConfig,
    ) -> None:
        self.root = root
        self.manager = manager
        self.config = config

    def index(
        self,
        force: bool = False,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> IndexResult:
        """
        Index (or re-index) the entire codebase.

        Args:
            force: If True, re-index all files even if unchanged.
            progress_callback: Called with (filepath, current, total) for progress reporting.

        Returns:
            IndexResult with statistics.
        """
        start = time.time()
        result = IndexResult()

        files = list(iter_indexable_files(self.root, self.config))
        total = len(files)

        for i, path in enumerate(files):
            if progress_callback:
                progress_callback(str(path.relative_to(self.root)), i + 1, total)

            try:
                rel_path = str(path.relative_to(self.root))
                content = path.read_text(encoding="utf-8", errors="replace")
                fhash = file_hash(content)

                if not force and not self.manager.needs_reindex(rel_path, fhash):
                    result.files_skipped += 1
                    continue

                language = detect_language(path)
                chunks, start_lines = chunk_file(
                    content,
                    language,
                    self.config.chunk_size,
                    self.config.chunk_overlap,
                )

                n = self.manager.add_chunks(
                    filepath=rel_path,
                    chunks=chunks,
                    language=language,
                    file_hash=fhash,
                    start_lines=start_lines,
                )
                result.files_indexed += 1
                result.chunks_added += n

            except Exception as e:
                result.errors.append(f"{path}: {e}")

        # Cleanup: remove indexed files that no longer exist
        current_rel_paths = {
            str(p.relative_to(self.root)) for p in files
        }
        for indexed_path in list(self.manager.get_indexed_files()):
            if indexed_path not in current_rel_paths:
                self.manager.remove_file(indexed_path)

        result.elapsed_seconds = time.time() - start
        return result

    def index_file(self, path: Path) -> int:
        """
        Index or re-index a single file. Returns number of chunks added.
        """
        try:
            rel_path = str(path.relative_to(self.root))
            content = path.read_text(encoding="utf-8", errors="replace")
            fhash = file_hash(content)
            language = detect_language(path)
            chunks, start_lines = chunk_file(
                content,
                language,
                self.config.chunk_size,
                self.config.chunk_overlap,
            )
            return self.manager.add_chunks(
                filepath=rel_path,
                chunks=chunks,
                language=language,
                file_hash=fhash,
                start_lines=start_lines,
            )
        except Exception:
            return 0
