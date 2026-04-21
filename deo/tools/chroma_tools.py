"""
deo/tools/chroma_tools.py

LangChain-compatible tools that expose ChromaDB semantic search to agents.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from deo.memory.chroma_manager import ChromaManager
    from deo.memory.indexer import CodebaseIndexer


_manager: "ChromaManager | None" = None
_indexer: "CodebaseIndexer | None" = None
_n_results: int = 8


def configure_chroma_tools(
    manager: "ChromaManager",
    indexer: "CodebaseIndexer",
    n_results: int = 8,
) -> None:
    """Inject the ChromaDB manager and indexer. Call once during Deo startup."""
    global _manager, _indexer, _n_results
    _manager = manager
    _indexer = indexer
    _n_results = n_results


@tool
def semantic_search(
    query: str,
    n_results: int = 0,
    language: str = "",
    filepath_contains: str = "",
) -> str:
    """
    Search the codebase semantically using natural language.

    Use this tool to find relevant code before writing new code, to understand
    existing patterns, or to locate where something is implemented.

    Args:
        query: Natural language description of what you're looking for.
        n_results: Number of results to return (default: uses project config, ~8).
        language: Optional — filter by language (e.g. "python", "typescript", "go").
        filepath_contains: Optional — filter results to paths containing this string.

    Returns:
        Formatted search results with file paths, line numbers, and code snippets.
    """
    if _manager is None:
        return "❌ ChromaDB not initialized. Run `deo index` first."

    count = _manager.count()
    if count == 0:
        return (
            "⚠️  The codebase index is empty. No files have been indexed yet.\n"
            "The index will be built automatically, or you can run `deo index` manually."
        )

    actual_n = n_results if n_results > 0 else _n_results

    where_filter: dict | None = None
    if language:
        where_filter = {"language": language.lower()}

    results = _manager.query(
        query_text=query,
        n_results=actual_n,
        where=where_filter,
    )

    if not results:
        return f"No results found for: '{query}'"

    if filepath_contains:
        results = [r for r in results if filepath_contains in r["filepath"]]

    if not results:
        return f"No results found for: '{query}' (with path filter: '{filepath_contains}')"

    lines = [f"🔍 Semantic search results for: '{query}'\n"]
    lines.append(f"Found {len(results)} relevant chunks:\n")

    for i, result in enumerate(results, 1):
        filepath = result["filepath"]
        start_line = result.get("start_line", 0)
        lang = result.get("language", "")
        distance = result.get("distance", 1.0)
        similarity_pct = round((1 - distance) * 100, 1)

        lines.append(f"{'─' * 60}")
        lines.append(f"[{i}] 📄 {filepath}  (line ~{start_line})  [{lang}]  {similarity_pct}% match")
        lines.append(f"{'─' * 60}")

        doc = result["document"]
        if len(doc) > 1200:
            doc = doc[:1200] + f"\n... [truncated, {len(result['document']) - 1200} more chars]"

        lines.append(f"```{lang}")
        lines.append(doc)
        lines.append("```\n")

    return "\n".join(lines)


@tool
def add_to_index(filepath: str) -> str:
    """
    Add or re-index a specific file in the ChromaDB codebase index.

    Use this immediately after creating or significantly modifying a file,
    so future semantic_search calls can find the new content.

    Args:
        filepath: Path to the file, relative to the project root.

    Returns:
        Confirmation message with number of chunks indexed.
    """
    if _indexer is None or _manager is None:
        return "❌ ChromaDB not initialized."

    path = Path(_indexer.root) / filepath
    if not path.exists():
        return f"❌ File not found: {filepath}"

    n = _indexer.index_file(path)
    if n > 0:
        return f"✅ Indexed '{filepath}' → {n} chunk(s) added to semantic search index."
    else:
        return f"⚠️  '{filepath}' produced no indexable content (empty or excluded)."


@tool
def index_status() -> str:
    """
    Show the current status of the ChromaDB codebase index.

    Returns information about number of indexed files, total chunks, and languages.
    """
    if _manager is None:
        return "❌ ChromaDB not initialized."

    total_chunks = _manager.count()
    indexed_files = _manager.get_indexed_files()
    n_files = len(indexed_files)

    if n_files == 0:
        return (
            "📭 Index is empty — no files indexed yet.\n"
            "Run `deo index` to build the initial index."
        )

    from deo.memory.indexer import detect_language
    lang_counts: dict[str, int] = {}
    for f in indexed_files:
        lang = detect_language(Path(f))
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    lang_summary = ", ".join(
        f"{lang}: {count}" for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1])
    )

    sample_files = sorted(indexed_files)[:20]
    file_list = "\n".join(f"  • {f}" for f in sample_files)
    if n_files > 20:
        file_list += f"\n  ... and {n_files - 20} more"

    return (
        f"📚 Index Status\n"
        f"{'─' * 40}\n"
        f"Files indexed:  {n_files}\n"
        f"Total chunks:   {total_chunks}\n"
        f"Languages:      {lang_summary}\n"
        f"\nIndexed files (sample):\n{file_list}"
    )


@tool
def find_file(pattern: str) -> str:
    """
    Find indexed files matching a path pattern.

    Use this when you know roughly where something is but need the exact path.

    Args:
        pattern: Partial path or filename to match (case-insensitive).

    Returns:
        List of matching file paths from the index.
    """
    if _manager is None:
        return "❌ ChromaDB not initialized."

    indexed = _manager.get_indexed_files()
    pattern_lower = pattern.lower()
    matches = [f for f in indexed if pattern_lower in f.lower()]

    if not matches:
        return f"No indexed files matching '{pattern}'."

    matches.sort()
    result = f"Files matching '{pattern}' ({len(matches)} found):\n"
    result += "\n".join(f"  📄 {f}" for f in matches[:50])
    if len(matches) > 50:
        result += f"\n  ... and {len(matches) - 50} more"

    return result


CHROMA_TOOLS = [semantic_search, add_to_index, index_status, find_file]
