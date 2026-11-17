"""
deo/memory/chroma_manager.py

ChromaDB Cloud collection management for Deo's semantic codebase search.
Credentials are read from environment variables (loaded from .env).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class ChromaManager:
    """
    Manages a persistent ChromaDB collection for codebase semantic search.

    Each document in the collection represents a *chunk* of a source file.
    Metadata stored per chunk:
      - filepath:   relative path from project root
      - language:   file extension / detected language
      - chunk_idx:  0-based index of this chunk within the file
      - start_line: approximate start line
      - end_line:   approximate end line
      - file_hash:  MD5 of the full file content at index time
      - indexed_at: unix timestamp
    """

    def __init__(self, persist_path: Path, collection_name: str = "codebase",
                 mode: str = "auto") -> None:
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb is not installed. Run: pip install chromadb"
            )

        self.persist_path = persist_path
        self.collection_name = collection_name

        has_cloud = all(os.environ.get(k) for k in ("CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE"))
        if mode == "cloud" or (mode == "auto" and has_cloud):
            self.backend = "cloud"
            self._client = chromadb.CloudClient(
                api_key=os.environ["CHROMA_API_KEY"],
                tenant=os.environ["CHROMA_TENANT"],
                database=os.environ["CHROMA_DATABASE"],
            )
        else:
            # No Cloud credentials: keep a local persistent store in .deo/chroma.
            self.backend = "local"
            persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_path / "vectors"))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # In-memory cache: filepath → file_hash (persisted locally)
        persist_path.mkdir(parents=True, exist_ok=True)
        self._hash_cache: dict[str, str] = self._load_hash_cache()
        from deo.memory.lexical import LexicalIndex
        self.lexical = LexicalIndex(persist_path / "lexical.sqlite")
        self.file_prior = None  # optional callable(filepath) -> float from deo.learning

    @property
    def client(self) -> Any:
        return self._client

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def add_chunks(
        self,
        filepath: str,
        chunks: list[str],
        language: str,
        file_hash: str,
        start_lines: list[int] | None = None,
        end_lines: list[int] | None = None,
        symbols: list[str] | None = None,
        kinds: list[str] | None = None,
    ) -> int:
        """
        Add or replace all chunks for a given file.

        Returns the number of chunks added.
        """
        # Remove stale chunks for this file first
        self._delete_file_chunks(filepath)

        if not chunks:
            return 0

        ids = [self._chunk_id(filepath, i) for i in range(len(chunks))]
        metadatas = [
            {
                "filepath": filepath,
                "language": language,
                "chunk_idx": i,
                "start_line": start_lines[i] if start_lines else 0,
                "end_line": (end_lines[i] if end_lines else
                             start_lines[i + 1] - 1 if start_lines and i + 1 < len(start_lines) else 0),
                "symbols": symbols[i] if symbols else "",
                "kind": kinds[i] if kinds else "",
                "file_hash": file_hash,
                "indexed_at": time.time(),
            }
            for i in range(len(chunks))
        ]

        # Chromadb has a max batch size of 5461; chunk into batches
        batch_size = 500
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            self._collection.add(
                ids=ids[batch_start:batch_end],
                documents=chunks[batch_start:batch_end],
                metadatas=metadatas[batch_start:batch_end],
            )

        self.lexical.replace_file(filepath, [
            {"id": ids[i], "document": chunks[i], **metadatas[i]} for i in range(len(chunks))
        ])
        self._hash_cache[filepath] = file_hash
        self._save_hash_cache()
        return len(chunks)

    def hybrid_query(
        self,
        query_text: str,
        n_results: int = 8,
        language: str = "",
        dense_weight: float = 0.6,
        max_per_file: int = 2,
    ) -> list[dict[str, Any]]:
        """Dense + BM25 (FTS5) retrieval fused with weighted RRF, then reranked.

        Boosts exact symbol hits and files that helped in successful episodes,
        and caps chunks per file so results cover more of the codebase.
        """
        from deo.memory.chunking import identifier_terms

        k = max(n_results * 4, 24)
        where = {"language": language.lower()} if language else None
        dense = []
        try:
            dense = self.query(query_text, n_results=k, where=where)
        except Exception:
            dense = []
        lexical = self.lexical.search(query_text, limit=k, language=language.lower())

        fused: dict[str, dict[str, Any]] = {}
        scores: dict[str, float] = {}
        for weight, results in ((dense_weight, dense), (1 - dense_weight, lexical)):
            for rank, r in enumerate(results):
                key = f"{r['filepath']}::{r.get('chunk_idx', 0)}"
                fused.setdefault(key, r)
                scores[key] = scores.get(key, 0.0) + weight / (60 + rank)
        q_idents = {t.lower() for t in query_text.replace(".", " ").split()} | set(identifier_terms(query_text))
        for key, r in fused.items():
            syms = {s.strip().lower().split(".")[-1] for s in str(r.get("symbols", "")).split(",") if s.strip()}
            if syms & q_idents:
                scores[key] += 0.01
            if self.file_prior is not None:
                scores[key] += 0.005 * self.file_prior(r["filepath"])
        per_file: dict[str, int] = {}
        out = []
        for key in sorted(scores, key=lambda x: -scores[x]):
            r = fused[key]
            if per_file.get(r["filepath"], 0) >= max_per_file:
                continue
            per_file[r["filepath"]] = per_file.get(r["filepath"], 0) + 1
            out.append({**r, "fused_score": round(scores[key], 5)})
            if len(out) >= n_results:
                break
        return out

    def query(
        self,
        query_text: str,
        n_results: int = 8,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic search over indexed code chunks.

        Returns a list of result dicts with keys:
          - document: the chunk text
          - filepath: source file path
          - language: file language
          - start_line: approximate start line
          - distance: cosine distance (lower = more similar)
        """
        count = self.count()
        if count == 0:
            return []

        actual_n = min(n_results, count)

        query_kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": actual_n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = self._collection.query(**query_kwargs)

        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            output.append({
                "document": doc,
                "filepath": meta.get("filepath", ""),
                "language": meta.get("language", ""),
                "chunk_idx": meta.get("chunk_idx", 0),
                "start_line": meta.get("start_line", 0),
                "end_line": meta.get("end_line", 0),
                "symbols": meta.get("symbols", ""),
                "distance": dist,
            })

        return output

    def needs_reindex(self, filepath: str, current_hash: str) -> bool:
        """Return True if the file has changed since last indexing."""
        return self._hash_cache.get(filepath) != current_hash

    def get_indexed_files(self) -> list[str]:
        """Return list of all currently indexed file paths."""
        return list(self._hash_cache.keys())

    def remove_file(self, filepath: str) -> None:
        """Remove all chunks for a given file from the index."""
        self._delete_file_chunks(filepath)
        self.lexical.delete_file(filepath)
        self._hash_cache.pop(filepath, None)
        self._save_hash_cache()

    def count(self) -> int:
        """Return total number of indexed chunks."""
        return self._collection.count()

    def clear(self) -> None:
        """Delete all chunks from the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._hash_cache = {}
        self._save_hash_cache()
        self.lexical.clear()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _chunk_id(self, filepath: str, chunk_idx: int) -> str:
        key = f"{filepath}::{chunk_idx}"
        return hashlib.md5(key.encode()).hexdigest()

    def _delete_file_chunks(self, filepath: str) -> None:
        """Delete all existing chunks for a file."""
        try:
            results = self._collection.get(where={"filepath": filepath})
            ids = results.get("ids", [])
            if ids:
                self._collection.delete(ids=ids)
        except Exception:
            pass  # Collection might be empty or file not indexed yet

    @property
    def _cache_file(self) -> Path:
        return self.persist_path / "file_hashes.json"

    def _load_hash_cache(self) -> dict[str, str]:
        if self._cache_file.exists():
            try:
                return json.loads(self._cache_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_hash_cache(self) -> None:
        try:
            self._cache_file.write_text(json.dumps(self._hash_cache, indent=2))
        except OSError:
            pass
