"""
deo/memory/lexical.py

Local BM25 lexical index (SQLite FTS5) mirrored alongside the vector store.

Why: dense embeddings are weak at exact identifiers, error strings and rare
tokens — exactly what engineers search for. FTS5 gives BM25 ranking over raw
text, chunk symbols and split identifier parts (getUserById → get user by id),
works offline, and powers exact symbol lookup. On Chroma Cloud with a sparse
BM25 index the fusion happens server-side; this index still serves symbol
lookup, neighbour expansion and offline fallback.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

from deo.memory.chunking import identifier_terms

_QUERY_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_STOP = {"the", "and", "for", "with", "that", "this", "from", "into", "what", "where",
         "how", "does", "are", "is", "of", "to", "in", "a", "an", "on", "it", "be", "or"}


def fts_query(text: str) -> str:
    """Build an OR query of quoted terms (FTS5 syntax-safe)."""
    terms: list[str] = []
    for tok in _QUERY_TOKEN.findall(text):
        low = tok.lower()
        if len(low) < 2 or low in _STOP:
            continue
        terms.append(low)
    terms.extend(identifier_terms(text))
    seen = list(dict.fromkeys(terms))[:32]
    return " OR ".join(f'"{t}"' for t in seen)


class LexicalIndex:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5("
            "chunk_id UNINDEXED, filepath, language UNINDEXED, symbols, content, terms, "
            "chunk_idx UNINDEXED, start_line UNINDEXED, end_line UNINDEXED, kind UNINDEXED, "
            "tokenize = \"unicode61 tokenchars '_'\")"
        )
        self._db.commit()

    def replace_file(self, filepath: str, rows: list[dict[str, Any]]) -> None:
        with self._lock:
            self._db.execute("DELETE FROM chunks WHERE filepath = ?", (filepath,))
            self._db.executemany(
                "INSERT INTO chunks (chunk_id, filepath, language, symbols, content, terms, "
                "chunk_idx, start_line, end_line, kind) VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(r["id"], filepath, r["language"], r.get("symbols", ""), r["document"],
                  " ".join(identifier_terms(r["document"]) + identifier_terms(filepath)),
                  r.get("chunk_idx", 0), r.get("start_line", 0), r.get("end_line", 0),
                  r.get("kind", "")) for r in rows],
            )
            self._db.commit()

    def delete_file(self, filepath: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM chunks WHERE filepath = ?", (filepath,))
            self._db.commit()

    def clear(self) -> None:
        with self._lock:
            self._db.execute("DELETE FROM chunks")
            self._db.commit()

    def count(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    _COLS = "chunk_id, filepath, language, symbols, content, chunk_idx, start_line, end_line, kind"

    def _rows(self, cur) -> list[dict[str, Any]]:
        out = []
        for row in cur.fetchall():
            out.append({
                "id": row[0], "filepath": row[1], "language": row[2], "symbols": row[3],
                "document": row[4], "chunk_idx": int(row[5] or 0), "start_line": int(row[6] or 0),
                "end_line": int(row[7] or 0), "kind": row[8], **({"score": row[9]} if len(row) > 9 else {}),
            })
        return out

    def search(self, query: str, limit: int = 24, language: str = "") -> list[dict[str, Any]]:
        match = fts_query(query)
        if not match:
            return []
        # bm25 column weights: chunk_id, filepath, language, symbols, content, terms, ...
        sql = (f"SELECT {self._COLS}, bm25(chunks, 0, 2.0, 0, 5.0, 1.0, 0.6) AS score "
               "FROM chunks WHERE chunks MATCH ?")
        params: list[Any] = [match]
        if language:
            sql += " AND language = ?"
            params.append(language)
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)
        with self._lock:
            try:
                return self._rows(self._db.execute(sql, params))
            except sqlite3.OperationalError:
                return []

    def find_symbol(self, name: str, limit: int = 20) -> list[dict[str, Any]]:
        name = name.strip()
        if not name:
            return []
        last = name.split(".")[-1]
        with self._lock:
            rows = self._rows(self._db.execute(
                f"SELECT {self._COLS} FROM chunks WHERE symbols LIKE ? OR symbols LIKE ? LIMIT ?",
                (f"%{name}%", f"%.{last}%", limit * 3)))
        exact = re.compile(rf"(^|[ ,.]){re.escape(last)}($|[ ,(])")
        rows.sort(key=lambda r: (0 if exact.search(r["symbols"]) else 1, r["filepath"], r["start_line"]))
        return rows[:limit]

    def get_chunks(self, filepath: str, indices: list[int]) -> list[dict[str, Any]]:
        if not indices:
            return []
        marks = ",".join("?" * len(indices))
        with self._lock:
            return self._rows(self._db.execute(
                f"SELECT {self._COLS} FROM chunks WHERE filepath = ? "
                f"AND CAST(chunk_idx AS INTEGER) IN ({marks}) ORDER BY CAST(chunk_idx AS INTEGER)",
                [filepath, *[int(i) for i in indices]]))
