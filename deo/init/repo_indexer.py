"""
deo/init/repo_indexer.py

Create a Chroma collection named after the repository and index the
architecturally important files into it.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _sanitize_collection_name(name: str) -> str:
    """
    Chroma collection names must be 3-63 chars, alphanumeric + hyphens,
    start/end with alphanumeric.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9-]", "-", name).strip("-")
    sanitized = re.sub(r"-{2,}", "-", sanitized)
    if len(sanitized) < 3:
        sanitized = sanitized + "-db"
    return sanitized[:63]


def index_important_files(
    root: Path,
    important_files: list[Path],
    collection_name: str | None = None,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> tuple[str, int]:
    """
    Create (or get) a Chroma Cloud collection named after the repo and
    index the given files into it.

    Returns:
      - collection_name: the name used
      - n_chunks: total chunks indexed
    """
    import chromadb
    from deo.memory.indexer import chunk_file, detect_language, file_hash

    api_key = os.environ["CHROMA_API_KEY"]
    tenant = os.environ["CHROMA_TENANT"]
    database = os.environ["CHROMA_DATABASE"]

    if collection_name is None:
        collection_name = _sanitize_collection_name(root.name)

    client = chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database,
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    n_chunks = 0
    for path in important_files:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        rel = str(path.relative_to(root))
        language = detect_language(path)
        fhash = file_hash(content)
        chunks, start_lines = chunk_file(content, language, chunk_size, chunk_overlap)

        if not chunks:
            continue

        ids = [f"{rel}::{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "filepath": rel,
                "language": language,
                "chunk_idx": i,
                "start_line": start_lines[i] if start_lines else 0,
                "file_hash": fhash,
            }
            for i in range(len(chunks))
        ]

        # Delete existing chunks for this file first
        try:
            existing = collection.get(where={"filepath": rel})
            if existing["ids"]:
                collection.delete(ids=existing["ids"])
        except Exception:
            pass

        batch_size = 500
        for b in range(0, len(chunks), batch_size):
            collection.add(
                ids=ids[b:b + batch_size],
                documents=chunks[b:b + batch_size],
                metadatas=metadatas[b:b + batch_size],
            )

        n_chunks += len(chunks)

    return collection_name, n_chunks
