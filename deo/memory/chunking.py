"""
deo/memory/chunking.py

Structure-aware chunking for code retrieval.

Retrieval quality is decided mostly at chunking time, so chunks follow the
code's own structure instead of fixed character windows:

  - Python: real AST boundaries. Top-level functions/classes become chunks
    (decorators included); oversized classes are split into a header chunk
    plus one chunk per method, named `Class.method`; module preambles
    (imports/constants) are kept together.
  - Brace / keyword languages (TS/JS, Go, Rust, Java, Kotlin, C#, Swift, Ruby,
    PHP, C/C++): declaration-boundary regexes, then small neighbouring
    declarations are packed together up to the size budget.
  - Markdown: heading sections carrying their heading breadcrumb.
  - Everything else: newline-aligned size windows with overlap.

Each chunk carries its symbol names, kind and exact line span; the indexer
stores those as metadata (for filtering, symbol lookup and boosting) and
prepends a short contextual header before embedding.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    start_line: int                     # 1-indexed, inclusive
    end_line: int                       # 1-indexed, inclusive
    symbols: list[str] = field(default_factory=list)
    kind: str = "block"                 # function | class | method | module | section | block

    @property
    def symbol(self) -> str:
        return ", ".join(self.symbols[:6])


# ---------------------------------------------------------------------------
# Size-based fallback
# ---------------------------------------------------------------------------

def chunk_by_size(content: str, chunk_size: int, chunk_overlap: int,
                  line_offset: int = 0, symbols: list[str] | None = None,
                  kind: str = "block") -> list[Chunk]:
    lines = content.splitlines(keepends=True)
    if not lines:
        return []
    chunks: list[Chunk] = []
    i = 0
    while i < len(lines):
        size, j = 0, i
        while j < len(lines) and (size + len(lines[j]) <= chunk_size or j == i):
            size += len(lines[j])
            j += 1
        text = "".join(lines[i:j])
        if text.strip():
            chunks.append(Chunk(text, line_offset + i + 1, line_offset + j, list(symbols or []), kind))
        if j >= len(lines):
            break
        # Step back by roughly chunk_overlap characters worth of lines.
        back, k = 0, j
        while k > i + 1 and back < chunk_overlap:
            k -= 1
            back += len(lines[k])
        i = max(k, i + 1)
    return chunks


def _pack(segments: list[Chunk], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Merge small adjacent segments and split oversized ones."""
    out: list[Chunk] = []
    buf: Chunk | None = None
    for seg in segments:
        if len(seg.text) > chunk_size:
            if buf:
                out.append(buf)
                buf = None
            out.extend(chunk_by_size(seg.text, chunk_size, chunk_overlap,
                                     line_offset=seg.start_line - 1, symbols=seg.symbols, kind=seg.kind))
            continue
        if buf and len(buf.text) + len(seg.text) <= chunk_size:
            buf = Chunk(buf.text + seg.text, buf.start_line, seg.end_line,
                        buf.symbols + seg.symbols,
                        buf.kind if buf.kind == seg.kind else "block")
        else:
            if buf:
                out.append(buf)
            buf = Chunk(seg.text, seg.start_line, seg.end_line, list(seg.symbols), seg.kind)
    if buf:
        out.append(buf)
    return [c for c in out if c.text.strip()]


def _segments_from_boundaries(lines: list[str], bounds: list[tuple[int, str, str]]) -> list[Chunk]:
    """bounds: sorted (0-indexed line, symbol, kind). Lines before the first are a preamble."""
    segs: list[Chunk] = []
    starts = [b[0] for b in bounds]
    if not starts or starts[0] > 0:
        first = starts[0] if starts else len(lines)
        segs.append(Chunk("".join(lines[:first]), 1, first, [], "module"))
    for idx, (start, symbol, kind) in enumerate(bounds):
        end = bounds[idx + 1][0] if idx + 1 < len(bounds) else len(lines)
        segs.append(Chunk("".join(lines[start:end]), start + 1, end, [symbol] if symbol else [], kind))
    return [s for s in segs if s.text.strip()]


# ---------------------------------------------------------------------------
# Python (AST)
# ---------------------------------------------------------------------------

def _node_start(node: ast.AST) -> int:
    decos = getattr(node, "decorator_list", None) or []
    return min([node.lineno, *(d.lineno for d in decos)])  # type: ignore[attr-defined]


def chunk_python(content: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError):
        return chunk_by_size(content, chunk_size, chunk_overlap)
    lines = content.splitlines(keepends=True)
    segments: list[Chunk] = []
    cursor = 0  # 0-indexed next unconsumed line

    def emit(start0: int, end0: int, symbols: list[str], kind: str) -> None:
        text = "".join(lines[start0:end0])
        if text.strip():
            segments.append(Chunk(text, start0 + 1, end0, symbols, kind))

    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in tree.body:
        if not isinstance(node, defs):
            continue
        start0, end0 = _node_start(node) - 1, node.end_lineno or node.lineno  # type: ignore[attr-defined]
        if start0 > cursor:
            emit(cursor, start0, [], "module")
        if isinstance(node, ast.ClassDef):
            body_text = "".join(lines[start0:end0])
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if len(body_text) > chunk_size and methods:
                header_end = _node_start(methods[0]) - 1
                emit(start0, header_end, [node.name], "class")
                for mi, m in enumerate(methods):
                    m_start = _node_start(m) - 1
                    m_end = (_node_start(methods[mi + 1]) - 1) if mi + 1 < len(methods) else end0
                    emit(m_start, m_end, [f"{node.name}.{m.name}"], "method")
            else:
                emit(start0, end0, [node.name], "class")
        else:
            emit(start0, end0, [node.name], "function")
        cursor = end0
    if cursor < len(lines):
        emit(cursor, len(lines), [], "module")
    return _pack(segments, chunk_size, chunk_overlap) or chunk_by_size(content, chunk_size, chunk_overlap)


# ---------------------------------------------------------------------------
# Regex-boundary languages
# ---------------------------------------------------------------------------

_DECL: dict[str, list[tuple[re.Pattern[str], str]]] = {
    "typescript": [
        (re.compile(r"^(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)"), "class"),
        (re.compile(r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\*?\s+([A-Za-z_$][\w$]*)"), "function"),
        (re.compile(r"^(?:export\s+)?(?:declare\s+)?(?:interface|type|enum|namespace)\s+([A-Za-z_$][\w$]*)"), "class"),
        (re.compile(r"^(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s+)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>"), "function"),
        (re.compile(r"^  (?:public |private |protected |static |async |readonly |override )*([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*(?::\s*[^={]+)?\{\s*$"), "method"),
    ],
    "go": [
        (re.compile(r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)"), "function"),
        (re.compile(r"^type\s+([A-Za-z_]\w*)"), "class"),
    ],
    "rust": [
        (re.compile(r"^\s{0,4}(?:pub(?:\([\w:]+\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:const\s+)?fn\s+([A-Za-z_]\w*)"), "function"),
        (re.compile(r"^(?:pub(?:\([\w:]+\))?\s+)?(?:struct|enum|trait|mod|union)\s+([A-Za-z_]\w*)"), "class"),
        (re.compile(r"^impl(?:<[^>]*>)?\s+([\w:<>, ]+?)(?:\s+for\s+([\w:<>]+))?\s*\{"), "class"),
    ],
    "java": [
        (re.compile(r"^\s{0,4}(?:@\w+\s+)*(?:public |private |protected |static |final |abstract |sealed |non-sealed )*(?:class|interface|enum|record)\s+([A-Za-z_]\w*)"), "class"),
        (re.compile(r"^\s{2,8}(?:@\w+\s+)*(?:public |private |protected |static |final |abstract |synchronized |default )+[\w<>\[\], ?]+\s+([a-zA-Z_]\w*)\s*\("), "method"),
    ],
    "kotlin": [
        (re.compile(r"^\s{0,4}(?:(?:public|private|internal|protected|data|sealed|open|abstract|enum|inner)\s+)*(?:class|interface|object)\s+([A-Za-z_]\w*)"), "class"),
        (re.compile(r"^\s{0,8}(?:(?:public|private|internal|protected|override|suspend|inline|open)\s+)*fun\s+(?:<[^>]+>\s*)?(?:[\w.]+\.)?([A-Za-z_]\w*)"), "function"),
    ],
    "csharp": [
        (re.compile(r"^\s{0,8}(?:public |private |protected |internal |static |sealed |abstract |partial )*(?:class|interface|struct|enum|record)\s+([A-Za-z_]\w*)"), "class"),
        (re.compile(r"^\s{4,12}(?:public |private |protected |internal |static |virtual |override |async |sealed )+[\w<>\[\], ?]+\s+([A-Za-z_]\w*)\s*\("), "method"),
    ],
    "swift": [
        (re.compile(r"^\s{0,4}(?:(?:public|private|internal|open|fileprivate|final)\s+)*(?:class|struct|enum|protocol|extension|actor)\s+([A-Za-z_]\w*)"), "class"),
        (re.compile(r"^\s{0,8}(?:(?:public|private|internal|open|fileprivate|static|override|mutating)\s+)*func\s+([A-Za-z_]\w*)"), "function"),
    ],
    "ruby": [
        (re.compile(r"^\s*(?:class|module)\s+([A-Z][\w:]*)"), "class"),
        (re.compile(r"^\s*def\s+(?:self\.)?([\w?!=]+)"), "method"),
    ],
    "php": [
        (re.compile(r"^\s*(?:abstract\s+|final\s+)?(?:class|interface|trait|enum)\s+([A-Za-z_]\w*)"), "class"),
        (re.compile(r"^\s*(?:public |private |protected |static |final |abstract )*function\s+&?([A-Za-z_]\w*)"), "function"),
    ],
    "c": [
        (re.compile(r"^(?:static\s+|inline\s+|extern\s+)*[A-Za-z_][\w\s\*]*?\b([A-Za-z_]\w*)\s*\([^;{]*\)\s*\{?\s*$"), "function"),
        (re.compile(r"^(?:typedef\s+)?(?:struct|enum|union)\s+([A-Za-z_]\w*)\s*\{"), "class"),
    ],
}
_DECL["javascript"] = _DECL["typescript"]
_DECL["cpp"] = _DECL["c"] + [
    (re.compile(r"^(?:template\s*<[^>]*>\s*)?(?:class|struct|namespace)\s+([A-Za-z_]\w*)"), "class"),
    (re.compile(r"^[\w:<>\*&\s]+?\b([A-Za-z_]\w*::~?[A-Za-z_]\w*)\s*\([^;]*\)\s*(?:const)?\s*\{?\s*$"), "method"),
]

_C_KEYWORDS = {"if", "for", "while", "switch", "return", "else", "do", "sizeof"}


def chunk_declarations(content: str, language: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    patterns = _DECL.get(language)
    if not patterns:
        return chunk_by_size(content, chunk_size, chunk_overlap)
    lines = content.splitlines(keepends=True)
    bounds: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        for pattern, kind in patterns:
            m = pattern.match(line)
            if not m:
                continue
            name = next((g for g in m.groups() if g), "").strip()
            if language in ("c", "cpp") and name in _C_KEYWORDS:
                continue
            # Pull preceding doc comments / decorators / attributes into the chunk.
            start = i
            while start > 0 and re.match(r"^\s*(//|/\*|\*|#\[|@|///|# )", lines[start - 1]):
                start -= 1
            if bounds and start <= bounds[-1][0]:
                start = i
            bounds.append((start, name, kind))
            break
    if not bounds:
        return chunk_by_size(content, chunk_size, chunk_overlap)
    return _pack(_segments_from_boundaries(lines, bounds), chunk_size, chunk_overlap)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def chunk_markdown(content: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    lines = content.splitlines(keepends=True)
    bounds: list[tuple[int, str, str]] = []
    trail: list[str] = []
    in_fence = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        if in_fence:
            continue
        m = _HEADING.match(line)
        if m:
            level = len(m.group(1))
            trail = trail[: level - 1] + [m.group(2).strip()]
            bounds.append((i, " > ".join(trail), "section"))
    if not bounds:
        return chunk_by_size(content, chunk_size, chunk_overlap)
    return _pack(_segments_from_boundaries(lines, bounds), chunk_size, chunk_overlap)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def chunk_source(content: str, language: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    if not content.strip():
        return []
    if language == "python":
        return chunk_python(content, chunk_size, chunk_overlap)
    if language == "markdown":
        return chunk_markdown(content, chunk_size, chunk_overlap)
    if language in _DECL:
        return chunk_declarations(content, language, chunk_size, chunk_overlap)
    return chunk_by_size(content, chunk_size, chunk_overlap)


def contextual_header(filepath: str, language: str, chunk: Chunk) -> str:
    """One-line context prefix embedded with each chunk (contextual retrieval)."""
    parts = [filepath, language]
    if chunk.symbols:
        parts.append(f"{chunk.kind}: {chunk.symbol}")
    parts.append(f"lines {chunk.start_line}-{chunk.end_line}")
    return "# " + " | ".join(parts) + "\n"


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def identifier_terms(text: str) -> list[str]:
    """Split identifiers so `getUserById` / `get_user_by_id` match `user`."""
    terms: list[str] = []
    for ident in _IDENT.findall(text):
        parts = [p for chunk in ident.split("_") for p in _CAMEL.findall(chunk)]
        if len(parts) > 1:
            terms.extend(p.lower() for p in parts if len(p) > 1)
    return terms
