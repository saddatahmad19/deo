"""
deo/learning/experience.py

Continuous self-improvement for Deo.

Every user turn becomes an *episode*: the route that was chosen, which model
FreeLLMAPI actually served (X-Routed-Via), which tools and harnesses ran, the
verification signals seen in tool output (test pass/fail counts, tracebacks),
the files touched and — when given — explicit user feedback (/good, /bad).

From those episodes Deo learns three things:

  1. Routing   — a Thompson-sampling bandit over FreeLLMAPI presets per task
                 category, so the router drifts toward the preset that
                 actually produces verified results in *this* repository.
  2. Retrieval — a per-file utility prior: files that were read/edited in
                 successful episodes get a small boost in semantic search.
  3. Lessons   — short, reusable rules ("run `uv run pytest`, not `pytest`")
                 recorded by agents, by /bad notes, or distilled from episodes;
                 the most relevant few are recalled into each new turn. When a
                 Chroma client is available lessons are mirrored into a
                 `<collection>-lessons` collection and recalled with dense +
                 lexical fusion.

Everything is stored as append-only JSONL under .deo/experience/ so it is
inspectable, diffable and safe to delete.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Outcome signals
# ---------------------------------------------------------------------------

_PASSED = re.compile(r"\b(\d+) passed\b")
_FAILED = re.compile(r"\b(\d+) (?:failed|failing|errors?)\b")
_TEST_FAIL_MARKERS = re.compile(r"(^FAILED |^ERROR |AssertionError|Tests?: .*\d+ failed|npm ERR!|error\[E\d+\])", re.M)
_TRACEBACK = re.compile(r"Traceback \(most recent call last\)")
_EXIT = re.compile(r"(?:exit code|exited with|returncode)[:= ]+(-?\d+)", re.I)

FILE_ARG_KEYS = ("file_path", "filepath", "path", "file")
EDIT_TOOLS = {"edit_file", "write_file", "add_to_index"}
READ_TOOLS = {"read_file"}


@dataclass
class Signals:
    tests_passed: int = 0
    tests_failed: int = 0
    last_test_ok: bool | None = None
    tracebacks: int = 0
    nonzero_exits: int = 0

    def observe(self, text: str) -> None:
        if not text:
            return
        passed = sum(int(n) for n in _PASSED.findall(text))
        failed = sum(int(n) for n in _FAILED.findall(text))
        if passed or failed or _TEST_FAIL_MARKERS.search(text):
            self.tests_passed += passed
            self.tests_failed += failed
            self.last_test_ok = failed == 0 and not _TEST_FAIL_MARKERS.search(text) and passed > 0
        self.tracebacks += len(_TRACEBACK.findall(text))
        self.nonzero_exits += sum(1 for code in _EXIT.findall(text) if code != "0")


def infer_outcome(signals: Signals, error: str | None, edited: bool) -> str:
    if error:
        return "failure"
    if signals.last_test_ok is True:
        return "success"
    if signals.last_test_ok is False:
        return "failure"
    if not edited and signals.tracebacks == 0:
        return "neutral"      # questions / research: nothing to verify
    return "unknown"


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    id: str
    ts: float
    prompt: str
    category: str
    preset: str
    model: str
    profile: str
    route_source: str
    routed_via: list[str] = field(default_factory=list)
    tools: dict[str, int] = field(default_factory=dict)
    harnesses: list[dict[str, Any]] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    files_edited: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)
    outcome: str = "unknown"          # success | failure | neutral | unknown
    feedback: str | None = None       # good | bad
    feedback_note: str = ""
    duration: float = 0.0
    error: str | None = None

    @property
    def reward(self) -> float | None:
        """1.0 success, 0.0 failure, None when unverified. Feedback wins."""
        if self.feedback == "good":
            return 1.0
        if self.feedback == "bad":
            return 0.0
        return {"success": 1.0, "failure": 0.0}.get(self.outcome)


def _jsonl_append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _jsonl_read(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


class ExperienceStore:
    """Append-only episode log with feedback events folded in on load."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.path = directory / "episodes.jsonl"

    def append(self, episode: Episode) -> None:
        _jsonl_append(self.path, {"type": "episode", **asdict(episode)})

    def feedback(self, episode_id: str, verdict: str, note: str = "") -> None:
        _jsonl_append(self.path, {"type": "feedback", "id": episode_id, "verdict": verdict,
                                  "note": note, "ts": time.time()})

    def load(self) -> list[Episode]:
        episodes: dict[str, Episode] = {}
        fields = set(Episode.__dataclass_fields__)
        for row in _jsonl_read(self.path):
            if row.get("type") == "episode":
                data = {k: v for k, v in row.items() if k in fields}
                try:
                    episodes[data["id"]] = Episode(**data)
                except TypeError:
                    continue
            elif row.get("type") == "feedback" and row.get("id") in episodes:
                ep = episodes[row["id"]]
                ep.feedback = row.get("verdict")
                ep.feedback_note = row.get("note", "")
        return sorted(episodes.values(), key=lambda e: e.ts)


# ---------------------------------------------------------------------------
# Route learning
# ---------------------------------------------------------------------------

class PresetBandit:
    """Thompson sampling over presets per category.

    The router's proposed preset gets an optimistic prior, so with no data the
    bandit agrees with the router; it only overrides once a different preset
    has a measurably better verified-success record in this category.
    """

    PRIOR_PROPOSED = (3.0, 1.0)
    PRIOR_OTHER = (1.0, 1.0)
    MIN_OBSERVATIONS = 4

    def __init__(self, seed: int | None = None) -> None:
        self.stats: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
        self.rng = random.Random(seed)

    def update(self, category: str, preset: str, reward: float | None) -> None:
        if reward is None:
            return
        s = self.stats[(category, preset)]
        s[0] += reward
        s[1] += 1.0 - reward

    def observations(self, category: str) -> int:
        return int(sum(sum(v) for (c, _), v in self.stats.items() if c == category))

    def choose_preset(self, category: str, proposed: str, candidates: tuple[str, ...]) -> tuple[str, str]:
        if self.observations(category) < self.MIN_OBSERVATIONS:
            return proposed, "not enough data"
        best, best_score = proposed, -1.0
        for preset in candidates:
            a0, b0 = self.PRIOR_PROPOSED if preset == proposed else self.PRIOR_OTHER
            wins, losses = self.stats.get((category, preset), (0.0, 0.0))
            score = self.rng.betavariate(a0 + wins, b0 + losses)
            if score > best_score:
                best, best_score = preset, score
        if best == proposed:
            return proposed, "router choice confirmed"
        wins, losses = self.stats.get((category, best), (0.0, 0.0))
        return best, f"{best} succeeded {int(wins)}/{int(wins + losses)} on {category}"

    def table(self) -> list[tuple[str, str, int, int]]:
        return sorted((c, p, int(v[0]), int(v[0] + v[1])) for (c, p), v in self.stats.items())


class FileUtility:
    """How often each file participated in a verified-successful episode."""

    def __init__(self) -> None:
        self.wins: Counter[str] = Counter()
        self.seen: Counter[str] = Counter()

    def update(self, files: Iterable[str], reward: float | None) -> None:
        if reward is None:
            return
        for f in set(files):
            self.seen[f] += 1
            self.wins[f] += reward

    def prior(self, filepath: str) -> float:
        """Smoothed success-weighted usage in [0, ~1): used as a small rerank boost."""
        seen = self.seen.get(filepath, 0)
        if not seen:
            return 0.0
        rate = (self.wins[filepath] + 1) / (seen + 2)
        return rate * (1 - math.exp(-seen / 3))


# ---------------------------------------------------------------------------
# Lessons
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def _tokens(text: str) -> set[str]:
    out: set[str] = set()
    for tok in _TOKEN.findall(text.lower()):
        out.add(tok)
        out.update(p for p in tok.split("_") if len(p) > 2)
    return {t for t in out if len(t) > 2}


@dataclass
class Lesson:
    id: str
    ts: float
    text: str
    tags: list[str] = field(default_factory=list)
    source: str = "agent"           # agent | feedback | distilled | user
    category: str = ""
    uses: int = 0


class LessonBook:
    def __init__(self, directory: Path, chroma_client: Any = None, collection_name: str = "") -> None:
        self.path = directory / "lessons.jsonl"
        self._lessons: list[Lesson] | None = None
        self._collection = None
        if chroma_client is not None and collection_name:
            try:
                self._collection = chroma_client.get_or_create_collection(
                    name=f"{collection_name}-lessons"[:63], metadata={"hnsw:space": "cosine"})
            except Exception:
                self._collection = None

    @property
    def lessons(self) -> list[Lesson]:
        if self._lessons is None:
            fields = set(Lesson.__dataclass_fields__)
            seen: dict[str, Lesson] = {}
            for row in _jsonl_read(self.path):
                if row.get("deleted"):
                    seen.pop(row.get("id", ""), None)
                    continue
                try:
                    seen[row["id"]] = Lesson(**{k: v for k, v in row.items() if k in fields})
                except (TypeError, KeyError):
                    continue
            self._lessons = list(seen.values())
        return self._lessons

    def add(self, text: str, tags: list[str] | None = None, source: str = "agent",
            category: str = "") -> Lesson | None:
        text = " ".join(text.split())[:600]
        if len(text) < 8:
            return None
        norm = text.lower()
        for existing in self.lessons:
            if existing.text.lower() == norm:
                return existing
        lesson = Lesson(uuid.uuid4().hex[:12], time.time(), text, list(tags or []), source, category)
        _jsonl_append(self.path, asdict(lesson))
        self.lessons.append(lesson)
        if self._collection is not None:
            try:
                self._collection.upsert(ids=[lesson.id], documents=[text], metadatas=[{
                    "source": source, "category": category or "", "tags": ",".join(lesson.tags)}])
            except Exception:
                pass
        return lesson

    def recall(self, query: str, k: int = 3, category: str = "") -> list[Lesson]:
        if not self.lessons or k <= 0:
            return []
        q = _tokens(query)
        now = time.time()
        scores: dict[str, float] = {}
        for rank_pos, lesson in enumerate(self.lessons):
            overlap = len(q & _tokens(lesson.text + " " + " ".join(lesson.tags)))
            if not overlap and lesson.category != category:
                continue
            age_days = (now - lesson.ts) / 86400
            score = overlap + (1.5 if category and lesson.category == category else 0.0)
            score *= 1.0 / (1.0 + age_days / 90)
            scores[lesson.id] = score
        if self._collection is not None:
            try:
                res = self._collection.query(query_texts=[query], n_results=min(k * 3, len(self.lessons)))
                for pos, lid in enumerate(res.get("ids", [[]])[0]):
                    scores[lid] = scores.get(lid, 0.0) + 3.0 / (1 + pos)
            except Exception:
                pass
        by_id = {lesson.id: lesson for lesson in self.lessons}
        ranked = [by_id[i] for i, s in sorted(scores.items(), key=lambda kv: -kv[1]) if i in by_id and s > 0]
        return ranked[:k]


# ---------------------------------------------------------------------------
# Skills (Hermes-style autonomous skill creation)
# ---------------------------------------------------------------------------

_SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}[a-z0-9]$")


def save_skill(skills_dir: Path, name: str, description: str, body: str, overwrite: bool = False) -> Path:
    """Write a Deep Agents SKILL.md that future sessions load automatically."""
    name = name.strip().lower()
    if not _SKILL_NAME.match(name):
        raise ValueError("skill name must be 3-50 chars of lowercase letters, digits and hyphens")
    description = " ".join(description.split())
    if not description or len(description) > 300:
        raise ValueError("description must be 1-300 characters")
    target = skills_dir / name / "SKILL.md"
    if target.exists() and not overwrite:
        raise FileExistsError(f"skill '{name}' already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body.strip()}\n",
        encoding="utf-8",
    )
    return target


# ---------------------------------------------------------------------------
# Session learner — the glue the CLI talks to
# ---------------------------------------------------------------------------

@dataclass
class _Turn:
    prompt: str
    decision: Any
    model: str
    started: float
    tools: Counter = field(default_factory=Counter)
    files_read: set = field(default_factory=set)
    files_edited: set = field(default_factory=set)
    harnesses: list = field(default_factory=list)
    signals: Signals = field(default_factory=Signals)
    routed_start: int = 0


class SessionLearner:
    def __init__(self, directory: Path, chroma_client: Any = None, collection_name: str = "",
                 seed: int | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.dir = directory
        self.store = ExperienceStore(directory)
        self.bandit = PresetBandit(seed=seed)
        self.files = FileUtility()
        self.lessons = LessonBook(directory, chroma_client, collection_name)
        self.harness_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.last_episode: Episode | None = None
        self._turn: _Turn | None = None
        for ep in self.store.load():
            self._learn_from(ep)

    # -- learning -------------------------------------------------------
    def _learn_from(self, ep: Episode) -> None:
        reward = ep.reward
        self.bandit.update(ep.category, ep.preset, reward)
        self.files.update([*ep.files_read, *ep.files_edited], reward)
        for h in ep.harnesses:
            stats = self.harness_stats[h.get("harness", "?")]
            stats[1] += 1
            stats[0] += 1 if h.get("ok") else 0

    # -- turn lifecycle -------------------------------------------------
    def begin_turn(self, prompt: str, decision: Any, model: str, routed_len: int = 0) -> None:
        self._turn = _Turn(prompt=prompt, decision=decision, model=model, started=time.time(),
                           routed_start=routed_len)

    def observe_tool_call(self, name: str, args: dict[str, Any] | None) -> None:
        if self._turn is None:
            return
        self._turn.tools[name] += 1
        for key in FILE_ARG_KEYS:
            value = (args or {}).get(key)
            if isinstance(value, str) and value:
                path = value.lstrip("/")
                if name in EDIT_TOOLS:
                    self._turn.files_edited.add(path)
                elif name in READ_TOOLS:
                    self._turn.files_read.add(path)

    def observe_tool_output(self, text: str) -> None:
        if self._turn is not None:
            self._turn.signals.observe(text)

    def observe_harness(self, harness: str, ok: bool, duration: float) -> None:
        if self._turn is not None:
            self._turn.harnesses.append({"harness": harness, "ok": ok, "duration": round(duration, 1)})
        stats = self.harness_stats[harness]
        stats[1] += 1
        stats[0] += 1 if ok else 0

    def end_turn(self, routed_via: list[str] | None = None, error: str | None = None) -> Episode | None:
        turn, self._turn = self._turn, None
        if turn is None:
            return None
        d = turn.decision
        edited = bool(turn.files_edited) or turn.tools.get("execute", 0) > 0
        ep = Episode(
            id=uuid.uuid4().hex[:12], ts=turn.started, prompt=turn.prompt[:500],
            category=getattr(d, "category", "unknown"), preset=getattr(d, "preset", "balanced"),
            model=turn.model, profile=getattr(d, "profile", ""), route_source=getattr(d, "source", ""),
            routed_via=list(dict.fromkeys((routed_via or [])[turn.routed_start:]))[:10],
            tools=dict(turn.tools), harnesses=turn.harnesses,
            files_read=sorted(turn.files_read)[:50], files_edited=sorted(turn.files_edited)[:50],
            signals=asdict(turn.signals), outcome=infer_outcome(turn.signals, error, edited),
            duration=round(time.time() - turn.started, 1), error=(error or None) and error[:300],
        )
        self.store.append(ep)
        self._learn_from(ep)
        self.last_episode = ep
        return ep

    def feedback(self, verdict: str, note: str = "") -> Episode | None:
        ep = self.last_episode
        if ep is None:
            return None
        # Undo the automatic reward, then apply the explicit one.
        old = ep.reward
        if old is not None:
            s = self.bandit.stats[(ep.category, ep.preset)]
            s[0] -= old
            s[1] -= 1.0 - old
        ep.feedback, ep.feedback_note = verdict, note
        self.bandit.update(ep.category, ep.preset, ep.reward)
        self.files.update([*ep.files_read, *ep.files_edited], ep.reward if old is None else None)
        self.store.feedback(ep.id, verdict, note)
        if note:
            prefix = "Avoid: " if verdict == "bad" else "Works well: "
            self.lessons.add(prefix + note, tags=[ep.category], source="feedback", category=ep.category)
        return ep

    # -- prompt context -------------------------------------------------
    def context_block(self, prompt: str, category: str, k: int) -> str:
        lessons = self.lessons.recall(prompt, k=k, category=category)
        if not lessons:
            return ""
        lines = "\n".join(f"- {lesson.text}" for lesson in lessons)
        return f"<deo_lessons>\nLessons learned in earlier sessions on this project:\n{lines}\n</deo_lessons>"

    def report(self) -> dict[str, Any]:
        episodes = self.store.load()
        outcomes = Counter(e.outcome for e in episodes)
        feedback = Counter(e.feedback for e in episodes if e.feedback)
        served = Counter(v for e in episodes for v in e.routed_via)
        return {
            "episodes": len(episodes),
            "outcomes": dict(outcomes),
            "feedback": dict(feedback),
            "presets": self.bandit.table(),
            "harnesses": {k: tuple(v) for k, v in self.harness_stats.items()},
            "served_models": served.most_common(10),
            "lessons": len(self.lessons.lessons),
            "top_files": sorted(((f, round(self.files.prior(f), 2)) for f in self.files.seen),
                                key=lambda kv: -kv[1])[:10],
        }


DISTILL_PROMPT = """\
You are improving a coding agent from its own history. Below are recent episodes
(task, route, outcome, verification signals, user feedback). Write up to {n}
short, specific, reusable lessons for THIS repository — commands that work,
pitfalls, conventions, which routes/harnesses worked for which tasks. Skip
anything generic. Output one lesson per line, no numbering, no preamble.

{episodes}
"""


def distill_lessons(model: Any, learner: SessionLearner, n: int = 5, last: int = 30) -> list[Lesson]:
    """Ask the model to turn recent episodes into lessons and store them."""
    episodes = learner.store.load()[-last:]
    if not episodes:
        return []
    rows = []
    for e in episodes:
        rows.append(json.dumps({
            "task": e.prompt[:200], "category": e.category, "preset": e.preset,
            "served": e.routed_via[:2], "outcome": e.outcome, "feedback": e.feedback,
            "note": e.feedback_note, "signals": {k: v for k, v in e.signals.items() if v},
            "harnesses": e.harnesses, "error": e.error,
        }, ensure_ascii=False))
    response = model.invoke(DISTILL_PROMPT.format(n=n, episodes="\n".join(rows)))
    text = response.content if isinstance(response.content, str) else str(response.content)
    added = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if line:
            lesson = learner.lessons.add(line, source="distilled")
            if lesson:
                added.append(lesson)
        if len(added) >= n:
            break
    return added
