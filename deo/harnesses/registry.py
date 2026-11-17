"""
deo/harnesses/registry.py

Adapters for external coding harnesses that Deo can delegate work to. Every
harness is pointed at the same FreeLLMAPI gateway, so delegation never needs
a separate paid key.

Each adapter records:
  - what the harness is best at (used by the router and shown to the model)
  - how to run it non-interactively (argv template, `{task}` / `{model}`)
  - which flags must appear in `<binary> --help` before Deo trusts the
    template (`probe_flags`); a harness whose CLI does not advertise them is
    reported as "needs configuration" instead of being run with guessed flags
  - the environment that points it at FreeLLMAPI
  - the `npx freellmapi setup-<id>` generator that writes its config file

Templates can be overridden per project in `.deo.toml`:

    [harnesses]
    commands.reasonix = ["reasonix", "run", "{task}"]
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

from deo.routing.freellmapi import gateway_api_key, gateway_base_url, gateway_origin


def _openai_env(model: str) -> dict[str, str]:
    key = gateway_api_key() or ""
    base = gateway_base_url()
    return {"OPENAI_API_BASE": base, "OPENAI_BASE_URL": base, "OPENAI_API_KEY": key,
            "FREELLMAPI_API_KEY": key, "FREELLMAPI_BASE_URL": base}


def _anthropic_env(model: str) -> dict[str, str]:
    # Claude Code speaks the Anthropic Messages wire format; FreeLLMAPI serves
    # it at the bare origin. ANTHROPIC_AUTH_TOKEN (not ANTHROPIC_API_KEY) is
    # what FreeLLMAPI's docs specify; CLAUDE_CODE_AUTO_MODE_SERVER=0 because the
    # gateway does not run Anthropic's server-side classifier.
    return {"ANTHROPIC_BASE_URL": gateway_origin(), "ANTHROPIC_AUTH_TOKEN": gateway_api_key() or "",
            "CLAUDE_CODE_AUTO_MODE_SERVER": "0"}


@dataclass(frozen=True)
class HarnessSpec:
    id: str
    name: str
    binary: str
    strengths: str
    best_for: tuple[str, ...]                 # router categories
    argv: tuple[str, ...]                     # non-interactive template
    probe_flags: tuple[str, ...] = ()         # must appear in `--help`
    env: Callable[[str], dict[str, str]] = _openai_env
    setup: str = ""                           # FreeLLMAPI generator id
    model_arg: Callable[[str], str] = staticmethod(lambda m: m)  # type: ignore[assignment]
    note: str = ""


HARNESSES: dict[str, HarnessSpec] = {h.id: h for h in [
    HarnessSpec(
        "aider", "Aider", "aider",
        "Fast, surgical one-shot edits to named files with a repo map; ideal for "
        "well-specified small changes, tests and docs.",
        ("quick_edit", "tests", "docs"),
        ("aider", "--message", "{task}", "--yes-always", "--no-auto-commits",
         "--no-pretty", "--no-stream", "--model", "{model}"),
        probe_flags=("--message", "--yes-always", "--no-auto-commits"),
        setup="aider",
        model_arg=lambda m: f"openai/{m}",
    ),
    HarnessSpec(
        "claude", "Claude Code", "claude",
        "Agentic multi-file work: planning, broad refactors, careful reasoning and "
        "repository-wide edits with its own todo/plan discipline.",
        ("feature", "refactor", "review", "debug"),
        ("claude", "-p", "{task}", "--output-format", "text", "--permission-mode", "acceptEdits"),
        probe_flags=("--print", "--output-format", "--permission-mode"),
        env=_anthropic_env, setup="claude",
        note="Requires ANTHROPIC_* gateway env (set automatically) or `npx freellmapi setup-claude`.",
    ),
    HarnessSpec(
        "codex", "Codex CLI", "codex",
        "Test-driven, sandboxed implementation and debugging: runs commands, iterates "
        "until checks pass, terse precise diffs.",
        ("debug", "feature", "tests"),
        ("codex", "exec", "--full-auto", "--skip-git-repo-check", "{task}"),
        probe_flags=("exec",),
        setup="codex",
        note="Run `npx freellmapi setup-codex` once so ~/.codex/config.toml uses the gateway.",
    ),
    HarnessSpec(
        "opencode", "OpenCode", "opencode",
        "General agentic coding with LSP awareness; good default second opinion.",
        ("feature", "refactor"),
        ("opencode", "run", "{task}"),
        probe_flags=("run",), setup="opencode",
    ),
    HarnessSpec(
        "hermes", "Hermes Agent", "hermes",
        "Long-running research and self-improving skills/memory; strong at "
        "exploration and multi-step tasks across sessions.",
        ("research", "feature"),
        ("hermes", "chat", "-q", "{task}"),
        probe_flags=("chat",), setup="hermes",
    ),
    HarnessSpec(
        "omh", "Oh My Hermes (via Hermes)", "hermes",
        "Parallel coding team on top of Hermes: ulw-loop (plan→build→review until a "
        "real gate passes), dag-ulw (task graph with verification nodes), per-lane "
        "model chains. Best for big features, refactors and stubborn bugs.",
        ("feature", "refactor", "debug"),
        ("hermes", "chat", "-q", "ulw-loop: {task}"),
        probe_flags=("chat",), setup="hermes",
        note="Requires oh-my-hermes installed into Hermes (`omh setup`); detected via the `omh` binary.",
    ),
    HarnessSpec(
        "reasonix", "Reasonix", "reasonix",
        "Cache-first DeepSeek-native agent: stable prefix + append-only log keeps prefix-cache "
        "hits high, so it is cheap and strong on very long context and big-codebase reading.",
        ("research", "refactor"),
        ("reasonix", "run", "{task}"),
        probe_flags=("run",), setup="reasonix",
        note="Non-interactive flags vary by version; override [harnesses].commands.reasonix if probing fails.",
    ),
    HarnessSpec(
        "dsh", "DeepSeek Harness", "dsh",
        "DeepSeek's harness: long-context reasoning with aggressive context reuse; good for "
        "understanding large modules before changing them.",
        ("research",),
        ("dsh", "run", "{task}"),
        probe_flags=("run",), setup="dsh",
        note="Override [harnesses].commands.dsh if your version uses different flags.",
    ),
    HarnessSpec(
        "qwen", "Qwen Code", "qwen",
        "Lightweight agentic coding CLI; quick one-shot prompts.",
        ("quick_edit", "docs"),
        ("qwen", "-p", "{task}"),
        probe_flags=("--prompt",), setup="qwen",
    ),
    HarnessSpec(
        "goose", "Goose", "goose",
        "Extensible agent with MCP extensions; good for tasks that need external tools.",
        ("feature",),
        ("goose", "run", "-t", "{task}"),
        probe_flags=("run",), setup="goose",
    ),
]}


@dataclass
class HarnessStatus:
    spec: HarnessSpec
    installed: bool
    ready: bool
    argv: tuple[str, ...]
    reason: str = ""
    custom: bool = False


@lru_cache(maxsize=64)
def _help_text(binary: str, sub: str = "") -> str:
    path = shutil.which(binary)
    if not path:
        return ""
    texts = []
    for args in ([path, "--help"], [path, sub, "--help"] if sub else None):
        if not args:
            continue
        try:
            out = subprocess.run(args, capture_output=True, text=True, timeout=10)
            texts.append((out.stdout or "") + (out.stderr or ""))
        except (OSError, subprocess.TimeoutExpired):
            continue
    return "\n".join(texts)


def harness_status(spec: HarnessSpec, overrides: dict[str, list[str]] | None = None,
                   disabled: list[str] | None = None) -> HarnessStatus:
    custom = tuple((overrides or {}).get(spec.id) or ())
    argv = custom or spec.argv
    if spec.id in (disabled or []):
        return HarnessStatus(spec, bool(shutil.which(argv[0])), False, argv, "disabled in config", bool(custom))
    if spec.id == "omh" and not shutil.which("omh"):
        return HarnessStatus(spec, False, False, argv, "oh-my-hermes (`omh`) not installed", bool(custom))
    if not shutil.which(argv[0]):
        return HarnessStatus(spec, False, False, argv, f"`{argv[0]}` not on PATH", bool(custom))
    if custom:
        return HarnessStatus(spec, True, True, argv, "custom command", True)
    sub = argv[1] if len(argv) > 1 and not argv[1].startswith("-") and "{" not in argv[1] else ""
    help_text = _help_text(argv[0], sub)
    short_ok = {"--print": "-p", "--prompt": "-p"}
    missing = [f for f in spec.probe_flags
               if f not in help_text and not (f in short_ok and f" {short_ok[f]}" in help_text)]
    if missing:
        return HarnessStatus(spec, True, False, argv,
                             f"`{argv[0]} --help` does not list {', '.join(missing)}; "
                             f"set [harnesses].commands.{spec.id}", False)
    return HarnessStatus(spec, True, True, argv)


def all_statuses(overrides: dict[str, list[str]] | None = None,
                 disabled: list[str] | None = None) -> list[HarnessStatus]:
    return [harness_status(s, overrides, disabled) for s in HARNESSES.values()]


@dataclass
class HarnessRun:
    harness: str
    argv: list[str]
    returncode: int
    output: str
    duration: float
    timed_out: bool = False
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def build_argv(status: HarnessStatus, task: str, model: str, files: list[str] | None = None) -> list[str]:
    model_value = status.spec.model_arg(model)
    argv = [part.replace("{task}", task).replace("{model}", model_value) for part in status.argv]
    if status.spec.id == "aider" and files:
        argv.extend(files)
    return argv


def run_harness(status: HarnessStatus, task: str, *, model: str, cwd: Path,
                timeout: int, files: list[str] | None = None) -> HarnessRun:
    import time

    argv = build_argv(status, task, model, files)
    env = {**os.environ, **status.spec.env(model)}
    start = time.time()
    try:
        proc = subprocess.run(argv, cwd=str(cwd), env=env, capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL)
        output = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        return HarnessRun(status.spec.id, argv, proc.returncode, output, time.time() - start)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        return HarnessRun(status.spec.id, argv, -1, out, time.time() - start, timed_out=True)
