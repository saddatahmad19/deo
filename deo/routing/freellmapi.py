"""
deo/routing/freellmapi.py

FreeLLMAPI integration: gateway settings, the decision API (POST /v1/decisions),
the model catalog (GET /v1/models) and a routed ChatOpenAI that picks its
FreeLLMAPI model id and routing headers per request.

FreeLLMAPI facts this module relies on (from the gateway's REST API docs and
router source):
  - model "auto" follows the active fallback chain; "auto:default" names the
    built-in Default profile (profile names match case-insensitively).
  - "auto:smart" / "auto:fast" / "auto:balanced" / "auto:reliable" /
    "auto:cheap" rank every enabled model by that axis.
  - X-FreeLLM-Task-Type: code|chat|auto biases quality vs. speed per request.
  - X-Session-Id keeps a conversation on one model (sticky routing).
  - X-FreeLLM-Compress: off|lossless|standard|aggressive controls compression.
  - Responses carry X-Routed-Via: <platform>/<model> (and X-Fallback-Trail
    when failover happened), X-FreeLLM-Routing-Strategy under decision routing.
  - POST /v1/decisions answers typed questions with probabilities instead of
    generated text; chat routing never spends chat quota on it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI
from pydantic import Field

DEFAULT_BASE_URL = "http://localhost:3001/v1"

PRESETS = ("smartest", "fastest", "balanced", "reliable")

# FreeLLMAPI's own descriptions of its four decision presets
# (server/src/services/routing-decision.ts). Reusing them keeps Deo's decision
# questions consistent with the gateway's 'decision' routing strategy.
PRESET_CRITERIA: dict[str, str] = {
    "smartest": "Hard or high-stakes: answer quality matters most and waiting a bit longer is fine.",
    "fastest": "Simple or short: a quick answer matters most and a smaller model will do.",
    "balanced": "Ordinary request: a sensible mix of quality and speed.",
    "reliable": (
        "Must work first time: strict formats, structured output or an automated step "
        "that cannot be retried by hand."
    ),
}


# ---------------------------------------------------------------------------
# Gateway settings
# ---------------------------------------------------------------------------

def gateway_base_url() -> str:
    """OpenAI-compatible base URL (with /v1)."""
    return (
        os.environ.get("FREELLMAPI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or DEFAULT_BASE_URL
    ).rstrip("/")


def gateway_origin(base_url: str | None = None) -> str:
    """Bare origin (no /v1), as Anthropic-wire clients such as Claude Code expect."""
    parsed = urlparse(base_url or gateway_base_url())
    return f"{parsed.scheme}://{parsed.netloc}"


def gateway_api_key() -> str | None:
    return os.environ.get("FREELLMAPI_API_KEY") or os.environ.get("OPENAI_API_KEY")


def _http_client():
    import httpx  # dependency of openai / langchain-openai

    return httpx


def _auth_headers() -> dict[str, str]:
    key = gateway_api_key()
    return {"Authorization": f"Bearer {key}"} if key else {}


# ---------------------------------------------------------------------------
# Decision API
# ---------------------------------------------------------------------------

@dataclass
class DecisionAnswer:
    choice: str
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None

    @property
    def top_probability(self) -> float:
        if self.choice in self.probabilities:
            return float(self.probabilities[self.choice])
        return float(self.confidence or 0.0)


class DecisionError(RuntimeError):
    pass


def ask_decisions(
    state: str,
    questions: dict[str, dict[str, Any]],
    *,
    model: str = "auto",
    timeout: float = 2.5,
    base_url: str | None = None,
) -> dict[str, DecisionAnswer]:
    """Call POST /v1/decisions and return parsed answers keyed by question id.

    `questions` follows the gateway schema, e.g.
    {"preset": {"type": "choice", "instructions": "...", "criteria": {...}}}.
    Raises DecisionError on any transport, HTTP or schema failure.
    """
    httpx = _http_client()
    url = f"{(base_url or gateway_base_url()).rstrip('/')}/decisions"
    body: dict[str, Any] = {"state": state[:20_000], "questions": questions}
    if model and model != "auto":
        body["model"] = model
    try:
        resp = httpx.post(url, json=body, headers=_auth_headers(), timeout=timeout)
    except Exception as exc:  # timeouts, connection refused, DNS
        raise DecisionError(f"decision request failed: {exc}") from exc
    if resp.status_code != 200:
        raise DecisionError(f"decision request returned HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        answers = resp.json().get("answers") or {}
    except ValueError as exc:
        raise DecisionError("decision response was not JSON") from exc

    parsed: dict[str, DecisionAnswer] = {}
    for qid, ans in answers.items():
        if not isinstance(ans, dict) or "choice" not in ans:
            continue
        probs = ans.get("probabilities") or {}
        parsed[qid] = DecisionAnswer(
            choice=str(ans["choice"]),
            probabilities={str(k): float(v) for k, v in probs.items() if isinstance(v, (int, float))},
            confidence=ans.get("confidence") if isinstance(ans.get("confidence"), (int, float)) else None,
        )
    if not parsed:
        raise DecisionError("decision response contained no usable answers")
    return parsed


def list_models(available_only: bool = True, timeout: float = 5.0) -> list[dict[str, Any]]:
    """GET /v1/models (optionally ?available=true). Raises on failure."""
    httpx = _http_client()
    url = f"{gateway_base_url()}/models"
    params = {"available": "true"} if available_only else None
    resp = httpx.get(url, params=params, headers=_auth_headers(), timeout=timeout)
    resp.raise_for_status()
    return list(resp.json().get("data") or [])


# ---------------------------------------------------------------------------
# Per-request routing state
# ---------------------------------------------------------------------------

@dataclass
class RouteState:
    """Mutable routing state shared by every model object of one Deo session.

    The CLI updates `turn_preset` / `task_type` once per user turn (from the
    decision router); each RoutedChatOpenAI reads it at request time, so the
    agent graph does not need to be rebuilt to change routes.
    """

    presets: dict[str, str]
    role_presets: dict[str, str]
    default_model: str = "auto:default"
    role_models: dict[str, str] = field(default_factory=dict)  # explicit FreeLLMAPI ids
    turn_preset: str | None = None
    task_type: str = "code"
    session_id: str = ""
    compress: str = ""
    routed_via: list[str] = field(default_factory=list)

    def model_for(self, role: str) -> str:
        if role in self.role_models:
            return self.role_models[role]
        preset = self.role_presets.get(role, "turn")
        if preset == "turn":
            preset = self.turn_preset or ""
        return self.presets.get(preset, self.default_model) if preset else self.default_model

    def headers(self, role: str) -> dict[str, str]:
        headers = {"X-FreeLLM-Task-Type": self.task_type or "auto"}
        if self.session_id:
            # One sticky bucket per role keeps e.g. the reviewer on one model
            # across a turn without pinning the orchestrator to it.
            headers["X-Session-Id"] = f"deo-{self.session_id}-{role}"
        if self.compress:
            headers["X-FreeLLM-Compress"] = self.compress
        return headers

    def note_routed(self, value: str | None) -> None:
        if value:
            self.routed_via.append(value)
            del self.routed_via[:-50]


class RoutedChatOpenAI(ChatOpenAI):
    """ChatOpenAI that resolves its FreeLLMAPI model id and headers per request."""

    route_state: Any = Field(default=None, exclude=True)
    route_role: str = "orchestrator"

    def _get_request_payload(self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        state: RouteState | None = self.route_state
        if state is not None:
            payload["model"] = state.model_for(self.route_role)
            extra = dict(payload.get("extra_headers") or {})
            extra.update(state.headers(self.route_role))
            payload["extra_headers"] = extra
        return payload
