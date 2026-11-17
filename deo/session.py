"""
deo/session.py

Per-session glue between the CLI, the decision router, the learning loop and
the agent pool. One DeoSession lives for one interactive or one-shot run.

Per turn:
  1. DecisionRouter picks category / preset / profile (heuristics →
     FreeLLMAPI /v1/decisions → learned bandit).
  2. RouteState is updated so every model object routes accordingly.
  3. If auto-profile is on (no --profile pinned), the agent for that profile
     is fetched from a pool that shares one checkpointer, so conversation
     history survives profile switches.
  4. Route hint + recalled lessons are prepended to the user message.
  5. Stream chunks are observed for tool calls, test signals and the
     X-Routed-Via header; the episode is recorded when the turn ends.
"""

from __future__ import annotations

from typing import Any

from deo.config import DeoConfig


class DeoSession:
    def __init__(self, config: DeoConfig, manager: Any, indexer: Any, session_id: str,
                 profile: str = "default", pinned: bool = False) -> None:
        from langgraph.checkpoint.memory import MemorySaver

        from deo.agents.factory import make_route_state
        from deo.routing.router import DecisionRouter

        self.config = config
        self.manager = manager
        self.indexer = indexer
        self.session_id = session_id
        self.profile = profile
        self.pinned = pinned
        self.checkpointer = MemorySaver()
        self.route_state = make_route_state(config, session_id)
        self.learner = None
        if config.learning.enabled:
            try:
                from deo.learning.experience import SessionLearner

                client = getattr(manager, "client", None) if manager is not None else None
                self.learner = SessionLearner(config.experience_dir, client, config.chroma.collection_name)
            except Exception:
                self.learner = None
        self.router = DecisionRouter(config.routing, learner=self.learner.bandit if self.learner else None)
        self._agents: dict[str, Any] = {}
        self.last_decision = None

    # -- agents ---------------------------------------------------------
    def agent_for(self, profile: str) -> Any:
        if profile not in self._agents:
            from deo.agents.factory import create_deo_agent
            from deo.agents.registry import get_profile

            try:
                p = get_profile(profile)
            except KeyError:
                p, profile = get_profile("default"), "default"
            sa = self.config.subagents
            sa.planner, sa.implementer = p.enable_planner, p.enable_implementer
            sa.reviewer, sa.tester, sa.documenter = p.enable_reviewer, p.enable_tester, p.enable_documenter
            self._agents[profile] = create_deo_agent(
                config=self.config, manager=self.manager, indexer=self.indexer,
                checkpointer=self.checkpointer, extra_system_prompt=p.system_prompt_addon,
                route_state=self.route_state, learner=self.learner,
            )
        return self._agents[profile]

    def pin(self, profile: str) -> None:
        self.profile, self.pinned = profile, True

    def reset_agents(self) -> None:
        self._agents.clear()

    # -- turns ----------------------------------------------------------
    def prepare_turn(self, prompt: str) -> tuple[Any, str, str]:
        """Return (agent, augmented_prompt, route_summary)."""
        decision = self.router.decide(prompt)
        self.last_decision = decision
        profile = self.profile if (self.pinned or not self.config.routing.auto_profile) else decision.profile
        if self.route_state is not None:
            self.route_state.turn_preset = decision.preset
            self.route_state.task_type = decision.task_type
        model = self.route_state.model_for("orchestrator") if self.route_state else self.config.model.model_string
        blocks = [f"<deo_route category=\"{decision.category}\" preset=\"{decision.preset}\" "
                  f"suggested_harnesses=\"{','.join(decision.harnesses)}\"/>"]
        if self.learner is not None and self.config.learning.inject_lessons:
            ctx = self.learner.context_block(prompt, decision.category, self.config.learning.inject_lessons)
            if ctx:
                blocks.append(ctx)
            self.learner.begin_turn(prompt, decision, model,
                                    len(self.route_state.routed_via) if self.route_state else 0)
        summary = f"↳ {decision.summary(self.config.routing.presets)} · agents: {profile}"
        return self.agent_for(profile), "\n".join(blocks) + "\n\n" + prompt, summary

    def observe(self, chunk: Any) -> None:
        if not (isinstance(chunk, tuple) and len(chunk) == 2):
            return
        msg = chunk[0]
        meta = getattr(msg, "response_metadata", None) or {}
        headers = meta.get("headers") or {}
        if self.route_state is not None and headers:
            via = headers.get("x-routed-via") or headers.get("X-Routed-Via")
            if via and (not self.route_state.routed_via or self.route_state.routed_via[-1] != via):
                self.route_state.note_routed(via)
        if self.learner is None:
            return
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc.get("name"):
                self.learner.observe_tool_call(tc["name"], tc.get("args") or {})
        if getattr(msg, "type", "") == "tool":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            self.learner.observe_tool_output(content)

    def end_turn(self, error: str | None = None) -> Any:
        if self.learner is None:
            return None
        return self.learner.end_turn(self.route_state.routed_via if self.route_state else None, error)
