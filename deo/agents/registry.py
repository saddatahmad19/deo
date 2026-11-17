"""
deo/agents/registry.py

Agent profile registry — allows registering named agent configurations
that can be activated by name (e.g. "deo --profile security-audit").

Built-in profiles:
  default      — full multi-agent pipeline (planner/implementer/reviewer/tester)
  fast         — orchestrator only, no subagents (fastest, cheapest)
  review       — orchestrator + reviewer only
  write-tests  — orchestrator + tester only

Users can define custom profiles in ~/.deo/config.toml or .deo.toml.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentProfile:
    """
    A named configuration profile for the Forge agent system.

    Profiles let you switch between different agent setups quickly,
    e.g. a "fast" profile for quick questions vs a "full" profile
    for complex feature work.
    """
    name: str
    description: str
    model_override: str | None = None           # "freellmapi:auto:default"
    system_prompt_addon: str = ""               # Extra instructions for this profile
    enable_planner: bool = True
    enable_implementer: bool = True
    enable_reviewer: bool = True
    enable_tester: bool = True
    enable_documenter: bool = False
    extra_tools: list[str] = field(default_factory=list)  # Tool names to activate


class AgentRegistry:
    """Registry of named agent profiles."""

    def __init__(self) -> None:
        self._profiles: dict[str, AgentProfile] = {}
        self._register_defaults()

    def register(self, profile: AgentProfile) -> None:
        self._profiles[profile.name] = profile

    def get(self, name: str) -> AgentProfile:
        if name not in self._profiles:
            raise KeyError(
                f"No agent profile '{name}'. "
                f"Available: {list(self._profiles)}"
            )
        return self._profiles[name]

    def list(self) -> list[AgentProfile]:
        return list(self._profiles.values())

    def _register_defaults(self) -> None:
        self.register(AgentProfile(
            name="default",
            description="Full multi-agent pipeline: planner + implementer + reviewer + tester.",
        ))

        self.register(AgentProfile(
            name="fast",
            description="Orchestrator only — no subagents. Fastest and cheapest.",
            enable_planner=False,
            enable_implementer=False,
            enable_reviewer=False,
            enable_tester=False,
        ))

        self.register(AgentProfile(
            name="review",
            description="Orchestrator + reviewer. Best for code review and security audits.",
            enable_planner=False,
            enable_implementer=False,
            enable_reviewer=True,
            enable_tester=False,
            system_prompt_addon=(
                "Focus primarily on code review. "
                "Use the reviewer-agent for all code analysis tasks."
            ),
        ))

        self.register(AgentProfile(
            name="write-tests",
            description="Orchestrator + tester. Best for adding tests to existing code.",
            enable_planner=False,
            enable_implementer=False,
            enable_reviewer=False,
            enable_tester=True,
            system_prompt_addon=(
                "Focus exclusively on writing tests. "
                "Always use the tester-agent to write comprehensive test suites."
            ),
        ))

        self.register(AgentProfile(
            name="docs",
            description="Orchestrator + documenter. Best for READMEs, docstrings and guides.",
            enable_planner=False,
            enable_implementer=False,
            enable_reviewer=False,
            enable_tester=False,
            enable_documenter=True,
        ))

        self.register(AgentProfile(
            name="full",
            description="All agents including documenter.",
            enable_documenter=True,
        ))


# Module-level singleton
_registry = AgentRegistry()


def get_profile(name: str) -> AgentProfile:
    return _registry.get(name)


def list_profiles() -> list[AgentProfile]:
    return _registry.list()


def register_profile(profile: AgentProfile) -> None:
    _registry.register(profile)
