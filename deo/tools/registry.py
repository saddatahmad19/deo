"""
deo/tools/registry.py

Tool registry for the Forge agent system.

Provides a central place to:
  - Register new tools (built-in or user-defined)
  - Retrieve tool subsets by tag
  - Disable tools without removing them
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolEntry:
    tool: Any                           # LangChain BaseTool or callable
    tags: list[str] = field(default_factory=list)  # e.g. ["search", "chroma"]
    enabled: bool = True
    description: str = ""


class ToolRegistry:
    """
    Central registry for all Forge tools.

    Usage:
        registry = ToolRegistry()
        registry.register(my_tool, tags=["custom", "api"])
        tools = registry.get_tools(tags=["search", "git"])
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}

    def register(
        self,
        tool: Any,
        tags: list[str] | None = None,
        enabled: bool = True,
        description: str = "",
    ) -> None:
        """Register a tool."""
        name = getattr(tool, "name", None) or getattr(tool, "__name__", str(tool))
        self._tools[name] = ToolEntry(
            tool=tool,
            tags=tags or [],
            enabled=enabled,
            description=description or getattr(tool, "description", ""),
        )

    def register_many(
        self,
        tools: list[Any],
        tags: list[str] | None = None,
    ) -> None:
        """Register multiple tools with the same tags."""
        for t in tools:
            self.register(t, tags=tags)

    def disable(self, name: str) -> None:
        """Disable a tool by name."""
        if name in self._tools:
            self._tools[name].enabled = False

    def enable(self, name: str) -> None:
        """Re-enable a disabled tool."""
        if name in self._tools:
            self._tools[name].enabled = True

    def get_tools(
        self,
        tags: list[str] | None = None,
        names: list[str] | None = None,
    ) -> list[Any]:
        """
        Retrieve enabled tools, optionally filtered by tags or names.

        If both tags and names are provided, returns the union.
        If neither is provided, returns all enabled tools.
        """
        results = []
        for name, entry in self._tools.items():
            if not entry.enabled:
                continue

            if tags is None and names is None:
                results.append(entry.tool)
            elif names and name in names:
                results.append(entry.tool)
            elif tags and any(t in entry.tags for t in tags):
                results.append(entry.tool)

        return results

    def list_tools(self) -> list[dict[str, Any]]:
        """Return a summary of all registered tools."""
        return [
            {
                "name": name,
                "tags": entry.tags,
                "enabled": entry.enabled,
                "description": entry.description[:80] if entry.description else "",
            }
            for name, entry in self._tools.items()
        ]

    def __repr__(self) -> str:
        enabled = sum(1 for e in self._tools.values() if e.enabled)
        return f"ToolRegistry({enabled} enabled / {len(self._tools)} total)"


# ---------------------------------------------------------------------------
# Global registry instance
# ---------------------------------------------------------------------------

_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    return _registry
