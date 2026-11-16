---
name: subagent-handoff
description: Delegation contract for assigning isolated tasks to subagents and integrating their outputs cleanly.
compatibility: Use in orchestrator flows with task tool and specialized subagents.
allowed-tools: task write_todos
---

# Subagent Handoff

## When To Use
- A task is complex, context-heavy, or naturally parallelizable.
- A specialized role (planner, implementer, reviewer, tester, documenter) can produce a better result faster.

## Delegation Contract
Every delegated task should include:
- Objective: what outcome is needed.
- Scope: files/components in and out of scope.
- Constraints: compatibility, style, performance, security, or deadlines.
- Deliverable format: exact sections expected in the response.

## Parallelization Rule
- If two delegated tasks are independent, launch them in parallel.
- If one depends on another, enforce sequencing and pass predecessor output forward.

## Integration Rule
After each subagent returns:
1. Extract key decisions and concrete outputs.
2. Resolve conflicts between subagent recommendations.
3. Produce one coherent user-facing result.

## Guardrails
- Do not delegate trivial tasks that can be done directly.
- Do not delegate without an explicit output contract.
- Do not expose raw subagent internals to the user; synthesize outcomes.
