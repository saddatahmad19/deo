---
name: safe-shell-execution
description: Reliable shell execution pattern for coding agents using absolute paths, scoped commands, and deterministic validation.
compatibility: Use in terminal-based coding sessions with execute tool access.
allowed-tools: execute ls read_file glob grep
---

# Safe Shell Execution

## When To Use
- Running tests, linters, formatters, build commands, or diagnostics.
- Reproducing or validating behavior after code edits.

## Command Strategy
- Use absolute paths when possible.
- Prefer targeted commands over full-repo sweeps when speed matters.
- Keep command chains deterministic and easy to retry.

## Failure Handling
1. Capture the exact failing command and error.
2. Isolate whether failure is environment, dependency, or code.
3. Apply a specific fix.
4. Re-run the same command to confirm resolution.

## Reporting Pattern
- Command run
- Exit status (or pass/fail)
- Key output lines that matter
- What changed as a result

## Guardrails
- Do not hide command failures.
- Do not switch to broad or unrelated commands when a targeted one exists.
- Avoid destructive commands unless explicitly requested.
