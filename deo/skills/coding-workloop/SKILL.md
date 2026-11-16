---
name: coding-workloop
description: End-to-end implementation loop for terminal coding tasks with discovery, focused edits, verification, and concise reporting.
compatibility: Use for repository-based coding tasks where filesystem and shell tools are available.
allowed-tools: semantic_search find_file glob grep read_file edit_file write_file execute git_status git_diff
---

# Coding Workloop

## When To Use
- The user asks for code changes, bug fixes, or feature work.
- The task touches multiple files and needs disciplined execution.
- You need reproducible verification before completion.

## Workflow
1. Discover context first.
- Use `semantic_search` and `find_file` to locate relevant code.
- Use `read_file` on candidate files before editing.

2. Plan minimal edits.
- Prefer reusing existing patterns and helpers.
- Keep scope tight and avoid speculative rewrites.

3. Implement with precision.
- Use `edit_file` for surgical changes.
- Use `write_file` only when creating genuinely new files.

4. Verify behavior.
- Run targeted commands with `execute` (tests, lint, type checks).
- If failures occur, fix root cause and rerun.

5. Summarize with evidence.
- Report changed files and behavioral impact.
- Include verification commands and outcomes.
- Call out known risks and follow-ups.

## Guardrails
- Never claim success without running at least one relevant verification step when feasible.
- Avoid introducing placeholders for requested behavior.
- Keep user-facing output concise and operational.
