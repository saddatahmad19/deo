# Startup UI Redesign

**Date:** 2026-04-20
**Status:** Approved

## Goal

Replace the current two-step startup (ASCII banner + separate session panel) with a single, cinematic framed centerpiece that has cyberpunk personality.

## Design

### Structure (top to bottom, single Rich Panel)

```
╔═════════════════════════════════════════════════╗  ← magenta border
║                                                 ║
║          ██████╗ ███████╗ ██████╗              ║
║          ██╔══██╗██╔════╝██╔═══██╗             ║
║          ██║  ██║█████╗  ██║   ██║             ║
║          ██║  ██║██╔══╝  ██║   ██║             ║
║          ██████╔╝███████╗╚██████╔╝             ║
║          ╚═════╝ ╚══════╝ ╚═════╝              ║
║                                                 ║
║      The AI that codes while you think.         ║
║                                                 ║
║  ───────────────────────────────────────────    ║
║  session abc7a8a5  ·  gpt-4o  ·  ~/deo  ·  1 chunk  ║
╚═════════════════════════════════════════════════╝
```

### Color palette

- DEO ASCII block: `bold cyan`
- Outer panel border: `magenta`
- Tagline: `bold magenta`
- Inner rule: `dim`
- Footer stats: `dim white`, `·` separators, no trailing symbols

### Constraints

- DEO block must be **explicitly centered** — use Rich `Align(CENTER)` or `Text(justify="center")` so it never left-hugs inside the panel
- Footer must be a single clean `Text` object with no stray `|` or other artifacts
- Path shortened with `~` substitution for home directory
- The separate `print_session_info()` call in cli.py is removed; all info merges into `print_banner()`

## Changes Required

### `deo/tui/display.py`

- `print_banner(version, session_id, model, root_dir, n_indexed)` — new signature, renders the full framed Panel
- `print_session_info()` — removed (merged into banner)

### `deo/cli.py`

- `print_banner()` call at startup: pass session/model/root/index args
- Remove the separate `print_session_info(...)` call in the interactive REPL path
- `print_banner()` call in `/clear` handler: can pass empty/None for session args (no session context available there), renders without footer
