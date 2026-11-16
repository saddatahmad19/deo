---
name: greenfield-bootstrap
description: Bootstrap a genuinely new repository with evidence-based project conventions, shared agent context, and safe defaults. Use when Deo initializes an empty or newly created project.
---

# Greenfield Bootstrap

Follow the greenfield bootstrap discipline inspired by the public Oh My Hermes omh-idea-to-deploy workflow and its project-bootstrap reference:
https://github.com/rlaope/oh-my-hermes/blob/main/skills/omh-idea-to-deploy/references/project-bootstrap.md

This is a Deo-native implementation, not a copy of the upstream skill.

## Required sequence

1. Inspect the directory and identify whether it is truly empty. Do not classify an established repository as greenfield.
2. Create or safely extend .gitignore. Always protect .env, .env.*, private keys, and certificates; re-include .env.example.
3. Create README.md with only known facts. Do not invent install, build, test, or deploy commands.
4. Create a concise AGENTS.md behavioral contract. Record commands as unknown until they have been observed working.
5. Create CLAUDE.md as a one-line @AGENTS.md import when no user-authored CLAUDE.md exists. Preserve custom files.
6. Ask the owner to choose the license. Never silently create an open-source license or declare one in package metadata.
7. Add CI only after a toolchain and exact verification commands are known. Keep CI commands identical to the documented commands.
8. Create docs/ only when real documentation topics exceed README length; do not scaffold empty directories.

## Existing projects

Use codebase evidence to generate or update AGENTS.md. Preserve user-authored AGENTS.md and CLAUDE.md unless an explicit force option is supplied. Index the whole project through Deo's normal incremental index so the index remains consistent with ongoing sync.

## Completion reporting

Separate files created from files preserved. Distinguish observed commands from prepared-but-not-run commands. Never claim Git initialization, commits, tests, CI, or deployment happened unless they were actually run.
