# ADR 0001 — Record architecture decisions

- **Status:** Accepted
- **Date:** 2026-09-25

## Context

This is a learning project. The *decisions* matter as much as the code, because
the goal is to understand *why* an ML platform is shaped the way it is. Six
months from now the reasoning behind a choice is invisible in the code itself.

## Decision

Use lightweight **Architecture Decision Records** (ADRs). One file per decision,
numbered, dated, immutable once accepted. Format:

- **Context** — the forces at play.
- **Decision** — what we chose.
- **Consequences** — what this buys us and what it costs.
- **Alternatives considered** — what we rejected and why.

## Consequences

- Every non-obvious choice has a paper trail.
- Superseding a decision means writing a new ADR that references the old one,
  never editing history.

## Alternatives considered

- **A single DECISIONS.md** — grows unwieldy, merge-conflict prone, no natural
  unit to link to.
- **Comments in code** — good for *local* why, bad for *cross-cutting* why.
