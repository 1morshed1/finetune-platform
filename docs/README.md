# Docs — read this as a course

This folder is the *why* behind the code. It is written so that someone who has
never built an ML platform can follow along and understand each decision.

Three kinds of document:

### 1. Build journal — `learning/`
A chronological log. Each entry covers one build phase: what I built, the
concepts involved, the mistakes/tradeoffs, and what to read next. Start here if
you want the story in order.

- [`learning/00-journal.md`](learning/00-journal.md) — the running index of phases
- [`learning/01-project-scaffold.md`](learning/01-project-scaffold.md)
- [`learning/02-data-validation.md`](learning/02-data-validation.md)
- [`learning/03-orchestration-and-training.md`](learning/03-orchestration-and-training.md)
- [`learning/04-registry-and-deploy.md`](learning/04-registry-and-deploy.md)
- [`learning/05-api-and-dashboard.md`](learning/05-api-and-dashboard.md)
- [`learning/06-infrastructure.md`](learning/06-infrastructure.md)

### 2. Architecture Decision Records — `adr/`
Short, dated records of *decisions* and their alternatives. Read one when you
wonder "why did they do X and not Y?".

- [`adr/0001-record-architecture-decisions.md`](adr/0001-record-architecture-decisions.md)
- [`adr/0002-dependency-strategy.md`](adr/0002-dependency-strategy.md)
- [`adr/0003-simulated-trainer-by-default.md`](adr/0003-simulated-trainer-by-default.md)
- [`adr/0004-local-first-infrastructure.md`](adr/0004-local-first-infrastructure.md)

### 3. Subsystem deep-dives — `*.md`
Longer explainers on one topic each. Read when you want the mechanics.

- [`architecture.md`](architecture.md) — one-page system map (control/data plane, the swappable seams)
- [`concepts-fine-tuning.md`](concepts-fine-tuning.md) — what fine-tuning / LoRA actually is
- [`data-validation.md`](data-validation.md) — the validation pipeline internals (MinHash, perplexity, cost)

---

## How to use this

1. Skim `learning/00-journal.md` for the map.
2. Read each phase entry before reading the code it describes.
3. When a phase references an ADR, read the ADR — that is where the real
   tradeoff lives.
