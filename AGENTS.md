# AGENTS.md

Permanent instructions for AI coding agents (Codex, Claude Code, etc.) working in this repository.
Read `README.md` for the product vision and `PLAN.md` for architecture, stages, and acceptance criteria.

## What this project is

A multimodal automatic guitar transcription system (audio + video of both hands → notation + tablature).
**We are currently in Stage 1: audio file → pretrained transcription → `PerformanceEvent`s → rhythm
quantization (known tempo/meter) → MIDI/MusicXML.**
Do not build later-stage features (vision, fusion, live capture, GUI) unless the task explicitly asks for them.

## Core architectural rules

1. **`PerformanceEvent` is the central domain object**, not tablature and not a rendered score.
   Every subsystem either produces evidence for `PerformanceEvent`s or consumes them.
   Tablature, MIDI, and MusicXML are *views* derived from events.
2. **`GuitarConfig`** (strings, tuning, capo, max fret) is passed explicitly; never hard-code standard tuning
   deep inside logic. Standard tuning may exist only as a named default/factory.
3. **Third-party models live behind our own interfaces.** Only the adapter module for a backend
   (e.g. `audio/backends/basic_pitch.py`) may import that backend's package. Nothing else may import
   `basic_pitch`, `tensorflow`, `mediapipe`, etc. Backend imports are lazy (inside the adapter) so the
   core package imports without heavy optional dependencies installed.
4. **Dependency direction:** `domain` depends on nothing in this project. `guitar` depends only on `domain`.
   `audio`, `rhythm` (and later `vision`, `fusion`) depend on `domain`/`guitar`, never on each other.
   `notation` may also depend on `rhythm`, because it renders rhythm's output. Only the pipeline/CLI layer
   wires modules together.
5. **Keep unknowns explicit.** Fields not yet inferred (string, fret, technique, picking direction,
   vision confidences) are `None`, not guessed defaults. Stage 1 must not pretend to know string/fret.
6. **Performance time ≠ musical time.** `PerformanceEvent` times are raw seconds as played and are never
   replaced by quantized values. Quantization creates new objects that reference their source events.
   Musical time (tempo, meter, measures, beat positions, note values, rests, tuplets) is inferred *only*
   in `rhythm`. Perception modules emit raw seconds only. `notation` consumes rhythm's output and never
   infers timing itself (MIDI ticks/MusicXML divisions are just encodings). See PLAN.md §2a.
7. **Pitch is a MIDI note number (int) in the domain**; pitch names are derived.

## Code conventions

- Python **3.11** (constraint from Basic Pitch; see PLAN.md). `pyproject.toml` packaging, `src/` layout,
  package name `guitar_transcription`.
- Type hints on all public functions. Prefer `@dataclass(frozen=True, slots=True)` for domain values.
- Prefer plain functions and small `typing.Protocol` interfaces over class hierarchies and frameworks.
- No web app, GUI, Docker, or computer-vision dependencies until the relevant stage.
- Add a dependency only with a stated reason; heavy/backend dependencies go in optional extras.
- Readability over cleverness. Small modules, clear names, docstrings on public APIs explaining *why*.

## Testing

- `pytest`. Tests live in `tests/`, mirroring the package layout.
- Unit tests must not require model weights, network, GPU, or large audio. Use synthetic events,
  tiny generated signals, or fakes implementing the backend protocol.
- Tests that run a real model are marked `@pytest.mark.integration` and skipped when the backend
  isn't installed.
- Tests that need external resources (e.g. the MusicXML schema) are also `integration` and skip with
  a reason when the resource isn't provided.
- Before declaring work complete, run `uv run pytest`, `uv run mypy`, `uv run ruff check` and
  `uv run ruff format --check`, and report failures honestly.

## Repository hygiene

- Never commit model weights, audio/video recordings, or generated outputs (MIDI, MusicXML, PDFs).
  `data/` and `outputs/` contents are git-ignored except READMEs describing them.
- Tiny test fixtures (< ~100 KB, preferably synthesized in code) are acceptable in `tests/fixtures/`.
- Update `PLAN.md` when a milestone's status or an architectural decision changes.
  Record non-obvious decisions briefly under "Decision log" in `PLAN.md`.
