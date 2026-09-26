# guitar_transcription

A research/engineering project toward **multimodal automatic guitar transcription**: turning a recording
of a guitarist, both audio and video, into standard notation and tablature that shows *what was actually played*.

> **Current status: Stage 1 (audio-only), milestone M0 (project skeleton) done.** No transcription code yet. See [PLAN.md](PLAN.md).

## Motivation

Transcribing guitar by ear is slow. Existing automatic tools are good at hearing *which pitches*
sounded, but a guitarist also needs to know *where on the neck* they were played, and with which technique.
That information is the difference between a score and usable tablature. Most of it can't be heard,
but you can see it.

## End-state vision

1. A guitarist sits in front of a webcam with the whole guitar visible, including both hands.
2. The system records synchronized audio and video.
3. The user specifies the guitar configuration: number of strings, tuning, and capo.
4. The system analyzes:
   - **audio**: pitches, onsets, offsets, rhythm, and polyphony
   - **fretting-hand video**: which strings and frets are held down
   - **picking-hand video**: which strings are actually plucked or strummed, and in which direction
5. These signals are **fused** to infer the physical performance.
6. Output: standard notation plus tablature with the real string/fret positions, and eventually
   chord symbols and techniques (bends, slides, hammer-ons, and so on).
7. Export to **MusicXML** so the result opens and can be edited in MuseScore and similar tools.

## Why audio alone can't determine tablature

The same pitch can be played in several places on a guitar. In standard tuning, E4 (MIDI 64) can be
played on the open high-E string, B string fret 5, G string fret 9, D string fret 14, or A string fret 19.
A typical note has 2–5 valid positions, so a chord or phrase has a combinatorial number of fingerings.

Audio carries only weak cues about string choice. Timbre differs slightly between strings, and the
available fingerings constrain which notes can ring together. Those cues are unreliable with real
microphones, effects, and different instruments. Algorithmic "playability" heuristics give a *plausible*
fingering, not the one the player *used*. For teaching, archiving, or reproducing a performance, the
actual fingering is what matters.

## Why both hands matter

- **The fretting hand** shows candidate positions: which frets are pressed on which strings. A held
  shape does not mean every string sounds, though. Players fret full chord shapes and pick only some
  strings, and fingers mute neighboring strings.
- **The picking hand** shows which strings are excited, when, and in which direction (down/up strokes,
  arpeggios, strums that skip strings).
- **Audio** gives precise timing and confirms what pitches sounded.

None of these is enough alone. Fretting-hand evidence says what *could* sound, picking-hand evidence says
what *was struck*, and audio says what *did* sound and when. Combining them resolves ambiguities that each
signal leaves open.

## Planned architecture

The central internal representation is a time-ordered sequence of **`PerformanceEvent`s**, not tablature
or a rendered score. Each event carries onset/offset times, MIDI pitch, and optional string, fret, velocity,
technique, picking direction, and per-modality confidences. A `GuitarConfig` (tuning, capo, number of strings,
max fret) defines what's physically possible.

```
            ┌──────────── capture/ (webcam, mic, A/V sync) ─────────────┐
            ▼                              ▼                            ▼
        audio/                       vision/ fretting              vision/ picking
   (pitch, onsets, offsets)      (hand pose, fretboard geom.)   (string excitation, direction)
            │                              │                            │
            └──────────── evidence ────────┼────────────────────────────┘
                                           ▼
                 guitar/  ──────────▶   fusion/   (candidates, scoring, temporal reasoning)
     (tuning, capo, pitch→string/fret)     │
                                           ▼
                             domain/  PerformanceEvent sequence   ← raw timing (seconds)
                                           │
                                           ▼
                  rhythm/  (tempo/beat grid, meter, measures, note values, rests, tuplets)
                                           │  quantized musical time
                                           ▼
                  notation/  (MusicXML standard notation + tablature; performance MIDI)
```

### Performance timing vs. notated rhythm

A recording tells us *when* things sounded ("7.183 s to 7.561 s"). A score needs *musical* rhythm
("a dotted eighth on beat 2 of measure 5"). These are kept as two separate representations:

- **`PerformanceEvent`s preserve raw timing** (onset/offset in seconds) exactly as played. That timing
  is never overwritten by quantized values, because alignment, fusion, evaluation, and expressive
  analysis all need it.
- **`rhythm/` interprets that timing musically.** It infers the tempo/beat grid, meter, measures, beat
  positions, note values, rests, and tuplets, and produces quantized events that link back to their
  source events. Early on, tempo and meter are supplied by the user. Automatic tempo, meter, and
  tempo/meter-change estimation come later.
- **`notation/` only renders** already-quantized musical time. It doesn't guess rhythm. Audio
  transcription produces raw timing, and notation consumes quantized rhythm.

Modules are loosely coupled. Every third-party model (Basic Pitch today, perhaps MediaPipe or a
guitar-specific transcription model later) sits behind a small interface we own, so it can be swapped
without changing the rest of the system.

## Stage 1 scope (current focus)

```
audio file → pretrained transcription (Spotify Basic Pitch) → PerformanceEvents (raw timing)
           → rhythm quantization (user-supplied tempo + meter) → MusicXML → notation
           (and PerformanceEvents → performance MIDI, unquantized)
```

Stage 1 includes:
- domain models `PerformanceEvent`, `Performance`, and `GuitarConfig`
- a backend-agnostic `AudioTranscriber` interface with a Basic Pitch adapter
- pitch → candidate (string, fret) enumeration for a `GuitarConfig`, as groundwork for tablature.
  Candidates are enumerated but **not** chosen.
- rhythm quantization onto a beat/measure grid from a **user-supplied** tempo and time signature
- export to performance MIDI (raw timing) and to quantized MusicXML that opens in MuseScore
- a minimal command-line entry point

Stage 1 does **not** include computer vision, fingering selection, technique detection, automatic
tempo/meter estimation, live capture, or any UI.

## Roadmap (high level)

1. **Stage 1:** audio-only transcription to notation *(next)*
2. Evaluation harness on a public guitar dataset (e.g. GuitarSet)
3. Tablature from audio with a heuristic/playability fingering baseline
4. Fretting-hand vision: hand tracking and fretboard geometry → fret/string evidence
5. Picking-hand vision: which strings are struck, and in which direction
6. Multimodal fusion: probabilistic scoring with temporal reasoning
7. Better polyphony and guitar-specific transcription models
8. Technique detection (bends, slides, hammer-ons/pull-offs, palm muting, …)
9. Automatic calibration (fretboard detection, tuning/capo detection)
10. Live, near-real-time transcription

Alongside these stages, a **rhythm track** progresses from known tempo/meter (Stage 1) to automatic
tempo/beat tracking, meter estimation, and tempo/meter changes.

Details, acceptance criteria, and open research questions are in [PLAN.md](PLAN.md).

## Development

Requires [uv](https://docs.astral.sh/uv/). Python is pinned to **3.11** (see PLAN.md, "Decision log").

```bash
uv python install 3.11      # once; uv sync will also fetch it automatically
uv sync                     # creates .venv with the package (editable) + dev tools
uv run pytest               # tests
uv run ruff check           # lint
uv run ruff format --check  # formatting
```
