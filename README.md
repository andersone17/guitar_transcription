# guitar_transcription

A research/engineering project toward **multimodal automatic guitar transcription**: turning a recording
of a guitarist, both audio and video, into standard notation and tablature that shows *what was actually played*.

> **Current status: Stage 1 (audio-only).** Audio → raw note events → quantized standard notation
> (MusicXML) works from the command line. The tempo is given by you or estimated from the audio,
> and the meter is given by you (see [Usage](#usage)). Tablature, performance MIDI, and meter
> estimation are not done yet. See [PLAN.md](PLAN.md).

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

Stage 1 does **not** include computer vision, fingering selection, technique detection, meter or
downbeat estimation, tempo changes, live capture, or any UI. (A basic, optional global tempo
estimate, `--auto-tempo`, was added early.)

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

## Stage 1 Quick Start

From a guitar recording to standard notation you can open in MuseScore. Read
[Stage 1 Limitations](#stage-1-limitations) first so the results aren't a surprise.

**1. Install** (Linux, macOS, or WSL, with [uv](https://docs.astral.sh/uv/); Python 3.11 is fetched automatically):

```bash
git clone <this repository> guitar_transcription && cd guitar_transcription
uv sync --extra basic-pitch --extra tempo      # ~2 GB (TensorFlow); first run is slower
```

**2. Record** something that suits Stage 1:
- **Solo guitar only:** no backing track, voice, drums, or metronome click bleeding in. Acoustic or
  clean electric; heavy distortion and effects hurt.
- **Standard tuning, up to fret 22** (E2–D6). For drop tunings, 7-string, or 24-fret playing, add
  `--full-range`.
- **A single melody or simple chords, at a steady tempo.** Start playing on beat 1, about 0.1 s in:
  the recording's start is treated as the first downbeat, so a long silence or a pickup shifts every
  barline.
- **WAV or FLAC, 30 s or less, recorded close to the guitar.** Save it in `data/raw/`, which is
  git-ignored.

**3. Transcribe** and check the raw notes:

```bash
uv run guitar-transcribe transcribe data/raw/take1.wav
```

The table shows each detected note with its **raw** start/end in seconds. Check that the pitches match
what you played before worrying about rhythm.

**4. Generate MusicXML.** You supply the time signature. Give the tempo if you know it, or let it be
estimated:

```bash
uv run guitar-transcribe transcribe data/raw/take1.wav \
    --tempo 90 --time-signature 4/4 --musicxml outputs/take1.musicxml

uv run guitar-transcribe transcribe data/raw/take1.wav \
    --auto-tempo --time-signature 4/4 --musicxml outputs/take1.musicxml
```

With `--auto-tempo`, read the "Estimated tempo" line. If the notation looks twice too fast or slow,
rerun with one of the `--tempo` values it suggests. For simple lines, `--grid eighth` often reads more
cleanly than the default sixteenth grid.

**5. Open it** in [MuseScore](https://musescore.org) (free) via File → Open → `outputs/take1.musicxml`,
or in Finale, Dorico, Sibelius, and similar. Expect one "Guitar" staff in treble clef with a small 8
below it, and no tablature yet.

For a step-by-step check against a known recording, see
[docs/manual-testing.md](docs/manual-testing.md#stage-1-end-to-end-with-a-real-guitar-recording).

## Stage 1 Limitations

Stage 1 is an audio-only baseline. Specifically, it:

- **Doesn't know which string and fret you used.** The same pitch can be played in several places, and
  audio alone can't tell them apart. `string`/`fret` stay empty, the output has **no tablature**, and
  candidate positions are computed internally but never chosen.
- **Doesn't use video.** Fretting-hand and picking-hand vision, which would resolve string/fret and
  pick direction, are later stages.
- **Requires you to give the meter.** `--time-signature` is mandatory. It isn't inferred, because
  3/4 vs 6/8 and similar choices are unreliable from solo guitar audio (see PLAN.md §2a).
- **Interprets rhythm only simply:**
  - one constant tempo, and the first downbeat is fixed at 0 s, so there's no pickup support;
  - notes snap to a straight grid, so triplets and swing come out wrong;
  - output is a single voice: notes that ring over the next one are cut short, and a bass line
    under a melody isn't shown separately;
  - `--auto-tempo` can land on half or double time;
  - note lengths come from the model's note ends, which are often early, so notes can look shorter
    or more staccato than played.
- **Doesn't handle expressive guitar techniques.** Bends, slides, hammer-ons, pull-offs, vibrato,
  harmonics, palm muting and dead notes aren't detected or notated. They may show up as wrong or
  extra notes: a bend can appear as two pitches, a harmonic as a high note.
- **Uses a general-purpose model.** Basic Pitch isn't guitar-specific. Expect missed notes in dense
  strums, occasional octave or ghost notes, and weaker results with distortion. Detection is limited
  to a standard-tuned 22-fret guitar's range unless you pass `--full-range`.
- **Isn't real-time.** It processes a finished recording file. Measured: about 6 s for a 30 s clip,
  including model start-up, tempo estimation and MusicXML, on an 8-core CPU. The first run after
  installing is slower while libraries compile.
- **Doesn't export performance MIDI or chord symbols, and has no GUI.**

## Installation

Requires [uv](https://docs.astral.sh/uv/) and Linux/macOS/WSL. Python is pinned to **3.11**; uv installs it
if needed.

```bash
git clone <this repository> guitar_transcription
cd guitar_transcription
uv sync --extra basic-pitch --extra tempo   # + Basic Pitch (TensorFlow 2.15, ~2 GB) + tempo estimation
```

Without `--extra basic-pitch` everything installs and the tests run, but `transcribe` will exit with
an error that tells you how to install the backend. `--extra tempo` (librosa, which Basic Pitch
already pulls in) is only needed for `--auto-tempo`. Note that a later plain `uv sync` *removes*
extras, so keep passing them.

## Usage

Transcribe an audio file (`.wav`, `.flac`, `.ogg`; `.mp3`/`.m4a` may need `ffmpeg`) into raw note events:

```bash
uv run guitar-transcribe transcribe path/to/audio.wav
uv run guitar-transcribe transcribe path/to/audio.wav --json outputs/audio.events.json
uv run guitar-transcribe transcribe --help
```

(`uv run` uses the project environment. After `source .venv/bin/activate`, `guitar-transcribe ...`
works directly.)

Example output:

```text
RAW PERFORMANCE TIMING: seconds from the start of the recording, as played. Not quantized to beats or note values.

  onset_s  offset_s  duration_s  midi  note  velocity
    0.499     0.975       0.476    55  G3        0.86
    0.998     1.474       0.476    60  C4        0.84
    1.498     1.962       0.464    64  E4        0.77
    1.998     2.474       0.476    67  G4        0.87

4 notes detected.
```

- Times are **raw performance timing** in seconds. There are no beats, bars, or note values yet; those
  come from the rhythm stage.
- `velocity` is Basic Pitch's normalized note amplitude (0–1), not a calibrated confidence.
- String/fret are not inferred at this stage.
- Detection is limited to a standard-tuned 22-fret guitar's range (E2–D6), which removes ghost notes
  from string harmonics. Add `--full-range` for drop/extended tunings or 24-fret playing.
- Progress and errors go to stderr, and the table goes to stdout.
- Exit codes: `0` success, `2` missing/unreadable input or bad arguments, `1` transcription failure
  (e.g. backend not installed).

`--json` writes the events for later pipeline stages. The format is `guitar-transcription/performance-events`,
version 1, defined in `src/guitar_transcription/domain/serialization.py`. Every event field is written,
with `null` for unknowns:

```json
{
  "format": "guitar-transcription/performance-events",
  "version": 1,
  "timing": "raw-performance-seconds",
  "source": "path/to/audio.wav",
  "events": [
    {"onset_seconds": 0.4992, "offset_seconds": 0.9752, "pitch_midi": 55, "velocity": 0.8558,
     "string": null, "fret": null, "pick_direction": null, "audio_confidence": null,
     "fretting_confidence": null, "picking_confidence": null, "confidence": null}
  ]
}
```

### Standard notation (MusicXML)

Add `--musicxml` to also quantize the notes and write standard notation. Open the file in
MuseScore, Finale, Dorico, or similar. The time signature is required (meter is not inferred yet).
The tempo is either given (`--tempo`) or estimated from the audio (`--auto-tempo`):

```bash
uv run guitar-transcribe transcribe path/to/audio.wav \
    --tempo 120 --time-signature 4/4 \
    --musicxml outputs/audio.musicxml

uv run guitar-transcribe transcribe path/to/audio.wav \
    --auto-tempo --time-signature 4/4 \
    --musicxml outputs/audio.musicxml
```

| Option | Meaning |
|---|---|
| `--musicxml PATH` | Write MusicXML (parent folders are created). |
| `--tempo BPM` | **Quarter notes** per minute, in every meter. In 6/8 with a dotted-quarter pulse of 80, pass `--tempo 120`. |
| `--auto-tempo` | Estimate the tempo by beat tracking (librosa) instead. The detected pulse is used as the quarter note. It can't be combined with `--tempo`; an explicit tempo always takes precedence. |
| `--time-signature N/D` | e.g. `4/4`, `3/4`, `6/8`, `2/2`. |
| `--grid VALUE` | Finest subdivision to snap to: `whole`, `half`, `quarter`, `eighth`, or `sixteenth` (default). |

How it works:
- The table and `--json` still show **raw** timing. Quantization only affects the MusicXML.
- Second 0 of the recording is beat 1 of measure 1 (there is no pickup), so silence before the first
  note becomes a rest. Trim leading silence, or count it in.
- Notes are snapped to the grid, overlapping notes (let-ring) are shortened into a single voice, and
  simultaneous notes become chords. Ties across barlines, dots, and rests are handled.
- The output is one "Guitar" part in treble clef 8vb (guitar sounds an octave lower than written).
  There is no tablature yet.

Choosing a grid: `sixteenth` keeps the most detail but shows every early note release as a short note
plus a rest. `eighth` reads more cleanly for simple lines, but notes closer together than an eighth
merge into chords. A tempo that's slightly off makes both worse, so get it as close as you can.

About `--auto-tempo`:
- It measures the tempo precisely (within ~1% on test click tracks at 60–160 BPM). It is often better
  than a tapped or guessed tempo.
- **It can pick the wrong metrical level:** half or double the tempo you'd write. A scale in steady
  eighth notes at ♩ = 100 is usually detected as 200, because every note gets its own beat. The CLI
  prints both alternatives. If the notation looks twice too fast or slow, rerun with the suggested
  `--tempo`.
- It assumes a steady tempo and doesn't find the downbeat. Measure 1 still starts at 0 s.
- In meters like 6/8 or 2/2 the detected pulse often isn't a quarter note. Prefer `--tempo` there.
- The first estimate in a session takes a few extra seconds while librosa compiles its code.

Keep recordings in `data/raw/` and outputs in `outputs/`. Both are git-ignored.

## Development

Requires [uv](https://docs.astral.sh/uv/). Python is pinned to **3.11** (see PLAN.md, "Decision log").

```bash
uv python install 3.11      # once; uv sync will also fetch it automatically
uv sync                     # creates .venv with the package (editable) + dev tools
uv run pytest               # fast tests (no model inference)
uv run ruff check           # lint
uv run ruff format --check  # formatting
```

Tests come in three tiers:

| Tier | Command | What it covers | Speed |
|---|---|---|---|
| Unit | `uv run pytest` | each module alone, with synthetic events or fakes | ~1 s total |
| Pipeline contract | `uv run pytest` (in `tests/test_pipeline.py`) | the real rhythm and notation modules chained through `pipeline.py`, with only the model faked through our own protocols; checks what crosses each boundary | included above |
| Integration | `uv run pytest -m integration` | real Basic Pitch and librosa, including audio → MusicXML through the actual CLI on a synthesized melody | ~10 s; needs `--extra basic-pitch --extra tempo` |

The default run skips integration tests. For a real guitar recording, follow
[docs/manual-testing.md](docs/manual-testing.md).
