# PLAN.md

Architecture, stage plan, and acceptance criteria. The long-term system is multimodal (audio + video);
**the current goal is only Stage 1: audio-only transcription to notation.** Sections marked *future* are
direction, not commitments.

---

## 1. Architectural principles

1. **The performance is the model; notation is a view.** The core representation is a time-ordered sequence
   of `PerformanceEvent`s in physical time (seconds). MIDI, MusicXML, and tablature are derived from it and
   never fed back in as a source of truth.
2. **Performance time is not musical time.** `PerformanceEvent`s carry *raw* timing (seconds) exactly as
   played, and that timing is never overwritten with quantized values. Musical rhythm (tempo, meter,
   measures, beat positions, note values, rests, tuplets) is a separate *interpretation* produced by
   `rhythm/` and linked back to the raw events. Perception (`audio/`, later `vision/`) produces only raw
   timing, and `notation/` consumes only already-quantized musical time. It never infers rhythm itself.
   See §2a.
3. **Evidence in, events out.** Perception modules (audio now, vision later) emit *modality-specific
   evidence* with confidences. Converting evidence into `PerformanceEvent`s is a separate step. In Stage 1
   that step is a trivial 1:1 mapping; later it becomes `fusion/`. Evidence and events are kept separate
   now so that adding vision later doesn't require redesigning the audio path.
4. **Unknown is explicit.** `string`, `fret`, `technique`, `picking_direction`, and vision confidences are
   `None` until something actually infers them. Nothing silently fills in a guess.
5. **Own the interfaces, rent the models.** Each third-party model sits behind a small `Protocol` we define.
   Only its adapter imports it, and it's installed as an optional extra.
6. **Physical configuration is data.** `GuitarConfig` is passed explicitly and determines which pitches
   and positions are possible.
7. **One-way dependencies.** `domain` ← `guitar` ← {`audio`, `rhythm`, later `vision`, `fusion`} ← `pipeline`/`cli`,
   and `notation` depends on `domain`, `guitar`, and `rhythm` (whose output it renders). Otherwise sibling
   modules never import each other. In particular, `rhythm` never imports `notation` or `audio`.
8. **Boring code.** Dataclasses, functions, Protocols. No plugin registries, DI frameworks, or config
   systems until a real need appears.

## 2. Module boundaries

```
src/guitar_transcription/
    domain/          # pure data; stdlib only
        pitch.py         MIDI pitch validation, MIDI -> pitch name (M2 adds name -> MIDI)
        guitar_config.py GuitarConfig, STANDARD_TUNING, STANDARD_GUITAR
        events.py        PerformanceEvent, PickDirection (Technique added in Stage 8)
        performance.py   Performance (sorted events + GuitarConfig)
        evidence.py      AudioNoteEvidence (M3; later FrettingEvidence, PickingEvidence)
    guitar/          # instrument reasoning; pure Python
        tunings.py       parsing "E2,A2,..." strings, named tunings (M2)
        positions.py     pitch -> candidate (string, fret) positions under a GuitarConfig
    audio/           # audio perception
        transcriber.py   AudioTranscriber Protocol, TranscriptionOptions
        backends/
            basic_pitch.py   BasicPitchTranscriber (sole importer of basic_pitch; lazy import)
    rhythm/          # performance time -> musical time (pure Python, no music21)
        quantize.py      raw events + known tempo/meter -> events placed on a beat/measure grid (M5)
    notation/        # rendering views; no rhythm inference
        midi.py          performance MIDI export from raw seconds (no quantization)
        musicxml.py      quantized rhythm output -> MusicXML via music21 (M6)
    pipeline.py      # wires audio -> events -> rhythm -> notation for a file
    cli.py           # argparse entry point: `guitar-transcribe`
tests/               # mirrors package layout; tests/fixtures/ for tiny synthesized inputs
data/                # git-ignored local datasets/recordings (README.md only is tracked)
docs/                # design notes, research notes, dataset notes
notebooks/           # exploration only; nothing the package depends on
```

Future modules, not created until their stage: `capture/`, `vision/`, `fusion/`, `notation/tab.py`.

Changes from the suggested layout:
- `domain/evidence.py` was added so that perception output has a modality-neutral home that `fusion/`
  can later consume without depending on `audio/` or `vision/`.
- `rhythm/` is split out of `notation/` so that rhythm inference (which later grows into tempo, beat, and
  meter estimation) is testable without music21 and isn't hidden inside an exporter. See §2a.
- `pipeline.py` is the only place that knows about more than one module. This keeps the coupling in one
  replaceable spot.
- There's no top-level `models/` or `weights/` directory. Basic Pitch ships its weights inside its wheel,
  and we don't train anything.

### Key domain types (implemented in M1)

```python
@dataclass(frozen=True, slots=True)
class GuitarConfig:                       # domain/guitar_config.py
    open_strings: tuple[int, ...]         # MIDI pitches, index 0 = string 1 = top line of tab
    capo: int = 0                         # fret number of capo; 0 = none
    max_fret: int = 22
    # derived: num_strings, min_fret (= capo); method pitch_at(string, fret)

@dataclass(frozen=True, slots=True)
class PerformanceEvent:                   # domain/events.py
    onset_seconds: float                  # seconds from start of recording, >= 0
    offset_seconds: float                 # > onset_seconds
    pitch_midi: int                       # MIDI note number (plain int, 0..127)
    velocity: float | None = None         # normalized loudness 0..1 (notation maps to MIDI 1..127)
    string: int | None = None             # 1-based; set together with fret
    fret: int | None = None               # physical fret (capo-inclusive); 0 = open
    pick_direction: PickDirection | None = None
    audio_confidence: float | None = None      # 0..1
    fretting_confidence: float | None = None   # 0..1 (future)
    picking_confidence: float | None = None    # 0..1 (future)
    confidence: float | None = None            # 0..1 overall
    # derived: pitch_name, duration_seconds

@dataclass(frozen=True, slots=True)
class Performance:                        # domain/performance.py
    events: tuple[PerformanceEvent, ...]  # always sorted by (onset, pitch)
    config: GuitarConfig                  # known string/fret must be playable and sound pitch_midi
```

Conventions: `fret` is the **physical** fret because that's what a camera sees. With a capo at 2, an
"open" string is `fret == 2`. Notation code converts to capo-relative numbers when rendering tab.
All times in the domain are raw seconds. Beats, measures, and note values first appear in `rhythm/`'s output.

## 2a. Performance timing vs. notated rhythm

Standard notation and tablature both need *musical* rhythm, such as "a dotted eighth on beat 2 of
measure 5". A recording only gives *performance* timing, such as "a note from 7.183 s to 7.561 s".
Getting from one to the other is an inference problem with no single right answer, so it gets its own
stage and module:

```
audio file
  → audio/     raw evidence (seconds)
  → PerformanceEvent[]           raw timing: onset_seconds, offset_seconds (duration derived)
  → rhythm/    tempo / beat grid, meter, measures, quantized onsets & note values, rests, tuplets
  → quantized musical events     each one links back to its source PerformanceEvent
  → notation/  MusicXML (standard notation + TAB), rendering only
```

Rules:
- **Raw timing is never replaced.** `PerformanceEvent` is frozen, and quantization produces *new*
  objects that reference their source events. Raw seconds stay available for performance MIDI,
  evaluation against annotations, audio/video alignment, fusion, and expressive-timing analysis.
- **`rhythm/` owns every timing decision**: the tempo/beat grid, meter, measure boundaries, beat
  positions, note values, which gaps are rests, tuplet grouping, and (later) swing or expressive
  deviation. It is pure Python with no music21 dependency, so it can be unit-tested with synthetic
  events.
- **`notation/` only spells what `rhythm/` decided.** That means splitting notes at barlines into tied
  notes, choosing note-value symbols and dots, beaming, voices, clefs, and the TAB staff. If `notation/`
  ever needs to *guess* timing, that logic belongs in `rhythm/`.
- **Musical time is exact.** Beat positions and durations should use exact rationals (`fractions.Fraction`,
  in quarter notes) rather than floats, so triplets and other tuplets don't accumulate rounding error.
- **Performance MIDI is exempt.** `notation/midi.py` writes raw seconds and needs no rhythm analysis.
  Its tempo exists only to convert seconds to ticks.

Rhythm capability levels (each level keeps the same output types, so notation doesn't change):
1. **Known tempo + known meter** (user-supplied), a constant grid, and quantization. *Stage 1, M5.*
2. **Automatic tempo / beat tracking** (possibly a varying beat grid).
3. **Automatic meter / downbeat estimation.**
4. **Tempo and meter changes**, tuplet detection beyond a fixed grid, swing, and rubato.

The output types (e.g. a quantized event and whatever represents tempo/meter) are **not designed yet**.
They're created in M5, shaped by what `quantize` and the MusicXML writer actually need, and they live in
`rhythm/` (not `domain/`) because they are an interpretation of the performance, not the performance itself.

## 3. Stage 1 plan: audio file → notation

Goal: given an audio file of solo guitar, produce (a) a performance MIDI file and (b) a MusicXML file
that opens in MuseScore as readable standard notation. The events must carry candidate-position
information hooks for later tablature work, without choosing fingerings.

### Proposed Stage 1 dependencies (not yet installed)

| Purpose | Package | Where | Why |
|---|---|---|---|
| Transcription backend | `basic-pitch==0.4.0` | optional extra `basic-pitch` | Pretrained polyphonic AMT, Apache-2.0, returns note events directly |
| MusicXML writing | `music21>=10` | core | Mature, BSD-3; handles durations, ties, measures, clefs, chords, and MusicXML export |
| MIDI writing | `mido` | core | Tiny, pure-Python (MIT); exact-time performance MIDI without quantization |
| Tests | `pytest` | `dev` dependency group | Requested standard |
| Lint/format | `ruff` | `dev` dependency group | One fast tool for lint and format |
| Type checking | `mypy` | `dev` dependency group (deferred until M1 adds typed code) | Enforces the type-hint policy |

Build backend: `hatchling`. Environment tool: `uv` (already installed), though a plain `pip install -e .[dev,basic-pitch]` also works.
`numpy` is not a direct core dependency until code needs it. It arrives transitively.

**Python version: 3.11, pinned as `requires-python = ">=3.11,<3.12"`.** Findings as of 2026-09:
- Basic Pitch's latest release is **0.4.0 (2024-08-16)**. Classifiers list 3.8–3.11, and there's no
  3.12 support. Open issues #188 and #203 report 3.12 install failures because the dependency is pinned
  to `tensorflow>=2.4.1,<2.15.1`.
- music21 10.x requires **Python ≥ 3.11**.
- Python 3.11 is therefore the only version both support. On Linux with Python 3.11, Basic Pitch pulls
  in TensorFlow 2.15 (large, requires `numpy<2`). music21 needs `numpy>=1.26.4`, so both resolve to numpy 1.26.x.
- The system Python here is 3.12. We'll use `uv python install 3.11` plus a `.python-version` file.
- The upper bound is loosened when the Basic Pitch backend is replaced or updated.

Basic Pitch facts relevant to the adapter (verified from source on `main`):
- `basic_pitch.inference.predict(audio_path, model_or_model_path=ICASSP_2022_MODEL_PATH, onset_threshold=0.5,
  frame_threshold=0.3, minimum_note_length=127.7 (ms), minimum_frequency=None, maximum_frequency=None,
  multiple_pitch_bends=False, melodia_trick=True, midi_tempo=120)` → `(model_output: dict[str, ndarray],
  midi_data: pretty_midi.PrettyMIDI, note_events: list[(start_s, end_s, midi_pitch, amplitude, pitch_bends|None)])`.
- A `basic_pitch.inference.Model(path)` can be constructed once and reused across calls.
- It ships TF SavedModel, CoreML, TFLite, and ONNX serializations. The runtime is chosen by what's installed
  (TF → CoreML → TFLite → ONNX).
- It is instrument-agnostic, trained on multiple instruments and not specifically on guitar. `amplitude`
  is a mean posteriorgram activation, not a calibrated probability.
- License: Apache-2.0. Weights ship inside the wheel, so there's nothing to download or commit.

### Milestones and acceptance criteria

**M0 — Project skeleton** — *done 2026-09-26*
- `pyproject.toml` (hatchling, src layout, PEP 735 `dev` dependency group), `.python-version` = 3.11,
  `data/README.md` (+ `raw/`, `processed/`), `docs/` and `notebooks/` placeholders, empty
  `domain/`, `guitar/`, `audio/`, `notation/` subpackages. The `basic-pitch` extra moves to M3 and
  the `guitar-transcribe` console script to M7, when they have code behind them.
- ✅ `uv sync` succeeds without installing TensorFlow or Basic Pitch.
- ✅ `python -c "import guitar_transcription"` works in that environment.
- ✅ `pytest` runs (placeholder test passes). `ruff check` and `ruff format --check` are clean.

**M1 — Domain model** — *done 2026-09-26*
- `GuitarConfig`, `PerformanceEvent`, `Performance`, `PickDirection`, `pitch_name`.
  `AudioNoteEvidence` moves to M3 (its fields are shaped by the adapter that produces it);
  `Technique` moves to Stage 8.
- ✅ Validation rejects `offset <= onset`, negative or non-finite times, non-int or out-of-range pitch,
  velocity/confidences outside [0, 1], and `string` without `fret` (or vice versa). Within a
  `Performance`, `fret` must be ≥ capo and ≤ max_fret, and string/fret must sound `pitch_midi`.
- ✅ `Performance` always exposes events sorted by (onset, pitch).
- ✅ `pitch_name` is derived (e.g. 64 → "E4"; sharps by default).
- ✅ `domain/` imports nothing outside the standard library (checked by a test).

**M2 — Guitar knowledge**
- Tuning parsing (`"E2,A2,D3,G3,B3,E4"` and a `STANDARD_6` default, including 7-string/drop tunings via strings),
  capo handling, playable pitch range, candidate positions.
- ✅ Standard tuning, capo 0, max_fret 24: `candidate_positions(64)` returns exactly
  {(1,0),(2,5),(3,9),(4,14),(5,19),(6,24)}.
- ✅ With capo 2, no candidate has `fret < 2`, and open-string pitches shift by +2.
- ✅ `pitch_range(config)` = (lowest open string + capo, highest open string + max_fret).
- ✅ Pitches outside the range return no candidates; they do not raise.

**M3 — Audio transcription interface and Basic Pitch adapter**
- `AudioTranscriber` Protocol: `transcribe(path, options) -> list[AudioNoteEvidence]`.
- `BasicPitchTranscriber` lazily imports `basic_pitch`, loads the model once, and passes the
  `GuitarConfig`'s frequency range as `minimum_frequency`/`maximum_frequency`.
- `evidence_to_events(evidence, config) -> Performance` performs the Stage 1 1:1 mapping
  (amplitude → `audio_confidence` and velocity; `string`/`fret` stay `None`).
- ✅ Unit tests use a fake transcriber and never import `basic_pitch`.
- ✅ Importing `guitar_transcription.audio` works without Basic Pitch installed. Using the backend
  without it raises a clear error that names the install extra.
- ✅ Integration test (`@pytest.mark.integration`, auto-skipped without the extra): a synthesized
  WAV (stdlib `wave` + plucked-string synthesis) of 3–4 known notes a few hundred ms apart is transcribed
  with the correct pitches, and onsets are within 50 ms.
- ✅ A grep/test confirms `basic_pitch` is imported only in `audio/backends/basic_pitch.py`.

**M4 — Performance MIDI export**
- `write_midi(performance, path)` writes unquantized events (tempo only for tick conversion).
- ✅ Round-trip test: reading the file back with `mido` recovers pitches exactly and times within 1 tick.
- ✅ Overlapping notes of the same pitch are handled without stuck notes.

**M5 — Rhythm quantization with known tempo and meter** (`rhythm/`, rhythm level 1)
- `quantize(performance, tempo_bpm, time_signature, grid)` is pure: raw seconds → exact beat positions
  snapped to a grid (default 16th notes), with a minimum duration of one grid step. Tempo and meter are
  user-supplied in Stage 1. The quantized output types are defined here, and each quantized event
  references its source `PerformanceEvent`.
- ✅ Exact-grid input is unchanged, jittered input (±20 ms at 120 BPM) snaps correctly, and nothing
  ends up with zero duration.
- ✅ Source events are unchanged: after quantization, raw `onset_seconds`/`offset_seconds` are
  identical and reachable from every quantized event.
- ✅ Measure/beat positions are correct for 4/4 and 3/4, and the positions are exact (`Fraction`, no float drift).
- ✅ `rhythm/` does not import music21 or `notation/` (checked by a test).

**M6 — MusicXML export** (`notation/`)
- `write_musicxml(quantized, path, config)` uses music21 with guitar conventions: treble-8vb clef and
  simultaneous onsets grouped into chords. Overlapping notes are clipped to the next onset in the
  Stage 1 single-voice simplification. It makes no timing decisions of its own.
- ✅ Output is well-formed MusicXML: it parses back with music21, and measure count and pitches match input.
- ✅ Notes crossing a barline come out as tied notes whose total duration equals the quantized duration.
- ✅ Manual check (documented in `docs/`): a sample output opens in MuseScore 4 and reads sensibly.

**M7 — Pipeline and CLI**
- `guitar-transcribe INPUT.wav --out outputs/ [--tuning E2,A2,D3,G3,B3,E4] [--capo N] [--strings N]
  [--tempo BPM] [--time-signature 4/4] [--onset-threshold …] [--frame-threshold …]`.
- Writes `<name>.mid`, `<name>.musicxml`, and `<name>.events.json` (the `Performance` serialized; useful
  for debugging and later evaluation).
- ✅ End-to-end integration test on the synthesized fixture produces all three files.
- ✅ README "Development" and "Usage" sections are updated with working commands.
- ✅ No generated files appear in `git status` after running the pipeline.

**Stage 1 done when** M0–M7 pass, and a real recorded guitar clip (not committed) produces MusicXML
that a guitarist judges roughly readable in MuseScore for simple monophonic and chordal material. The
known limitations are written down in `docs/stage1-notes.md`.

## 4. Future stages (*direction, not commitments*)

**Stage 2 — Evaluation harness.** Add a loader for GuitarSet (which has hexaphonic per-string annotations
and mic audio) and note-level metrics via `mir_eval` (onset/offset F1). This establishes a baseline so
that later changes are measured, not guessed. It also gives string/fret ground truth for later stages.

**Stage 3 — Audio-only tablature baseline.** Choose among `candidate_positions` with a playability cost
(hand span, position shifts, open-string preference), for example with Viterbi/dynamic programming over time.
Output `notation/tab.py` and MusicXML tab staff. Measure string/fret accuracy on GuitarSet. This is the
baseline that vision must beat.

**Stage 4 — Fretting-hand vision.** Add `capture/` (webcam + mic recording with a shared clock) and `vision/`:
hand landmarks (MediaPipe or a replacement behind a Protocol), fretboard detection with a homography to
fret/string coordinates, and fingertip → (string, fret) evidence with confidence. Emit `FrettingEvidence`
over time.

**Stage 5 — Picking-hand vision.** Estimate which strings are excited, when, and pick direction from
picking-hand motion relative to the string plane. This is likely hard at webcam frame rates
(see risks). Emit `PickingEvidence`.

**Stage 6 — Multimodal fusion.** `fusion/` generates candidates (audio notes × positions), scores them
using each modality's likelihood and priors, and runs temporal reasoning (HMM/Viterbi or similar) over
hand-position continuity. It fills in `string`, `fret`, `pick_direction`, and the per-modality and overall
confidences.

**Stage 7 — Polyphony improvements.** Evaluate guitar-specific transcription models behind
`AudioTranscriber`. Use the fretting-hand shape as a prior for which pitches can co-occur. Improve
strum handling (near-simultaneous onsets).

**Stage 8 — Techniques.** Detect bends and vibrato from pitch contours (Basic Pitch already emits pitch bends),
slides, hammer-ons, and pull-offs (onset without a pick event plus fretting change), palm muting, harmonics.
Map them to `Technique` and MusicXML technical notations.

**Stage 9 — Automatic calibration.** Detect the fretboard and frets without manual setup, estimate tuning
and capo from audio and video, and handle camera angle and left-handed players.

**Stage 10 — Live transcription.** Streaming audio and video with bounded latency, incremental fusion,
and a rolling notation view. This may require replacing batch backends.

**Rhythm track (parallel to Stages 2–10).** Advance `rhythm/` through capability levels 2–4 of §2a:
beat tracking, then meter/downbeat estimation, then tempo/meter changes, tuplets, and swing. It's
evaluated against annotated beats/downbeats where datasets provide them. This track depends only on raw
`PerformanceEvent`s (plus audio features if needed, passed in as evidence), so it can proceed
independently of the vision stages.

Also later: chord-symbol inference, multi-voice notation.

## 5. Risks and open research questions

- **Basic Pitch maintenance.** There has been no release since 2024-08, it is pinned to old TensorFlow,
  and it doesn't support Python ≥3.12. *Mitigation:* confine it to its adapter and an optional extra.
  Evaluate the ONNX model path with `onnxruntime` (unverified) or alternative models early.
- **Instrument-agnostic model quality on guitar.** Expect octave errors, missed notes in dense strums,
  and spurious notes from string resonance. Stage 2 metrics will quantify this.
- **Timing precision.** An open upstream issue (#190) reports frame-level temporal drift in Basic Pitch.
  This must be verified before relying on audio onsets for audio/video alignment.
- **Quantization without known tempo.** Stage 1 sidesteps this with user-supplied tempo and meter. Rubato
  and free playing will render poorly until the rhythm track matures.
- **Unreliable offsets.** Transcribed note *ends* are much less precise than onsets (decay, let-ring,
  damping). Rhythm inference should lean on onsets, and notated durations may need musical rules
  (e.g. extend to the next onset) rather than raw offsets. Raw offsets are still preserved.
- **Notation of polyphony.** A single-voice simplification loses independent bass/melody lines.
  Proper voice separation is an open problem.
- **Webcam frame rate vs. picking speed.** At 30 fps a frame lasts 33 ms, while fast picking and strums
  happen faster than that. Picking evidence may need to be temporal/probabilistic, not frame-exact,
  or require higher-fps capture.
- **Occlusion and viewpoint.** The fretting hand occludes strings, and the picking hand is small and blurred.
  Fretboard homography degrades at steep angles.
- **A/V synchronization.** Consumer webcams and microphones drift and have variable latency, so we need a
  sync strategy (shared clock, clap/onset alignment).
- **Calibrated confidences.** Per-modality scores must be comparable before fusion. How do we calibrate
  model outputs into likelihoods?
- **Evaluation data for multimodal work.** GuitarSet has audio and annotations but no suitable video. We
  may need to record our own annotated audio+video dataset. Its size and annotation cost are open questions.
- **Fingering ambiguity even with vision.** Held-but-unplayed strings, ghosted/muted notes, and let-ring
  sustain are hard. How should truly ambiguous events be represented (keep top-k candidates)?

## 6. Decision log

- 2026-09-26 — `PerformanceEvent` (seconds, MIDI pitch, optional physical string/fret) is the core
  representation. Notation formats are derived views.
- 2026-09-26 — Perception output is modality-specific *evidence* (`domain/evidence.py`), mapped to events
  by a separate step (trivial in Stage 1, `fusion/` later).
- 2026-09-26 — `fret` is the physical fret (capo-inclusive), and string 1 is the highest-pitched string.
- 2026-09-26 — Python pinned to 3.11: Basic Pitch 0.4.0 supports ≤3.11 and music21 10.x requires ≥3.11.
- 2026-09-26 — Basic Pitch is an optional extra and imported lazily in a single adapter module.
- 2026-09-26 — Stage 1 tempo and time signature are user-supplied. Tempo estimation is deferred.
- 2026-09-26 — Re-verified for M0: Basic Pitch is still 0.4.0 on PyPI and `main` still pins
  `tensorflow<2.15.1` (TF 2.15.0 wheels: cp39–cp311 only). An upstream PR adding 3.12 (Jan 2026) is unmerged.
- 2026-09-26 — Dev tools use a PEP 735 `[dependency-groups] dev` (installed by default by `uv sync`)
  rather than an extra, so they're never part of the published package metadata. Runtime backends stay extras.
- 2026-09-26 — `ruff format` is scoped to Python sources (`*.md` excluded) so ruff ≥0.16 doesn't rewrite
  the hand-aligned code sketches in docs.
- 2026-09-26 — Tests use pytest `--import-mode=importlib`, so `tests/` mirrors the package without `__init__.py` files.
- 2026-09-26 — `GuitarConfig` and MIDI → name live in `domain/`, not `guitar/`: `Performance` holds a
  config and `PerformanceEvent.pitch_name` needs names, and `domain` may not import `guitar`.
  `guitar/` keeps instrument *reasoning* (tuning parsing, candidate positions).
- 2026-09-26 — Event fields carry units in their names (`onset_seconds`, `pitch_midi`). `velocity` is
  a normalized float 0..1, not MIDI 1..127: backends report loudness on their own scales, and MIDI
  velocity is an export concern.
- 2026-09-26 — `pitch_midi` must be a plain `int` (numpy/float rejected), which forces adapters to
  convert third-party output at the boundary.
- 2026-09-26 — `technique` is omitted until Stage 8. Whether it's a single value, a set, or carries
  parameters (bend amount, slide target) is undecided, and a placeholder enum would be a guess.
- 2026-09-26 — `Performance` rejects a known string/fret that doesn't sound `pitch_midi`. Harmonics
  and bends (Stage 8) will need this rule relaxed via technique.
- 2026-09-26 — Raw performance timing and notated rhythm are separate representations. A new `rhythm/`
  module (split out of `notation/`) owns all timing inference, and `notation/` only renders its output.
  Quantized types live in `rhythm/`, link to their source `PerformanceEvent`s, and are designed in M5.
  `PerformanceEvent` needed no field changes: it already stores raw onset/offset seconds and is frozen.
  The old M5 was split into M5 (rhythm) and M6 (MusicXML), so pipeline/CLI is now M7.
