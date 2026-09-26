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
        position.py      FretboardPosition (string, fret)
        events.py        PerformanceEvent, PickDirection (Technique added in Stage 8)
        performance.py   Performance (sorted events + GuitarConfig)
        serialization.py raw events <-> JSON document ("performance-events" v1)
        evidence.py      AudioNoteEvidence (M3; later FrettingEvidence, PickingEvidence)
    guitar/          # instrument reasoning; pure Python
        positions.py     pitch_for_position, candidate_positions, pitch_range
        tunings.py       parse_tuning ("E2,A2,..." low-to-high or a preset), describe_tuning
    audio/           # audio perception
        transcriber.py   AudioTranscriber Protocol, TranscriptionOptions
        backends/
            basic_pitch.py   BasicPitchTranscriber (sole importer of basic_pitch; lazy import)
    rhythm/          # performance time -> musical time (pure Python, no music21)
        values.py        NoteValue, RhythmicDuration, TimeSignature (measure length, locate)
        quantized.py     QuantizedEvent (source event + onset/duration in quarters), Rest, QuantizedPerformance
        chords.py        group_simultaneous: strum onsets -> one chord group (before snapping)
        quantize.py      quantize(events, quarter_note_bpm, time_signature, grid, chord_window_seconds)
        voices.py        to_single_voice / is_single_voice (Stage 1 notation simplification)
        tempo.py         TempoEstimator Protocol, TempoEstimate, tempo_from_beat_times, resolve_tempo
        backends/
            librosa_tempo.py LibrosaTempoEstimator (sole importer of librosa; lazy import)
    notation/        # rendering views; no rhythm inference
        midi.py          performance MIDI export from raw seconds (no quantization; not built yet)
        musicxml.py      write_musicxml(QuantizedPerformance) via music21 (sole music21 importer)
    pipeline.py      # Stage 1 wiring: detection_range, notate (tempo -> quantize -> single voice),
                     #   write_events_json, write_notation. The only module that knows every stage.
    cli.py           # argparse + reporting only: `guitar-transcribe transcribe ...` calls pipeline.py
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
2. **Automatic tempo / beat tracking** (possibly a varying beat grid). *Started 2026-09-26:* one global
   tempo from librosa beat tracking (`--auto-tempo`). Beat times are kept in `TempoEstimate` but not
   yet used to align the grid (no downbeat/phase), and there's no varying tempo.
3. **Automatic meter / downbeat estimation.** *Evaluated 2026-09-26: deferred; see "Meter inference:
   findings" below. The time signature stays user-specified.*
4. **Tempo and meter changes**, tuplet detection beyond a fixed grid, swing, and rubato.

**Triplets (deferred, agreed 2026-09-26).** Today the grid only has straight subdivisions, so triplets are
silently snapped to the nearest sixteenth and come out as the wrong rhythm (an eighth triplet becomes
sixteenth–eighth–sixteenth, and sextuplets collide into fake chords). The model already supports them
(exact `Fraction` thirds; `rhythmic_duration` returns `None` for ⅓), and music21 renders ⅓ as a triplet.
Planned steps:
(1) user-chosen triplet grids (⅓, ⅙) plus a CLI `--grid` option;
(2) per-beat automatic duple-vs-triplet choice with a penalty against spurious triplets, tuned on
Stage 2 data;
(3) explicit tuplet groups in the rhythm output if music21's automatic bracketing is not enough.
Swing (straight eighths written with a swing marking) is a separate decision on top of (2).
Workaround until then: all-triplet music can use 12/8 with an eighth grid and `quarter_note_bpm` = 1.5 × the dotted-quarter pulse.

The output types live in `rhythm/` (not `domain/`) because they are an interpretation of the
performance, not the performance itself. As of M5 they are deliberately small:
- `QuantizedEvent(source, onset_quarters, duration_quarters)`. `source` is the untouched `PerformanceEvent`;
  pitch, offset, and the single-symbol name of the duration (`rhythmic_duration`, e.g. dotted eighth,
  or `None` if it needs a tie) are derived.
- `QuantizedPerformance(events, quarter_note_bpm, time_signature, grid)`, with `rests()`, `end_quarters`,
  and `measure_count`.
- Measure number and position within the measure are *not stored*. `TimeSignature.locate(quarters)`
  derives them, so they can't disagree with the onset.
- There is no TempoMap/MeterMap. One tempo and one meter are fields; levels 2–4 will replace those
  fields, not the event type.

**How recording time maps to the grid (level 1):** a *downbeat* (raw seconds of a beat 1) anchors the
grid. It's user-given (`--downbeat`), or else the first detected note's onset.
`quarters = (seconds − downbeat) × quarter_note_bpm / 60`, snapped. If any note snaps before the
downbeat, everything shifts by whole measures, so the pickup sits at the end of measure 1 after leading
rests. `QuantizedPerformance.origin_seconds` records the raw time of measure 1's downbeat, and
`seconds_at(q)` maps back, which audio/video alignment will need. A true partial first measure
(anacrusis) is a later notation refinement.

### Meter inference: findings and decision (2026-09-26)

**Decision: keep the time signature user-specified for the MVP (option C).** Don't implement meter
inference now (A). Pursue downbeat tracking as the future route (B), but only once Stage 2 can measure
it. No `MeterEstimator` was added: no candidate is both simple to integrate and reliable enough to
be worth an interface today.

**Four different problems** (only the first two are solved in this project):

| Problem | Output | Status here |
|---|---|---|
| Tempo estimation | pulse rate (BPM) | `--auto-tempo` (librosa), with half/double ambiguity |
| Beat tracking | pulse *times* | librosa beats (kept in `TempoEstimate`, not yet used for phase) |
| Downbeat tracking | which beats start bars → bar length in beats + phase | none |
| Meter inference | time signature: beats per bar **and** how beats subdivide (e.g. 3×2 vs 2×3 eighths) | user-specified |

Downbeats give bar length and phase but not the time signature. **3/4 and 6/8** both have six
eighths per bar, grouped 3×2 versus 2×3. The same audio can also be written as 2/4 or 4/4 (bar length
vs. hypermeter), 4/4 or 2/2, swung 4/4 or 12/8, or 3/4 at double tempo instead of 6/8. Some of that is
notational intent, not audible fact, and trained musicians disagree on it. Any estimator must return
alternatives, not one answer.

**Tools surveyed (Sep 2026):**

| Tool | Beats | Downbeats | Meter | Status / fit |
|---|---|---|---|---|
| **Beat This!** (CPJKU, ISMIR 2024) | ✅ | ✅ | ✗ (derive from downbeats) | MIT code+checkpoints, active (May 2026). Needs PyTorch (~830 MB CPU-only) and numpy 2, which conflicts with TF 2.15/Basic Pitch (numpy < 2), so it would need a separate environment/process. Checkpoint (81 MB) is downloaded at runtime from a university server. |
| madmom | ✅ | ✅ | among given candidates (`beats_per_bar=[3, 4]`) | No release since 2018 (repo still has commits). **Models CC BY-NC-SA (non-commercial).** Build issues on modern Python. |
| BeatNet | ✅ | ✅ | ✅ (joint) | CC-BY-4.0. Pins numba 0.54 (Python < 3.10), so it can't be installed here. |
| all-in-one (allin1) | ✅ | ✅ | ✗ | Needs demucs, NATTEN, madmom. Last push 2024. |
| Essentia | ✅ | limited | ✗ | AGPL-3.0. |
| librosa | ✅ | ✗ | ✗ | Already used for tempo. |
| Symbolic, from our `PerformanceEvent`s (accent/onset periodicity at 2/3/4-beat lags, bass notes, chord changes) | n/a | possible | duple vs. triple | No dependencies. Literature reports useful but imperfect duple/triple discrimination on melodies. **Untested here.** |

**Published evidence (Beat This! paper, arXiv 2407.21658):**
- GuitarSet comping: beat F1 92.0, downbeat F1 88.1. This is 8-fold cross-validation, so it's *in-domain*
  (GuitarSet is in its training data). Its three progressions (12-bar blues, Autumn Leaves, Pachelbel)
  are conventionally 4/4. The dataset docs don't state meter; check the annotations in Stage 2.
  Either way, this number says nothing about telling 3/4 from 4/4 on guitar.
- Held-out GTZAN: beat F1 89.1, downbeat F1 78.3. Downbeats trail beats by about 11 points, and are
  worst on classical/solo-style material.

**Local probe** (`docs/research/meter_beat_this_experiment.py`): Beat This! on synthetic plucked-guitar
accompaniment, 16 bars per meter. One example each, so it's indicative, not a benchmark.

| Played | Beats | Downbeats | Meter derivable? |
|---|---|---|---|
| 4/4 bass/chord pattern, ♩=100 | ✅ 100.0 BPM | 14/21 on true bar starts; right for 10 bars, then a spurious 1+3 split | 4 by majority vote |
| 3/4 waltz, ♩=120 | ✅ 120.0 BPM | ✗ 38 "downbeats" for 16 bars; nearly every beat is marked | **no** |
| 6/8 arpeggio, dotted-♩=60 | pulse at 90.9 BPM (every 2 eighths) | ✅ 16/17 on true bar starts | bar right, grouping wrong: **reads as 3/4, not 6/8** |

**Assessment for solo guitar:** there are no drums, so bar cues come from bass notes, harmony changes
and accents, which vary by player and style. Pickups and rubato are common. Expect downbeat accuracy
well below beat accuracy, and 3/4-vs-6/8 or 2/4-vs-4/4 decisions to be unreliable from audio alone.
A wrong meter is worse than asking: every barline, tie and beam in the MusicXML depends on it.

**What would change this decision / next steps:**
1. The bigger practical gap is *phase*, not the meter label. Measure 1 starting at 0 s breaks any
   recording with a lead-in or pickup. A user-specified first-downbeat offset (or pickup length) is
   cheap and deterministic, and worth doing before any automatic downbeat work. *Done 2026-09-26:*
   `--downbeat`, and the default is now the first note.
2. In Stage 2, evaluate Beat This! (isolated environment/subprocess) and the dependency-free symbolic
   baseline on annotated guitar data that includes real 3/4 and 6/8 material, not just GuitarSet's 4/4.
3. Only if that shows useful accuracy, add a `MeterEstimator` returning ranked candidates
   (beats per bar, grouping, first-downbeat time) that are shown as suggestions. It must never
   silently override `--time-signature`, which stays supported.
4. Revisit the integration cost when the Basic Pitch/TF stack is replaced (numpy 2 becomes possible),
   which would let Beat This! run in-process.

## 3. Stage 1 plan: audio file → notation

Goal: given an audio file of solo guitar, produce (a) a performance MIDI file and (b) a MusicXML file
that opens in MuseScore as readable standard notation. The events must carry candidate-position
information hooks for later tablature work, without choosing fingerings.

### Stage 1 dependencies

| Purpose | Package | Where | Why |
|---|---|---|---|
| Transcription backend | `basic-pitch==0.4.0` (+ `setuptools<82`) | optional extra `basic-pitch` (**installed M3**) | Pretrained polyphonic AMT, Apache-2.0, returns note events directly |
| MusicXML writing | `music21>=10.5,<11` | core (**installed M6**) | Mature, BSD-3; handles durations, ties, measures, clefs, chords, and MusicXML export |
| MIDI writing | `mido` | core | Tiny, pure-Python (MIT); exact-time performance MIDI without quantization |
| Tests | `pytest` | `dev` dependency group | Requested standard |
| Lint/format | `ruff` | `dev` dependency group | One fast tool for lint and format |
| Tempo estimation | `librosa>=0.10,<1.0` | optional extra `tempo` (**installed**; also comes with `basic-pitch`) | Beat tracking for `--auto-tempo`; ISC; no new weight |
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

Basic Pitch facts relevant to the adapter (re-verified 2026-09-26 against the installed 0.4.0 wheel,
upstream `main` (last commit 2025-11, no API changes since 0.4.0), and a real inference run):
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
- Note-event values are **numpy scalars** (`float64` times, `int64` pitch, `float32` amplitude), and
  pitch bends are a list even with `multiple_pitch_bends=False`. `amplitude` is the mean frame
  activation over the note, and Basic Pitch's own MIDI export uses it as velocity (`round(127 * amplitude)`).
- `minimum_frequency`/`maximum_frequency` are rounded to note bins and applied as `[:min]` / `[max:]`,
  so the minimum is **inclusive** and the maximum is **exclusive**. The adapter passes `high + 1` so
  that `pitch_range=(low, high)` means inclusive on both ends.
- Input is decoded with `librosa.load` (documented: .mp3 .ogg .wav .flac .m4a), downmixed to mono, and
  resampled to 22050 Hz. Undecodable files raise `audioread.exceptions.NoBackendError`.
- `predict()` prints a progress line to stdout, and importing the package logs warnings about
  missing optional runtimes.
- **Install breakage:** basic-pitch pins `resampy<0.4.3`, and resampy 0.4.2 imports `pkg_resources`,
  which setuptools 82 removed. A fresh install then fails at `import basic_pitch.inference`, so the
  extra also pins `setuptools<82`.

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

**M2 — Guitar knowledge** — *done 2026-09-26*
- `FretboardPosition` (domain), and `pitch_for_position`, `candidate_positions`, `pitch_range` (guitar),
  handling arbitrary tunings, string counts, capo, and max fret. There's no fingering choice.
  Tuning-string parsing (`"E2,A2,D3,G3,B3,E4"`) moves to M7, where the CLI is its first consumer.
- ✅ Standard tuning, capo 0, max_fret 24: `candidate_positions(64)` returns exactly
  {(1,0),(2,5),(3,9),(4,14),(5,19),(6,24)}.
- ✅ Invariant, checked exhaustively over several tunings: every candidate sounds the requested pitch,
  and every playable position appears among its pitch's candidates.
- ✅ With capo 2, no candidate has `fret < 2`, and open-string pitches shift by +2.
- ✅ `pitch_range(config)` = (lowest open string + capo, highest open string + max_fret).
- ✅ Pitches outside the range return no candidates; they do not raise.

**M3 — Audio transcription interface and Basic Pitch adapter** — *done 2026-09-26*
- `audio/transcriber.py`: `AudioTranscriber` Protocol, `transcribe(path) -> list[PerformanceEvent]`
  (raw timing, sorted), plus errors `TranscriptionError` ⊃ {`UnsupportedAudioError`,
  `BackendUnavailableError`}. A missing file raises `FileNotFoundError`.
- `audio/backends/basic_pitch.py`: `BasicPitchTranscriber(onset_threshold, frame_threshold,
  minimum_note_length_ms, pitch_range)` imports `basic_pitch` and loads the model once, at construction.
  `pitch_range` (MIDI, e.g. `guitar.pitch_range(config)`) becomes `minimum_frequency`/`maximum_frequency`.
  amplitude → `velocity` only; `string`/`fret`/confidences stay `None`.
- ✅ Unit tests use a fake `basic_pitch` package injected into `sys.modules` (no TensorFlow/numpy).
  The fake returns numpy-like scalar types.
- ✅ Importing `guitar_transcription.audio` works without Basic Pitch installed. Using the backend
  without it raises a clear error that names the install extra.
- ✅ Integration test (`@pytest.mark.integration`, auto-skipped without the extra): a synthesized
  WAV (stdlib `wave` + plucked-string synthesis) of 3–4 known notes a few hundred ms apart is transcribed
  with the correct pitches, and onsets are within 50 ms.
- ✅ A test confirms `basic_pitch` is imported only in `audio/backends/basic_pitch.py` (static and
  `importlib` imports), and that importing the adapter module loads no heavy packages.
- ✅ Integration tests are opt-in (`pytest -m integration`), so the default run stays fast.
  The manual procedure for a real recording is in `docs/manual-testing.md`.

**M4 — Performance MIDI export**
- `write_midi(performance, path)` writes unquantized events (tempo only for tick conversion).
- ✅ Round-trip test: reading the file back with `mido` recovers pitches exactly and times within 1 tick.
- ✅ Overlapping notes of the same pitch are handled without stuck notes.

**M5 — Rhythm quantization with known tempo and meter** (`rhythm/`, rhythm level 1) — *done 2026-09-26*
- `quantize(events, quarter_note_bpm, time_signature, grid)` is pure: raw seconds → exact beat positions
  snapped to a grid (default 16th notes), with a minimum duration of one grid step. Tempo and meter are
  user-supplied in Stage 1. The quantized output types are defined here, and each quantized event
  references its source `PerformanceEvent`.
- ✅ Exact-grid input is unchanged, jittered input (±20 ms at 120 BPM) snaps correctly, and nothing
  ends up with zero duration.
- ✅ Source events are unchanged: after quantization, raw `onset_seconds`/`offset_seconds` are
  identical and reachable from every quantized event.
- ✅ Measure/beat positions are correct for 4/4 and 3/4, and the positions are exact (`Fraction`, no float drift).
- ✅ `rhythm/` does not import music21 or `notation/` (checked by a test).

**M6 — MusicXML export** (`notation/`) — *done 2026-09-26 (manual MuseScore check pending: not installed)*
- `write_musicxml(quantized, path, *, title)` uses music21 with guitar conventions: a "Guitar" part,
  treble-8vb clef with sounding pitches, quarter-note metronome mark, and simultaneous onsets as chords.
  It makes no timing decisions. Overlap clipping is `rhythm.to_single_voice`, and the exporter *rejects*
  input that isn't single-voice. Rests come from `QuantizedPerformance.rests()`. music21's
  `makeNotation` does the spelling: measures, barline ties, dots, beams, final-measure fill.
  `GuitarConfig` isn't a parameter yet because only TAB needs it.
- CLI: `transcribe AUDIO --musicxml PATH --tempo BPM --time-signature N/D [--grid VALUE]`. Tempo and meter
  are required with `--musicxml` (never defaulted), and notation flags without it are usage errors.
- ✅ Output validates against the official MusicXML 4.0 XSD (checked manually with lxml for 4/4 with
  ties/chords/dots, 3/4, 6/8 at a fractional tempo, and an empty performance). The procedure is in
  `docs/manual-testing.md`; it's not in CI because the schema is 380 KB and lxml isn't a dependency.
- ✅ Output is well-formed MusicXML: it parses back with music21, and measure count and pitches match input.
- ✅ Notes crossing a barline come out as tied notes whose total duration equals the quantized duration.
- ✅ Manual check (documented in `docs/`): a sample output opens in MuseScore 4 and reads sensibly.

**M7 — Pipeline and CLI** — *integration path done 2026-09-26; tuning options and performance MIDI open*
- `pipeline.py` holds the Stage 1 wiring, and the CLI only parses arguments and reports. One command
  covers the whole path:
  `guitar-transcribe transcribe AUDIO [--json PATH] [--tuning T] [--capo N] [--max-fret N] [--full-range]
  [--musicxml PATH (--tempo BPM | --auto-tempo) --time-signature N/D [--grid VALUE] [--downbeat S]]`. Exit codes: 0 ok, 2 bad input/usage, 1 processing failure.
  (This replaces the earlier idea of a separate pipeline subcommand with `--out DIR`: the per-output
  flags cover the same ground.)
- Detection is limited to the instrument's range, `guitar.pitch_range(config)` for the `GuitarConfig`
  built from `--tuning/--capo/--max-fret` (default: standard, no capo, 22 frets = E2–D6), and passed
  to the transcriber. `--full-range` disables this. It was found necessary by the end-to-end test, where
  Basic Pitch reported G6/E6 ghost notes from string harmonics.
- *Done 2026-09-26:* `guitar/tunings.py` + `--tuning/--capo/--max-fret`. Transcribed events become a
  `Performance(events, config)` (validates any string/fret), which `notate` and `write_events_json`
  take, and the JSON records the guitar.
- *Still open:* `--onset-threshold/--frame-threshold`, and performance MIDI (M4).
- ✅ Integration strategy in three tiers: fast unit tests; fast pipeline contract tests
  (`tests/test_pipeline.py`: real rhythm and notation chained through `pipeline.py`, model faked through
  our protocols, checking what crosses each boundary, including source-event identity); and opt-in
  `-m integration` tests (`tests/test_stage1_end_to_end.py`: real Basic Pitch + librosa through the
  CLI's `main`, synthesized melody → MusicXML with every note on its beat).
- ✅ README "Stage 1 Quick Start" and "Stage 1 Limitations"; manual real-guitar procedure with
  per-stage pass criteria in `docs/manual-testing.md`.
- ✅ No generated files appear in `git status` after running the pipeline (outputs/ and data/ are ignored).
- ⏳ Not done: running that manual procedure on a real reference recording (needs the user's guitar).

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
- **Meter and downbeat ambiguity.** 3/4 vs 6/8, 2/4 vs 4/4, 4/4 vs 12/8 swing, and where bar 1 starts
  are partly notational choices and hard to hear on solo guitar. Downbeat trackers trail beat trackers
  by ~10 F1 points even on ensemble music. Kept user-specified; see §2a.
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
- 2026-09-26 — `FretboardPosition` lives in `domain/`: it's a config-independent value that events,
  future vision evidence, and fusion all share. Playability and candidate enumeration (config-dependent
  reasoning) live in `guitar/`. `guitar.pitch_for_position` is the public API but delegates to
  `GuitarConfig.pitch_at`, so the position → pitch arithmetic exists once (`Performance` validation in
  `domain` needs it too).
- 2026-09-26 — String numbering is positional (tab order): string N = `open_strings[N-1]`, and string 1 is
  the highest-pitched string on conventional tunings. Pitch order isn't enforced, because re-entrant
  tunings (Nashville) are real. The cost is that a low-to-high tuple is silently accepted as a valid but
  different instrument, so tuning parsing (M7) must do the reversal.
- 2026-09-26 — `candidate_positions` returns an empty tuple for pitches the instrument can't play (they are
  legitimate backend output) but raises for invalid MIDI numbers. Candidates are ordered by string number.
- 2026-09-26 — `AudioTranscriber.transcribe(path)` returns raw `PerformanceEvent`s directly. There is no
  `AudioNoteEvidence` yet: with one modality it would be a field-for-field copy. Principle 3 still
  holds, since events from audio carry only audio-derived fields. A modality-specific evidence type is
  introduced with `fusion/`, when a second modality exists to combine.
- 2026-09-26 — Basic Pitch `amplitude` maps to `velocity`, not `audio_confidence`. It is an uncalibrated
  mean activation that upstream itself uses as MIDI velocity, and putting the same number in two fields
  would double-count it in fusion. `audio_confidence` stays `None` until a backend exposes a real one.
- 2026-09-26 — Basic Pitch model tuning (thresholds, min note length) are constructor arguments of
  `BasicPitchTranscriber`, not part of the `AudioTranscriber` protocol. Other backends will have
  different knobs.
- 2026-09-26 — Decoder failures are recognized by the exception's module (`audioread`/`soundfile`/`librosa`)
  and reported as `UnsupportedAudioError`, which avoids importing those libraries. Other failures raise
  `TranscriptionError`, and backend output that violates domain invariants raises instead of being repaired.
- 2026-09-26 — pytest excludes `integration` tests by default (`-m "not integration"` in addopts).
- 2026-09-26 — The CLI uses subcommands (`guitar-transcribe transcribe …`) so raw transcription, and later
  the full pipeline, evaluation, etc., live under one entry point. It's plain argparse, with no CLI dependency.
- 2026-09-26 — Event JSON serialization lives in `domain/serialization.py` (stdlib `json`-ready dicts):
  it's the hand-off format between stages. It's self-describing (`format`, `version`,
  `timing: raw-performance-seconds`), writes unknowns as `null`, omits derived values, and the loader
  re-validates every event and rejects unknown fields/versions.
- 2026-09-26 — The Basic Pitch adapter silences known-irrelevant backend chatter (TF C++ logs via
  `TF_CPP_MIN_LOG_LEVEL` default 3, optional-runtime warnings, `pkg_resources` and decoder-fallback
  warnings), so CLI stderr shows only our progress and errors. Real failures still raise.
- 2026-09-26 — Rhythm tempo is **quarter notes per minute** (`quarter_note_bpm`) for every meter, as in
  MIDI. "Beat" is ambiguous in 6/8 (eighth or dotted quarter) and 2/2 (half), so the parameter name says
  the unit. Callers (CLI) convert from other conventions.
- 2026-09-26 — Quantization snaps onset and offset independently to the nearest grid line (half rounds
  later) with a minimum of one grid step. Durations are grid multiples, not forced into single note
  symbols. Rests are the gaps where no note sounds, so a legato gap under half a step disappears.
  Overlapping notes are kept, because voicing them is a notation decision. The grid must divide the
  measure evenly (validated).
- 2026-09-26 — `rhythm` accepts any iterable of `PerformanceEvent`s, not a `Performance`, because
  quantization needs no `GuitarConfig`. Notation receives the config separately.
- 2026-09-26 — MusicXML via **music21 10.5** (BSD-3, maintained, Python ≥3.11). Its `makeNotation` does
  the spelling we'd otherwise hand-write (barline ties, dotted/tied values, rests, beams). It also has
  the TAB vocabulary for later: `StringIndication`, `FretIndication`, `HammerOn`, `PullOff`, `FretBend`.
  Hand-written XML would re-implement spelling, and partitura/abjad fit analysis/LilyPond better.
  music21 is imported only by `notation/musicxml.py`, and the CLI imports that lazily, since music21
  takes ~1 s to import.
- 2026-09-26 — Single-voice reduction (clip let-ring at the next onset, chord = shared duration capped at
  the next onset, duplicate pitch kept once) is a timing decision, so it lives in `rhythm/voices.py`,
  not the exporter. This supersedes the earlier M6 wording ("clipped in notation").
- 2026-09-26 — Guitar pitches are written at **sounding** pitch under a treble-8vb clef
  (`clef-octave-change -1`), with no `<transpose>`. That's the MusicXML convention for octave clefs.
  music21's default "Music21" composer credit is replaced with an honest
  `creator type="transcriber"` = guitar-transcription.
- 2026-09-26 — TAB will attach without changing this design. `QuantizedEvent.source` already carries
  `string`/`fret` (and later technique), which map to MusicXML `<technical>` via music21 in
  `_sounding_element`. A TAB staff is a second staff built from the same events plus `GuitarConfig`
  (tuning, capo-relative frets).
- 2026-09-26 — Observed on a real scale recording: Basic Pitch notes end ~25% early, so a sixteenth grid
  shows eighths as sixteenth + rest, and an approximate tempo on an eighth grid merges neighbours into
  chords. Both are known risks (unreliable offsets, user-supplied tempo). A later rhythm improvement
  should derive notated durations from inter-onset intervals, not raw offsets.
- 2026-09-26 — Tempo estimation backend: **librosa 0.11** beat tracker. Compared (Sep 2026):
  - madmom: last release 2018, no wheels, and its models are CC BY-NC-SA (non-commercial).
  - Essentia: AGPL-3.0, and the newest wheels are cp314 only.
  - beat_this: MIT and state of the art, but needs PyTorch (GBs).
  - BeatNet, tempocnn, aubio: stale, or GPL/AGPL, or pin TF 2.17.
  - librosa is ISC and already installed with Basic Pitch.
  - librosa ≥1.0 (Aug 2026) needs Python 3.12 and numpy 2, which the TF 2.15 stack can't use, hence `<1.0`.
  - Revisit beat_this if accuracy on real guitar recordings is poor.
- 2026-09-26 — The estimated tempo comes from a least-squares fit of the tracker's beat *times*
  (numbered locally so missed beats don't shift the count), not from librosa's own tempo value. That
  value is quantized to tempogram bins (117.45 for a true 120), and a 2% error drifts a beat every
  ~50 beats.
- 2026-09-26 — `TempoEstimator` lives in `rhythm/` (tempo is rhythm's decision), and its librosa backend
  in `rhythm/backends/`. That makes this the one place `rhythm` reads audio, through a lazily imported
  backend, so `rhythm` stays independent of `audio`. `TempoEstimate` has no confidence field, because
  librosa provides no meaningful one.
- 2026-09-26 — Explicit tempo wins: `resolve_tempo` never calls the estimator when a BPM is given, and
  the CLI makes `--tempo`/`--auto-tempo` mutually exclusive. The detected pulse is treated as a quarter
  note, with a warning for non-/4 meters. Estimates are rounded to 0.01 BPM.
- 2026-09-26 — Metrical-level ambiguity (half/double time) is surfaced, not hidden. On a real
  eighth-note scale, librosa reported ~201 BPM, and the CLI's suggested `--tempo 100.62` gave clean
  notation (better than a hand guess of 100, which merged two notes into a chord).
- 2026-09-26 — **Meter stays user-specified for the MVP** (`--time-signature` required). The survey and
  a local Beat This! probe showed reliable beats but unreliable downbeats on synthetic guitar: the 3/4
  waltz failed, and 6/8 was read as 3/4. The best tool (Beat This!) also needs PyTorch and numpy 2,
  which is incompatible with the TF 2.15 stack. No `MeterEstimator` was added. Next: a user-specified
  first-downbeat/pickup option, then a Stage 2 evaluation of downbeat tracking and a symbolic baseline
  on data that includes 3/4 and 6/8. Details in §2a "Meter inference: findings".
- 2026-09-26 — Stage 1 wiring moved from `cli.py` into `pipeline.py` (as the module layout always
  intended). The CLI keeps argument parsing, input validation before model load, and reporting.
  Behaviour is unchanged: all existing CLI tests pass untouched, except that transcriber factories now
  receive the detection range.
- 2026-09-26 — By default the transcriber gets the standard 22-fret guitar's range (E2–D6). This is
  physically grounded (`GuitarConfig`), not a guess, and it removes harmonic ghost notes. It trades
  away notes below E2 (drop/7-string) and above D6 unless `--full-range` is given. Proper
  `--tuning/--capo` options are the real fix.
- 2026-09-26 — Integration tests use `tests/conftest.py`'s `write_plucked_notes` fixture (synthesized in
  code), so no audio is committed. Measured: the full pipeline takes ~6 s for a 30 s clip on an 8-core
  CPU (warm).
- 2026-09-26 — Fixed (review CRITICAL 1): the Basic Pitch adapter passed the top note's own frequency
  as `maximum_frequency`, which Basic Pitch treats as exclusive, so the default E2–D6 range silently
  dropped D6 (fret 22, high E). It now passes the next semitone. Guarded by unit tests that replicate
  Basic Pitch's bin arithmetic, and by a real-model integration test at both range edges; both fail on
  the old code.
- 2026-09-26 — Fixed (review CRITICAL 2): the quantizer snapped each note alone, so a 100 ms strum was
  split into a 4-note and a 2-note chord. Now `rhythm.chords.group_simultaneous` groups onsets
  first, and each group snaps as one, anchored at its earliest raw onset (each note keeps its own
  offset; raw per-string onsets stay in `source` for future strum-direction evidence). A note joins
  a group only if all of these hold:
  - it starts ≤ 50 ms after the previous note (strum string gap vs. melodic spacing);
  - it starts ≤ 100 ms after the group's first note, capped at 0.8 of a grid step;
  - the group is still sounding;
  - its pitch is new to the group.

  The consecutive-gap rule is what keeps let-ring sixteenth arpeggios (85–125 ms apart) from
  becoming fake chords.
- 2026-09-26 — Limitation found while fixing it: on real Basic Pitch output of synthesized strums,
  chord-note onsets jitter by up to ~±60 ms (one G-chord spanned 116 ms, with a 58 ms gap), and
  ringing strings produce spurious re-attacks. On a sixteenth grid at 120 BPM that can still split a
  chord, and no constant fixes it without merging sixteenth arpeggios. Advice: use `--grid eighth`
  for strummed material. Tune `DEFAULT_CHORD_WINDOW_SECONDS`/`MAX_STRING_GAP_SECONDS` in Stage 2 on
  GuitarSet comping (annotated strums), not on synthetic audio.
- 2026-09-26 — Fixed (review IMPORTANT 7): bar 1 no longer starts at 0 s.
  - **Default change:** beat 1 is now the first detected note, so silence before playing (the normal
    webcam/mic case) doesn't shift barlines. `--downbeat SECONDS` (`quantize(downbeat_seconds=)`)
    sets a real downbeat, and earlier notes become a pickup bar.
  - The pickup decision uses *snapped* positions, so a chord struck 20 ms early doesn't create an
    empty measure.
  - A pickup is written as a full measure with leading rests (valid, readable), not as an anacrusis.
  - `origin_seconds` / `seconds_at()` give an exact musical → recording time map.
  - The downbeat is never inferred (see the meter findings). A stray noise before the music will
    become beat 1, and `--downbeat` is the fix.
- 2026-09-26 — Done (review IMPORTANT 3): `GuitarConfig` now flows through the pipeline. The CLI builds
  it from `--tuning` (written low-to-high like guitarists write it, reversed *by position* so
  re-entrant tunings keep their string numbers; or a preset), `--capo` and `--max-fret`, and
  validates it before any slow work (exit 2). The detection range follows it, which fixes drop/7-string
  low notes being cut. Events are wrapped in `Performance(events, config)` right after transcription,
  so string/fret added by future fusion are checked against the real instrument. `performance-events`
  JSON gained an optional `guitar` object (still v1: loaders ignore it, and `guitar_from_dict` reads
  it). Notation doesn't take the config yet; TAB will, for capo-relative frets.
- 2026-09-26 — Done (review IMPORTANT 9): the CLI no longer validates file formats with Basic Pitch's
  list. `check_audio_path(path)` without suffixes does the backend-neutral early check (exists, is a
  file, non-empty), and each backend enforces its own formats in `transcribe()`. The cost: an
  unsupported extension is reported after the model loads (~3.7 s, measured) instead of instantly,
  with the same message and exit code 2.
- 2026-09-26 — Done (review IMPORTANT 8):
  - `mypy` (2.3) is a dev dependency, configured in `pyproject.toml` (untyped defs disallowed,
    `warn_return_any`, `warn_unused_ignores`) over `src/` and `tests/`. Missing-stub exceptions
    cover only the optional backends (basic_pitch, librosa, lxml).
  - It found `2.0 ** x` typed as `Any` in `_midi_to_hz` and music21's untyped `makeNotation()`
    result; both are now explicit.
  - New tests: a rest crossing a barline in MusicXML; backend-edge semantics (range test, strums).
  - Schema validation is an opt-in `integration` test (`MUSICXML_XSD` + `uv run --with lxml`),
    because vendoring the 380 KB XSD would break the fixture-size rule and lxml isn't needed at runtime.
  - AGENTS.md now requires mypy and ruff alongside pytest.
- 2026-09-26 — Review MINOR items:
  - **Fixed:**
    - `--json`/`--musicxml` can't overwrite the input recording (compared after resolving the
      path), can't be the same file, and can't be a directory (exit 2, before any work).
    - Unexpected MusicXML-writer errors (music21's own exception types) become a clean exit 1, and
      earlier outputs are kept.
    - `TF_CPP_MIN_LOG_LEVEL` is set only while importing Basic Pitch, then restored (a user's value
      is left alone). TensorFlow reads it at load time, so the import stays quiet without leaking
      into the process or subprocesses.
    - `GuitarConfig` rejects non-integer capo/max_fret.
    - `tempo_from_beat_times` skips spurious beats (at most half a median interval after the previous
      kept beat) instead of counting them as whole beats, which had biased the tempo (120 → 108.8 in
      the regression test).
    - The librosa tempo backend rejects empty files like the audio adapter does.
    - `velocity` is documented as a backend-reported note strength (Basic Pitch: mean activation),
      not calibrated loudness or a probability.
    - `build_score` is now private.
  - **Kept, deliberately:**
    - `RhythmicDuration`/`rhythmic_duration`: part of rhythm's public model (rhythm owns note
      values, §2a) and exposed on `QuantizedEvent`.
    - music21's matplotlib dependency in the core install: MusicXML is the main output, and moving
      music21 to an extra would make the default install unable to produce notation. Revisit if
      install size becomes a problem.
