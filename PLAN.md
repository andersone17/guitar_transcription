# PLAN.md

Architecture, stage plan, and acceptance criteria. The long-term system is multimodal (audio + video);
**the current goal is only Stage 1: audio-only transcription to notation.** Sections marked *future* are
direction, not commitments.

---

## 1. Architectural principles

1. **The performance is the model; notation is a view.** The core representation is a time-ordered sequence
   of `PerformanceEvent`s in physical time (seconds). MIDI, MusicXML, and tablature are derived from it and
   never fed back in as a source of truth.
2. **Evidence in, events out.** Perception modules (audio now, vision later) emit *modality-specific
   evidence* with confidences. Converting evidence into `PerformanceEvent`s is a separate step. In Stage 1
   that step is a trivial 1:1 mapping; later it becomes `fusion/`. Evidence and events are kept separate
   now so that adding vision later doesn't require redesigning the audio path.
3. **Unknown is explicit.** `string`, `fret`, `technique`, `picking_direction`, and vision confidences are
   `None` until something actually infers them. Nothing silently fills in a guess.
4. **Own the interfaces, rent the models.** Each third-party model sits behind a small `Protocol` we define.
   Only its adapter imports it, and it's installed as an optional extra.
5. **Physical configuration is data.** `GuitarConfig` is passed explicitly and determines which pitches
   and positions are possible.
6. **One-way dependencies.** `domain` ← `guitar` ← {`audio`, `notation`, later `vision`, `fusion`} ← `pipeline`/`cli`.
   Sibling modules never import each other.
7. **Boring code.** Dataclasses, functions, Protocols. No plugin registries, DI frameworks, or config
   systems until a real need appears.

## 2. Module boundaries

```
src/guitar_transcription/
    domain/          # pure data; no third-party imports
        events.py        PerformanceEvent, Technique, PickDirection
        performance.py   Performance (ordered events + metadata + GuitarConfig)
        evidence.py      AudioNoteEvidence (later: FrettingEvidence, PickingEvidence)
    guitar/          # instrument knowledge; pure Python
        config.py        GuitarConfig, standard/named tunings, parsing "E2,A2,..." strings
        pitch.py         MIDI <-> pitch name helpers
        positions.py     pitch -> candidate (string, fret) positions under a GuitarConfig
    audio/           # audio perception
        transcriber.py   AudioTranscriber Protocol, TranscriptionOptions
        backends/
            basic_pitch.py   BasicPitchTranscriber (sole importer of basic_pitch; lazy import)
    notation/        # rendering views of a Performance
        midi.py          performance (unquantized) MIDI export
        quantize.py      seconds -> beat-grid quantization (pure, backend-free)
        musicxml.py      quantized events -> MusicXML via music21
    pipeline.py      # wires audio -> events -> notation for a file
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
- `pipeline.py` is the only place that knows about more than one module. This keeps the coupling in one
  replaceable spot.
- There's no top-level `models/` or `weights/` directory. Basic Pitch ships its weights inside its wheel,
  and we don't train anything.

### Key domain types (sketch — finalized in M1)

```python
@dataclass(frozen=True, slots=True)
class GuitarConfig:
    open_strings: tuple[int, ...]   # MIDI pitches, index 0 = string 1 = highest-pitched (tab/MusicXML convention)
    capo: int = 0                   # fret number of capo; 0 = none
    max_fret: int = 22
    # num_strings is derived: len(open_strings)

@dataclass(frozen=True, slots=True)
class PerformanceEvent:
    onset: float                    # seconds from start of recording
    offset: float                   # seconds; > onset
    pitch: int                      # MIDI note number
    velocity: int | None = None     # 1..127
    string: int | None = None       # 1-based, 1 = highest-pitched string
    fret: int | None = None         # physical fret on the neck (not relative to capo); 0 = open
    technique: Technique | None = None
    pick_direction: PickDirection | None = None
    audio_confidence: float | None = None      # 0..1
    fretting_confidence: float | None = None   # 0..1 (future)
    picking_confidence: float | None = None    # 0..1 (future)
    confidence: float | None = None            # 0..1 overall
    # pitch_name is a derived property, not a stored field, so it can never disagree with `pitch`.
```

Conventions: `fret` is the **physical** fret because that's what a camera sees. With a capo at 2, an
"open" string is `fret == 2`. Notation code converts to capo-relative numbers when rendering tab.
All times in the domain are seconds, and only `notation/` knows about beats.

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
| Tests | `pytest` | dev extra | Requested standard |
| Lint/format | `ruff` | dev extra | One fast tool for lint and format |
| Type checking | `mypy` | dev extra | Enforces the type-hint policy |

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

**M0 — Project skeleton**
- `pyproject.toml` (hatchling, src layout, extras `dev` and `basic-pitch`, console script `guitar-transcribe`),
  `.python-version` = 3.11, `data/README.md`, empty `docs/` and `notebooks/` placeholders.
- ✅ `uv sync --extra dev` succeeds without installing TensorFlow or Basic Pitch.
- ✅ `python -c "import guitar_transcription"` works in that environment.
- ✅ `pytest` runs (placeholder test passes). `ruff check` and `ruff format --check` are clean.

**M1 — Domain model**
- `GuitarConfig`, `PerformanceEvent`, `Performance`, `Technique`, `PickDirection`, `AudioNoteEvidence`.
- ✅ Validation rejects `offset <= onset`, confidences outside [0, 1], velocity outside 1–127, and
  `string` without `fret` (or vice versa). `fret` must be ≥ capo and ≤ max_fret.
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

**M5 — Quantization and MusicXML export**
- `quantize(performance, tempo_bpm, time_signature, grid)` is pure: seconds → beat positions snapped to
  a grid (default 16th notes), with minimum duration of one grid step. Tempo is user-supplied in Stage 1.
- `write_musicxml(quantized, path, config)` uses music21 with guitar conventions: treble-8vb clef and
  simultaneous onsets grouped into chords. Overlapping notes are clipped to the next onset in the
  Stage 1 single-voice simplification.
- ✅ Quantizer unit tests: exact-grid input is unchanged, jittered input (±20 ms at 120 BPM) snaps
  correctly, and nothing ends up with zero duration.
- ✅ Output is well-formed MusicXML: it parses back with music21, and measure count and pitches match input.
- ✅ Manual check (documented in `docs/`): a sample output opens in MuseScore 4 and reads sensibly.

**M6 — Pipeline and CLI**
- `guitar-transcribe INPUT.wav --out outputs/ [--tuning E2,A2,D3,G3,B3,E4] [--capo N] [--strings N]
  [--tempo BPM] [--time-signature 4/4] [--onset-threshold …] [--frame-threshold …]`.
- Writes `<name>.mid`, `<name>.musicxml`, and `<name>.events.json` (the `Performance` serialized; useful
  for debugging and later evaluation).
- ✅ End-to-end integration test on the synthesized fixture produces all three files.
- ✅ README "Development" and "Usage" sections are updated with working commands.
- ✅ No generated files appear in `git status` after running the pipeline.

**Stage 1 done when** M0–M6 pass, and a real recorded guitar clip (not committed) produces MusicXML
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

Also later: rhythm/tempo/meter estimation (instead of user-supplied tempo), chord-symbol inference,
multi-voice notation.

## 5. Risks and open research questions

- **Basic Pitch maintenance.** There has been no release since 2024-08, it is pinned to old TensorFlow,
  and it doesn't support Python ≥3.12. *Mitigation:* confine it to its adapter and an optional extra.
  Evaluate the ONNX model path with `onnxruntime` (unverified) or alternative models early.
- **Instrument-agnostic model quality on guitar.** Expect octave errors, missed notes in dense strums,
  and spurious notes from string resonance. Stage 2 metrics will quantify this.
- **Timing precision.** An open upstream issue (#190) reports frame-level temporal drift in Basic Pitch.
  This must be verified before relying on audio onsets for audio/video alignment.
- **Quantization without known tempo.** Stage 1 sidesteps this with user-supplied tempo. Rubato and free
  playing will render poorly.
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
