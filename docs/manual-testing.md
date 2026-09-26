# Manual testing

## Stage 1 end to end with a real guitar recording

The automated integration tests use synthesized audio. This procedure checks the whole Stage 1
pipeline (audio → raw events → tempo → quantization → MusicXML) on a real guitar, with a reference
you know exactly, so you can tell *which stage* is at fault when something is off.

**Setup:** `uv sync --extra basic-pitch --extra tempo`, plus [MuseScore 4](https://musescore.org)
(free) to view the result.

**1. Record the reference take**, saved as `data/raw/g_major_ref.wav` (git-ignored):
- Standard tuning, clean tone, one guitar, no other sound. Use a metronome **in headphones only**.
- At ♩ = 80, play a G major scale in steady **eighth notes**: G2 A2 B2 C3 D3 E3 F#3 G3 A3 B3 C4 D4 E4 F#4
  G4, then back down to G2. Let the last note ring for a bar.
- Start the first note on a click; leading silence doesn't matter (the first note is beat 1).
- Optional pickup check: record a second take that starts with one D3 on beat 4 *before* the scale's
  first G2, and run step b) with `--downbeat <time of the G2 from the table>`. The D3 should appear
  as a pickup at the end of measure 1, and the G2 on beat 1 of measure 2.

**2. Stage by stage:**

```bash
# a) raw transcription only
uv run guitar-transcribe transcribe data/raw/g_major_ref.wav --json outputs/ref.events.json

# b) notation with the known tempo
uv run guitar-transcribe transcribe data/raw/g_major_ref.wav \
    --tempo 80 --time-signature 4/4 --grid eighth --musicxml outputs/ref_known.musicxml

# c) notation with the estimated tempo
uv run guitar-transcribe transcribe data/raw/g_major_ref.wav \
    --auto-tempo --time-signature 4/4 --grid eighth --musicxml outputs/ref_auto.musicxml
```

**3. Check each stage** (a failure in an earlier stage explains failures in later ones):

| Stage | Pass if | If it fails, the likely cause is |
|---|---|---|
| a) Transcription | 29 notes, pitches G2…G4…G2 in order; onsets about 0.375 s apart; no notes above D6 | Basic Pitch: missed low notes, octave errors, or extra notes from ringing strings. Try a cleaner or louder take. |
| b) Known-tempo notation | 4–5 measures of 4/4 at ♩ = 80, all eighth notes (the last one longer), no chords, no rests between scale notes | Rhythm/notation. If every barline is shifted, the first detected note wasn't beat 1 (a stray noise, or a pickup): pass `--downbeat`. |
| c) Auto tempo | "Estimated tempo" within ~2% of 80, 160 or 40. Using the suggested value that's near 80 gives the same result as b) | Tempo estimation. 160 is expected here (every eighth note gets a beat); rerun with the suggested `--tempo`. |
| Open in MuseScore | Title `g_major_ref`, one Guitar staff with a treble-8 clef, ♩ = 80, notes readable | MusicXML/viewer. Validate with the schema check below. |

**4. Record the outcome** in a note in `docs/` (date, guitar, results per stage). A take that fails is a
useful regression case: keep the WAV locally in `data/raw/`, never in git.

Known Stage 1 behaviour that is **not** a failure: string/fret are unknown (no TAB); note lengths may
look slightly short; `--auto-tempo` may report double or half time.

## Audio transcription on a real recording (Basic Pitch)

Checks that real model inference works on your machine and that the output looks plausible for a
recording you know. Nothing here is committed: keep recordings in `data/raw/` (git-ignored).

1. Install the backend extra (pulls in TensorFlow 2.15; roughly 2 GB in `.venv`):

   ```bash
   uv sync --extra basic-pitch
   ```

   Note: a plain `uv sync` afterwards *removes* the extra again. Keep passing `--extra basic-pitch`.

2. Run the automated integration test (synthesized plucked notes, no recording needed):

   ```bash
   uv run pytest -m integration
   ```

3. Transcribe your own clip (`.wav`, `.flac`, `.ogg`; `.mp3`/`.m4a` may need `ffmpeg`):

   ```bash
   uv run python - data/raw/my-clip.wav <<'EOF'
   import sys
   from guitar_transcription.audio.backends.basic_pitch import BasicPitchTranscriber

   # (40, 88) = E2..E6: standard-tuned guitar with 24 frets. Omit for the model's full range.
   events = BasicPitchTranscriber(pitch_range=(40, 88)).transcribe(sys.argv[1])
   print(f"{len(events)} notes")
   for e in events:
       print(f"{e.onset_seconds:8.3f} {e.offset_seconds:8.3f}  {e.pitch_name:4} "
             f"(MIDI {e.pitch_midi:3})  velocity {e.velocity:.2f}")
   EOF
   ```

   TensorFlow prints startup warnings to stderr (CUDA not found, oneDNN, `pkg_resources`
   deprecation); they are harmless on a CPU-only machine.

What to check, e.g. for a slow single-note melody you played:

- The pitches match what you played (octave errors are the most likely mistake).
- Onsets fall where the notes start (within a few tens of ms).
- Times are raw seconds, not snapped to any beat grid.
- `string`/`fret` are not shown because they are unknown (`None`) at this stage.

Expect weaker results on fast strums, dense chords, heavy distortion, or long let-ring: Basic Pitch
is instrument-agnostic, not guitar-specific (see PLAN.md, "Risks").

## MusicXML output in a notation program

MuseScore is not needed for the automated tests (they parse the MusicXML back with music21). To
check the rendering by eye:

```bash
uv run guitar-transcribe transcribe data/raw/my-clip.wav \
    --tempo 100 --time-signature 4/4 --musicxml outputs/my-clip.musicxml
```

Open `outputs/my-clip.musicxml` in MuseScore 4 (File → Open), then check:

- The title is the file name, and there's one "Guitar" staff with a treble clef that has a small 8 below it.
- The time signature and the "♩ = 100" tempo mark are at the start.
- Pitches read as played. A low open E hangs just below the third ledger line under the staff.
- Notes crossing a barline are tied, and the last measure is filled with rests.

Headless alternative (if MuseScore's CLI is installed): `mscore -o outputs/my-clip.pdf outputs/my-clip.musicxml`.

## Validating MusicXML against the official schema

Optional; needs network access and `lxml` in a throwaway environment (it is not a project dependency):

```bash
mkdir -p /tmp/musicxml-xsd && cd /tmp/musicxml-xsd
for f in musicxml.xsd xlink.xsd xml.xsd; do
  curl -sfLO "https://raw.githubusercontent.com/w3c/musicxml/v4.0/schema/$f"; done
sed -i 's#http://www.musicxml.org/xsd/##' musicxml.xsd   # use the local xml.xsd/xlink.xsd
cd - && uv run --with lxml python -c "
from lxml import etree
schema = etree.XMLSchema(etree.parse('/tmp/musicxml-xsd/musicxml.xsd'))
doc = etree.parse('outputs/my-clip.musicxml')
print('valid' if schema.validate(doc) else schema.error_log)"
```
