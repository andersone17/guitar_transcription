# Manual testing

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
