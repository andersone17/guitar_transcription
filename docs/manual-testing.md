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
