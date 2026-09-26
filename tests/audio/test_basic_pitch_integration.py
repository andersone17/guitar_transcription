"""Real Basic Pitch inference on a synthesized clip. Slow (TensorFlow); excluded by default.

Run with: ``uv sync --extra basic-pitch && uv run pytest -m integration``
"""

import random
import struct
import wave
from pathlib import Path

import pytest

# Safe without the extra: the adapter imports basic_pitch lazily, at construction.
from guitar_transcription.audio.backends.basic_pitch import BasicPitchTranscriber

pytestmark = pytest.mark.integration

SAMPLE_RATE = 22050
# (onset seconds, MIDI pitch): G3, C4, E4, G4, a few hundred ms apart.
NOTES = [(0.5, 55), (1.0, 60), (1.5, 64), (2.0, 67)]


def karplus_strong(pitch: int, seconds: float, seed: int) -> list[float]:
    """Plucked-string synthesis: a noise burst circulating through a damped delay line."""
    period = round(SAMPLE_RATE / (440.0 * 2 ** ((pitch - 69) / 12)))
    rng = random.Random(seed)
    buffer = [rng.uniform(-1.0, 1.0) for _ in range(period)]
    out = []
    for i in range(int(SAMPLE_RATE * seconds)):
        current = buffer[i % period]
        buffer[i % period] = 0.996 * 0.5 * (current + buffer[(i + 1) % period])
        out.append(current)
    return out


def write_plucked_notes(path: Path) -> None:
    samples = [0.0] * int(SAMPLE_RATE * 3.0)
    for onset, pitch in NOTES:
        start = int(onset * SAMPLE_RATE)
        for i, value in enumerate(karplus_strong(pitch, 0.45, seed=pitch)):
            samples[start + i] += 0.5 * value
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(
            b"".join(struct.pack("<h", int(max(-1.0, min(1.0, v)) * 32000)) for v in samples)
        )


@pytest.fixture(scope="module")
def transcriber() -> BasicPitchTranscriber:
    pytest.importorskip("basic_pitch", reason="requires the basic-pitch extra")
    return BasicPitchTranscriber(pitch_range=(40, 88))


def test_transcribes_synthesized_plucked_notes(
    transcriber: BasicPitchTranscriber, tmp_path: Path
) -> None:
    audio = tmp_path / "plucks.wav"
    write_plucked_notes(audio)

    events = transcriber.transcribe(audio)

    assert [e.pitch_midi for e in events] == [pitch for _, pitch in NOTES]
    for event, (onset, _) in zip(events, NOTES, strict=True):
        assert event.onset_seconds == pytest.approx(onset, abs=0.05)
        assert event.offset_seconds > event.onset_seconds
        assert event.velocity is not None and 0.0 < event.velocity <= 1.0
        assert event.string is None and event.fret is None
