"""Real Basic Pitch inference on a synthesized clip. Slow (TensorFlow); excluded by default.

Run with: ``uv sync --extra basic-pitch && uv run pytest -m integration``
"""

from pathlib import Path
from typing import Any

import pytest

# Safe without the extra: the adapter imports basic_pitch lazily, at construction.
from guitar_transcription.audio.backends.basic_pitch import BasicPitchTranscriber

pytestmark = pytest.mark.integration

# (onset seconds, MIDI pitch): G3, C4, E4, G4, a few hundred ms apart.
NOTES = [(0.5, 55), (1.0, 60), (1.5, 64), (2.0, 67)]


@pytest.fixture(scope="module")
def transcriber() -> BasicPitchTranscriber:
    pytest.importorskip("basic_pitch", reason="requires the basic-pitch extra")
    return BasicPitchTranscriber(pitch_range=(40, 88))


def test_transcribes_synthesized_plucked_notes(
    transcriber: BasicPitchTranscriber, tmp_path: Path, write_plucked_notes: Any
) -> None:
    audio = write_plucked_notes(tmp_path / "plucks.wav", NOTES)

    events = transcriber.transcribe(audio)

    assert [e.pitch_midi for e in events] == [pitch for _, pitch in NOTES]
    for event, (onset, _) in zip(events, NOTES, strict=True):
        assert event.onset_seconds == pytest.approx(onset, abs=0.05)
        assert event.offset_seconds > event.onset_seconds
        assert event.velocity is not None and 0.0 < event.velocity <= 1.0
        assert event.string is None and event.fret is None


# Lowest and highest notes of a standard-tuned 22-fret guitar: open low E (E2) and D6.
RANGE_EDGE_NOTES = [(0.0, 40), (0.6, 86), (1.2, 86)]


@pytest.mark.parametrize(
    ("pitch_range", "expected"),
    [
        ((40, 86), [40, 86, 86]),  # both ends are inclusive
        ((41, 85), []),  # one semitone tighter on each side excludes both
    ],
)
def test_pitch_range_limits_are_inclusive_with_the_real_model(
    tmp_path: Path, write_plucked_notes: Any, pitch_range: tuple[int, int], expected: list[int]
) -> None:
    # Regression: Basic Pitch's maximum frequency is exclusive; D6 used to be dropped silently.
    pytest.importorskip("basic_pitch", reason="requires the basic-pitch extra")
    audio = write_plucked_notes(tmp_path / "edges.wav", RANGE_EDGE_NOTES)

    events = BasicPitchTranscriber(pitch_range=pitch_range).transcribe(audio)

    assert [e.pitch_midi for e in events] == expected
