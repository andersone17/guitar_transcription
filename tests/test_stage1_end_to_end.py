"""Stage 1 end to end with the real models, through the real CLI. Slow; opt in with
``uv run pytest -m integration`` (needs ``uv sync --extra basic-pitch --extra tempo``).

Nothing is faked: ``guitar-transcribe``'s ``main`` builds its default backends (Basic Pitch,
librosa), so this covers audio -> PerformanceEvent[] -> tempo -> quantization -> MusicXML across
every module boundary. Input is a synthesized plucked-string melody (see ``conftest.py``); a real
guitar recording is covered by the manual procedure in docs/manual-testing.md.
"""

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from guitar_transcription.cli import EXIT_OK, main
from guitar_transcription.domain.serialization import events_from_dict

pytestmark = pytest.mark.integration

G_MAJOR = [55, 57, 59, 60, 62, 64, 66, 67]  # G3 .. G4
BPM = 120
QUARTER = 60 / BPM


def melody(pitches: list[int]) -> list[tuple[float, int]]:
    """One quarter note per pitch at 120 BPM, starting on the downbeat at 0 s."""
    return [(i * QUARTER, pitch) for i, pitch in enumerate(pitches)]


def exported(path: Path) -> list[tuple[str, Fraction]]:
    from music21 import converter

    score = converter.parse(path).stripTies()
    return [
        (
            "+".join(p.nameWithOctave for p in n.pitches),  # a chord would show as "A3+G6"
            Fraction(n.getOffsetInHierarchy(score)).limit_denominator(16),
        )
        for n in score.recurse().notes
    ]


@pytest.fixture(autouse=True)
def _require_backends() -> None:
    pytest.importorskip("basic_pitch", reason="requires the basic-pitch extra")
    pytest.importorskip("librosa", reason="requires the tempo (or basic-pitch) extra")


def test_explicit_tempo_audio_to_musicxml(
    tmp_path: Path, write_plucked_notes: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    audio = write_plucked_notes(tmp_path / "g_major.wav", melody(G_MAJOR))
    xml, events_json = tmp_path / "out" / "g_major.musicxml", tmp_path / "out" / "g_major.json"

    code = main(
        ["transcribe", str(audio), "--json", str(events_json)]
        + ["--tempo", str(BPM), "--time-signature", "4/4", "--musicxml", str(xml)]
    )

    assert code == EXIT_OK
    # Raw boundary: the transcriber's events, in raw seconds, near the synthesized onsets.
    raw = events_from_dict(json.loads(events_json.read_text()))
    assert [e.pitch_midi for e in raw] == G_MAJOR
    for event, (onset, _) in zip(raw, melody(G_MAJOR), strict=True):
        assert event.onset_seconds == pytest.approx(onset, abs=0.05)
        assert event.string is None and event.fret is None
    # Notation boundary: every note lands exactly on its beat, two measures of 4/4.
    assert exported(xml) == [
        (name, Fraction(i))
        for i, name in enumerate(["G3", "A3", "B3", "C4", "D4", "E4", "F#4", "G4"])
    ]
    out = capsys.readouterr().out
    assert out.startswith("RAW PERFORMANCE TIMING")


def test_auto_tempo_audio_to_musicxml(
    tmp_path: Path, write_plucked_notes: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    pitches = G_MAJOR + G_MAJOR[-2::-1]  # up and back down: 15 notes
    audio = write_plucked_notes(tmp_path / "g_major_updown.wav", melody(pitches))
    xml = tmp_path / "g_major_updown.musicxml"

    code = main(
        [
            "transcribe",
            str(audio),
            "--auto-tempo",
            "--time-signature",
            "4/4",
            "--musicxml",
            str(xml),
        ]
    )

    assert code == EXIT_OK
    err = capsys.readouterr().err
    assert "Estimated tempo:" in err
    notes = exported(xml)
    assert [name for name, _ in notes] == [
        "G3", "A3", "B3", "C4", "D4", "E4", "F#4", "G4", "F#4", "E4", "D4", "C4", "B3", "A3", "G3",
    ]  # fmt: skip
    # Metrical-level ambiguity is expected: the pulse may be read at 60, 120, or 240 BPM, which
    # scales note spacing to 1/2, 1, or 2 quarters. Whatever was chosen, spacing must be uniform.
    spacing = {b - a for (_, a), (_, b) in zip(notes, notes[1:], strict=False)}
    assert len(spacing) == 1
    assert spacing.pop() in {Fraction(1, 2), Fraction(1), Fraction(2)}
