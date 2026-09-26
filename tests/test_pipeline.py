"""Stage 1 pipeline contract tests (fast).

Only the *model* is faked, through the project's own protocols (``AudioTranscriber``,
``TempoEstimator``). Rhythm, single-voice reduction, and MusicXML export are the real modules, so
these tests check what crosses each boundary rather than bypassing it.
"""

import copy
import json
from fractions import Fraction as F
from os import PathLike
from pathlib import Path

import pytest
from music21 import converter, stream, tempo

from guitar_transcription.audio import AudioTranscriber
from guitar_transcription.domain import PerformanceEvent
from guitar_transcription.domain.serialization import events_from_dict
from guitar_transcription.pipeline import (
    NotationRequest,
    notate,
    write_events_json,
    write_notation,
)
from guitar_transcription.rhythm import (
    NoteValue,
    TempoEstimate,
    TempoEstimationError,
    TempoEstimator,
    TimeSignature,
    is_single_voice,
)

FOUR_FOUR = TimeSignature(4, 4)

# G major line at quarter = 120 (0.5 s), slightly human: G3 A3 B3 C4 | D4 (half) E4 ringing into F#4
RAW = [
    PerformanceEvent(onset_seconds=0.012, offset_seconds=0.47, pitch_midi=55, velocity=0.8),
    PerformanceEvent(onset_seconds=0.508, offset_seconds=0.96, pitch_midi=57, velocity=0.7),
    PerformanceEvent(onset_seconds=0.991, offset_seconds=1.49, pitch_midi=59, velocity=0.7),
    PerformanceEvent(onset_seconds=1.503, offset_seconds=1.98, pitch_midi=60, velocity=0.7),
    PerformanceEvent(onset_seconds=2.010, offset_seconds=2.99, pitch_midi=62, velocity=0.8),
    PerformanceEvent(onset_seconds=3.002, offset_seconds=4.10, pitch_midi=64, velocity=0.6),
    PerformanceEvent(onset_seconds=3.497, offset_seconds=3.98, pitch_midi=66, velocity=0.6),
]


class FakeTranscriber:
    """Stands in for the model at the AudioTranscriber boundary."""

    def __init__(self, events: list[PerformanceEvent]) -> None:
        self.events = events

    def transcribe(self, audio_path: str | PathLike[str]) -> list[PerformanceEvent]:
        return self.events


class FakeTempoEstimator:
    def __init__(self, bpm: float) -> None:
        self.bpm = bpm
        self.calls = 0

    def estimate(self, audio_path: str | PathLike[str]) -> TempoEstimate:
        self.calls += 1
        return TempoEstimate(self.bpm, tuple(0.01 + i * 60 / self.bpm for i in range(8)))


def run(
    tmp_path: Path,
    request: NotationRequest,
    estimator: TempoEstimator | None = None,
) -> tuple[list[PerformanceEvent], Path, Path]:
    """The full Stage 1 path as the CLI runs it, returning raw events and both output files."""
    transcriber: AudioTranscriber = FakeTranscriber(RAW)
    audio = tmp_path / "take.wav"
    events = transcriber.transcribe(audio)
    json_path = write_events_json(events, tmp_path / "out" / "take.events.json", source=str(audio))
    result = notate(events, audio, request, tempo_estimator=estimator)
    xml_path = write_notation(result, tmp_path / "out" / "take.musicxml", title="take")
    return events, json_path, xml_path


def test_end_to_end_with_explicit_tempo(tmp_path: Path) -> None:
    events, json_path, xml_path = run(tmp_path, NotationRequest(FOUR_FOUR, tempo_bpm=120))

    # Raw boundary: JSON holds exactly the transcriber's events, raw seconds intact.
    assert events_from_dict(json.loads(json_path.read_text())) == RAW
    # Notation boundary: the MusicXML carries the quantized rhythm, in 2 measures of 4/4.
    score = converter.parse(xml_path)
    assert [p.nameWithOctave for p in score.stripTies().pitches] == [
        "G3", "A3", "B3", "C4", "D4", "E4", "F#4",
    ]  # fmt: skip
    measures = score.parts[0].getElementsByClass(stream.Measure)
    assert len(measures) == 2
    # E4 rang on past F#4's onset; the single-voice step cut it to a quarter.
    durations = [float(n.quarterLength) for n in score.stripTies().recurse().notes]
    assert durations == [1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0]
    assert score.recurse().getElementsByClass(tempo.MetronomeMark)[0].number == 120


def test_raw_events_are_passed_through_not_copied_or_changed(tmp_path: Path) -> None:
    before = copy.deepcopy(RAW)
    audio = tmp_path / "take.wav"

    result = notate(RAW, audio, NotationRequest(FOUR_FOUR, tempo_bpm=120))

    assert RAW == before
    # Every quantized event points at one of the transcriber's own objects.
    assert all(any(e.source is raw for raw in RAW) for e in result.quantized.events)
    assert all(any(e.source is raw for raw in RAW) for e in result.voiced.events)


def test_rhythm_output_is_single_voice_and_counts_what_changed(tmp_path: Path) -> None:
    result = notate(RAW, tmp_path / "take.wav", NotationRequest(FOUR_FOUR, tempo_bpm=120))

    assert not is_single_voice(result.quantized)  # E4 rings into F#4
    assert is_single_voice(result.voiced)
    assert result.shortened_count == 1
    assert result.dropped_count == 0
    e4 = next(e for e in result.voiced.events if e.pitch_midi == 64)
    assert (e4.onset_quarters, e4.duration_quarters) == (F(6), F(1))
    assert e4.source.offset_seconds == 4.10  # raw let-ring still known


def test_explicit_tempo_wins_over_estimator(tmp_path: Path) -> None:
    estimator = FakeTempoEstimator(bpm=61.0)

    result = notate(
        RAW, tmp_path / "t.wav", NotationRequest(FOUR_FOUR, 120), tempo_estimator=estimator
    )

    assert estimator.calls == 0
    assert (result.tempo_bpm, result.tempo_estimate) == (120, None)


def test_estimated_tempo_is_used_rounded_and_reported(tmp_path: Path) -> None:
    estimator = FakeTempoEstimator(bpm=119.98765)

    events, _, xml_path = run(tmp_path, NotationRequest(FOUR_FOUR), estimator)
    result = notate(
        events, tmp_path / "take.wav", NotationRequest(FOUR_FOUR), tempo_estimator=estimator
    )

    assert result.tempo_bpm == 119.99
    assert result.tempo_estimate is not None and result.tempo_estimate.bpm == 119.98765
    assert (
        converter.parse(xml_path).recurse().getElementsByClass(tempo.MetronomeMark)[0].number
        == 119.99
    )


def test_missing_tempo_and_estimator_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no tempo"):
        notate(RAW, tmp_path / "t.wav", NotationRequest(FOUR_FOUR))


def test_estimation_failure_propagates(tmp_path: Path) -> None:
    class Failing:
        def estimate(self, audio_path: str | PathLike[str]) -> TempoEstimate:
            raise TempoEstimationError("need at least 4 beats")

    with pytest.raises(TempoEstimationError):
        notate(RAW, tmp_path / "t.wav", NotationRequest(FOUR_FOUR), tempo_estimator=Failing())


def test_meter_and_grid_are_applied(tmp_path: Path) -> None:
    request = NotationRequest(TimeSignature(3, 4), tempo_bpm=120, grid=NoteValue.EIGHTH)

    result = notate(RAW, tmp_path / "t.wav", request)

    assert (result.voiced.time_signature, result.voiced.grid) == (
        TimeSignature(3, 4),
        NoteValue.EIGHTH,
    )
    assert result.voiced.measure_count == 3  # 8 quarters of material in 3/4


def test_no_notes_still_gives_valid_notation(tmp_path: Path) -> None:
    result = notate([], tmp_path / "t.wav", NotationRequest(FOUR_FOUR, tempo_bpm=90))
    path = write_notation(result, tmp_path / "empty.musicxml", title="empty")

    assert len(converter.parse(path).parts[0].getElementsByClass(stream.Measure)) == 1
