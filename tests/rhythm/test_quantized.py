import dataclasses
from fractions import Fraction as F

import pytest

from guitar_transcription.domain import PerformanceEvent
from guitar_transcription.rhythm import (
    NoteValue,
    QuantizedEvent,
    QuantizedPerformance,
    Rest,
    TimeSignature,
)

SOURCE = PerformanceEvent(onset_seconds=1.013, offset_seconds=1.49, pitch_midi=64)


def test_derived_properties() -> None:
    event = QuantizedEvent(SOURCE, F(2), F(1))

    assert event.pitch_midi == 64
    assert event.offset_quarters == F(3)


def test_is_immutable() -> None:
    event = QuantizedEvent(SOURCE, F(2), F(1))

    with pytest.raises(dataclasses.FrozenInstanceError):
        event.onset_quarters = F(3)  # type: ignore[misc]


def test_rejects_float_positions() -> None:
    with pytest.raises(TypeError, match="Fraction"):
        QuantizedEvent(SOURCE, 2.0, F(1))  # type: ignore[arg-type]


def test_rejects_negative_onset_and_non_positive_duration() -> None:
    with pytest.raises(ValueError, match="onset_quarters"):
        QuantizedEvent(SOURCE, F(-1), F(1))
    with pytest.raises(ValueError, match="duration_quarters"):
        QuantizedEvent(SOURCE, F(0), F(0))


def test_rests_include_gaps_only_up_to_the_last_note() -> None:
    events = (QuantizedEvent(SOURCE, F(1), F(1)), QuantizedEvent(SOURCE, F(5), F(1, 2)))
    performance = QuantizedPerformance(events, 120.0, TimeSignature(4, 4), NoteValue.SIXTEENTH)

    assert performance.rests() == (Rest(F(0), F(1)), Rest(F(2), F(3)))
    assert performance.rests()[1].offset_quarters == F(5)
    assert performance.end_quarters == F(11, 2)
    assert performance.measure_count == 2
