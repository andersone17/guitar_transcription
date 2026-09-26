from fractions import Fraction as F

import pytest

from guitar_transcription.rhythm import (
    NoteValue,
    RhythmicDuration,
    TimeSignature,
    rhythmic_duration,
)


def test_note_values_in_quarters() -> None:
    assert [v.quarters for v in NoteValue] == [F(4), F(2), F(1), F(1, 2), F(1, 4)]


@pytest.mark.parametrize(
    ("quarters", "expected"),
    [
        (F(4), RhythmicDuration(NoteValue.WHOLE)),
        (F(3), RhythmicDuration(NoteValue.HALF, dotted=True)),
        (F(2), RhythmicDuration(NoteValue.HALF)),
        (F(3, 2), RhythmicDuration(NoteValue.QUARTER, dotted=True)),
        (F(1), RhythmicDuration(NoteValue.QUARTER)),
        (F(3, 4), RhythmicDuration(NoteValue.EIGHTH, dotted=True)),
        (F(1, 2), RhythmicDuration(NoteValue.EIGHTH)),
        (F(1, 4), RhythmicDuration(NoteValue.SIXTEENTH)),
        (F(6), RhythmicDuration(NoteValue.WHOLE, dotted=True)),
    ],
)
def test_rhythmic_duration_names_single_symbols(quarters: F, expected: RhythmicDuration) -> None:
    assert rhythmic_duration(quarters) == expected
    assert expected.quarters == quarters


@pytest.mark.parametrize("quarters", [F(5, 4), F(5, 2), F(7, 4), F(5), F(1, 3)])
def test_durations_needing_ties_or_tuplets_have_no_single_name(quarters: F) -> None:
    assert rhythmic_duration(quarters) is None


@pytest.mark.parametrize(
    ("numerator", "denominator", "quarters"),
    [
        (4, 4, F(4)),
        (3, 4, F(3)),
        (2, 4, F(2)),
        (6, 8, F(3)),
        (3, 8, F(3, 2)),
        (2, 2, F(4)),
        (7, 8, F(7, 2)),
    ],
)
def test_measure_length(numerator: int, denominator: int, quarters: F) -> None:
    assert TimeSignature(numerator, denominator).measure_quarters == quarters


def test_str() -> None:
    assert str(TimeSignature(6, 8)) == "6/8"


@pytest.mark.parametrize(
    ("signature", "quarters", "expected"),
    [
        (TimeSignature(4, 4), F(0), (1, F(0))),
        (TimeSignature(4, 4), F(3), (1, F(3))),
        (TimeSignature(4, 4), F(4), (2, F(0))),  # barline belongs to the next measure
        (TimeSignature(4, 4), F(9, 2), (2, F(1, 2))),
        (TimeSignature(3, 4), F(3), (2, F(0))),
        (TimeSignature(3, 4), F(7), (3, F(1))),
        (TimeSignature(6, 8), F(7, 2), (2, F(1, 2))),
    ],
)
def test_locate(signature: TimeSignature, quarters: F, expected: tuple[int, F]) -> None:
    assert signature.locate(quarters) == expected


def test_locate_rejects_negative_time() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        TimeSignature(4, 4).locate(F(-1))


@pytest.mark.parametrize(("numerator", "denominator"), [(0, 4), (4, 0), (4, 3), (3, 6), (4, -4)])
def test_rejects_invalid_time_signatures(numerator: int, denominator: int) -> None:
    with pytest.raises(ValueError):
        TimeSignature(numerator, denominator)


def test_rejects_non_int_time_signature() -> None:
    with pytest.raises(TypeError):
        TimeSignature(4.0, 4)  # type: ignore[arg-type]
