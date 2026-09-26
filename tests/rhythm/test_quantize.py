"""Quantization tests on deterministic synthetic timing (no audio, no model)."""

import copy
from fractions import Fraction as F

import pytest

from guitar_transcription.domain import PerformanceEvent
from guitar_transcription.rhythm import (
    NoteValue,
    QuantizedPerformance,
    Rest,
    RhythmicDuration,
    TimeSignature,
    quantize,
)
from guitar_transcription.rhythm.quantize import seconds_to_quarters, snap_to_grid

FOUR_FOUR = TimeSignature(4, 4)
THREE_FOUR = TimeSignature(3, 4)
SIX_EIGHT = TimeSignature(6, 8)
CUT_TIME = TimeSignature(2, 2)

# At 120 BPM: quarter = 0.5 s, eighth = 0.25 s, sixteenth = 0.125 s.
BPM = 120.0


def ev(onset: float, offset: float, pitch: int = 60) -> PerformanceEvent:
    return PerformanceEvent(onset_seconds=onset, offset_seconds=offset, pitch_midi=pitch)


def q(
    events: list[PerformanceEvent],
    signature: TimeSignature = FOUR_FOUR,
    *,
    bpm: float = BPM,
    grid: NoteValue = NoteValue.SIXTEENTH,
) -> QuantizedPerformance:
    return quantize(events, quarter_note_bpm=bpm, time_signature=signature, grid=grid)


def positions(result: QuantizedPerformance) -> list[tuple[F, F]]:
    return [(e.onset_quarters, e.duration_quarters) for e in result.events]


# --- helpers -------------------------------------------------------------------------------


def test_seconds_to_quarters() -> None:
    assert seconds_to_quarters(1.0, 120) == 2.0
    assert seconds_to_quarters(1.0, 60) == 1.0
    assert seconds_to_quarters(0.75, 90) == pytest.approx(1.125)


@pytest.mark.parametrize(
    ("quarters", "grid", "expected"),
    [
        (2.026, NoteValue.SIXTEENTH, F(2)),  # 1.013 s at 120 BPM -> beat 3 (quarter 2)
        (1.96, NoteValue.SIXTEENTH, F(2)),
        (0.124, NoteValue.SIXTEENTH, F(0)),
        (0.125, NoteValue.SIXTEENTH, F(1, 4)),  # exactly halfway rounds later
        (0.3, NoteValue.EIGHTH, F(1, 2)),
        (1.4, NoteValue.QUARTER, F(1)),
        (2.9, NoteValue.HALF, F(2)),
        (3.0, NoteValue.HALF, F(4)),
        (1.9, NoteValue.WHOLE, F(0)),
    ],
)
def test_snap_to_grid(quarters: float, grid: NoteValue, expected: F) -> None:
    result = snap_to_grid(quarters, grid)

    assert result == expected
    assert isinstance(result, F)


# --- exact grid input ---------------------------------------------------------------------


def test_exact_quarter_notes() -> None:
    events = [ev(i * 0.5, (i + 1) * 0.5, 60 + i) for i in range(4)]

    result = q(events)

    assert positions(result) == [(F(i), F(1)) for i in range(4)]
    assert all(e.rhythmic_duration == RhythmicDuration(NoteValue.QUARTER) for e in result.events)


def test_exact_eighth_notes() -> None:
    events = [ev(i * 0.25, (i + 1) * 0.25) for i in range(8)]

    assert positions(q(events)) == [(F(i, 2), F(1, 2)) for i in range(8)]


def test_exact_sixteenth_notes() -> None:
    events = [ev(i * 0.125, (i + 1) * 0.125) for i in range(16)]

    result = q(events)

    assert positions(result) == [(F(i, 4), F(1, 4)) for i in range(16)]
    assert result.events[-1].rhythmic_duration == RhythmicDuration(NoteValue.SIXTEENTH)


def test_positions_are_exact_fractions() -> None:
    result = q([ev(0.125 * i, 0.125 * (i + 1)) for i in range(100)])

    assert all(
        isinstance(e.onset_quarters, F) and isinstance(e.duration_quarters, F)
        for e in result.events
    )
    assert result.events[-1].onset_quarters == F(99, 4)  # no float drift after 100 steps


# --- human timing -------------------------------------------------------------------------


def test_example_from_spec_onset_1_013_is_beat_3() -> None:
    [event] = q([ev(1.013, 1.49)]).events

    assert event.onset_quarters == F(2)  # quarter 2 = beat 3 of measure 1
    assert FOUR_FOUR.locate(event.onset_quarters) == (1, F(2))


@pytest.mark.parametrize("jitter", [-0.020, -0.010, -0.001])
def test_slightly_early_notes_snap_forward(jitter: float) -> None:
    events = [ev(i * 0.5 + jitter if i else 0.0, (i + 1) * 0.5 + jitter) for i in range(4)]

    assert [e.onset_quarters for e in q(events).events] == [F(0), F(1), F(2), F(3)]


@pytest.mark.parametrize("jitter", [0.001, 0.010, 0.020])
def test_slightly_late_notes_snap_back(jitter: float) -> None:
    events = [ev(i * 0.5 + jitter, (i + 1) * 0.5 + jitter) for i in range(4)]

    assert [e.onset_quarters for e in q(events).events] == [F(0), F(1), F(2), F(3)]


def test_mixed_jitter_sixteenths() -> None:
    offsets_ms = [12, -18, 7, -3, 20, -20, 0, 15]
    events = [
        ev(max(0.0, i * 0.125 + ms / 1000), i * 0.125 + ms / 1000 + 0.11)
        for i, ms in enumerate(offsets_ms)
    ]

    assert [e.onset_quarters for e in q(events).events] == [F(i, 4) for i in range(8)]


def test_large_deviation_lands_on_neighbouring_grid_line() -> None:
    # 70 ms late at 120 BPM exceeds half a sixteenth (62.5 ms): documented limitation.
    [event] = q([ev(0.570, 1.0)]).events

    assert event.onset_quarters == F(5, 4)


def test_coarser_grid_tolerates_larger_deviation() -> None:
    [event] = q([ev(0.570, 1.0)], grid=NoteValue.EIGHTH).events

    assert event.onset_quarters == F(1)


@pytest.mark.parametrize(
    ("duration_s", "expected"),
    [
        (0.47, RhythmicDuration(NoteValue.QUARTER)),  # a bit short of a quarter (0.5 s)
        (0.53, RhythmicDuration(NoteValue.QUARTER)),
        (0.23, RhythmicDuration(NoteValue.EIGHTH)),
        (0.74, RhythmicDuration(NoteValue.QUARTER, dotted=True)),  # ~0.75 s
        (0.97, RhythmicDuration(NoteValue.HALF)),
        (1.52, RhythmicDuration(NoteValue.HALF, dotted=True)),
        (1.98, RhythmicDuration(NoteValue.WHOLE)),
        (0.36, RhythmicDuration(NoteValue.EIGHTH, dotted=True)),  # ~0.375 s
    ],
)
def test_durations_close_to_standard_values(duration_s: float, expected: RhythmicDuration) -> None:
    [event] = q([ev(0.0, duration_s)]).events

    assert event.rhythmic_duration == expected


def test_duration_needing_a_tie_has_no_single_symbol() -> None:
    [event] = q([ev(0.0, 0.625)]).events  # 5 sixteenths

    assert event.duration_quarters == F(5, 4)
    assert event.rhythmic_duration is None


@pytest.mark.parametrize("offset", [0.01, 0.05, 0.06])
def test_very_short_note_gets_one_grid_step(offset: float) -> None:
    [event] = q([ev(0.0, offset)]).events

    assert event.duration_quarters == F(1, 4)


def test_note_whose_ends_round_together_keeps_one_step() -> None:
    [event] = q([ev(0.49, 0.55)]).events  # both ends snap to quarter 1

    assert (event.onset_quarters, event.duration_quarters) == (F(1), F(1, 4))


# --- rests --------------------------------------------------------------------------------


def test_rest_between_notes() -> None:
    # quarter, quarter rest, quarter
    result = q([ev(0.0, 0.5), ev(1.0, 1.5)])

    assert result.rests() == (Rest(F(1), F(1)),)


def test_leading_silence_is_a_rest_because_time_zero_is_the_downbeat() -> None:
    result = q([ev(0.5, 1.0)])

    assert result.rests() == (Rest(F(0), F(1)),)


def test_legato_gap_smaller_than_half_a_step_is_not_a_rest() -> None:
    # Note released 40 ms early: the offset snaps to the next onset.
    result = q([ev(0.0, 0.46), ev(0.5, 1.0)])

    assert result.rests() == ()
    assert result.events[0].duration_quarters == F(1)


def test_overlapping_notes_leave_no_rest() -> None:
    result = q([ev(0.0, 2.0, 40), ev(0.5, 1.0, 64), ev(1.5, 2.0, 67)])

    assert result.rests() == ()


def test_no_rests_for_contiguous_or_empty() -> None:
    assert q([ev(i * 0.5, (i + 1) * 0.5) for i in range(4)]).rests() == ()
    assert q([]).rests() == ()


# --- measures and meters ------------------------------------------------------------------


def test_measure_boundaries_in_four_four() -> None:
    events = [ev(i * 0.5, (i + 1) * 0.5) for i in range(9)]  # 9 quarters

    result = q(events)

    located = [FOUR_FOUR.locate(e.onset_quarters) for e in result.events]
    assert located[:5] == [(1, F(0)), (1, F(1)), (1, F(2)), (1, F(3)), (2, F(0))]
    assert located[8] == (3, F(0))
    assert result.measure_count == 3


def test_note_crossing_barline_keeps_full_duration() -> None:
    # Starts on beat 4 of measure 1, lasts a half note -> crosses into measure 2 (notation ties it).
    [event] = q([ev(1.5, 2.5)]).events

    assert event.onset_quarters == F(3)
    assert event.duration_quarters == F(2)
    assert FOUR_FOUR.locate(event.offset_quarters) == (2, F(1))


def test_three_four_waltz() -> None:
    events = [ev(i * 0.5, (i + 1) * 0.5) for i in range(7)]

    result = q(events, THREE_FOUR)

    located = [THREE_FOUR.locate(e.onset_quarters) for e in result.events]
    assert located == [(1, F(0)), (1, F(1)), (1, F(2)), (2, F(0)), (2, F(1)), (2, F(2)), (3, F(0))]
    assert result.measure_count == 3


def test_six_eight_eighth_notes() -> None:
    # Tempo counts quarter notes: at 120 quarter BPM an eighth is 0.25 s; a 6/8 bar is 1.5 s.
    events = [ev(i * 0.25 + 0.01, (i + 1) * 0.25) for i in range(12)]

    result = q(events, SIX_EIGHT)

    located = [SIX_EIGHT.locate(e.onset_quarters) for e in result.events]
    assert located[0] == (1, F(0))
    assert located[5] == (1, F(5, 2))
    assert located[6] == (2, F(0))
    assert result.measure_count == 2


def test_cut_time_half_notes() -> None:
    events = [ev(i * 1.0, (i + 1) * 1.0) for i in range(3)]

    result = q(events, CUT_TIME)

    assert [CUT_TIME.locate(e.onset_quarters) for e in result.events] == [
        (1, F(0)),
        (1, F(2)),
        (2, F(0)),
    ]
    assert all(e.rhythmic_duration == RhythmicDuration(NoteValue.HALF) for e in result.events)


def test_tempo_scales_positions() -> None:
    # At 60 BPM a quarter is 1 s.
    [event] = q([ev(3.0, 4.0)], bpm=60.0).events

    assert (event.onset_quarters, event.duration_quarters) == (F(3), F(1))


def test_measure_count_of_empty_performance_is_zero() -> None:
    assert q([]).measure_count == 0


# --- raw timing is preserved --------------------------------------------------------------


def test_source_events_are_unchanged_and_reachable() -> None:
    raw = [ev(1.013, 1.49, 64), ev(0.02, 0.51, 60), ev(0.02, 0.51, 55)]
    before = copy.deepcopy(raw)

    result = q(raw)

    assert raw == before  # inputs untouched (they are frozen, and still equal)
    assert sorted(
        (e.source for e in result.events), key=lambda e: (e.onset_seconds, e.pitch_midi)
    ) == sorted(raw, key=lambda e: (e.onset_seconds, e.pitch_midi))
    by_pitch = {e.pitch_midi: e for e in result.events}
    assert by_pitch[64].source.onset_seconds == 1.013  # raw seconds, not 1.0
    assert by_pitch[64].source.offset_seconds == 1.49
    assert by_pitch[64].source is raw[0]


def test_events_are_sorted_by_onset_then_pitch() -> None:
    result = q([ev(0.5, 1.0, 64), ev(0.0, 0.5, 67), ev(0.5, 1.0, 48)])

    assert [(e.onset_quarters, e.pitch_midi) for e in result.events] == [
        (F(0), 67),
        (F(1), 48),
        (F(1), 64),
    ]


def test_chord_members_share_an_onset() -> None:
    # A slightly rolled chord (strum) within half a grid step lands on one onset.
    result = q([ev(0.000, 1.0, 43), ev(0.015, 1.0, 47), ev(0.030, 1.0, 50), ev(0.045, 1.0, 55)])

    assert {e.onset_quarters for e in result.events} == {F(0)}


def test_result_records_its_parameters() -> None:
    result = q([ev(0.0, 0.5)], THREE_FOUR, grid=NoteValue.EIGHTH)

    assert (result.quarter_note_bpm, result.time_signature, result.grid) == (
        BPM,
        THREE_FOUR,
        NoteValue.EIGHTH,
    )


# --- validation ---------------------------------------------------------------------------


@pytest.mark.parametrize("bpm", [0.0, -120.0, float("nan"), float("inf")])
def test_rejects_invalid_tempo(bpm: float) -> None:
    with pytest.raises(ValueError, match="quarter_note_bpm"):
        q([ev(0.0, 0.5)], bpm=bpm)


@pytest.mark.parametrize(
    ("signature", "grid"),
    [
        (SIX_EIGHT, NoteValue.HALF),
        (THREE_FOUR, NoteValue.WHOLE),
        (TimeSignature(3, 8), NoteValue.QUARTER),
    ],
)
def test_rejects_grid_that_does_not_divide_the_measure(
    signature: TimeSignature, grid: NoteValue
) -> None:
    with pytest.raises(ValueError, match="does not divide"):
        q([ev(0.0, 0.5)], signature, grid=grid)
