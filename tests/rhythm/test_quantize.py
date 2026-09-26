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
    downbeat: float | None = 0.0,
) -> QuantizedPerformance:
    # Most tests check snapping against a known grid, so they pin the downbeat at 0 s explicitly;
    # the default anchor (first note) and pickups are tested separately below.
    return quantize(
        events,
        quarter_note_bpm=bpm,
        time_signature=signature,
        grid=grid,
        downbeat_seconds=downbeat,
    )


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


def test_leading_silence_is_a_rest_when_the_downbeat_is_at_zero() -> None:
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


# --- strums (review CRITICAL 2) --------------------------------------------------------------

E_MAJOR = [40, 47, 52, 56, 59, 64]


def strummed(start: float, spread: float, ring: float = 0.95) -> list[PerformanceEvent]:
    step = spread / (len(E_MAJOR) - 1)
    return [ev(start + i * step, start + ring, p) for i, p in enumerate(E_MAJOR)]


def test_100ms_strum_stays_one_chord_on_sixteenth_grid() -> None:
    # Regression: this strum used to become a 4-note sixteenth chord plus a 2-note chord.
    result = q(strummed(0.0, 0.1) + [ev(0.5, 1.0, 64)])

    chord = [e for e in result.events if e.onset_quarters == F(0)]
    assert sorted(e.pitch_midi for e in chord) == E_MAJOR
    assert {e.onset_quarters for e in result.events} == {F(0), F(1)}


@pytest.mark.parametrize("spread", [0.02, 0.05, 0.08])
def test_strums_of_typical_spread_share_one_onset(spread: float) -> None:
    result = q(strummed(1.0, spread))

    assert {e.onset_quarters for e in result.events} == {F(2)}


def test_strum_anchors_at_its_earliest_onset() -> None:
    # Starts 30 ms before beat 2 and spans 80 ms: lands on beat 2, not 1/4 later.
    result = q(strummed(0.47, 0.08))

    assert {e.onset_quarters for e in result.events} == {F(1)}


def test_strum_notes_keep_their_own_offsets_and_raw_onsets() -> None:
    events = [ev(0.0, 0.5, 40), ev(0.04, 1.0, 47), ev(0.08, 1.0, 52)]

    result = q(events)

    assert [(e.onset_quarters, e.duration_quarters) for e in result.events] == [
        (F(0), F(1)),
        (F(0), F(2)),
        (F(0), F(2)),
    ]
    assert [e.source.onset_seconds for e in result.events] == [0.0, 0.04, 0.08]


def test_accurate_fast_sixteenths_are_not_merged() -> None:
    # 150 BPM sixteenths (100 ms apart), slightly overlapping as ringing notes do.
    line = [ev(i * 0.1, i * 0.1 + 0.12, 60 + i) for i in range(8)]

    result = q(line, bpm=150)

    assert [e.onset_quarters for e in result.events] == [F(i, 4) for i in range(8)]


def test_chord_window_is_capped_below_a_grid_step() -> None:
    # 240 BPM sixteenth = 62.5 ms, so the window is capped at ~47 ms: overlapping notes 55 ms
    # apart are separate sixteenths, not a chord.
    result = q([ev(0.0, 0.2, 60), ev(0.055, 0.2, 64)], bpm=240)

    assert [e.onset_quarters for e in result.events] == [F(0), F(1, 4)]


def test_grouping_can_be_disabled() -> None:
    result = quantize(
        strummed(0.0, 0.1),
        quarter_note_bpm=120,
        time_signature=FOUR_FOUR,
        chord_window_seconds=0,
    )

    assert len({e.onset_quarters for e in result.events}) == 2  # the old, split behaviour


@pytest.mark.parametrize("window", [-0.01, float("nan"), float("inf")])
def test_rejects_invalid_chord_window(window: float) -> None:
    with pytest.raises(ValueError, match="chord_window_seconds"):
        quantize([], quarter_note_bpm=120, time_signature=FOUR_FOUR, chord_window_seconds=window)


# --- downbeat anchor and pickups (review IMPORTANT 7) ---------------------------------------


def default_anchor(
    events: list[PerformanceEvent], downbeat_seconds: float | None = None
) -> QuantizedPerformance:
    return quantize(
        events, quarter_note_bpm=BPM, time_signature=FOUR_FOUR, downbeat_seconds=downbeat_seconds
    )


def test_default_downbeat_is_the_first_note_so_leading_silence_is_ignored() -> None:
    # 2.3 s of silence before playing starts (webcam/mic case).
    result = default_anchor([ev(2.3, 2.8, 60), ev(2.8, 3.3, 62), ev(3.3, 4.3, 64)])

    assert positions(result) == [(F(0), F(1)), (F(1), F(1)), (F(2), F(2))]
    assert result.rests() == ()
    assert result.origin_seconds == 2.3


def test_default_downbeat_with_no_notes_is_zero() -> None:
    assert default_anchor([]).origin_seconds == 0.0


def test_explicit_downbeat_overrides_the_first_note() -> None:
    result = default_anchor([ev(2.3, 2.8, 60)], downbeat_seconds=1.8)

    assert positions(result) == [(F(1), F(1))]  # one beat after the given downbeat
    assert result.origin_seconds == 1.8


def test_notes_before_the_downbeat_form_a_pickup_bar() -> None:
    # Pickup on beat 4 (0.5 s), downbeat at 1.0 s: "and-4 | 1".
    result = default_anchor([ev(0.5, 1.0, 55), ev(1.0, 2.0, 60)], downbeat_seconds=1.0)

    assert positions(result) == [(F(3), F(1)), (F(4), F(2))]
    assert [FOUR_FOUR.locate(e.onset_quarters) for e in result.events] == [(1, F(3)), (2, F(0))]
    assert result.rests() == (Rest(F(0), F(3)),)  # measure 1 = 3 beats rest + pickup note
    assert result.origin_seconds == -1.0  # measure 1 "starts" 2 s before the downbeat


def test_pickup_longer_than_a_measure_adds_whole_measures() -> None:
    # Notes start 6 beats before the downbeat -> 2 pickup measures, first note on beat 3.
    result = default_anchor([ev(0.0, 0.5, 60), ev(3.0, 3.5, 64)], downbeat_seconds=3.0)

    assert [e.onset_quarters for e in result.events] == [F(2), F(8)]
    assert FOUR_FOUR.locate(F(8)) == (3, F(0))


def test_pickup_in_three_four() -> None:
    result = quantize(
        [ev(0.5, 1.0, 55), ev(1.0, 1.5, 60)],
        quarter_note_bpm=BPM,
        time_signature=THREE_FOUR,
        downbeat_seconds=1.0,
    )

    assert [THREE_FOUR.locate(e.onset_quarters) for e in result.events] == [(1, F(2)), (2, F(0))]


def test_slightly_early_chord_does_not_create_an_empty_pickup_bar() -> None:
    # Strum starts 20 ms before the stated downbeat: it snaps onto the downbeat, no pickup.
    strum = [ev(0.98 + 0.01 * i, 1.9, p) for i, p in enumerate([40, 47, 52])]

    result = default_anchor(strum, downbeat_seconds=1.0)

    assert {e.onset_quarters for e in result.events} == {F(0)}
    assert result.origin_seconds == 1.0


def test_seconds_at_maps_musical_time_back_to_the_recording() -> None:
    raw = [ev(0.52, 1.0, 55), ev(1.013, 2.0, 60), ev(2.49, 3.0, 64)]

    result = default_anchor(raw, downbeat_seconds=1.0)

    for event in result.events:
        # Within half a sixteenth (62.5 ms) of the raw onset, since snapping moved it at most that.
        assert result.seconds_at(event.onset_quarters) == pytest.approx(
            event.source.onset_seconds, abs=0.0625
        )


def test_downbeat_does_not_change_raw_timing() -> None:
    raw = [ev(0.52, 1.0, 55), ev(1.013, 2.0, 60)]
    before = copy.deepcopy(raw)

    result = default_anchor(raw, downbeat_seconds=1.0)

    assert raw == before
    assert [e.source.onset_seconds for e in result.events] == [0.52, 1.013]


@pytest.mark.parametrize("downbeat", [-0.1, float("nan"), float("inf")])
def test_rejects_invalid_downbeat(downbeat: float) -> None:
    with pytest.raises(ValueError, match="downbeat_seconds"):
        default_anchor([ev(0.0, 0.5, 60)], downbeat_seconds=downbeat)
