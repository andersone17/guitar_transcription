"""Onset grouping (strums -> chords) on raw events; no quantization involved."""

from guitar_transcription.domain import PerformanceEvent
from guitar_transcription.rhythm.chords import (
    DEFAULT_CHORD_WINDOW_SECONDS,
    MAX_STRING_GAP_SECONDS,
    group_simultaneous,
)

OPEN_E_MAJOR = [40, 47, 52, 56, 59, 64]  # low to high, as a downstroke sounds them


def ev(onset: float, offset: float, pitch: int) -> PerformanceEvent:
    return PerformanceEvent(onset_seconds=onset, offset_seconds=offset, pitch_midi=pitch)


def strum(
    start: float, spread: float, pitches: list[int], ring: float = 1.0
) -> list[PerformanceEvent]:
    step = spread / (len(pitches) - 1)
    return [ev(start + i * step, start + ring, p) for i, p in enumerate(pitches)]


def pitches(groups: list[list[PerformanceEvent]]) -> list[list[int]]:
    return [[e.pitch_midi for e in group] for group in groups]


def test_defaults() -> None:
    assert DEFAULT_CHORD_WINDOW_SECONDS == 0.1
    assert MAX_STRING_GAP_SECONDS == 0.05


def test_downstroke_strum_is_one_group() -> None:
    groups = group_simultaneous(strum(0.0, 0.07, OPEN_E_MAJOR), 0.08)

    assert pitches(groups) == [OPEN_E_MAJOR]


def test_upstroke_strum_is_one_group() -> None:
    groups = group_simultaneous(strum(0.5, 0.05, OPEN_E_MAJOR[::-1]), 0.08)

    assert len(groups) == 1 and sorted(pitches(groups)[0]) == OPEN_E_MAJOR


def test_consecutive_strums_are_separate_groups() -> None:
    events = strum(0.0, 0.06, OPEN_E_MAJOR, ring=0.49) + strum(0.5, 0.06, OPEN_E_MAJOR, ring=0.49)

    assert pitches(group_simultaneous(events, 0.08)) == [OPEN_E_MAJOR, OPEN_E_MAJOR]


def test_window_is_measured_from_the_first_note_so_slow_rolls_do_not_chain() -> None:
    # 40 ms between strings: each note is within 80 ms of the previous one, but the whole roll
    # spans 200 ms. Only the first three strings (0, 40, 80 ms) fit the window.
    groups = group_simultaneous(strum(0.0, 0.2, OPEN_E_MAJOR), 0.08)

    assert pitches(groups) == [[40, 47, 52], [56, 59, 64]]


def test_100ms_six_string_strum_is_one_group() -> None:
    groups = group_simultaneous(strum(0.0, 0.1, OPEN_E_MAJOR), DEFAULT_CHORD_WINDOW_SECONDS)

    assert pitches(groups) == [OPEN_E_MAJOR]


def test_let_ring_arpeggio_is_not_merged_even_when_notes_overlap() -> None:
    # Fingerpicked sixteenths at 120 BPM (125 ms apart), played up to 20 ms early/late and left
    # ringing: consecutive gaps of 85-105 ms are melodic spacing, not a strum.
    onsets = [0.0, 0.105, 0.25, 0.355, 0.5, 0.605]
    arpeggio = [ev(t, 1.0, p) for t, p in zip(onsets, OPEN_E_MAJOR, strict=True)]

    assert len(group_simultaneous(arpeggio, DEFAULT_CHORD_WINDOW_SECONDS)) == 6


def test_consecutive_gap_limit_splits_a_hesitant_strum() -> None:
    # Two strings, a 60 ms pause, then the rest: the pause is longer than a strum's string gap.
    events = [ev(0.0, 1, 40), ev(0.01, 1, 47), ev(0.07, 1, 52), ev(0.08, 1, 56)]

    assert pitches(group_simultaneous(events, 0.1)) == [[40, 47], [52, 56]]


def test_fast_melody_without_overlap_is_not_merged() -> None:
    # Legato line / hammer-ons 60 ms apart: each note ends as the next begins.
    line = [ev(i * 0.06, (i + 1) * 0.06, p) for i, p in enumerate([64, 66, 67, 69])]

    assert pitches(group_simultaneous(line, 0.08)) == [[64], [66], [67], [69]]


def test_repeated_pitch_starts_a_new_group() -> None:
    tremolo = [ev(0.0, 0.2, 64), ev(0.05, 0.2, 64)]

    assert pitches(group_simultaneous(tremolo, 0.08)) == [[64], [64]]


def test_zero_window_disables_grouping() -> None:
    assert len(group_simultaneous(strum(0.0, 0.05, OPEN_E_MAJOR), 0.0)) == 6


def test_input_order_does_not_matter_and_events_are_untouched() -> None:
    events = strum(0.0, 0.07, OPEN_E_MAJOR)
    shuffled = [events[i] for i in (3, 0, 5, 1, 4, 2)]

    [group] = group_simultaneous(shuffled, 0.08)

    assert group == events  # same objects, raw onsets preserved, in onset order
    assert all(a is b for a, b in zip(group, events, strict=True))


def test_empty() -> None:
    assert group_simultaneous([], 0.08) == []
