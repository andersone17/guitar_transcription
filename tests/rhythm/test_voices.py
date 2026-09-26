from fractions import Fraction as F

from guitar_transcription.domain import PerformanceEvent
from guitar_transcription.rhythm import (
    NoteValue,
    QuantizedEvent,
    QuantizedPerformance,
    TimeSignature,
    is_single_voice,
    to_single_voice,
)


def qe(onset: F, duration: F, pitch: int = 60, raw_onset: float = 0.0) -> QuantizedEvent:
    source = PerformanceEvent(
        onset_seconds=raw_onset, offset_seconds=raw_onset + 1, pitch_midi=pitch
    )
    return QuantizedEvent(source, onset, duration)


def perf(*events: QuantizedEvent) -> QuantizedPerformance:
    return QuantizedPerformance(tuple(events), 120.0, TimeSignature(4, 4), NoteValue.SIXTEENTH)


def spans(performance: QuantizedPerformance) -> list[tuple[F, F, int]]:
    return [(e.onset_quarters, e.duration_quarters, e.pitch_midi) for e in performance.events]


def test_monophonic_line_is_unchanged() -> None:
    melody = perf(qe(F(0), F(1)), qe(F(1), F(1, 2), 62), qe(F(2), F(1), 64))

    assert to_single_voice(melody) == melody
    assert is_single_voice(melody)


def test_let_ring_is_cut_at_the_next_onset() -> None:
    ringing = perf(qe(F(0), F(3), 40), qe(F(1), F(1), 64))

    result = to_single_voice(ringing)

    assert spans(result) == [(F(0), F(1), 40), (F(1), F(1), 64)]
    assert not is_single_voice(ringing)
    assert is_single_voice(result)


def test_chord_members_share_the_longest_duration_up_to_next_onset() -> None:
    chord = perf(qe(F(0), F(1), 43), qe(F(0), F(2), 47), qe(F(0), F(4), 50), qe(F(3), F(1), 55))

    # Longest member is 4 quarters, but the next onset is at 3.
    assert spans(to_single_voice(chord)) == [
        (F(0), F(3), 43),
        (F(0), F(3), 47),
        (F(0), F(3), 50),
        (F(3), F(1), 55),
    ]


def test_final_chord_keeps_its_longest_duration() -> None:
    assert spans(to_single_voice(perf(qe(F(0), F(1), 43), qe(F(0), F(2), 47)))) == [
        (F(0), F(2), 43),
        (F(0), F(2), 47),
    ]


def test_duplicate_pitch_in_a_chord_is_kept_once_earliest_raw_onset_wins() -> None:
    result = to_single_voice(
        perf(qe(F(0), F(1), 60, raw_onset=0.04), qe(F(0), F(1), 60, raw_onset=0.01))
    )

    [event] = result.events
    assert event.source.onset_seconds == 0.01
    assert not is_single_voice(perf(qe(F(0), F(1), 60), qe(F(0), F(1), 60)))


def test_sources_and_parameters_are_carried_over_untouched() -> None:
    original = perf(qe(F(0), F(4), 40, raw_onset=0.013), qe(F(1), F(1), 64))

    result = to_single_voice(original)

    assert [e.source for e in result.events] == [e.source for e in original.events]
    assert result.events[0].source.onset_seconds == 0.013
    assert (result.quarter_note_bpm, result.time_signature, result.grid) == (
        original.quarter_note_bpm,
        original.time_signature,
        original.grid,
    )
    assert original.events[0].duration_quarters == F(4)  # input not modified


def test_empty() -> None:
    assert to_single_voice(perf()) == perf()
    assert is_single_voice(perf())
