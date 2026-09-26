import dataclasses

import pytest

from guitar_transcription.domain import (
    STANDARD_GUITAR,
    STANDARD_TUNING,
    GuitarConfig,
    Performance,
    PerformanceEvent,
)


def note(onset: float, pitch: int, **kwargs: object) -> PerformanceEvent:
    return PerformanceEvent(
        onset_seconds=onset, offset_seconds=onset + 0.5, pitch_midi=pitch, **kwargs
    )  # type: ignore[arg-type]


def test_events_are_sorted_by_onset_then_pitch() -> None:
    performance = Performance([note(1.0, 64), note(0.0, 60), note(1.0, 55)], STANDARD_GUITAR)

    assert [(e.onset_seconds, e.pitch_midi) for e in performance.events] == [
        (0.0, 60),
        (1.0, 55),
        (1.0, 64),
    ]


def test_events_are_stored_as_tuple_from_any_iterable() -> None:
    performance = Performance((note(t, 60) for t in (2.0, 1.0)), STANDARD_GUITAR)

    assert isinstance(performance.events, tuple)
    assert len(performance) == 2


def test_empty_performance() -> None:
    performance = Performance([], STANDARD_GUITAR)

    assert len(performance) == 0
    assert performance.end_seconds == 0.0


def test_end_seconds_is_latest_offset_not_last_onset() -> None:
    long_note = PerformanceEvent(onset_seconds=0.0, offset_seconds=5.0, pitch_midi=40)
    performance = Performance([long_note, note(1.0, 64)], STANDARD_GUITAR)

    assert performance.end_seconds == 5.0


def test_is_immutable_and_comparable() -> None:
    a = Performance([note(0.0, 60)], STANDARD_GUITAR)

    with pytest.raises(dataclasses.FrozenInstanceError):
        a.config = STANDARD_GUITAR  # type: ignore[misc]
    assert a == Performance([note(0.0, 60)], STANDARD_GUITAR)


def test_overlapping_same_pitch_events_are_allowed() -> None:
    # Re-plucking a ringing note is real; export code, not the domain, resolves overlaps.
    performance = Performance([note(0.0, 64), note(0.2, 64)], STANDARD_GUITAR)

    assert len(performance) == 2


def test_unknown_positions_need_no_config_checks() -> None:
    # Pitch 30 is below this guitar's range, but Stage 1 audio may still report it.
    assert len(Performance([note(0.0, 30)], STANDARD_GUITAR)) == 1


def test_accepts_consistent_known_position() -> None:
    performance = Performance([note(0.0, 64, string=2, fret=5)], STANDARD_GUITAR)

    assert performance.events[0].fret == 5


def test_rejects_position_that_does_not_sound_the_pitch() -> None:
    with pytest.raises(ValueError, match="string 2 fret 4 sounds MIDI 63, but pitch_midi is 64"):
        Performance([note(0.0, 64, string=2, fret=4)], STANDARD_GUITAR)


def test_rejects_string_not_on_instrument() -> None:
    with pytest.raises(ValueError, match="string must be in 1..6"):
        Performance([note(0.0, 64, string=7, fret=0)], STANDARD_GUITAR)


def test_rejects_fret_beyond_max_fret() -> None:
    with pytest.raises(ValueError, match="fret must be in 0..22"):
        Performance([note(0.0, 88, string=1, fret=24)], STANDARD_GUITAR)


def test_rejects_fret_below_capo() -> None:
    capo_3 = GuitarConfig(STANDARD_TUNING, capo=3)

    with pytest.raises(ValueError, match="fret must be in 3..22"):
        Performance([note(0.0, 64, string=1, fret=0)], capo_3)
    assert len(Performance([note(0.0, 67, string=1, fret=3)], capo_3)) == 1
