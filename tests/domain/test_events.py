import dataclasses
import math

import pytest

from guitar_transcription.domain import PerformanceEvent, PickDirection

CONFIDENCE_FIELDS = [
    "velocity",
    "audio_confidence",
    "fretting_confidence",
    "picking_confidence",
    "confidence",
]


def make_event(**overrides: object) -> PerformanceEvent:
    fields: dict[str, object] = {"onset_seconds": 1.0, "offset_seconds": 1.5, "pitch_midi": 64}
    fields.update(overrides)
    return PerformanceEvent(**fields)  # type: ignore[arg-type]


def test_minimal_event_leaves_unknowns_as_none() -> None:
    event = make_event()

    assert event.velocity is None
    assert event.string is None
    assert event.fret is None
    assert event.pick_direction is None
    for name in CONFIDENCE_FIELDS:
        assert getattr(event, name) is None


def test_pitch_name_is_derived() -> None:
    assert make_event(pitch_midi=64).pitch_name == "E4"
    assert make_event(pitch_midi=61).pitch_name == "C#4"


def test_pitch_name_follows_pitch_after_replace() -> None:
    event = dataclasses.replace(make_event(pitch_midi=64), pitch_midi=40)

    assert event.pitch_name == "E2"


def test_duration() -> None:
    assert make_event(onset_seconds=0.25, offset_seconds=1.0).duration_seconds == 0.75


def test_fully_specified_event() -> None:
    event = make_event(
        velocity=0.8,
        string=2,
        fret=5,
        pick_direction=PickDirection.DOWN,
        audio_confidence=0.9,
        fretting_confidence=0.7,
        picking_confidence=0.6,
        confidence=0.75,
    )

    assert (event.string, event.fret) == (2, 5)
    assert event.pick_direction == "down"


def test_event_is_immutable_and_hashable() -> None:
    event = make_event()

    with pytest.raises(dataclasses.FrozenInstanceError):
        event.pitch_midi = 65  # type: ignore[misc]
    assert hash(event) == hash(make_event())


def test_onset_at_zero_is_allowed() -> None:
    assert make_event(onset_seconds=0.0, offset_seconds=0.1).onset_seconds == 0.0


def test_integer_times_are_accepted() -> None:
    assert make_event(onset_seconds=1, offset_seconds=2).duration_seconds == 1


def test_rejects_negative_onset() -> None:
    with pytest.raises(ValueError, match="onset_seconds must be >= 0"):
        make_event(onset_seconds=-0.01)


@pytest.mark.parametrize("offset", [1.0, 0.5])
def test_rejects_offset_not_after_onset(offset: float) -> None:
    with pytest.raises(ValueError, match="must be after"):
        make_event(onset_seconds=1.0, offset_seconds=offset)


@pytest.mark.parametrize("field", ["onset_seconds", "offset_seconds"])
@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_rejects_non_finite_times(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        make_event(**{field: value})


@pytest.mark.parametrize("pitch", [-1, 128])
def test_rejects_out_of_range_pitch(pitch: int) -> None:
    with pytest.raises(ValueError, match="pitch_midi"):
        make_event(pitch_midi=pitch)


def test_rejects_float_pitch() -> None:
    # Adapters must convert backend output (often floats/numpy scalars) at the boundary.
    with pytest.raises(TypeError, match="pitch_midi"):
        make_event(pitch_midi=64.0)


@pytest.mark.parametrize(("string", "fret"), [(1, None), (None, 0)])
def test_rejects_string_without_fret_and_vice_versa(string: int | None, fret: int | None) -> None:
    with pytest.raises(ValueError, match="both be set or both be None"):
        make_event(string=string, fret=fret)


def test_rejects_string_zero() -> None:
    with pytest.raises(ValueError, match="string must be >= 1"):
        make_event(string=0, fret=0)


def test_rejects_negative_fret() -> None:
    with pytest.raises(ValueError, match="fret must be >= 0"):
        make_event(string=1, fret=-1)


@pytest.mark.parametrize("field", CONFIDENCE_FIELDS)
@pytest.mark.parametrize("value", [-0.01, 1.01, math.nan])
def test_rejects_unit_interval_fields_out_of_range(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        make_event(**{field: value})


@pytest.mark.parametrize("field", CONFIDENCE_FIELDS)
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_accepts_unit_interval_bounds(field: str, value: float) -> None:
    assert getattr(make_event(**{field: value}), field) == value


def test_pick_direction_values_are_stable_strings() -> None:
    # Values are part of the serialized form (e.g. events.json), so they must not drift.
    assert [d.value for d in PickDirection] == ["down", "up"]
    assert PickDirection("up") is PickDirection.UP
