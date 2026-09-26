import json
from typing import Any

import pytest

from guitar_transcription.domain import (
    STANDARD_TUNING,
    GuitarConfig,
    PerformanceEvent,
    PickDirection,
)
from guitar_transcription.domain.serialization import (
    FORMAT,
    TIMING,
    VERSION,
    events_from_dict,
    events_to_dict,
    guitar_from_dict,
)

RAW = PerformanceEvent(
    onset_seconds=0.4992290249, offset_seconds=0.9752380952, pitch_midi=55, velocity=0.8558
)
FULL = PerformanceEvent(
    onset_seconds=1.0,
    offset_seconds=1.5,
    pitch_midi=64,
    velocity=0.5,
    string=2,
    fret=5,
    pick_direction=PickDirection.UP,
    audio_confidence=0.9,
    fretting_confidence=0.8,
    picking_confidence=0.7,
    confidence=0.85,
)


def test_document_header_describes_raw_timing() -> None:
    document = events_to_dict([RAW], source="clip.wav")

    assert document["format"] == FORMAT
    assert document["version"] == VERSION
    assert document["timing"] == TIMING == "raw-performance-seconds"
    assert document["source"] == "clip.wav"


def test_unknown_fields_are_written_as_null() -> None:
    [event] = events_to_dict([RAW])["events"]

    assert event["string"] is None
    assert event["fret"] is None
    assert event["confidence"] is None
    assert "pitch_name" not in event  # derived values aren't stored
    assert "duration_seconds" not in event


def test_fields_are_in_declaration_order() -> None:
    [event] = events_to_dict([RAW])["events"]

    assert list(event)[:4] == ["onset_seconds", "offset_seconds", "pitch_midi", "velocity"]


@pytest.mark.parametrize("events", [[], [RAW], [RAW, FULL]])
def test_json_round_trip_is_lossless(events: list[PerformanceEvent]) -> None:
    text = json.dumps(events_to_dict(events, source="clip.wav"))

    assert events_from_dict(json.loads(text)) == events


def test_raw_timing_survives_exactly() -> None:
    [event] = events_from_dict(json.loads(json.dumps(events_to_dict([RAW]))))

    assert event.onset_seconds == 0.4992290249
    assert event.offset_seconds == 0.9752380952


def test_pick_direction_is_stored_as_string() -> None:
    [event] = json.loads(json.dumps(events_to_dict([FULL])))["events"]

    assert event["pick_direction"] == "up"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"format": "something-else"}, "not a"),
        ({"version": 2}, "not a"),
        ({"timing": "beats"}, "timing"),
    ],
)
def test_rejects_unknown_documents(change: dict[str, Any], message: str) -> None:
    document = {**events_to_dict([RAW]), **change}

    with pytest.raises(ValueError, match=message):
        events_from_dict(document)


def test_rejects_unknown_event_fields() -> None:
    document = events_to_dict([RAW])
    document["events"][0]["beat"] = 1.5

    with pytest.raises(ValueError, match=r"event 0: unknown fields \['beat'\]"):
        events_from_dict(document)


def test_rejects_invalid_events() -> None:
    document = events_to_dict([RAW])
    document["events"][0]["offset_seconds"] = 0.1  # before onset

    with pytest.raises(ValueError, match="event 0"):
        events_from_dict(document)


# --- guitar object ---------------------------------------------------------------------------


def test_guitar_is_recorded_when_given() -> None:
    capo_2 = GuitarConfig(STANDARD_TUNING, capo=2, max_fret=24)

    document = json.loads(json.dumps(events_to_dict([RAW], guitar=capo_2)))

    assert document["guitar"] == {"open_strings": list(STANDARD_TUNING), "capo": 2, "max_fret": 24}
    assert guitar_from_dict(document) == capo_2
    assert events_from_dict(document) == [RAW]  # events still load as before


def test_guitar_is_absent_when_not_given() -> None:
    document = events_to_dict([RAW])

    assert "guitar" not in document
    assert guitar_from_dict(document) is None


@pytest.mark.parametrize(
    "guitar",
    [
        {"open_strings": [64, 59], "capo": 30, "max_fret": 22},  # capo off the neck
        {"open_strings": [], "capo": 0, "max_fret": 22},
        {"capo": 0, "max_fret": 22},
    ],
)
def test_invalid_guitar_object(guitar: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="invalid guitar"):
        guitar_from_dict({**events_to_dict([RAW]), "guitar": guitar})
