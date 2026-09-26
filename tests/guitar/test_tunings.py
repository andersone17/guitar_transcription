import pytest

from guitar_transcription.domain import STANDARD_TUNING, GuitarConfig
from guitar_transcription.guitar import (
    NAMED_TUNINGS,
    describe_tuning,
    parse_tuning,
    pitch_for_position,
)


def test_written_tuning_is_reversed_into_string_1_first_order() -> None:
    # Guitarists write low string first; GuitarConfig lists string 1 (high E) first.
    assert parse_tuning("E2,A2,D3,G3,B3,E4") == STANDARD_TUNING
    config = GuitarConfig(parse_tuning("E2,A2,D3,G3,B3,E4"))
    assert pitch_for_position(config, 6, 0) == 40  # string 6 = low E
    assert pitch_for_position(config, 1, 0) == 64  # string 1 = high E


@pytest.mark.parametrize(
    "text", ["E2 A2 D3 G3 B3 E4", "E2, A2, D3, G3, B3, E4", " e2,a2,d3,g3,b3,e4 "]
)
def test_separators_and_case(text: str) -> None:
    assert parse_tuning(text) == STANDARD_TUNING


def test_drop_d_lowers_string_6_only() -> None:
    tuning = parse_tuning("D2,A2,D3,G3,B3,E4")

    assert tuning == NAMED_TUNINGS["drop-d"]
    assert tuning[5] == 38 and tuning[:5] == STANDARD_TUNING[:5]


def test_seven_string() -> None:
    tuning = parse_tuning("B1,E2,A2,D3,G3,B3,E4")

    assert len(tuning) == 7
    assert tuning[6] == 35  # string 7 = low B


def test_reentrant_tuning_keeps_string_positions() -> None:
    # Nashville: strings 6..3 an octave up. Written from string 6 to 1; not sorted by pitch.
    tuning = parse_tuning("E3,A3,D4,G4,B3,E4")

    assert tuning == (64, 59, 67, 62, 57, 52)
    assert tuning[2] > tuning[0]  # string 3 (G4) is higher than string 1 (E4)


def test_flats_and_sharps() -> None:
    assert parse_tuning("Eb2,Ab2,Db3,Gb3,Bb3,Eb4") == NAMED_TUNINGS["half-step-down"]
    assert parse_tuning("D#2,G#2,C#3,F#3,A#3,D#4") == NAMED_TUNINGS["half-step-down"]


@pytest.mark.parametrize(
    ("name", "written"),
    [
        ("standard", "E2 A2 D3 G3 B3 E4"),
        ("drop-d", "D2 A2 D3 G3 B3 E4"),
        ("half-step-down", "D#2 G#2 C#3 F#3 A#3 D#4"),
        ("dadgad", "D2 A2 D3 G3 A3 D4"),
        ("open-g", "D2 G2 D3 G3 B3 D4"),
        ("open-d", "D2 A2 D3 F#3 A3 D4"),
    ],
)
def test_presets(name: str, written: str) -> None:
    assert describe_tuning(parse_tuning(name)) == written
    assert parse_tuning(name.upper()) == parse_tuning(written)


def test_describe_is_the_inverse_of_parse() -> None:
    assert describe_tuning(STANDARD_TUNING) == "E2 A2 D3 G3 B3 E4"
    assert parse_tuning(describe_tuning((62, 59, 55, 50, 43, 38))) == (62, 59, 55, 50, 43, 38)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("EADGBE", "not a pitch name"),  # octaves are required
        ("E A D G B E", "not a pitch name"),
        ("E2,A2,X3", "not a pitch name"),
        ("drop-q", "not a pitch name"),
        ("", "empty tuning"),
        (" , ", "empty tuning"),
    ],
)
def test_invalid_tunings_explain_the_format(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_tuning(text)


def test_error_lists_the_presets() -> None:
    with pytest.raises(ValueError, match="standard, drop-d"):
        parse_tuning("EADGBE")
