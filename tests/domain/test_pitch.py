import pytest

from guitar_transcription.domain.pitch import parse_pitch_name, pitch_name, validate_midi_pitch


@pytest.mark.parametrize(
    ("midi", "name"),
    [
        (0, "C-1"),
        (12, "C0"),
        (40, "E2"),
        (45, "A2"),
        (59, "B3"),
        (60, "C4"),
        (61, "C#4"),
        (64, "E4"),
        (69, "A4"),
        (70, "A#4"),
        (71, "B4"),
        (72, "C5"),
        (127, "G9"),
    ],
)
def test_pitch_name(midi: int, name: str) -> None:
    assert pitch_name(midi) == name


def test_pitch_names_are_unique_across_the_midi_range() -> None:
    names = [pitch_name(midi) for midi in range(128)]

    assert len(set(names)) == 128


@pytest.mark.parametrize("midi", [-1, 128])
def test_out_of_range_pitch_is_rejected(midi: int) -> None:
    with pytest.raises(ValueError, match="0..127"):
        pitch_name(midi)


@pytest.mark.parametrize("value", [64.0, "64", True, None])
def test_non_int_pitch_is_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="int MIDI note number"):
        validate_midi_pitch(value)  # type: ignore[arg-type]


def test_error_message_names_the_field() -> None:
    with pytest.raises(ValueError, match="open pitch of string 3"):
        validate_midi_pitch(200, what="open pitch of string 3")


# --- parse_pitch_name ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "midi"),
    [
        ("E2", 40),
        ("A2", 45),
        ("E4", 64),
        ("C4", 60),
        ("F#2", 42),
        ("Bb3", 58),
        ("Eb2", 39),
        ("db3", 49),
        ("B1", 35),
        ("C-1", 0),
        ("G9", 127),
        (" D2 ", 38),
    ],
)
def test_parse_pitch_name(name: str, midi: int) -> None:
    assert parse_pitch_name(name) == midi


def test_enharmonic_spellings_are_the_same_pitch() -> None:
    assert parse_pitch_name("C#3") == parse_pitch_name("Db3")
    assert parse_pitch_name("B#3") == parse_pitch_name("C4")
    assert parse_pitch_name("Cb4") == parse_pitch_name("B3")


def test_round_trips_with_pitch_name_for_every_midi_note() -> None:
    assert all(parse_pitch_name(pitch_name(m)) == m for m in range(128))


@pytest.mark.parametrize("name", ["E", "H2", "E#", "E2.5", "", "EE2", "E 2", "2E"])
def test_rejects_malformed_names(name: str) -> None:
    with pytest.raises(ValueError, match="not a pitch name"):
        parse_pitch_name(name)


@pytest.mark.parametrize("name", ["Cb-1", "G#9", "C10"])
def test_rejects_names_outside_midi(name: str) -> None:
    with pytest.raises(ValueError, match="0..127"):
        parse_pitch_name(name)
