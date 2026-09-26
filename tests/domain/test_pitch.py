import pytest

from guitar_transcription.domain.pitch import pitch_name, validate_midi_pitch


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
