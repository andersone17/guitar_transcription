import dataclasses

import pytest

from guitar_transcription.domain import STANDARD_GUITAR, STANDARD_TUNING, GuitarConfig, pitch_name


def test_standard_tuning_is_e_a_d_g_b_e_from_string_1() -> None:
    assert [pitch_name(p) for p in STANDARD_TUNING] == ["E4", "B3", "G3", "D3", "A2", "E2"]


def test_standard_guitar_defaults() -> None:
    assert STANDARD_GUITAR.open_strings == STANDARD_TUNING
    assert STANDARD_GUITAR.num_strings == 6
    assert STANDARD_GUITAR.capo == 0
    assert STANDARD_GUITAR.min_fret == 0
    assert STANDARD_GUITAR.max_fret == 22


def test_num_strings_is_derived_from_tuning() -> None:
    seven_string = GuitarConfig(open_strings=(*STANDARD_TUNING, 35))  # low B1

    assert seven_string.num_strings == 7


def test_config_is_immutable_and_hashable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        STANDARD_GUITAR.capo = 2  # type: ignore[misc]

    assert hash(GuitarConfig(STANDARD_TUNING)) == hash(STANDARD_GUITAR)
    assert GuitarConfig(STANDARD_TUNING) == STANDARD_GUITAR


def test_reentrant_tuning_is_allowed() -> None:
    # Ukulele-style re-entrant G: string order is not required to be monotonic.
    config = GuitarConfig(open_strings=(69, 64, 60, 67), max_fret=15)

    assert config.num_strings == 4


@pytest.mark.parametrize(
    ("string", "fret", "expected"),
    [
        (1, 0, 64),  # open high E
        (2, 5, 64),  # B string fret 5 = E4
        (6, 0, 40),  # open low E
        (6, 22, 62),
        (3, 9, 64),
    ],
)
def test_pitch_at_standard_tuning(string: int, fret: int, expected: int) -> None:
    assert STANDARD_GUITAR.pitch_at(string, fret) == expected


def test_capo_raises_lowest_playable_fret() -> None:
    capo_2 = GuitarConfig(STANDARD_TUNING, capo=2)

    assert capo_2.min_fret == 2
    assert capo_2.pitch_at(6, 2) == 42  # "open" low string with capo 2 is F#2
    with pytest.raises(ValueError, match="fret must be in 2..22"):
        capo_2.pitch_at(6, 1)
    with pytest.raises(ValueError, match="fret must be in 2..22"):
        capo_2.pitch_at(6, 0)


@pytest.mark.parametrize(("string", "fret"), [(0, 0), (7, 0), (-1, 3)])
def test_pitch_at_rejects_missing_string(string: int, fret: int) -> None:
    with pytest.raises(ValueError, match="string must be in 1..6"):
        STANDARD_GUITAR.pitch_at(string, fret)


@pytest.mark.parametrize("fret", [-1, 23])
def test_pitch_at_rejects_fret_off_the_neck(fret: int) -> None:
    with pytest.raises(ValueError, match="fret must be in 0..22"):
        STANDARD_GUITAR.pitch_at(1, fret)


def test_rejects_empty_tuning() -> None:
    with pytest.raises(ValueError, match="at least one string"):
        GuitarConfig(open_strings=())


def test_rejects_list_tuning() -> None:
    # Must be a tuple so configs stay immutable and hashable.
    with pytest.raises(TypeError, match="tuple"):
        GuitarConfig(open_strings=[64, 59])  # type: ignore[arg-type]


def test_rejects_invalid_open_pitch() -> None:
    with pytest.raises(ValueError, match="string 2"):
        GuitarConfig(open_strings=(64, -5))
    with pytest.raises(TypeError, match="string 1"):
        GuitarConfig(open_strings=(64.0,))  # type: ignore[arg-type]


@pytest.mark.parametrize("capo", [-1, 22, 30])
def test_rejects_capo_outside_neck(capo: int) -> None:
    with pytest.raises(ValueError, match="capo"):
        GuitarConfig(STANDARD_TUNING, capo=capo)


def test_rejects_non_positive_max_fret() -> None:
    with pytest.raises(ValueError, match="max_fret"):
        GuitarConfig(STANDARD_TUNING, max_fret=0)


def test_rejects_range_above_midi() -> None:
    with pytest.raises(ValueError, match="MIDI pitch range"):
        GuitarConfig(open_strings=(120,), max_fret=10)


@pytest.mark.parametrize(("capo", "max_fret"), [(1.0, 22), (True, 22), (0, 22.5), (0, "22")])
def test_rejects_non_int_capo_or_max_fret(capo: object, max_fret: object) -> None:
    with pytest.raises(TypeError, match="must be an int"):
        GuitarConfig(STANDARD_TUNING, capo=capo, max_fret=max_fret)  # type: ignore[arg-type]
