import dataclasses

import pytest

from guitar_transcription.domain import FretboardPosition


def test_holds_string_and_fret() -> None:
    position = FretboardPosition(string=2, fret=5)

    assert (position.string, position.fret) == (2, 5)


def test_is_immutable_hashable_and_compared_by_value() -> None:
    position = FretboardPosition(2, 5)

    with pytest.raises(dataclasses.FrozenInstanceError):
        position.fret = 6  # type: ignore[misc]
    assert position == FretboardPosition(2, 5)
    assert len({position, FretboardPosition(2, 5), FretboardPosition(1, 0)}) == 2


def test_open_string_is_fret_zero() -> None:
    assert FretboardPosition(string=1, fret=0).fret == 0


@pytest.mark.parametrize("string", [0, -1])
def test_rejects_string_below_one(string: int) -> None:
    with pytest.raises(ValueError, match="string must be >= 1"):
        FretboardPosition(string=string, fret=0)


def test_rejects_negative_fret() -> None:
    with pytest.raises(ValueError, match="fret must be >= 0"):
        FretboardPosition(string=1, fret=-1)


@pytest.mark.parametrize(("string", "fret"), [(1.0, 0), (1, 0.0), (True, 0), (1, "3")])
def test_rejects_non_int_values(string: object, fret: object) -> None:
    with pytest.raises(TypeError, match="must be an int"):
        FretboardPosition(string=string, fret=fret)  # type: ignore[arg-type]
