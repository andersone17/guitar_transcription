"""``FretboardPosition``: a (string, fret) location on the neck.

String numbering follows tablature convention: string 1 is the string drawn on the top line of
tab, which on a conventionally tuned guitar is the highest-pitched string (string 6 is the lowest
on a six-string). Formally, string N is ``GuitarConfig.open_strings[N - 1]``.

Fret numbers are physical (capo-inclusive): 0 is the nut, and with a capo at fret 2 the lowest
sounding fret is 2. Whether a position is playable depends on a ``GuitarConfig``, so only the
config-independent bounds are checked here.
"""

from dataclasses import dataclass


def require_int(value: int, name: str) -> None:
    """Raise ``TypeError`` unless ``value`` is a plain ``int``; ``bool`` and floats are rejected."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {value!r}")


@dataclass(frozen=True, slots=True)
class FretboardPosition:
    string: int
    fret: int

    def __post_init__(self) -> None:
        require_int(self.string, "string")
        require_int(self.fret, "fret")
        if self.string < 1:
            raise ValueError(f"string must be >= 1, got {self.string}")
        if self.fret < 0:
            raise ValueError(f"fret must be >= 0, got {self.fret}")
