"""Musical-time value types: note values and time signatures.

All musical time in ``rhythm`` is measured in **quarter notes** as exact ``Fraction``s, so a dotted
eighth is ``Fraction(3, 4)`` and nothing drifts the way float beats would.
"""

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction


class NoteValue(Enum):
    """Undotted note values, valued by their length in quarter notes."""

    WHOLE = Fraction(4)
    HALF = Fraction(2)
    QUARTER = Fraction(1)
    EIGHTH = Fraction(1, 2)
    SIXTEENTH = Fraction(1, 4)

    @property
    def quarters(self) -> Fraction:
        return self.value


@dataclass(frozen=True, slots=True)
class RhythmicDuration:
    """A duration expressible as one note symbol: a note value, optionally dotted (x 1.5)."""

    base: NoteValue
    dotted: bool = False

    @property
    def quarters(self) -> Fraction:
        return self.base.quarters * (Fraction(3, 2) if self.dotted else 1)


def rhythmic_duration(quarters: Fraction) -> RhythmicDuration | None:
    """Name ``quarters`` as a single (possibly dotted) note value, or ``None`` if it needs ties.

    E.g. 1 -> quarter, 3/4 -> dotted eighth, 5/4 -> None (quarter tied to sixteenth). Splitting
    such durations into tied symbols is notation's job.
    """
    for base in NoteValue:
        for dotted in (False, True):
            candidate = RhythmicDuration(base, dotted)
            if candidate.quarters == quarters:
                return candidate
    return None


@dataclass(frozen=True, slots=True)
class TimeSignature:
    """A fixed meter such as 4/4, 3/4, or 6/8.

    Measure 1 starts at quarter 0 (time zero); there is no pickup measure in this version.
    """

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        for name in ("numerator", "denominator"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {value!r}")
        if self.numerator < 1:
            raise ValueError(f"numerator must be >= 1, got {self.numerator}")
        if self.denominator < 1 or self.denominator & (self.denominator - 1):
            raise ValueError(f"denominator must be a power of two, got {self.denominator}")

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"

    @property
    def measure_quarters(self) -> Fraction:
        """Length of one measure in quarter notes (4/4 -> 4, 3/4 -> 3, 6/8 -> 3, 2/2 -> 4)."""
        return Fraction(4 * self.numerator, self.denominator)

    def locate(self, quarters: Fraction) -> tuple[int, Fraction]:
        """Return (1-based measure number, position within that measure in quarter notes)."""
        if quarters < 0:
            raise ValueError(f"musical time must be >= 0, got {quarters}")
        index, position = divmod(Fraction(quarters), self.measure_quarters)
        return int(index) + 1, position
