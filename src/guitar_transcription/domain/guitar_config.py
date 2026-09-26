"""The physical configuration of the instrument being transcribed.

``GuitarConfig`` lives in ``domain`` (not ``guitar``) because a ``Performance`` is only meaningful
relative to the instrument it was played on, and ``domain`` may not import other packages.
Instrument *reasoning* (tuning parsing, candidate positions) belongs in ``guitar``.
"""

from dataclasses import dataclass

from guitar_transcription.domain.pitch import MIDI_MAX, validate_midi_pitch
from guitar_transcription.domain.position import require_int


@dataclass(frozen=True, slots=True)
class GuitarConfig:
    """Strings, tuning, capo, and fret range.

    Attributes:
        open_strings: MIDI pitch of each unfretted string without capo, listed from string 1
            (top line of tablature) downward. For a conventionally tuned guitar that means
            highest-pitched first: standard tuning is (E4, B3, G3, D3, A2, E2), the *reverse* of
            the low-to-high way tunings are usually written. Order is not enforced, so re-entrant
            tunings (e.g. Nashville tuning) are representable.
        capo: Fret the capo is clamped at; 0 means no capo. Fret numbers elsewhere are physical
            (capo-inclusive), so with ``capo=2`` the lowest playable fret is 2.
        max_fret: Highest playable fret.
    """

    open_strings: tuple[int, ...]
    capo: int = 0
    max_fret: int = 22

    def __post_init__(self) -> None:
        if not isinstance(self.open_strings, tuple):
            raise TypeError(f"open_strings must be a tuple, got {type(self.open_strings).__name__}")
        if not self.open_strings:
            raise ValueError("a guitar needs at least one string")
        for index, pitch in enumerate(self.open_strings):
            validate_midi_pitch(pitch, what=f"open pitch of string {index + 1}")
        require_int(self.capo, "capo")
        require_int(self.max_fret, "max_fret")
        if self.max_fret < 1:
            raise ValueError(f"max_fret must be >= 1, got {self.max_fret}")
        if not 0 <= self.capo < self.max_fret:
            raise ValueError(f"capo must be in 0..{self.max_fret - 1}, got {self.capo}")
        if max(self.open_strings) + self.max_fret > MIDI_MAX:
            raise ValueError("highest string at max_fret would exceed the MIDI pitch range")

    @property
    def num_strings(self) -> int:
        return len(self.open_strings)

    @property
    def min_fret(self) -> int:
        """Lowest fret that can sound: the capo position (0 without a capo)."""
        return self.capo

    def pitch_at(self, string: int, fret: int) -> int:
        """MIDI pitch sounded by ``string`` (1-based) at physical ``fret``.

        Raises ``ValueError`` if the position is not playable on this configuration. This is the
        single implementation of position -> pitch; ``guitar.pitch_for_position`` is the public API.
        """
        require_int(string, "string")
        require_int(fret, "fret")
        if not 1 <= string <= self.num_strings:
            raise ValueError(f"string must be in 1..{self.num_strings}, got {string}")
        if not self.min_fret <= fret <= self.max_fret:
            raise ValueError(f"fret must be in {self.min_fret}..{self.max_fret}, got {fret}")
        return self.open_strings[string - 1] + fret


STANDARD_TUNING: tuple[int, ...] = (64, 59, 55, 50, 45, 40)
"""Standard six-string tuning E4 B3 G3 D3 A2 E2, listed from string 1 to string 6."""

STANDARD_GUITAR = GuitarConfig(open_strings=STANDARD_TUNING)
"""Six strings in standard tuning, no capo, 22 frets."""
