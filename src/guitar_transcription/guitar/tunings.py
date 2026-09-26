"""Tuning input: named presets and note lists as guitarists write them.

Guitarists write tunings from the lowest-sounding (thickest) string up: standard is
"E2 A2 D3 G3 B3 E4". ``GuitarConfig.open_strings`` is in *tablature* order, string 1 first, which is
the reverse. The conversion reverses by position, not by sorting pitches, so re-entrant tunings
(e.g. Nashville, where string 3 is higher than string 1) keep their string numbers.
"""

import re

from guitar_transcription.domain.guitar_config import STANDARD_TUNING
from guitar_transcription.domain.pitch import parse_pitch_name, pitch_name

NAMED_TUNINGS: dict[str, tuple[int, ...]] = {
    "standard": STANDARD_TUNING,  # E2 A2 D3 G3 B3 E4
    "drop-d": (64, 59, 55, 50, 45, 38),  # D2 A2 D3 G3 B3 E4
    "half-step-down": (63, 58, 54, 49, 44, 39),  # Eb2 Ab2 Db3 Gb3 Bb3 Eb4
    "dadgad": (62, 57, 55, 50, 45, 38),  # D2 A2 D3 G3 A3 D4
    "open-g": (62, 59, 55, 50, 43, 38),  # D2 G2 D3 G3 B3 D4
    "open-d": (62, 57, 54, 50, 45, 38),  # D2 A2 D3 F#3 A3 D4
}
"""Common six-string tunings, in ``GuitarConfig.open_strings`` order (string 1 first)."""


def parse_tuning(text: str) -> tuple[int, ...]:
    """Open-string pitches in ``GuitarConfig`` order from a preset name or a written note list.

    ``text`` is a name from ``NAMED_TUNINGS`` (case-insensitive) or pitch names from the lowest
    string (string 6 on a six-string) to string 1, separated by commas and/or spaces, e.g.
    ``"E2,A2,D3,G3,B3,E4"`` or ``"B1 E2 A2 D3 G3 B3 E4"`` (seven strings). Octaves are required,
    because "E" alone doesn't say which E. Raises ``ValueError`` with a usable message.
    """
    key = text.strip().lower()
    if key in NAMED_TUNINGS:
        return NAMED_TUNINGS[key]
    names = [part for part in re.split(r"[,\s]+", text.strip()) if part]
    if not names:
        raise ValueError("empty tuning")
    try:
        written = [parse_pitch_name(name) for name in names]
    except ValueError as error:
        presets = ", ".join(NAMED_TUNINGS)
        raise ValueError(
            f"invalid tuning {text!r}: {error}. Give notes with octaves from the lowest string up "
            f"(e.g. E2,A2,D3,G3,B3,E4) or a preset: {presets}"
        ) from None
    return tuple(reversed(written))


def describe_tuning(open_strings: tuple[int, ...]) -> str:
    """Tuning as guitarists write it, lowest string first: "E2 A2 D3 G3 B3 E4"."""
    return " ".join(pitch_name(pitch) for pitch in reversed(open_strings))
