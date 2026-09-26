"""MIDI pitch helpers.

Pitch is stored as a MIDI note number everywhere in the domain; names are derived on demand so a
stored name can never disagree with the number it describes.
"""

import re

MIDI_MIN = 0
MIDI_MAX = 127

# Sharps only: spelling (C# vs Db) depends on key context, which only notation knows about.
_PITCH_CLASS_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

_LETTER_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACCIDENTAL_SEMITONES = {"": 0, "#": 1, "b": -1}
_NAME_PATTERN = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")


def validate_midi_pitch(pitch: int, what: str = "pitch") -> None:
    """Raise if ``pitch`` is not an ``int`` in the MIDI range 0..127.

    Floats (including numpy scalars) are rejected on purpose: backend adapters must convert
    third-party output to plain Python types at the boundary.
    """
    if isinstance(pitch, bool) or not isinstance(pitch, int):
        raise TypeError(f"{what} must be an int MIDI note number, got {pitch!r}")
    if not MIDI_MIN <= pitch <= MIDI_MAX:
        raise ValueError(f"{what} must be in {MIDI_MIN}..{MIDI_MAX}, got {pitch}")


def pitch_name(pitch: int) -> str:
    """Return the scientific pitch name of a MIDI note, e.g. 60 -> "C4", 64 -> "E4", 61 -> "C#4".

    Uses the common convention that middle C (MIDI 60) is C4, so MIDI 0 is "C-1".
    """
    validate_midi_pitch(pitch)
    octave, pitch_class = divmod(pitch, 12)
    return f"{_PITCH_CLASS_NAMES[pitch_class]}{octave - 1}"


def parse_pitch_name(name: str) -> int:
    """MIDI number of a scientific pitch name: "E2" -> 40, "F#2" -> 42, "Bb3" -> 58, "C-1" -> 0.

    Letter (any case), optional ``#`` or ``b``, then the octave (middle C is C4). Enharmonic
    spellings are accepted: "Db3" == "C#3", "B#3" == "C4", "Cb4" == "B3". Raises ``ValueError``
    for anything else or a pitch outside MIDI 0..127.
    """
    match = _NAME_PATTERN.match(name.strip())
    if match is None:
        raise ValueError(f"not a pitch name like 'E2', 'F#3' or 'Bb3': {name!r}")
    letter, accidental, octave = match.groups()
    pitch = (int(octave) + 1) * 12 + _LETTER_SEMITONES[letter.upper()]
    pitch += _ACCIDENTAL_SEMITONES[accidental]
    validate_midi_pitch(pitch, what=f"pitch {name!r}")
    return pitch
