"""MIDI pitch helpers.

Pitch is stored as a MIDI note number everywhere in the domain; names are derived on demand so a
stored name can never disagree with the number it describes.
"""

MIDI_MIN = 0
MIDI_MAX = 127

# Sharps only: spelling (C# vs Db) depends on key context, which only notation knows about.
_PITCH_CLASS_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


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
