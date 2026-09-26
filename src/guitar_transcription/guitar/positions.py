"""Deterministic fretboard geometry: which (string, fret) positions produce which pitches.

This enumerates what is *physically possible* on a ``GuitarConfig``. It never decides which
candidate a guitarist actually used; that is the job of later fingering/fusion stages, which
choose among these candidates using playability, vision, or other evidence.

String numbering (see ``domain.position``): string 1 is the top line of tablature, which is the
highest-pitched string on a conventionally tuned guitar; string 6 is the lowest on a six-string.
Frets are physical (capo-inclusive).
"""

from guitar_transcription.domain.guitar_config import GuitarConfig
from guitar_transcription.domain.pitch import validate_midi_pitch
from guitar_transcription.domain.position import FretboardPosition


def pitch_for_position(config: GuitarConfig, string: int, fret: int) -> int:
    """MIDI pitch sounded at ``string`` / physical ``fret`` on ``config``.

    Raises ``ValueError`` if the string doesn't exist or the fret is below the capo or above
    ``max_fret``; raises ``TypeError`` for non-int arguments.
    """
    return config.pitch_at(string, fret)


def candidate_positions(config: GuitarConfig, pitch_midi: int) -> tuple[FretboardPosition, ...]:
    """Every position on ``config`` that sounds ``pitch_midi``, ordered by string number.

    A pitch the instrument can't produce (below the lowest capo-adjusted open string, or above
    what ``max_fret`` allows) yields an empty tuple rather than an error: transcription backends
    can legitimately report such pitches, and callers decide what to do with them. Each string
    contributes at most one candidate because a string's pitch rises strictly with fret.

    Raises ``TypeError``/``ValueError`` only if ``pitch_midi`` is not a valid MIDI note number.
    """
    validate_midi_pitch(pitch_midi, what="pitch_midi")
    candidates = []
    for string, open_pitch in enumerate(config.open_strings, start=1):
        fret = pitch_midi - open_pitch
        if config.min_fret <= fret <= config.max_fret:
            candidates.append(FretboardPosition(string=string, fret=fret))
    return tuple(candidates)


def pitch_range(config: GuitarConfig) -> tuple[int, int]:
    """Lowest and highest MIDI pitch ``config`` can sound, inclusive.

    Useful for bounding transcription (e.g. a backend's frequency range) to what the instrument
    can actually play. Not every pitch inside the range is guaranteed playable on an unusual
    tuning with large gaps between strings; use ``candidate_positions`` for exact answers.
    """
    lowest = min(config.open_strings) + config.min_fret
    highest = max(config.open_strings) + config.max_fret
    return lowest, highest
