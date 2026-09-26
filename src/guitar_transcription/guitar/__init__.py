"""Instrument reasoning: tuning, capo, and pitch -> string/fret candidates.

Depends only on ``domain``. Guitar-specific physical rules live here (and in ``domain``'s
``GuitarConfig``), not in ``audio``, ``rhythm``, ``notation``, or future vision modules.
"""

from guitar_transcription.guitar.positions import (
    candidate_positions,
    pitch_for_position,
    pitch_range,
)
from guitar_transcription.guitar.tunings import NAMED_TUNINGS, describe_tuning, parse_tuning

__all__ = [
    "NAMED_TUNINGS",
    "candidate_positions",
    "describe_tuning",
    "parse_tuning",
    "pitch_for_position",
    "pitch_range",
]
