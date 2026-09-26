"""Instrument reasoning: tuning, capo, and pitch -> string/fret candidates.

Depends only on ``domain``. Guitar-specific physical rules live here (and in ``domain``'s
``GuitarConfig``), not in ``audio``, ``rhythm``, ``notation``, or future vision modules.
"""

from guitar_transcription.guitar.positions import (
    candidate_positions,
    pitch_for_position,
    pitch_range,
)

__all__ = ["candidate_positions", "pitch_for_position", "pitch_range"]
