"""Rhythm: interprets raw performance timing (seconds) as musical time (quarter notes, measures).

Owns every timing decision (grid, meter, measure positions, rests); ``notation`` only renders the
result. Output objects reference their source ``PerformanceEvent``s and never modify them.
Pure Python; depends only on ``domain``.
"""

from guitar_transcription.rhythm.quantize import quantize
from guitar_transcription.rhythm.quantized import QuantizedEvent, QuantizedPerformance, Rest
from guitar_transcription.rhythm.values import (
    NoteValue,
    RhythmicDuration,
    TimeSignature,
    rhythmic_duration,
)

__all__ = [
    "NoteValue",
    "QuantizedEvent",
    "QuantizedPerformance",
    "Rest",
    "RhythmicDuration",
    "TimeSignature",
    "quantize",
    "rhythmic_duration",
]
