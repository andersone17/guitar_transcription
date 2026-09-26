"""Rhythm: interprets raw performance timing (seconds) as musical time (quarter notes, measures).

Owns every timing decision (tempo, grid, meter, measure positions, rests); ``notation`` only
renders the result. Output objects reference their source ``PerformanceEvent``s and never modify
them. Pure Python and depends only on ``domain``, except optional tempo backends in
``rhythm.backends`` (lazily imported; not re-exported here).
"""

from guitar_transcription.rhythm.quantize import quantize
from guitar_transcription.rhythm.quantized import QuantizedEvent, QuantizedPerformance, Rest
from guitar_transcription.rhythm.tempo import (
    TempoEstimate,
    TempoEstimationError,
    TempoEstimator,
    resolve_tempo,
    tempo_from_beat_times,
)
from guitar_transcription.rhythm.values import (
    NoteValue,
    RhythmicDuration,
    TimeSignature,
    rhythmic_duration,
)
from guitar_transcription.rhythm.voices import is_single_voice, to_single_voice

__all__ = [
    "NoteValue",
    "QuantizedEvent",
    "QuantizedPerformance",
    "Rest",
    "RhythmicDuration",
    "TempoEstimate",
    "TempoEstimationError",
    "TempoEstimator",
    "TimeSignature",
    "is_single_voice",
    "quantize",
    "resolve_tempo",
    "rhythmic_duration",
    "tempo_from_beat_times",
    "to_single_voice",
]
