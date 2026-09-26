"""The quantized (notated-rhythm) view of a performance.

These objects are an *interpretation* of raw ``PerformanceEvent``s under a tempo, meter, and grid.
Each ``QuantizedEvent`` keeps its source event, so raw seconds are never lost or overwritten.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

from guitar_transcription.domain import PerformanceEvent
from guitar_transcription.rhythm.values import (
    NoteValue,
    RhythmicDuration,
    TimeSignature,
    rhythmic_duration,
)


@dataclass(frozen=True, slots=True)
class QuantizedEvent:
    """A note placed on the musical grid.

    Attributes:
        source: The raw event this was derived from (raw timing lives here, untouched).
        onset_quarters: Start, in quarter notes from the downbeat of measure 1.
        duration_quarters: Length in quarter notes; always > 0.
    """

    source: PerformanceEvent
    onset_quarters: Fraction
    duration_quarters: Fraction

    def __post_init__(self) -> None:
        for name in ("onset_quarters", "duration_quarters"):
            if not isinstance(getattr(self, name), Fraction):
                raise TypeError(f"{name} must be a Fraction, got {getattr(self, name)!r}")
        if self.onset_quarters < 0:
            raise ValueError(f"onset_quarters must be >= 0, got {self.onset_quarters}")
        if self.duration_quarters <= 0:
            raise ValueError(f"duration_quarters must be > 0, got {self.duration_quarters}")

    @property
    def pitch_midi(self) -> int:
        return self.source.pitch_midi

    @property
    def offset_quarters(self) -> Fraction:
        return self.onset_quarters + self.duration_quarters

    @property
    def rhythmic_duration(self) -> RhythmicDuration | None:
        """Single note symbol for the duration, or ``None`` if it needs tied notes."""
        return rhythmic_duration(self.duration_quarters)


@dataclass(frozen=True, slots=True)
class Rest:
    """A span, in quarter notes, during which no quantized note sounds."""

    onset_quarters: Fraction
    duration_quarters: Fraction

    @property
    def offset_quarters(self) -> Fraction:
        return self.onset_quarters + self.duration_quarters


@dataclass(frozen=True, slots=True)
class QuantizedPerformance:
    """Quantized events plus the tempo, meter, and grid used to produce them.

    ``events`` are sorted by (onset, pitch). Overlapping notes are kept as-is (e.g. a held bass
    note under a melody); how to voice them is a notation decision.

    ``origin_seconds`` is the raw recording time of quarter 0, the downbeat of measure 1, so
    ``seconds = origin_seconds + quarters * 60 / quarter_note_bpm`` maps musical time back to the
    recording (e.g. for audio/video alignment). It is negative when a pickup bar was added before
    a downbeat near the start of the recording.
    """

    events: tuple[QuantizedEvent, ...]
    quarter_note_bpm: float
    time_signature: TimeSignature
    grid: NoteValue
    origin_seconds: float = 0.0

    def seconds_at(self, quarters: Fraction) -> float:
        """Raw recording time of a musical position (inverse of quantization, before snapping)."""
        return self.origin_seconds + float(quarters) * 60 / self.quarter_note_bpm

    @property
    def end_quarters(self) -> Fraction:
        """Latest offset of any event; 0 when empty."""
        return max((event.offset_quarters for event in self.events), default=Fraction(0))

    @property
    def measure_count(self) -> int:
        """Measures needed to hold every event (the last one may be partly empty)."""
        return math.ceil(self.end_quarters / self.time_signature.measure_quarters)

    def rests(self) -> tuple[Rest, ...]:
        """Gaps between time zero and ``end_quarters`` where no note sounds.

        A rest can cross a barline; splitting it per measure is notation's job. Trailing space in
        the final measure is not included.
        """
        return _gaps(self.events)


def _gaps(events: Iterable[QuantizedEvent]) -> tuple[Rest, ...]:
    rests = []
    covered_until = Fraction(0)
    for event in sorted(events, key=lambda e: e.onset_quarters):
        if event.onset_quarters > covered_until:
            rests.append(Rest(covered_until, event.onset_quarters - covered_until))
        covered_until = max(covered_until, event.offset_quarters)
    return tuple(rests)
