"""``PerformanceEvent``: one sounded note, the central domain object.

An event records what physically happened, in seconds and MIDI pitch. Anything not yet inferred
(string/fret, pick direction, per-modality confidences) is ``None``, never a guessed default.

Times are raw performance timing and are never snapped to a beat grid. Musical rhythm (beats,
measures, note values) is inferred by ``rhythm`` into separate objects that reference these events;
see PLAN.md section 2a.
"""

import math
from dataclasses import dataclass
from enum import StrEnum

from guitar_transcription.domain.pitch import pitch_name, validate_midi_pitch
from guitar_transcription.domain.position import FretboardPosition


class PickDirection(StrEnum):
    DOWN = "down"
    UP = "up"


@dataclass(frozen=True, slots=True)
class PerformanceEvent:
    """A single note as performed.

    Attributes:
        onset_seconds: Start time, in seconds from the start of the recording.
        offset_seconds: End time, in seconds; strictly after ``onset_seconds``.
        pitch_midi: Sounding MIDI note number (0..127).
        velocity: Normalized loudness in 0..1. Notation maps it to MIDI velocity when exporting.
        string: 1-based string number (1 = top line of tablature). Set together with ``fret``.
        fret: Physical fret (capo-inclusive; 0 = open string without capo). Set with ``string``.
        pick_direction: Picking stroke direction, when known.
        audio_confidence: Evidence strength from audio, 0..1.
        fretting_confidence: Evidence strength from fretting-hand vision, 0..1.
        picking_confidence: Evidence strength from picking-hand vision, 0..1.
        confidence: Overall confidence after combining evidence, 0..1.

    Whether ``string``/``fret`` are playable and consistent with ``pitch_midi`` depends on the
    instrument, so that is checked by ``Performance``, which knows the ``GuitarConfig``.
    """

    onset_seconds: float
    offset_seconds: float
    pitch_midi: int
    velocity: float | None = None
    string: int | None = None
    fret: int | None = None
    pick_direction: PickDirection | None = None
    audio_confidence: float | None = None
    fretting_confidence: float | None = None
    picking_confidence: float | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        for name in ("onset_seconds", "offset_seconds"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite, got {getattr(self, name)}")
        if self.onset_seconds < 0:
            raise ValueError(f"onset_seconds must be >= 0, got {self.onset_seconds}")
        if self.offset_seconds <= self.onset_seconds:
            raise ValueError(
                f"offset_seconds ({self.offset_seconds}) must be after "
                f"onset_seconds ({self.onset_seconds})"
            )
        validate_midi_pitch(self.pitch_midi, what="pitch_midi")

        if (self.string is None) != (self.fret is None):
            raise ValueError("string and fret must both be set or both be None")
        if self.string is not None and self.fret is not None:
            FretboardPosition(self.string, self.fret)  # validates types and lower bounds

        for name in (
            "velocity",
            "audio_confidence",
            "fretting_confidence",
            "picking_confidence",
            "confidence",
        ):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @property
    def pitch_name(self) -> str:
        """Scientific pitch name, e.g. "E4"; derived from ``pitch_midi``."""
        return pitch_name(self.pitch_midi)

    @property
    def duration_seconds(self) -> float:
        return self.offset_seconds - self.onset_seconds
