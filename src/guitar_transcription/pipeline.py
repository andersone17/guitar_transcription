"""Stage 1 pipeline: the one place that connects the stages.

    audio file --AudioTranscriber (range from GuitarConfig)--> PerformanceEvent[] (raw seconds)
        --Performance(events, GuitarConfig)--> events bound to (and checked against) the instrument
        --tempo (explicit, or TempoEstimator)--> quarter-note BPM
        --quantize + to_single_voice (known meter)--> QuantizedPerformance
        --write_musicxml--> MusicXML

Each stage is reached only through its own public API, and backends arrive as objects implementing
the project's protocols, so any backend can be swapped (or faked in tests). The pipeline makes no
musical decisions itself: tempo and meter come from the caller or an estimator, timing decisions
from ``rhythm``, and spelling from ``notation``. Raw events are passed along unchanged; quantized
events reference the very same objects.
"""

import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from guitar_transcription.domain import STANDARD_GUITAR, GuitarConfig, Performance
from guitar_transcription.domain.pitch import pitch_name
from guitar_transcription.domain.serialization import events_to_dict
from guitar_transcription.guitar import pitch_range
from guitar_transcription.rhythm import (
    NoteValue,
    QuantizedPerformance,
    TempoEstimate,
    TempoEstimator,
    TimeSignature,
    quantize,
    resolve_tempo,
    to_single_voice,
)

DEFAULT_GUITAR = STANDARD_GUITAR
"""Instrument assumed when the caller gives none: 6 strings, standard tuning, no capo, 22 frets."""

ESTIMATED_TEMPO_DECIMALS = 2  # beyond 0.01 BPM is noise; keeps the tempo mark readable


@dataclass(frozen=True, slots=True)
class NotationRequest:
    """What the caller knows about the music. Meter is required; tempo may be estimated.

    ``tempo_bpm`` is in quarter notes per minute. When it is given, it always wins and no
    estimator is consulted. ``downbeat_seconds`` is the raw time of a beat 1 (``None``: the first
    note); notes before it become a pickup bar.
    """

    time_signature: TimeSignature
    tempo_bpm: float | None = None
    grid: NoteValue = NoteValue.SIXTEENTH
    downbeat_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class NotationResult:
    """Everything the notation step decided, kept separate from the raw events.

    Attributes:
        tempo_bpm: Quarter-note BPM actually used for quantization.
        tempo_estimate: The estimate it came from, or ``None`` if the tempo was explicit.
        quantized: All events on the grid, before single-voice reduction.
        voiced: The single-voice version that is exported.
    """

    tempo_bpm: float
    tempo_estimate: TempoEstimate | None
    quantized: QuantizedPerformance
    voiced: QuantizedPerformance

    @property
    def shortened_count(self) -> int:
        """Notes whose notated duration the single-voice reduction cut short."""
        before = {id(event.source): event.duration_quarters for event in self.quantized.events}
        return sum(1 for e in self.voiced.events if e.duration_quarters != before[id(e.source)])

    @property
    def dropped_count(self) -> int:
        """Duplicate chord pitches removed by the single-voice reduction."""
        return len(self.quantized.events) - len(self.voiced.events)


def detection_range(guitar: GuitarConfig = DEFAULT_GUITAR) -> tuple[int, int]:
    """MIDI range worth detecting for ``guitar``: anything outside it can't have been played on it.

    Passing this to the transcriber removes, e.g., high "ghost" notes that models report from a
    plucked string's upper harmonics.
    """
    return pitch_range(guitar)


def describe_range(low: int, high: int) -> str:
    return f"{pitch_name(low)}-{pitch_name(high)}"


def notate(
    performance: Performance,
    audio_path: str | PathLike[str],
    request: NotationRequest,
    *,
    tempo_estimator: TempoEstimator | None = None,
) -> NotationResult:
    """Resolve the tempo, then quantize and reduce to one voice. Events are not modified.

    ``audio_path`` is only read by the tempo estimator, and only if no tempo was given.
    Raises ``TempoEstimationError`` if estimation fails, ``ValueError`` for an unusable grid/meter.
    """
    bpm, estimate = resolve_tempo(
        audio_path,
        explicit_bpm=request.tempo_bpm,
        estimator=tempo_estimator if request.tempo_bpm is None else None,
    )
    if estimate is not None:
        bpm = round(bpm, ESTIMATED_TEMPO_DECIMALS)
    quantized = quantize(
        performance.events,
        quarter_note_bpm=bpm,
        time_signature=request.time_signature,
        grid=request.grid,
        downbeat_seconds=request.downbeat_seconds,
    )
    return NotationResult(bpm, estimate, quantized, to_single_voice(quantized))


def write_events_json(performance: Performance, path: str | PathLike[str], *, source: str) -> Path:
    """Write raw events and the guitar they were played on as ``performance-events`` JSON."""
    document = events_to_dict(performance.events, source=source, guitar=performance.config)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n")
    return output


def write_notation(result: NotationResult, path: str | PathLike[str], *, title: str) -> Path:
    """Write the single-voice result as MusicXML (creating parent folders)."""
    # Imported here: music21 takes ~1 s to import and is only needed for notation output.
    from guitar_transcription.notation.musicxml import write_musicxml

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return write_musicxml(result.voiced, output, title=title)
