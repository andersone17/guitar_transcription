"""Grid quantization of raw performance timing, with tempo and meter supplied by the caller.

How time maps to the grid (rhythm level 1, see PLAN.md section 2a):

- Time zero of the recording is the downbeat of measure 1. There is no pickup and no lead-in
  offset, so silence before the first note becomes a leading rest.
- The tempo is constant and counts **quarter notes** per minute, whatever the meter, so
  ``quarters = seconds * quarter_note_bpm / 60``.
- Onsets and offsets are each rounded to the nearest grid line (halfway rounds later). A note
  that would collapse to zero length gets one grid step. Durations are therefore multiples of the
  grid, not necessarily single note symbols; ties are left to notation.

With a sixteenth grid at 120 BPM, a grid step is 125 ms, so timing errors under ±62.5 ms land on the
intended sixteenth. Larger errors land on a neighbouring one.
"""

import math
from collections.abc import Iterable
from fractions import Fraction

from guitar_transcription.domain import PerformanceEvent
from guitar_transcription.rhythm.quantized import QuantizedEvent, QuantizedPerformance
from guitar_transcription.rhythm.values import NoteValue, TimeSignature


def quantize(
    events: Iterable[PerformanceEvent],
    *,
    quarter_note_bpm: float,
    time_signature: TimeSignature,
    grid: NoteValue = NoteValue.SIXTEENTH,
) -> QuantizedPerformance:
    """Place raw events on a musical grid. Pure; the input events are not modified.

    Raises ``ValueError`` if the tempo isn't a positive finite number, or if measures don't
    contain a whole number of grid steps (e.g. a quarter-note grid in 6/8).
    """
    if not (math.isfinite(quarter_note_bpm) and quarter_note_bpm > 0):
        raise ValueError(f"quarter_note_bpm must be a positive number, got {quarter_note_bpm}")
    check_grid(time_signature, grid)

    quantized = []
    for event in events:
        onset = snap_to_grid(seconds_to_quarters(event.onset_seconds, quarter_note_bpm), grid)
        offset = snap_to_grid(seconds_to_quarters(event.offset_seconds, quarter_note_bpm), grid)
        duration = max(offset - onset, grid.quarters)
        quantized.append(QuantizedEvent(event, onset, duration))
    quantized.sort(key=lambda e: (e.onset_quarters, e.pitch_midi))

    return QuantizedPerformance(
        events=tuple(quantized),
        quarter_note_bpm=quarter_note_bpm,
        time_signature=time_signature,
        grid=grid,
    )


def check_grid(time_signature: TimeSignature, grid: NoteValue) -> None:
    """Raise ``ValueError`` unless a measure holds a whole number of grid steps."""
    if time_signature.measure_quarters % grid.quarters != 0:
        raise ValueError(
            f"a {grid.name.lower()} grid does not divide a {time_signature} measure evenly; "
            "barlines would fall between grid lines"
        )


def seconds_to_quarters(seconds: float, quarter_note_bpm: float) -> float:
    """Unquantized musical position of a raw time, in quarter notes (float, not yet exact)."""
    return seconds * quarter_note_bpm / 60


def snap_to_grid(quarters: float, grid: NoteValue) -> Fraction:
    """Nearest grid line as an exact ``Fraction``; exactly halfway rounds to the later line."""
    steps = math.floor(quarters / grid.quarters + 0.5)
    return steps * grid.quarters
