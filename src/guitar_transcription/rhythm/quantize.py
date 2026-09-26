"""Grid quantization of raw performance timing, with tempo and meter supplied by the caller.

How time maps to the grid (rhythm level 1, see PLAN.md section 2a):

- A **downbeat** (raw seconds of some beat 1) anchors the grid. By default it is the first note's
  onset, so silence before playing starts doesn't shift barlines. Callers can pass the real one
  (``downbeat_seconds``), e.g. when the piece starts with a pickup.
- Notes that land *before* that downbeat (after snapping) form a pickup: the whole piece moves
  later by whole measures, so measure 1 holds leading rests and then the pickup notes. (A partial
  first measure, i.e. true anacrusis notation, is a later notation refinement.) Deciding this on
  snapped positions means a chord played slightly early doesn't create an empty extra measure.
- The tempo is constant and counts **quarter notes** per minute, whatever the meter, so
  ``quarters = (seconds - downbeat) * quarter_note_bpm / 60`` (plus any pickup shift).
- Near-simultaneous onsets (a strum) are first grouped into chords (see ``rhythm.chords``); a
  group is anchored at its earliest raw onset and snapped as one, so a strum can't be split
  across grid lines. Each note keeps its own offset.
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
from guitar_transcription.rhythm.chords import DEFAULT_CHORD_WINDOW_SECONDS, group_simultaneous
from guitar_transcription.rhythm.quantized import QuantizedEvent, QuantizedPerformance
from guitar_transcription.rhythm.values import NoteValue, TimeSignature


def quantize(
    events: Iterable[PerformanceEvent],
    *,
    quarter_note_bpm: float,
    time_signature: TimeSignature,
    grid: NoteValue = NoteValue.SIXTEENTH,
    chord_window_seconds: float = DEFAULT_CHORD_WINDOW_SECONDS,
    downbeat_seconds: float | None = None,
) -> QuantizedPerformance:
    """Place raw events on a musical grid. Pure; the input events are not modified.

    ``chord_window_seconds`` is the longest strum treated as one chord (0 disables grouping). It
    is capped at 0.8 of a grid step, so notes played a full step apart are never merged.

    ``downbeat_seconds`` is the raw time of a beat 1; ``None`` means the first note's onset (0 s
    when there are no notes).

    Raises ``ValueError`` if the tempo isn't a positive finite number, or if measures don't
    contain a whole number of grid steps (e.g. a quarter-note grid in 6/8).
    """
    if not (math.isfinite(quarter_note_bpm) and quarter_note_bpm > 0):
        raise ValueError(f"quarter_note_bpm must be a positive number, got {quarter_note_bpm}")
    check_grid(time_signature, grid)
    if not (math.isfinite(chord_window_seconds) and chord_window_seconds >= 0):
        raise ValueError(f"chord_window_seconds must be >= 0, got {chord_window_seconds}")
    if downbeat_seconds is not None and not (
        math.isfinite(downbeat_seconds) and downbeat_seconds >= 0
    ):
        raise ValueError(f"downbeat_seconds must be >= 0, got {downbeat_seconds}")
    events = list(events)
    if downbeat_seconds is None:
        downbeat_seconds = min((e.onset_seconds for e in events), default=0.0)

    def position(seconds: float) -> Fraction:
        return snap_to_grid(seconds_to_quarters(seconds - downbeat_seconds, quarter_note_bpm), grid)

    step_seconds = float(grid.quarters) * 60 / quarter_note_bpm
    window = min(chord_window_seconds, 0.8 * step_seconds)

    placed: list[tuple[PerformanceEvent, Fraction, Fraction]] = []
    for group in group_simultaneous(events, window):
        onset = position(group[0].onset_seconds)
        for event in group:
            duration = max(position(event.offset_seconds) - onset, grid.quarters)
            placed.append((event, onset, duration))

    # Pickup: shift by whole measures so nothing starts before measure 1.
    earliest = min((onset for _, onset, _ in placed), default=Fraction(0))
    pickup_measures = math.ceil(-earliest / time_signature.measure_quarters) if earliest < 0 else 0
    shift = pickup_measures * time_signature.measure_quarters

    quantized = [
        QuantizedEvent(event, onset + shift, duration) for event, onset, duration in placed
    ]
    quantized.sort(key=lambda e: (e.onset_quarters, e.pitch_midi))

    return QuantizedPerformance(
        events=tuple(quantized),
        quarter_note_bpm=quarter_note_bpm,
        time_signature=time_signature,
        grid=grid,
        origin_seconds=downbeat_seconds - float(shift) * 60 / quarter_note_bpm,
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
