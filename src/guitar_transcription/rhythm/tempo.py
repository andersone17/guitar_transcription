"""Tempo estimation interface (project-owned) and backend-independent tempo arithmetic.

Estimation is optional: an explicit, user-supplied tempo always wins (see ``resolve_tempo``).
Backends (e.g. ``rhythm.backends.librosa_tempo``) implement ``TempoEstimator``; nothing else here
depends on a particular library, and quantization only ever receives a plain BPM number.

Known limitation of every beat tracker: the *metrical level* is ambiguous. A track at 90 BPM can
be reported as 180 (every eighth note) or 45, and which one is "right" depends on how the music is
meant to be written. Estimates are therefore presented with their half/double alternatives.
"""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from typing import Protocol

MIN_BEATS = 4
"""Fewer detected beats than this is too little evidence for a tempo."""


class TempoEstimationError(Exception):
    """Tempo could not be estimated (backend missing, unreadable audio, or too few beats)."""


@dataclass(frozen=True, slots=True)
class TempoEstimate:
    """A detected pulse rate and the beat times it was derived from.

    Attributes:
        bpm: Detected pulses per minute. Callers decide which note value the pulse is (the CLI
            treats it as a quarter note).
        beat_times_seconds: Detected beat positions in raw recording time, ascending. Kept for
            later use (e.g. aligning the grid to the first beat); not used by quantization yet.

    There is deliberately no confidence field: the current backend doesn't produce a meaningful one.
    """

    bpm: float
    beat_times_seconds: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (math.isfinite(self.bpm) and self.bpm > 0):
            raise ValueError(f"bpm must be a positive number, got {self.bpm}")
        times = self.beat_times_seconds
        if any(t < 0 or not math.isfinite(t) for t in times):
            raise ValueError("beat times must be finite and >= 0")
        if any(b < a for a, b in zip(times, times[1:], strict=False)):
            raise ValueError("beat times must be ascending")

    @property
    def half_time_bpm(self) -> float:
        return self.bpm / 2

    @property
    def double_time_bpm(self) -> float:
        return self.bpm * 2


class TempoEstimator(Protocol):
    def estimate(self, audio_path: str | PathLike[str]) -> TempoEstimate:
        """Estimate the tempo of a local audio file.

        Raises:
            FileNotFoundError: the file does not exist.
            TempoEstimationError: the tempo could not be estimated.
        """
        ...


def tempo_from_beat_times(beat_times: Sequence[float]) -> float:
    """Average tempo (BPM) of a beat sequence, robust to a few missed beats.

    Beat trackers report beats on a coarse frame grid (~23 ms), so the tempo implied by one
    interval, or by a tracker's own tempo histogram, can be off by 2-3%, which drifts by a beat
    every few dozen beats. Instead each beat is numbered relative to the previous one: an
    interval of about k median intervals advances the count by k (so a skipped beat doesn't
    shift later numbers), and the tempo is the least-squares slope of time against beat number.
    Numbering locally matters: counting from the first beat with a slightly-off median interval
    would drift by a whole beat over a long take.
    """
    if len(beat_times) < MIN_BEATS:
        raise TempoEstimationError(
            f"need at least {MIN_BEATS} beats to estimate a tempo, got {len(beat_times)}"
        )
    intervals = [b - a for a, b in zip(beat_times, beat_times[1:], strict=False)]
    typical = statistics.median(intervals)
    if typical <= 0:
        raise TempoEstimationError("beat times do not advance")
    numbers = [0]
    for interval in intervals:
        numbers.append(numbers[-1] + max(1, round(interval / typical)))
    mean_n = statistics.fmean(numbers)
    mean_t = statistics.fmean(beat_times)
    spread = sum((n - mean_n) ** 2 for n in numbers)
    if spread == 0:
        raise TempoEstimationError("beat times are all at the same position")
    seconds_per_beat = (
        sum((n - mean_n) * (t - mean_t) for n, t in zip(numbers, beat_times, strict=True)) / spread
    )
    return 60.0 / seconds_per_beat


def resolve_tempo(
    audio_path: str | PathLike[str],
    *,
    explicit_bpm: float | None,
    estimator: TempoEstimator | None,
) -> tuple[float, TempoEstimate | None]:
    """Pick the tempo to quantize with: the explicit BPM if given, otherwise an estimate.

    The estimator is not called at all when an explicit tempo is given. Returns the BPM and the
    estimate it came from (``None`` for an explicit tempo).
    """
    if explicit_bpm is not None:
        if not (math.isfinite(explicit_bpm) and explicit_bpm > 0):
            raise ValueError(f"explicit tempo must be a positive number, got {explicit_bpm}")
        return explicit_bpm, None
    if estimator is None:
        raise ValueError("no tempo given and no estimator to infer one")
    estimate = estimator.estimate(audio_path)
    return estimate.bpm, estimate
