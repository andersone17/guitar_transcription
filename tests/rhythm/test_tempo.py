"""Tempo arithmetic and precedence; no audio library needed."""

import random
from os import PathLike

import pytest

from guitar_transcription.rhythm import (
    TempoEstimate,
    TempoEstimationError,
    resolve_tempo,
    tempo_from_beat_times,
)

FRAME = 512 / 22050  # librosa's default hop: beats are reported on this ~23 ms grid


def beats(bpm: float, count: int = 40, start: float = 0.3) -> list[float]:
    return [start + i * 60 / bpm for i in range(count)]


def on_frame_grid(times: list[float]) -> list[float]:
    return [round(t / FRAME) * FRAME for t in times]


# --- TempoEstimate --------------------------------------------------------------------------


def test_estimate_holds_bpm_and_beats() -> None:
    estimate = TempoEstimate(bpm=120.0, beat_times_seconds=(0.5, 1.0, 1.5))

    assert estimate.half_time_bpm == 60.0
    assert estimate.double_time_bpm == 240.0


@pytest.mark.parametrize("bpm", [0.0, -1.0, float("nan"), float("inf")])
def test_estimate_rejects_invalid_bpm(bpm: float) -> None:
    with pytest.raises(ValueError, match="bpm"):
        TempoEstimate(bpm=bpm, beat_times_seconds=())


@pytest.mark.parametrize("times", [(1.0, 0.5), (-0.1, 0.5), (0.5, float("nan"))])
def test_estimate_rejects_invalid_beat_times(times: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="beat times"):
        TempoEstimate(bpm=120.0, beat_times_seconds=times)


# --- tempo_from_beat_times -------------------------------------------------------------------


@pytest.mark.parametrize("bpm", [60, 90, 120, 160])
def test_exact_beats(bpm: float) -> None:
    assert tempo_from_beat_times(beats(bpm)) == pytest.approx(bpm, rel=1e-9)


@pytest.mark.parametrize("bpm", [60, 90, 117, 120, 143, 160, 200])
def test_frame_quantized_beats_are_recovered_within_half_a_percent(bpm: float) -> None:
    # The motivating case: a single frame-quantized interval can be ~2-3% off.
    assert tempo_from_beat_times(on_frame_grid(beats(bpm))) == pytest.approx(bpm, rel=0.005)


@pytest.mark.parametrize("bpm", [60, 90, 120, 160])
def test_human_jitter_averages_out(bpm: float) -> None:
    rng = random.Random(bpm)
    jittered = [t + rng.uniform(-0.015, 0.015) for t in beats(bpm, count=60)]

    assert tempo_from_beat_times(jittered) == pytest.approx(bpm, rel=0.01)


def test_missed_beats_do_not_bias_the_tempo() -> None:
    times = beats(120, count=40)
    del times[25]
    del times[10]

    assert tempo_from_beat_times(times) == pytest.approx(120, rel=1e-6)


def test_too_few_beats() -> None:
    with pytest.raises(TempoEstimationError, match="at least 4 beats"):
        tempo_from_beat_times([0.0, 0.5, 1.0])


def test_non_advancing_beats() -> None:
    with pytest.raises(TempoEstimationError):
        tempo_from_beat_times([1.0, 1.0, 1.0, 1.0])


# --- resolve_tempo (precedence) --------------------------------------------------------------


class FakeEstimator:
    def __init__(self, bpm: float = 97.5) -> None:
        self.bpm = bpm
        self.calls: list[object] = []

    def estimate(self, audio_path: str | PathLike[str]) -> TempoEstimate:
        self.calls.append(audio_path)
        return TempoEstimate(self.bpm, (0.1, 0.7, 1.3, 1.9))


def test_explicit_tempo_wins_and_estimator_is_not_called() -> None:
    estimator = FakeEstimator()

    assert resolve_tempo("clip.wav", explicit_bpm=120.0, estimator=estimator) == (120.0, None)
    assert estimator.calls == []


def test_estimate_is_used_without_explicit_tempo() -> None:
    estimator = FakeEstimator(97.5)

    bpm, estimate = resolve_tempo("clip.wav", explicit_bpm=None, estimator=estimator)

    assert bpm == 97.5
    assert estimate is not None and estimate.bpm == 97.5
    assert estimator.calls == ["clip.wav"]


def test_no_tempo_source_is_an_error() -> None:
    with pytest.raises(ValueError, match="no tempo"):
        resolve_tempo("clip.wav", explicit_bpm=None, estimator=None)


def test_invalid_explicit_tempo() -> None:
    with pytest.raises(ValueError, match="positive"):
        resolve_tempo("clip.wav", explicit_bpm=0.0, estimator=None)


def test_estimation_errors_propagate() -> None:
    class Failing:
        def estimate(self, audio_path: str | PathLike[str]) -> TempoEstimate:
            raise TempoEstimationError("too few beats")

    with pytest.raises(TempoEstimationError):
        resolve_tempo("clip.wav", explicit_bpm=None, estimator=Failing())


@pytest.mark.parametrize("bpm", [90, 120, 160])
def test_long_take_does_not_drift(bpm: float) -> None:
    # Regression: numbering beats from the first one with a frame-biased median interval
    # miscounted by a whole beat after a few dozen beats.
    times = on_frame_grid(beats(bpm, count=200))

    assert tempo_from_beat_times(times) == pytest.approx(bpm, rel=0.002)


def test_spurious_extra_beats_do_not_bias_the_tempo() -> None:
    # Regression: an extra beat halfway between two real ones used to count as a whole beat.
    times = beats(120, count=40)
    for extra in (5.3 + 0.25, 10.3 + 0.25, 15.3 + 0.25):  # off-beats between real beats
        times.append(extra)
    times.sort()

    assert tempo_from_beat_times(times) == pytest.approx(120, rel=1e-6)


def test_too_few_consistent_beats() -> None:
    # Four timestamps with a ~0.9-1.0 s period, but 1.0 is a spurious beat right after 0.9:
    # only three real beats remain.
    with pytest.raises(TempoEstimationError, match="consistent beats"):
        tempo_from_beat_times([0.0, 0.9, 1.0, 2.0])
