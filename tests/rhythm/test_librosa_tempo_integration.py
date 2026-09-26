"""Real librosa beat tracking on synthesized click tracks. Opt-in: ``pytest -m integration``.

The first estimate in a process is slow (numba compilation, several seconds).
"""

import math
import struct
import wave
from pathlib import Path

import pytest

# Safe without the extra: the adapter imports librosa at construction.
from guitar_transcription.rhythm.backends.librosa_tempo import LibrosaTempoEstimator

pytestmark = pytest.mark.integration

SAMPLE_RATE = 22050


def write_click_track(path: Path, bpm: float, seconds: float = 15.0) -> None:
    """Short decaying 1 kHz blips at every beat, starting at 0.25 s."""
    samples = [0.0] * int(SAMPLE_RATE * seconds)
    click = [
        math.sin(2 * math.pi * 1000 * i / SAMPLE_RATE) * math.exp(-i / (0.005 * SAMPLE_RATE))
        for i in range(int(0.03 * SAMPLE_RATE))
    ]
    beat = 0.25
    while beat < seconds - 0.05:
        start = int(beat * SAMPLE_RATE)
        for i, value in enumerate(click):
            if start + i < len(samples):
                samples[start + i] += value
        beat += 60 / bpm
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(b"".join(struct.pack("<h", int(v * 30000)) for v in samples))


@pytest.fixture(scope="module")
def estimator() -> LibrosaTempoEstimator:
    pytest.importorskip("librosa", reason="requires the tempo (or basic-pitch) extra")
    return LibrosaTempoEstimator()


@pytest.mark.parametrize("bpm", [60, 90, 120, 160])
def test_recovers_click_track_tempo(
    estimator: LibrosaTempoEstimator, tmp_path: Path, bpm: int
) -> None:
    audio = tmp_path / f"click{bpm}.wav"
    write_click_track(audio, bpm)

    estimate = estimator.estimate(audio)

    assert estimate.bpm == pytest.approx(bpm, rel=0.01)
    # Beats are in phase with the clicks (librosa may skip the first/last beat or two).
    period = 60 / bpm
    for beat in estimate.beat_times_seconds:
        clicks_since_start = (beat - 0.25) / period
        assert abs(clicks_since_start - round(clicks_since_start)) * period < 0.05


def test_fast_tempo_may_be_reported_at_half_time(
    estimator: LibrosaTempoEstimator, tmp_path: Path
) -> None:
    # Documented metrical-level ambiguity: librosa's tracker prefers tempi near 120, so a 180 BPM
    # click is currently reported as 90. Either level is accepted; the CLI shows both alternatives.
    audio = tmp_path / "click180.wav"
    write_click_track(audio, 180)

    bpm = estimator.estimate(audio).bpm

    assert bpm == pytest.approx(180, rel=0.01) or bpm == pytest.approx(90, rel=0.01)
