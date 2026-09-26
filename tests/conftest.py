"""Shared test fixtures.

``write_plucked_notes`` synthesizes a tiny guitar-like WAV in code (no committed audio), for the
opt-in integration tests that run real models.
"""

import random
import struct
import wave
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

SAMPLE_RATE = 22050

WritePluckedNotes = Callable[[Path, Sequence[tuple[float, int]]], Path]


def karplus_strong(pitch: int, seconds: float, seed: int) -> list[float]:
    """Plucked-string synthesis: a noise burst circulating through a damped delay line."""
    period = round(SAMPLE_RATE / (440.0 * 2 ** ((pitch - 69) / 12)))
    rng = random.Random(seed)
    buffer = [rng.uniform(-1.0, 1.0) for _ in range(period)]
    out = []
    for i in range(int(SAMPLE_RATE * seconds)):
        current = buffer[i % period]
        buffer[i % period] = 0.996 * 0.5 * (current + buffer[(i + 1) % period])
        out.append(current)
    return out


def _write_plucked_notes(
    path: Path, notes: Sequence[tuple[float, int]], note_seconds: float = 0.45
) -> Path:
    """Write ``(onset_seconds, midi_pitch)`` plucks to a mono 16-bit WAV at 22.05 kHz."""
    total = max(onset for onset, _ in notes) + note_seconds + 0.5
    samples = [0.0] * int(SAMPLE_RATE * total)
    for onset, pitch in notes:
        start = int(onset * SAMPLE_RATE)
        for i, value in enumerate(karplus_strong(pitch, note_seconds, seed=pitch)):
            samples[start + i] += 0.5 * value
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(
            b"".join(struct.pack("<h", int(max(-1.0, min(1.0, v)) * 32000)) for v in samples)
        )
    return path


@pytest.fixture
def write_plucked_notes() -> WritePluckedNotes:
    return _write_plucked_notes
