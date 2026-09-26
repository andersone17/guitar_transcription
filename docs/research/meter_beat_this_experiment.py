"""Meter-inference feasibility probe: Beat This! downbeats on synthetic guitar in known meters.

NOT part of the package and not run by the test suite. Beat This! needs PyTorch (~830 MB
CPU-only) and numpy 2, which conflicts with the Basic Pitch / TensorFlow 2.15 stack (numpy < 2),
so run it in a throwaway environment:

    uv venv -p 3.11 /tmp/bt-env
    VIRTUAL_ENV=/tmp/bt-env uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    VIRTUAL_ENV=/tmp/bt-env uv pip install beat-this soundfile
    /tmp/bt-env/bin/python docs/research/meter_beat_this_experiment.py

The first run downloads the "final0" checkpoint (~81 MB) to ~/.cache/torch/hub/checkpoints/.
Results from 2026-09-26 are recorded in PLAN.md ("Meter inference: findings").
"""

import random
import statistics

import numpy as np
import soundfile as sf
from beat_this.inference import File2Beats

SR = 22050
C, G, AM, F = [48, 55, 60, 64], [43, 50, 55, 59], [45, 52, 57, 60], [41, 48, 53, 57]
PROGRESSION = [C, G, AM, F]
BARS = 16
START = 0.5  # seconds of silence before the first downbeat


def pluck(midi: int, seconds: float, amp: float, seed: int) -> np.ndarray:
    """Karplus-Strong plucked string."""
    period = max(2, int(SR / (440 * 2 ** ((midi - 69) / 12))))
    buffer = np.random.default_rng(seed).uniform(-1, 1, period)
    out = np.empty(int(SR * seconds))
    for i in range(len(out)):
        j = i % period
        out[i] = buffer[j]
        buffer[j] = 0.996 * 0.5 * (buffer[j] + buffer[(j + 1) % period])
    return amp * out


def render(events: list[tuple[float, list[int], float]], seconds: float, path: str) -> None:
    """Events are (time, pitches, amplitude); chords are strummed 12 ms apart with jitter."""
    y = np.zeros(int(SR * seconds))
    rng = random.Random(1)
    for t, pitches, amp in events:
        for k, pitch in enumerate(pitches):
            start = int((t + 0.012 * k + rng.uniform(-0.008, 0.008)) * SR)
            x = pluck(pitch, 0.9, amp, start + pitch)
            y[start : start + len(x)] += x[: len(y) - start]
    sf.write(path, 0.25 * y / np.max(np.abs(y)), SR)


def four_four() -> tuple[list[tuple[float, list[int], float]], float]:
    """Boom-chick at quarter = 100: bass on 1 (accented) and 3, chord on 2 and 4."""
    q = 0.6
    events = []
    for b in range(BARS):
        chord, t0 = PROGRESSION[b % 4], START + b * 4 * q
        events += [
            (t0, chord[:1], 1.0),
            (t0 + q, chord[1:], 0.55),
            (t0 + 2 * q, [chord[0] + 7], 0.8),
            (t0 + 3 * q, chord[1:], 0.55),
        ]
    return events, 4 * q


def three_four() -> tuple[list[tuple[float, list[int], float]], float]:
    """Waltz oom-pah-pah at quarter = 120."""
    q = 0.5
    events = []
    for b in range(BARS):
        chord, t0 = PROGRESSION[b % 4], START + b * 3 * q
        events += [(t0, chord[:1], 1.0), (t0 + q, chord[1:], 0.5), (t0 + 2 * q, chord[1:], 0.5)]
    return events, 3 * q


def six_eight() -> tuple[list[tuple[float, list[int], float]], float]:
    """Eighth-note arpeggio, dotted quarter = 60: bass on 1, weaker accent on 4."""
    e = 1 / 3
    events = []
    for b in range(BARS):
        chord, t0 = PROGRESSION[b % 4], START + b * 6 * e
        notes = [chord[0], chord[1], chord[2], chord[0] + 12, chord[2], chord[1]]
        amps = [1.0, 0.45, 0.45, 0.75, 0.45, 0.45]
        events += [(t0 + i * e, [p], a) for i, (p, a) in enumerate(zip(notes, amps, strict=True))]
    return events, 6 * e


def main() -> None:
    tracker = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    cases = [("4/4", 4, four_four), ("3/4", 3, three_four), ("6/8", 2, six_eight)]
    for name, true_beats_per_bar, build in cases:
        events, bar_seconds = build()
        path = f"meter_{name.replace('/', '_')}.wav"
        render(events, START + BARS * bar_seconds + 1, path)
        beats, downbeats = (list(x) for x in tracker(path))
        true_downbeats = [START + i * bar_seconds for i in range(BARS)]
        on_bar_start = sum(1 for d in downbeats if min(abs(d - t) for t in true_downbeats) < 0.07)
        beats_per_bar = [
            sum(1 for b in beats if d0 - 0.03 <= b < d1 - 0.03)
            for d0, d1 in zip(downbeats, downbeats[1:], strict=False)
        ]
        interval = statistics.median(np.diff(beats))
        print(
            f"{name}: true beats/bar={true_beats_per_bar} | beat period {interval:.3f}s "
            f"({60 / interval:.1f} BPM) | downbeats on bar starts {on_bar_start}/{len(downbeats)} "
            f"| beats per detected bar {beats_per_bar}"
        )


if __name__ == "__main__":
    main()
