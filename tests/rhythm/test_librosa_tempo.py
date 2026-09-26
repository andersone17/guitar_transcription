"""librosa tempo adapter against a fake ``librosa`` module (no audio decoding or numba)."""

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from guitar_transcription.rhythm import TempoEstimationError
from guitar_transcription.rhythm.backends.librosa_tempo import LibrosaTempoEstimator


class FakeArray(list[Any]):
    """Stands in for a numpy array: librosa results are converted with ``.tolist()``."""

    def tolist(self) -> list[Any]:
        return list(self)


def install_fake_librosa(
    monkeypatch: pytest.MonkeyPatch,
    beat_times: list[float],
    load_error: Exception | None = None,
) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def load(path: str, sr: int, mono: bool) -> tuple[list[float], int]:
        calls["load"] = (path, sr, mono)
        if load_error:
            raise load_error
        return [0.0] * 10, sr

    def beat_track(y: list[float], sr: int) -> tuple[float, FakeArray]:
        calls["beat_track"] = sr
        return 117.45, FakeArray(range(len(beat_times)))  # tempo deliberately coarse, like librosa

    def frames_to_time(frames: FakeArray, sr: int) -> FakeArray:
        return FakeArray(beat_times[i] for i in frames)

    librosa = types.ModuleType("librosa")
    librosa.load = load  # type: ignore[attr-defined]
    librosa.frames_to_time = frames_to_time  # type: ignore[attr-defined]
    beat = types.ModuleType("librosa.beat")
    beat.beat_track = beat_track  # type: ignore[attr-defined]
    librosa.beat = beat  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "librosa", librosa)
    return calls


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "clip.wav"
    path.write_bytes(b"RIFF")
    return path


def test_tempo_comes_from_beat_times_not_librosa_tempo_value(
    monkeypatch: pytest.MonkeyPatch, audio_file: Path
) -> None:
    beat_times = [0.25 + i * 0.5 for i in range(20)]  # exactly 120 BPM
    calls = install_fake_librosa(monkeypatch, beat_times)

    estimate = LibrosaTempoEstimator().estimate(audio_file)

    assert estimate.bpm == pytest.approx(120.0)  # not librosa's 117.45
    assert estimate.beat_times_seconds == tuple(beat_times)
    assert all(type(t) is float for t in estimate.beat_times_seconds)
    assert calls["load"] == (str(audio_file), 22050, True)


def test_too_few_beats_is_an_estimation_error(
    monkeypatch: pytest.MonkeyPatch, audio_file: Path
) -> None:
    install_fake_librosa(monkeypatch, [0.5, 1.0])

    with pytest.raises(TempoEstimationError, match="at least 4 beats"):
        LibrosaTempoEstimator().estimate(audio_file)


def test_missing_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = install_fake_librosa(monkeypatch, [])

    with pytest.raises(FileNotFoundError):
        LibrosaTempoEstimator().estimate(tmp_path / "missing.wav")
    assert "load" not in calls


def test_decode_failure_is_wrapped(monkeypatch: pytest.MonkeyPatch, audio_file: Path) -> None:
    install_fake_librosa(monkeypatch, [], load_error=RuntimeError("cannot decode"))

    with pytest.raises(TempoEstimationError, match="cannot decode") as info:
        LibrosaTempoEstimator().estimate(audio_file)
    assert isinstance(info.value.__cause__, RuntimeError)


def test_missing_librosa_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "librosa", None)

    with pytest.raises(TempoEstimationError, match="--extra tempo"):
        LibrosaTempoEstimator()


def test_empty_file_is_an_estimation_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = install_fake_librosa(monkeypatch, [])
    empty = tmp_path / "empty.wav"
    empty.touch()

    with pytest.raises(TempoEstimationError, match="empty"):
        LibrosaTempoEstimator().estimate(empty)
    assert "load" not in calls
