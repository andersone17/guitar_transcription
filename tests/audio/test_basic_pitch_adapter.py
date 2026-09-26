"""Unit tests for the Basic Pitch adapter against a fake ``basic_pitch`` package.

The fake is installed in ``sys.modules`` so the adapter's real lazy-import path runs, but no
model, TensorFlow, or numpy is needed. Fake scalar types mimic the numpy types Basic Pitch 0.4.0
actually returns (int64 pitch, float32 amplitude).
"""

import importlib
import math
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from guitar_transcription.audio import (
    BackendUnavailableError,
    TranscriptionError,
    UnsupportedAudioError,
)
from guitar_transcription.audio.backends.basic_pitch import (
    BasicPitchTranscriber,
    note_events_to_performance_events,
)


class FakeInt64:
    """Like ``numpy.int64``: usable as an index, but not an ``int`` subclass."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


class FakeFloat32:
    """Like ``numpy.float32``: convertible with ``float()``, but not a ``float`` subclass."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __float__(self) -> float:
        return self.value


def note(start: float, end: float, pitch: int, amplitude: float) -> tuple[Any, ...]:
    return (start, end, FakeInt64(pitch), FakeFloat32(amplitude), [0, 1, 0])


@dataclass
class FakeBackend:
    note_events: list[tuple[Any, ...]] = field(default_factory=list)
    predict_error: Exception | None = None
    model_error: Exception | None = None
    predict_calls: list[dict[str, Any]] = field(default_factory=list)
    models_loaded: list[Any] = field(default_factory=list)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = self

        class Model:
            def __init__(self, path: Any) -> None:
                if backend.model_error:
                    raise backend.model_error
                backend.models_loaded.append(path)

        def predict(audio_path: Any, **kwargs: Any) -> tuple[Any, Any, list[tuple[Any, ...]]]:
            print("Predicting MIDI for ...")  # the real predict() prints to stdout
            backend.predict_calls.append({"audio_path": audio_path, **kwargs})
            if backend.predict_error:
                raise backend.predict_error
            return {}, object(), list(backend.note_events)

        package = types.ModuleType("basic_pitch")
        package.ICASSP_2022_MODEL_PATH = "fake/icassp_2022/nmp"  # type: ignore[attr-defined]
        inference = types.ModuleType("basic_pitch.inference")
        inference.Model = Model  # type: ignore[attr-defined]
        inference.predict = predict  # type: ignore[attr-defined]
        package.inference = inference  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "basic_pitch", package)
        monkeypatch.setitem(sys.modules, "basic_pitch.inference", inference)


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    fake = FakeBackend()
    fake.install(monkeypatch)
    return fake


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "clip.wav"
    path.write_bytes(b"RIFF not really audio; the fake backend never reads it")
    return path


# --- conversion ----------------------------------------------------------------------------


def test_converts_note_events_to_raw_performance_events(
    backend: FakeBackend, audio_file: Path
) -> None:
    backend.note_events = [note(0.4992, 0.9751, 55, 0.856)]

    [event] = BasicPitchTranscriber().transcribe(audio_file)

    assert event.onset_seconds == 0.4992  # raw timing, untouched
    assert event.offset_seconds == 0.9751
    assert event.pitch_midi == 55
    assert type(event.pitch_midi) is int
    assert event.velocity == pytest.approx(0.856)
    assert type(event.velocity) is float


def test_does_not_invent_unmeasured_fields(backend: FakeBackend, audio_file: Path) -> None:
    backend.note_events = [note(0.5, 1.0, 64, 0.7)]

    [event] = BasicPitchTranscriber().transcribe(audio_file)

    assert event.string is None
    assert event.fret is None
    assert event.pick_direction is None
    assert event.audio_confidence is None  # amplitude is not a calibrated confidence
    assert event.fretting_confidence is None
    assert event.picking_confidence is None
    assert event.confidence is None


def test_events_are_sorted_by_onset_then_pitch() -> None:
    events = note_events_to_performance_events(
        [note(1.0, 1.5, 64, 0.5), note(0.0, 0.5, 60, 0.5), note(1.0, 1.5, 55, 0.5)]
    )

    assert [(e.onset_seconds, e.pitch_midi) for e in events] == [(0.0, 60), (1.0, 55), (1.0, 64)]


def test_timing_is_not_quantized() -> None:
    # Awkward, off-grid times must survive exactly.
    [event] = note_events_to_performance_events([note(1.23456789, 2.3456789, 60, 0.5)])

    assert (event.onset_seconds, event.offset_seconds) == (1.23456789, 2.3456789)


def test_empty_output_gives_empty_list(backend: FakeBackend, audio_file: Path) -> None:
    assert BasicPitchTranscriber().transcribe(audio_file) == []


@pytest.mark.parametrize(
    "bad_note",
    [
        (1.0, 1.0, FakeInt64(60), FakeFloat32(0.5), None),  # zero length
        (-0.1, 1.0, FakeInt64(60), FakeFloat32(0.5), None),  # negative onset
        (0.0, 1.0, 60.0, FakeFloat32(0.5), None),  # float pitch must not be truncated
        (0.0, 1.0, FakeInt64(200), FakeFloat32(0.5), None),  # pitch outside MIDI
        (0.0, 1.0, FakeInt64(60), FakeFloat32(1.5), None),  # amplitude outside [0, 1]
        (0.0, 1.0, FakeInt64(60), FakeFloat32(math.nan), None),
        (0.0, 1.0, FakeInt64(60)),  # wrong tuple shape (API change)
    ],
)
def test_invalid_backend_output_raises_instead_of_being_repaired(bad_note: tuple[Any, ...]) -> None:
    with pytest.raises(TranscriptionError, match="invalid note"):
        note_events_to_performance_events([bad_note])


# --- calling the backend ------------------------------------------------------------------


def test_passes_defaults_model_and_path_to_predict(backend: FakeBackend, audio_file: Path) -> None:
    BasicPitchTranscriber().transcribe(str(audio_file))

    [call] = backend.predict_calls
    assert call["audio_path"] == audio_file
    assert call["onset_threshold"] == 0.5
    assert call["frame_threshold"] == 0.3
    assert call["minimum_note_length"] == 127.7
    assert call["minimum_frequency"] is None
    assert call["maximum_frequency"] is None
    assert call["model_or_model_path"] is not None


BASIC_PITCH_LOWEST_NOTE = 21  # its note bins cover MIDI 21..108 (piano range)


def notes_basic_pitch_keeps(min_frequency: float, max_frequency: float) -> list[int]:
    """Replicates basic_pitch.note_creation.constrain_frequency (0.4.0): each limit is rounded to
    a note-bin index, then bins ``[:min_index]`` and ``[max_index:]`` are zeroed. So the minimum
    is inclusive and the maximum is exclusive.
    """

    def index(frequency: float) -> int:
        return round(12 * math.log2(frequency / 440) + 69 - BASIC_PITCH_LOWEST_NOTE)

    bins = range(88)
    kept = [b for b in bins if index(min_frequency) <= b < index(max_frequency)]
    return [b + BASIC_PITCH_LOWEST_NOTE for b in kept]


@pytest.mark.parametrize("pitch_range", [(40, 86), (40, 88), (35, 88), (60, 60), (21, 107)])
def test_pitch_range_is_inclusive_at_both_ends(
    backend: FakeBackend, audio_file: Path, pitch_range: tuple[int, int]
) -> None:
    # Regression: the maximum used to be the top note's own frequency, which Basic Pitch treats
    # as exclusive, so D6 (fret 22 on the high E string) could never be detected.
    BasicPitchTranscriber(pitch_range=pitch_range).transcribe(audio_file)

    [call] = backend.predict_calls
    low, high = pitch_range
    kept = notes_basic_pitch_keeps(call["minimum_frequency"], call["maximum_frequency"])
    assert kept == list(range(low, high + 1))


def test_standard_guitar_range_limits(backend: FakeBackend, audio_file: Path) -> None:
    BasicPitchTranscriber(pitch_range=(40, 86)).transcribe(audio_file)  # E2..D6

    [call] = backend.predict_calls
    assert call["minimum_frequency"] == pytest.approx(82.407, abs=1e-3)  # E2 itself
    assert call["maximum_frequency"] == pytest.approx(1244.508, abs=1e-3)  # D#6: first excluded


def test_custom_thresholds_are_forwarded(backend: FakeBackend, audio_file: Path) -> None:
    transcriber = BasicPitchTranscriber(
        onset_threshold=0.6, frame_threshold=0.2, minimum_note_length_ms=50
    )
    transcriber.transcribe(audio_file)

    [call] = backend.predict_calls
    assert (call["onset_threshold"], call["frame_threshold"], call["minimum_note_length"]) == (
        0.6,
        0.2,
        50,
    )


def test_model_is_loaded_once_and_reused(backend: FakeBackend, audio_file: Path) -> None:
    transcriber = BasicPitchTranscriber()
    transcriber.transcribe(audio_file)
    transcriber.transcribe(audio_file)

    assert backend.models_loaded == ["fake/icassp_2022/nmp"]
    first, second = backend.predict_calls
    assert first["model_or_model_path"] is second["model_or_model_path"]


def test_backend_stdout_is_suppressed(
    backend: FakeBackend, audio_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    BasicPitchTranscriber().transcribe(audio_file)

    assert capsys.readouterr().out == ""


# --- errors ---------------------------------------------------------------------------------


def test_missing_file_fails_before_running_model(backend: FakeBackend, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        BasicPitchTranscriber().transcribe(tmp_path / "missing.wav")

    assert backend.predict_calls == []


def test_unsupported_suffix_fails_before_running_model(
    backend: FakeBackend, tmp_path: Path
) -> None:
    video = tmp_path / "take1.mp4"
    video.write_bytes(b"video")

    with pytest.raises(UnsupportedAudioError, match=r"'\.mp4'"):
        BasicPitchTranscriber().transcribe(video)
    assert backend.predict_calls == []


@pytest.mark.parametrize("suffix", [".wav", ".flac", ".ogg", ".mp3", ".m4a"])
def test_documented_formats_are_accepted(backend: FakeBackend, tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"clip{suffix}"
    path.write_bytes(b"data")

    assert BasicPitchTranscriber().transcribe(path) == []


def test_decoder_error_becomes_unsupported_audio(backend: FakeBackend, audio_file: Path) -> None:
    # Real corrupt files surface as audioread.exceptions.NoBackendError via librosa.
    decoder_error = type("NoBackendError", (Exception,), {"__module__": "audioread.exceptions"})
    backend.predict_error = decoder_error()

    with pytest.raises(UnsupportedAudioError, match="could not decode") as info:
        BasicPitchTranscriber().transcribe(audio_file)
    assert info.value.__cause__ is backend.predict_error


def test_model_failure_becomes_transcription_error(backend: FakeBackend, audio_file: Path) -> None:
    backend.predict_error = RuntimeError("out of memory")

    with pytest.raises(TranscriptionError, match="out of memory") as info:
        BasicPitchTranscriber().transcribe(audio_file)
    assert not isinstance(info.value, UnsupportedAudioError)
    assert info.value.__cause__ is backend.predict_error


def test_model_load_failure(backend: FakeBackend) -> None:
    backend.model_error = ValueError("cannot load saved model")

    with pytest.raises(TranscriptionError, match="could not load the Basic Pitch model"):
        BasicPitchTranscriber()


def test_missing_backend_names_the_install_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "basic_pitch", None)  # makes the import fail
    monkeypatch.setitem(sys.modules, "basic_pitch.inference", None)

    with pytest.raises(BackendUnavailableError, match=r"--extra basic-pitch"):
        BasicPitchTranscriber()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"onset_threshold": 0.0}, "onset_threshold"),
        ({"frame_threshold": 1.0}, "frame_threshold"),
        ({"minimum_note_length_ms": 0}, "minimum_note_length_ms"),
        ({"pitch_range": (88, 40)}, "low must be <= high"),
        ({"pitch_range": (40, 200)}, "pitch_range high"),
    ],
)
def test_invalid_options_are_rejected(
    backend: FakeBackend, kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BasicPitchTranscriber(**kwargs)


# --- process environment (review MINOR) -----------------------------------------------------


def test_tf_log_level_is_not_left_set_after_loading(
    backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TF_CPP_MIN_LOG_LEVEL", raising=False)

    BasicPitchTranscriber()

    assert "TF_CPP_MIN_LOG_LEVEL" not in os.environ


def test_user_tf_log_level_is_respected(
    backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "1")

    BasicPitchTranscriber()

    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "1"


def test_tf_log_level_is_quiet_during_the_backend_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TF_CPP_MIN_LOG_LEVEL", raising=False)
    seen: list[str | None] = []
    fake = FakeBackend()
    fake.install(monkeypatch)
    real_import = importlib.import_module

    def spying_import(name: str, package: str | None = None) -> types.ModuleType:
        seen.append(os.environ.get("TF_CPP_MIN_LOG_LEVEL"))
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", spying_import)

    BasicPitchTranscriber()

    assert seen and all(value == "3" for value in seen)
