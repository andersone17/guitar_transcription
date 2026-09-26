"""Spotify Basic Pitch backend (the only module allowed to import ``basic_pitch``).

Verified against basic-pitch 0.4.0 (Apache-2.0). ``basic_pitch.inference.predict`` returns
``(model_output, midi_data, note_events)`` where each note event is
``(start_s, end_s, pitch_midi, amplitude, pitch_bends | None)`` with numpy scalar types.

Mapping to ``PerformanceEvent``:
- ``start_s`` / ``end_s`` -> ``onset_seconds`` / ``offset_seconds``, unchanged: raw performance
  timing, never quantized.
- ``pitch_midi`` -> ``pitch_midi``.
- ``amplitude`` -> ``velocity``. It is the mean note-activation of the model over the note (0..1);
  Basic Pitch itself uses it as MIDI velocity (``round(127 * amplitude)``). It is not a calibrated
  probability, so ``audio_confidence`` stays ``None`` rather than reusing the same number.
- ``pitch_bends`` are dropped for now; they are technique evidence for a later stage.
Everything else (string, fret, pick direction, confidences) stays ``None``.
"""

import contextlib
import importlib
import io
import logging
import operator
import os
import warnings
from collections.abc import Sequence
from os import PathLike
from typing import Any

from guitar_transcription.audio.transcriber import (
    BackendUnavailableError,
    TranscriptionError,
    UnsupportedAudioError,
    check_audio_path,
)
from guitar_transcription.domain import PerformanceEvent
from guitar_transcription.domain.pitch import validate_midi_pitch

# Formats Basic Pitch documents as supported (decoded via librosa; .mp3/.m4a may need ffmpeg).
SUPPORTED_SUFFIXES = frozenset({".wav", ".flac", ".ogg", ".mp3", ".m4a"})

# Basic Pitch decodes with librosa, which raises soundfile/audioread errors for unreadable audio.
# Matching on the exception's module keeps those libraries out of our imports.
_DECODER_MODULES = frozenset({"audioread", "soundfile", "librosa"})

INSTALL_HINT = (
    "Basic Pitch is not installed. Install the optional extra: "
    "`uv sync --extra basic-pitch` (or `pip install 'guitar-transcription[basic-pitch]'`)."
)


class BasicPitchTranscriber:
    """``AudioTranscriber`` backed by Basic Pitch's pretrained ICASSP 2022 model.

    The backend is imported and the model loaded once, at construction, so a missing install
    fails fast and repeated ``transcribe`` calls don't reload the model.

    Args:
        onset_threshold: Basic Pitch onset activation threshold (upstream default 0.5).
        frame_threshold: Basic Pitch frame activation threshold (upstream default 0.3).
        minimum_note_length_ms: Shortest note Basic Pitch will emit (upstream default 127.7 ms).
        pitch_range: Optional inclusive (low, high) MIDI range to restrict detection to, e.g. the
            playable range of the instrument. ``None`` means the model's full range.
    """

    def __init__(
        self,
        *,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        minimum_note_length_ms: float = 127.7,
        pitch_range: tuple[int, int] | None = None,
    ) -> None:
        for name, value in (
            ("onset_threshold", onset_threshold),
            ("frame_threshold", frame_threshold),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {value}")
        if minimum_note_length_ms <= 0:
            raise ValueError(f"minimum_note_length_ms must be > 0, got {minimum_note_length_ms}")
        if pitch_range is not None:
            low, high = pitch_range
            validate_midi_pitch(low, what="pitch_range low")
            validate_midi_pitch(high, what="pitch_range high")
            if low > high:
                raise ValueError(f"pitch_range low must be <= high, got {pitch_range}")

        self._predict_kwargs: dict[str, Any] = {
            "onset_threshold": onset_threshold,
            "frame_threshold": frame_threshold,
            "minimum_note_length": minimum_note_length_ms,
            **_frequency_limits(pitch_range),
        }
        self._inference, self._model = _load_backend()

    def transcribe(self, audio_path: str | PathLike[str]) -> list[PerformanceEvent]:
        path = check_audio_path(audio_path, SUPPORTED_SUFFIXES)
        try:
            # predict() prints a progress line to stdout, and librosa warns when it falls back
            # between decoders; failures are reported by our own error below instead.
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="PySoundFile failed")
                warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
                _, _, note_events = self._inference.predict(
                    path, model_or_model_path=self._model, **self._predict_kwargs
                )
        except Exception as error:
            detail = f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
            if type(error).__module__.split(".")[0] in _DECODER_MODULES:
                raise UnsupportedAudioError(
                    f"could not decode audio file {path} ({detail}); it may be corrupt, or "
                    "need ffmpeg installed for compressed formats like .mp3/.m4a"
                ) from error
            raise TranscriptionError(f"Basic Pitch failed on {path} ({detail})") from error
        return note_events_to_performance_events(note_events)


def note_events_to_performance_events(
    note_events: Sequence[Sequence[Any]],
) -> list[PerformanceEvent]:
    """Convert Basic Pitch note-event tuples into ``PerformanceEvent``s sorted by (onset, pitch).

    Numpy scalars are converted to plain Python numbers here, at the boundary. Output that
    violates domain invariants raises ``TranscriptionError`` instead of being silently repaired.
    """
    events = []
    for note in note_events:
        try:
            start_s, end_s, pitch, amplitude, _pitch_bends = note
            event = PerformanceEvent(
                onset_seconds=float(start_s),
                offset_seconds=float(end_s),
                pitch_midi=operator.index(pitch),  # accepts numpy ints, rejects floats
                velocity=float(amplitude),
            )
        except (TypeError, ValueError) as error:
            raise TranscriptionError(
                f"Basic Pitch returned an invalid note {note!r}: {error}"
            ) from error
        events.append(event)
    events.sort(key=lambda e: (e.onset_seconds, e.pitch_midi))
    return events


def _load_backend() -> tuple[Any, Any]:
    try:
        with _quiet_backend_import():
            inference = importlib.import_module("basic_pitch.inference")
            model_path = importlib.import_module("basic_pitch").ICASSP_2022_MODEL_PATH
    except ImportError as error:
        raise BackendUnavailableError(INSTALL_HINT) from error
    try:
        model = inference.Model(model_path)
    except Exception as error:
        raise TranscriptionError(
            f"could not load the Basic Pitch model from {model_path}: {error}"
        ) from error
    return inference, model


def _frequency_limits(pitch_range: tuple[int, int] | None) -> dict[str, float | None]:
    """Basic Pitch frequency limits that keep MIDI ``low``..``high`` *inclusive*.

    Basic Pitch rounds each limit to a note index and zeroes activations with
    ``[:min_index]`` and ``[max_index:]``, so its minimum is inclusive but its maximum is
    **exclusive**. Passing ``high``'s own frequency would silently drop ``high`` (e.g. D6 at fret
    22 of the high E string), so the maximum is the next semitone up.
    """
    if pitch_range is None:
        return {"minimum_frequency": None, "maximum_frequency": None}
    low, high = pitch_range
    return {"minimum_frequency": _midi_to_hz(low), "maximum_frequency": _midi_to_hz(high + 1)}


def _midi_to_hz(pitch: int) -> float:
    return 440.0 * 2.0 ** ((pitch - 69) / 12)


@contextlib.contextmanager
def _quiet_backend_import() -> Any:
    """Silence known-irrelevant import chatter: TensorFlow's C++ startup logs (CUDA/oneDNN/CPU
    notices), Basic Pitch's warnings about optional runtimes we don't use, and resampy's
    ``pkg_resources`` deprecation warning. Real errors still raise. An explicit
    ``TF_CPP_MIN_LOG_LEVEL`` set by the user is respected.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    previous_disable = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
            yield
    finally:
        logging.disable(previous_disable)
