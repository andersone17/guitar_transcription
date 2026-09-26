"""The project-owned interface every audio transcription backend implements.

Callers depend on ``AudioTranscriber`` and these error types, never on a backend package, so a
backend (Basic Pitch today) can be replaced or benchmarked against others without changes elsewhere.
"""

from os import PathLike
from pathlib import Path
from typing import Protocol

from guitar_transcription.domain import PerformanceEvent


class TranscriptionError(Exception):
    """Transcription failed: the model could not run or returned unusable output."""


class UnsupportedAudioError(TranscriptionError):
    """The input exists but is not audio the backend can read."""


class BackendUnavailableError(TranscriptionError):
    """The backend's optional dependency is not installed."""


class AudioTranscriber(Protocol):
    def transcribe(self, audio_path: str | PathLike[str]) -> list[PerformanceEvent]:
        """Transcribe a local audio file into raw performance events.

        Events carry raw performance timing in seconds (never quantized) and only the fields the
        backend actually measures; string, fret, technique, etc. stay ``None``. Events are sorted
        by (onset, pitch).

        Raises:
            FileNotFoundError: ``audio_path`` does not exist.
            UnsupportedAudioError: the path is not a readable audio file of a supported type.
            BackendUnavailableError: the backend's optional dependency is missing.
            TranscriptionError: the model failed or produced invalid output.
        """
        ...


def check_audio_path(audio_path: str | PathLike[str], supported_suffixes: frozenset[str]) -> Path:
    """Validate a local audio path before handing it to a backend; returns it as a ``Path``.

    Backends report unreadable input with confusing library-specific errors (or, worse, after a
    slow model load), so obvious problems are caught up front with a clear message.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"audio file not found: {path}")
    if not path.is_file():
        raise UnsupportedAudioError(f"not a file: {path}")
    if path.suffix.lower() not in supported_suffixes:
        supported = ", ".join(sorted(supported_suffixes))
        raise UnsupportedAudioError(
            f"unsupported audio type {path.suffix or '(no extension)'!r} for {path}; "
            f"supported: {supported}"
        )
    if path.stat().st_size == 0:
        raise UnsupportedAudioError(f"audio file is empty: {path}")
    return path
