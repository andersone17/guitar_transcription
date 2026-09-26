"""Audio perception: recordings -> raw performance events, behind a backend-agnostic interface.

Backend packages (e.g. Basic Pitch) are imported only inside their adapter module, lazily, so
importing this package never requires them.
"""

from guitar_transcription.audio.transcriber import (
    AudioTranscriber,
    BackendUnavailableError,
    TranscriptionError,
    UnsupportedAudioError,
)

__all__ = [
    "AudioTranscriber",
    "BackendUnavailableError",
    "TranscriptionError",
    "UnsupportedAudioError",
]
