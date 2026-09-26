from pathlib import Path

import pytest

from guitar_transcription.audio import TranscriptionError, UnsupportedAudioError
from guitar_transcription.audio.transcriber import check_audio_path

WAV_ONLY = frozenset({".wav"})


def test_returns_path_for_valid_file(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")

    assert check_audio_path(str(audio), WAV_ONLY) == audio


def test_suffix_check_is_case_insensitive(tmp_path: Path) -> None:
    audio = tmp_path / "CLIP.WAV"
    audio.write_bytes(b"RIFF")

    assert check_audio_path(audio, WAV_ONLY) == audio


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="audio file not found"):
        check_audio_path(tmp_path / "missing.wav", WAV_ONLY)


def test_directory_is_unsupported(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedAudioError, match="not a file"):
        check_audio_path(tmp_path, WAV_ONLY)


@pytest.mark.parametrize("name", ["notes.txt", "video.mp4", "noextension"])
def test_unsupported_suffix(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_bytes(b"data")

    with pytest.raises(UnsupportedAudioError, match="unsupported audio type.*supported: .wav"):
        check_audio_path(path, WAV_ONLY)


def test_empty_file(tmp_path: Path) -> None:
    audio = tmp_path / "empty.wav"
    audio.touch()

    with pytest.raises(UnsupportedAudioError, match="empty"):
        check_audio_path(audio, WAV_ONLY)


def test_error_hierarchy() -> None:
    # Callers can catch every backend problem with one except clause.
    assert issubclass(UnsupportedAudioError, TranscriptionError)
