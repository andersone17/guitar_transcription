"""librosa beat-tracking backend for ``TempoEstimator`` (the only module that imports librosa).

Uses ``librosa.beat.beat_track`` (dynamic-programming beat tracker, Ellis 2007) for beat *times*,
then derives the tempo from those times with ``tempo_from_beat_times``, not from librosa's own
tempo value, which is quantized to its tempogram resolution (e.g. 117.45 for a true 120 BPM).

Verified against librosa 0.11. librosa >= 1.0 needs Python 3.12 and numpy 2, which the Basic Pitch
stack can't use, hence the ``<1.0`` pin in the ``tempo`` extra.
"""

import importlib
import warnings
from os import PathLike
from pathlib import Path
from typing import Any

from guitar_transcription.rhythm.tempo import (
    TempoEstimate,
    TempoEstimationError,
    tempo_from_beat_times,
)

SAMPLE_RATE = 22050  # librosa's default analysis rate; plenty for onset/beat detection

INSTALL_HINT = (
    "librosa is not installed. Install the optional extra: `uv sync --extra tempo` "
    "(it is also included with `--extra basic-pitch`)."
)


class LibrosaTempoEstimator:
    """``TempoEstimator`` using librosa's beat tracker.

    librosa is imported at construction so a missing install fails fast. The first estimate in a
    process is slow (numba compiles librosa's kernels: several seconds); later ones are sub-second.
    """

    def __init__(self) -> None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # import-time deprecation chatter from dependencies
                self._librosa: Any = importlib.import_module("librosa")
        except ImportError as error:
            raise TempoEstimationError(INSTALL_HINT) from error

    def estimate(self, audio_path: str | PathLike[str]) -> TempoEstimate:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"audio file not found: {path}")
        librosa = self._librosa
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="PySoundFile failed")
                warnings.filterwarnings("ignore", category=FutureWarning)
                samples, rate = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
                _, frames = librosa.beat.beat_track(y=samples, sr=rate)
                beat_times = [float(t) for t in librosa.frames_to_time(frames, sr=rate).tolist()]
        except Exception as error:
            detail = f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
            raise TempoEstimationError(f"librosa could not analyse {path} ({detail})") from error
        return TempoEstimate(
            bpm=tempo_from_beat_times(beat_times), beat_times_seconds=tuple(beat_times)
        )
