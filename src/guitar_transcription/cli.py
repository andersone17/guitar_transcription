"""Command-line entry point: ``guitar-transcribe``.

This is wiring only: it validates arguments, chooses a transcription backend, and prints or saves
the result. It talks to backends through ``AudioTranscriber``, never to a model package directly.
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from guitar_transcription.audio import (
    AudioTranscriber,
    TranscriptionError,
    UnsupportedAudioError,
)
from guitar_transcription.audio.backends.basic_pitch import (
    SUPPORTED_SUFFIXES,
    BasicPitchTranscriber,
)
from guitar_transcription.audio.transcriber import check_audio_path
from guitar_transcription.domain import PerformanceEvent
from guitar_transcription.domain.serialization import events_to_dict

EXIT_OK = 0
EXIT_FAILURE = 1  # the tool ran but transcription failed (backend missing, model error)
EXIT_BAD_INPUT = 2  # the input file is missing or unreadable (argparse also uses 2 for usage)

TIMING_NOTE = (
    "RAW PERFORMANCE TIMING: seconds from the start of the recording, as played. "
    "Not quantized to beats or note values."
)

TranscriberFactory = Callable[[], AudioTranscriber]


def main(
    argv: Sequence[str] | None = None,
    *,
    make_transcriber: TranscriberFactory = BasicPitchTranscriber,
) -> int:
    """Run the CLI and return an exit code. ``make_transcriber`` lets tests supply a fake."""
    args = build_parser().parse_args(argv)
    return _transcribe(args.audio, args.json, make_transcriber)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="guitar-transcribe",
        description="Automatic guitar transcription (Stage 1: audio only).",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    transcribe = commands.add_parser(
        "transcribe",
        help="detect notes in an audio file and print raw performance events",
        description=(
            "Detect notes in an audio file and print them as raw performance events. "
            f"{TIMING_NOTE} String/fret are not inferred."
        ),
    )
    transcribe.add_argument(
        "audio",
        type=Path,
        help=f"local audio file ({', '.join(sorted(SUPPORTED_SUFFIXES))})",
    )
    transcribe.add_argument(
        "--json",
        type=Path,
        metavar="PATH",
        help="also write the events to PATH as JSON, for later pipeline stages",
    )
    return parser


def format_events_table(events: Sequence[PerformanceEvent]) -> str:
    """Render events as a fixed-width table with a heading that states the timing is raw."""
    lines = [TIMING_NOTE, ""]
    if not events:
        lines.append("No notes detected.")
        return "\n".join(lines)
    lines.append(_row("onset_s", "offset_s", "duration_s", "midi", "note", "velocity"))
    for event in events:
        lines.append(
            _row(
                f"{event.onset_seconds:.3f}",
                f"{event.offset_seconds:.3f}",
                f"{event.duration_seconds:.3f}",
                str(event.pitch_midi),
                event.pitch_name,
                "-" if event.velocity is None else f"{event.velocity:.2f}",
            )
        )
    lines.append("")
    lines.append(f"{len(events)} note{'s' if len(events) != 1 else ''} detected.")
    return "\n".join(lines)


def _row(onset: str, offset: str, duration: str, midi: str, note: str, velocity: str) -> str:
    return f"{onset:>9} {offset:>9} {duration:>11} {midi:>5}  {note:<5} {velocity:>8}"


def _transcribe(audio: Path, json_path: Path | None, make_transcriber: TranscriberFactory) -> int:
    # Validate before loading the model, which takes seconds.
    try:
        path = check_audio_path(audio, SUPPORTED_SUFFIXES)
    except (FileNotFoundError, UnsupportedAudioError) as error:
        return _fail(error, EXIT_BAD_INPUT)

    print(f"Transcribing {path} ...", file=sys.stderr)
    try:
        events = make_transcriber().transcribe(path)
    except UnsupportedAudioError as error:  # e.g. the decoder couldn't read the file
        return _fail(error, EXIT_BAD_INPUT)
    except TranscriptionError as error:
        return _fail(error, EXIT_FAILURE)

    print(format_events_table(events))

    if json_path is not None:
        try:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            document = events_to_dict(events, source=str(path))
            json_path.write_text(json.dumps(document, indent=2) + "\n")
        except OSError as error:
            return _fail(error, EXIT_FAILURE)
        print(f"Wrote {len(events)} events to {json_path}", file=sys.stderr)
    return EXIT_OK


def _fail(error: Exception, code: int) -> int:
    print(f"guitar-transcribe: error: {error}", file=sys.stderr)
    return code
