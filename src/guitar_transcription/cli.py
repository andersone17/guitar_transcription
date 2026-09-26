"""Command-line entry point: ``guitar-transcribe``.

This is wiring only: it validates arguments, chooses a transcription backend, and connects the
stages: audio -> raw events -> (rhythm quantization -> MusicXML). It talks to backends through
``AudioTranscriber``, never to a model package directly, and makes no musical decisions itself:
tempo and meter come from the user, timing decisions from ``rhythm``, spelling from ``notation``.
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
from guitar_transcription.rhythm import NoteValue, TimeSignature, quantize, to_single_voice
from guitar_transcription.rhythm.quantize import check_grid

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
    parser = build_parser()
    args = parser.parse_args(argv)
    _check_notation_options(parser, args)
    return _transcribe(args, make_transcriber)


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
    notation = transcribe.add_argument_group(
        "notation",
        "Quantize the events and write standard notation. Tempo and meter are not inferred yet, "
        "so --musicxml requires --tempo and --time-signature. Measure 1 starts at 0 s.",
    )
    notation.add_argument(
        "--musicxml",
        type=Path,
        metavar="PATH",
        help="write quantized notation to PATH as MusicXML (open it in e.g. MuseScore)",
    )
    notation.add_argument(
        "--tempo",
        type=_positive_float,
        metavar="BPM",
        help="tempo in quarter notes per minute, in any meter (e.g. 120)",
    )
    notation.add_argument(
        "--time-signature",
        type=_time_signature,
        metavar="N/D",
        help="meter, e.g. 4/4, 3/4, 6/8",
    )
    notation.add_argument(
        "--grid",
        type=str.lower,
        choices=[value.name.lower() for value in NoteValue],
        help="finest rhythmic subdivision to snap to (default: sixteenth)",
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


def _check_notation_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Usage errors (exit 2) before any slow work. Nothing musical is defaulted or guessed."""
    notation_only = {
        "--tempo": args.tempo,
        "--time-signature": args.time_signature,
        "--grid": args.grid,
    }
    if args.musicxml is None:
        given = [flag for flag, value in notation_only.items() if value is not None]
        if given:
            parser.error(f"{', '.join(given)} only apply to notation output; add --musicxml PATH")
        return
    missing = [flag for flag in ("--tempo", "--time-signature") if notation_only[flag] is None]
    if missing:
        parser.error(
            f"--musicxml requires {' and '.join(missing)} (tempo and meter are not inferred yet)"
        )
    try:
        check_grid(args.time_signature, _grid(args))
    except ValueError as error:
        parser.error(str(error))


def _transcribe(args: argparse.Namespace, make_transcriber: TranscriberFactory) -> int:
    # Validate before loading the model, which takes seconds.
    try:
        path = check_audio_path(args.audio, SUPPORTED_SUFFIXES)
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

    json_path: Path | None = args.json
    if json_path is not None:
        try:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            document = events_to_dict(events, source=str(path))
            json_path.write_text(json.dumps(document, indent=2) + "\n")
        except OSError as error:
            return _fail(error, EXIT_FAILURE)
        print(f"Wrote {len(events)} events to {json_path}", file=sys.stderr)

    if args.musicxml is not None:
        return _write_notation(events, args, title=path.stem)
    return EXIT_OK


def _write_notation(
    events: Sequence[PerformanceEvent], args: argparse.Namespace, title: str
) -> int:
    # Imported here: music21 takes ~1 s to import and is only needed for notation output.
    from guitar_transcription.notation.musicxml import write_musicxml

    quantized = quantize(
        events,
        quarter_note_bpm=args.tempo,
        time_signature=args.time_signature,
        grid=_grid(args),
    )
    voiced = to_single_voice(quantized)
    original_duration = {id(event.source): event.duration_quarters for event in quantized.events}
    shortened = sum(
        1
        for event in voiced.events
        if event.duration_quarters != original_duration[id(event.source)]
    )
    dropped = len(quantized.events) - len(voiced.events)
    try:
        args.musicxml.parent.mkdir(parents=True, exist_ok=True)
        write_musicxml(voiced, args.musicxml, title=title)
    except OSError as error:
        return _fail(error, EXIT_FAILURE)

    print(
        f"Wrote MusicXML to {args.musicxml}: {max(voiced.measure_count, 1)} measure(s) of "
        f"{voiced.time_signature} at quarter = {args.tempo:g}, {_grid(args).name.lower()} grid",
        file=sys.stderr,
    )
    if dropped or shortened:
        print(
            "Note: single-voice notation shortened overlapping notes"
            + (f" and dropped {dropped} duplicate pitch(es)" if dropped else "")
            + "; raw timing is unchanged in the table/JSON.",
            file=sys.stderr,
        )
    return EXIT_OK


def _grid(args: argparse.Namespace) -> NoteValue:
    return NoteValue[args.grid.upper()] if args.grid else NoteValue.SIXTEENTH


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not (value > 0 and value != float("inf")):
        raise argparse.ArgumentTypeError(f"must be a positive number, got {text!r}")
    return value


def _time_signature(text: str) -> TimeSignature:
    numerator, slash, denominator = text.partition("/")
    try:
        if not slash:
            raise ValueError("expected N/D")
        return TimeSignature(int(numerator), int(denominator))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"invalid time signature {text!r}: {error}") from None


def _fail(error: Exception, code: int) -> int:
    print(f"guitar-transcribe: error: {error}", file=sys.stderr)
    return code
