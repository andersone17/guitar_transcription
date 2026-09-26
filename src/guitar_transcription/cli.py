"""Command-line entry point: ``guitar-transcribe``.

Argument parsing and reporting only: it validates input, chooses the default backends, and runs
the stages through ``guitar_transcription.pipeline``. It talks to backends through the project's
protocols (``AudioTranscriber``, ``TempoEstimator``), never to a model package directly, and makes
no musical decisions itself.
"""

import argparse
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
from guitar_transcription.domain import GuitarConfig, Performance, PerformanceEvent
from guitar_transcription.guitar import NAMED_TUNINGS, describe_tuning, parse_tuning
from guitar_transcription.pipeline import (
    DEFAULT_GUITAR,
    NotationRequest,
    NotationResult,
    describe_range,
    detection_range,
    notate,
    write_events_json,
    write_notation,
)
from guitar_transcription.rhythm import (
    NoteValue,
    TempoEstimate,
    TempoEstimationError,
    TempoEstimator,
    TimeSignature,
)
from guitar_transcription.rhythm.backends.librosa_tempo import LibrosaTempoEstimator
from guitar_transcription.rhythm.quantize import check_grid

EXIT_OK = 0
EXIT_FAILURE = 1  # the tool ran but transcription failed (backend missing, model error)
EXIT_BAD_INPUT = 2  # the input file is missing or unreadable (argparse also uses 2 for usage)

TIMING_NOTE = (
    "RAW PERFORMANCE TIMING: seconds from the start of the recording, as played. "
    "Not quantized to beats or note values."
)

TranscriberFactory = Callable[[tuple[int, int] | None], AudioTranscriber]
"""Builds the transcriber; receives the MIDI detection range (``None`` = the model's full range)."""
TempoEstimatorFactory = Callable[[], TempoEstimator]


def main(
    argv: Sequence[str] | None = None,
    *,
    make_transcriber: TranscriberFactory = lambda pitch_range: BasicPitchTranscriber(
        pitch_range=pitch_range
    ),
    make_tempo_estimator: TempoEstimatorFactory = LibrosaTempoEstimator,
) -> int:
    """Run the CLI and return an exit code. The factories let tests supply fakes."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.guitar = _build_guitar(parser, args)
    _check_notation_options(parser, args)
    return _transcribe(args, make_transcriber, make_tempo_estimator)


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
        help=(
            "local audio file; the default backend (Basic Pitch) reads "
            f"{', '.join(sorted(SUPPORTED_SUFFIXES))}"
        ),
    )
    transcribe.add_argument(
        "--json",
        type=Path,
        metavar="PATH",
        help="also write the events to PATH as JSON, for later pipeline stages",
    )
    guitar = transcribe.add_argument_group(
        "guitar",
        "The instrument played. Notes it can't produce are not detected. Default: "
        f"{describe_tuning(DEFAULT_GUITAR.open_strings)}, no capo, {DEFAULT_GUITAR.max_fret} frets "
        f"({describe_range(*detection_range())}).",
    )
    guitar.add_argument(
        "--tuning",
        type=_tuning,
        metavar="NOTES|NAME",
        help=(
            "open strings from the lowest string up, with octaves, e.g. E2,A2,D3,G3,B3,E4 or "
            f"B1,E2,A2,D3,G3,B3,E4 (7-string); or a preset: {', '.join(NAMED_TUNINGS)}"
        ),
    )
    guitar.add_argument(
        "--capo", type=_non_negative_int, metavar="FRET", help="capo position (default: none)"
    )
    guitar.add_argument(
        "--max-fret",
        type=_positive_int,
        metavar="N",
        help=f"highest fret on the neck (default: {DEFAULT_GUITAR.max_fret})",
    )
    guitar.add_argument(
        "--full-range",
        action="store_true",
        help="detect over the model's full pitch range instead of the guitar's",
    )
    notation = transcribe.add_argument_group(
        "notation",
        "Quantize the events and write standard notation. --musicxml requires --time-signature "
        "(meter is not inferred) and either --tempo or --auto-tempo. Beat 1 of the first measure "
        "is the first detected note unless --downbeat says otherwise.",
    )
    notation.add_argument(
        "--musicxml",
        type=Path,
        metavar="PATH",
        help="write quantized notation to PATH as MusicXML (open it in e.g. MuseScore)",
    )
    tempo_source = notation.add_mutually_exclusive_group()
    tempo_source.add_argument(
        "--tempo",
        type=_positive_float,
        metavar="BPM",
        help="tempo in quarter notes per minute, in any meter (e.g. 120)",
    )
    tempo_source.add_argument(
        "--auto-tempo",
        action="store_true",
        help=(
            "estimate the tempo from the audio (beat tracking); the detected pulse is treated as a "
            "quarter note. It may come out at half or double the intended tempo."
        ),
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
    notation.add_argument(
        "--downbeat",
        type=_non_negative_float,
        metavar="SECONDS",
        help=(
            "raw time (as in the table) of a beat 1; notes before it become a pickup bar "
            "(default: the first detected note; use 0 to start measure 1 at the recording start)"
        ),
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


def _build_guitar(parser: argparse.ArgumentParser, args: argparse.Namespace) -> GuitarConfig:
    """The instrument from --tuning/--capo/--max-fret (default: DEFAULT_GUITAR); errors exit 2."""
    try:
        return GuitarConfig(
            open_strings=args.tuning or DEFAULT_GUITAR.open_strings,
            capo=DEFAULT_GUITAR.capo if args.capo is None else args.capo,
            max_fret=args.max_fret or DEFAULT_GUITAR.max_fret,
        )
    except ValueError as error:
        parser.error(f"invalid guitar: {error}")


def _describe_guitar(guitar: GuitarConfig) -> str:
    capo = f"capo {guitar.capo}" if guitar.capo else "no capo"
    return f"{describe_tuning(guitar.open_strings)}, {capo}, {guitar.max_fret} frets"


def _check_notation_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Usage errors (exit 2) before any slow work. Nothing musical is defaulted or guessed."""
    notation_only = {
        "--tempo": args.tempo,
        "--auto-tempo": args.auto_tempo or None,
        "--time-signature": args.time_signature,
        "--grid": args.grid,
        "--downbeat": args.downbeat,
    }
    if args.musicxml is None:
        given = [flag for flag, value in notation_only.items() if value is not None]
        if given:
            parser.error(f"{', '.join(given)} only apply to notation output; add --musicxml PATH")
        return
    if args.time_signature is None:
        parser.error("--musicxml requires --time-signature (meter is not inferred yet)")
    if args.tempo is None and not args.auto_tempo:
        parser.error("--musicxml requires --tempo BPM or --auto-tempo")
    try:
        check_grid(args.time_signature, _grid(args))
    except ValueError as error:
        parser.error(str(error))


def _transcribe(
    args: argparse.Namespace,
    make_transcriber: TranscriberFactory,
    make_tempo_estimator: TempoEstimatorFactory,
) -> int:
    # Validate before loading the model, which takes seconds.
    try:
        # Backend-neutral early check; each backend enforces its own formats in transcribe().
        path = check_audio_path(args.audio)
    except (FileNotFoundError, UnsupportedAudioError) as error:
        return _fail(error, EXIT_BAD_INPUT)

    guitar: GuitarConfig = args.guitar
    detect = None if args.full_range else detection_range(guitar)
    scope = "full model range" if detect is None else f"notes {describe_range(*detect)}"
    print(f"Transcribing {path} (guitar: {_describe_guitar(guitar)}; {scope}) ...", file=sys.stderr)
    try:
        events = make_transcriber(detect).transcribe(path)
    except UnsupportedAudioError as error:  # e.g. the decoder couldn't read the file
        return _fail(error, EXIT_BAD_INPUT)
    except TranscriptionError as error:
        return _fail(error, EXIT_FAILURE)

    try:
        performance = Performance(events, guitar)  # checks any string/fret against the guitar
    except ValueError as error:
        return _fail(error, EXIT_FAILURE)
    print(format_events_table(performance.events))

    if args.json is not None:
        try:
            write_events_json(performance, args.json, source=str(path))
        except OSError as error:
            return _fail(error, EXIT_FAILURE)
        print(f"Wrote {len(events)} events to {args.json}", file=sys.stderr)

    if args.musicxml is not None:
        return _write_notation(performance, args, path, make_tempo_estimator)
    return EXIT_OK


def _write_notation(
    performance: Performance,
    args: argparse.Namespace,
    audio_path: Path,
    make_tempo_estimator: TempoEstimatorFactory,
) -> int:
    request = NotationRequest(
        args.time_signature, tempo_bpm=args.tempo, grid=_grid(args), downbeat_seconds=args.downbeat
    )
    try:
        result = notate(
            performance,
            audio_path,
            request,
            # Only built when needed: an explicit tempo always wins.
            tempo_estimator=make_tempo_estimator() if args.tempo is None else None,
        )
    except TempoEstimationError as error:
        return _fail(error, EXIT_FAILURE)
    if result.tempo_estimate is not None:
        _report_estimate(result.tempo_estimate, result.tempo_bpm, request.time_signature)

    try:
        write_notation(result, args.musicxml, title=audio_path.stem)
    except OSError as error:
        return _fail(error, EXIT_FAILURE)
    _report_notation(result, args.musicxml)
    _report_downbeat(result, performance.events, args.downbeat)
    return EXIT_OK


def _report_downbeat(
    result: NotationResult, events: Sequence[PerformanceEvent], downbeat: float | None
) -> None:
    if downbeat is None:
        downbeat = min((e.onset_seconds for e in events), default=0.0)
        source = "the first detected note; set --downbeat if the piece starts with a pickup"
    else:
        source = "--downbeat"
    line = f"Beat 1 of the first full measure is at {downbeat:.3f} s ({source})."
    if result.voiced.origin_seconds < downbeat - 1e-9:
        line += " Earlier notes are written as a pickup in measure 1."
    print(line, file=sys.stderr)


def _report_notation(result: NotationResult, path: Path) -> None:
    voiced = result.voiced
    print(
        f"Wrote MusicXML to {path}: {max(voiced.measure_count, 1)} measure(s) of "
        f"{voiced.time_signature} at quarter = {result.tempo_bpm:g}, "
        f"{voiced.grid.name.lower()} grid",
        file=sys.stderr,
    )
    if result.dropped_count or result.shortened_count:
        print(
            "Note: single-voice notation shortened overlapping notes"
            + (
                f" and dropped {result.dropped_count} duplicate pitch(es)"
                if result.dropped_count
                else ""
            )
            + "; raw timing is unchanged in the table/JSON.",
            file=sys.stderr,
        )


def _report_estimate(estimate: TempoEstimate, bpm: float, time_signature: TimeSignature) -> None:
    print(
        f"Estimated tempo: {bpm:g} BPM from {len(estimate.beat_times_seconds)} detected beats, "
        f"used as quarter = {bpm:g}.",
        file=sys.stderr,
    )
    print(
        "  Beat trackers can lock onto half or double the intended pulse. If the notation looks "
        f"twice too fast or slow, rerun with --tempo {estimate.half_time_bpm:.2f} or "
        f"--tempo {estimate.double_time_bpm:.2f}.",
        file=sys.stderr,
    )
    if time_signature.denominator != 4:
        print(
            f"  In {time_signature} the pulse is often not a quarter note (e.g. a dotted quarter "
            "in 6/8); pass --tempo in quarter notes per minute if the result looks wrong.",
            file=sys.stderr,
        )


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


def _tuning(text: str) -> tuple[int, ...]:
    try:
        return parse_tuning(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {text!r}")
    return value


def _positive_int(text: str) -> int:
    value = _non_negative_int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {text!r}")
    return value


def _non_negative_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not (0 <= value < float("inf")):
        raise argparse.ArgumentTypeError(f"must be a number >= 0, got {text!r}")
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
