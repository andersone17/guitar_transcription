"""CLI tests with a fake ``AudioTranscriber``; no model or backend package is needed."""

import json
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from guitar_transcription.audio import (
    BackendUnavailableError,
    TranscriptionError,
    UnsupportedAudioError,
)
from guitar_transcription.cli import (
    EXIT_BAD_INPUT,
    EXIT_FAILURE,
    EXIT_OK,
    format_events_table,
    main,
)
from guitar_transcription.domain import PerformanceEvent
from guitar_transcription.domain.serialization import events_from_dict
from guitar_transcription.rhythm import TempoEstimate, TempoEstimationError

EVENTS = [
    PerformanceEvent(onset_seconds=0.4992, offset_seconds=0.9752, pitch_midi=55, velocity=0.856),
    PerformanceEvent(onset_seconds=1.4981, offset_seconds=1.9623, pitch_midi=64, velocity=0.774),
]


class FakeTranscriber:
    def __init__(
        self, events: list[PerformanceEvent] | None = None, error: Exception | None = None
    ) -> None:
        self.events = EVENTS if events is None else events
        self.error = error
        self.calls: list[Path] = []

    def transcribe(self, audio_path: object) -> list[PerformanceEvent]:
        self.calls.append(Path(str(audio_path)))
        if self.error:
            raise self.error
        return self.events


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "clip.wav"
    path.write_bytes(b"RIFF fake audio")
    return path


def run(argv: list[str], transcriber: FakeTranscriber) -> int:
    return main(argv, make_transcriber=lambda _pitch_range: transcriber)


# --- help ---------------------------------------------------------------------------------


def test_top_level_help_lists_transcribe(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "usage: guitar-transcribe" in out
    assert "transcribe" in out


def test_transcribe_help_describes_raw_timing_and_json(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["transcribe", "--help"])

    assert exit_info.value.code == 0
    out = " ".join(capsys.readouterr().out.split())  # undo argparse line wrapping
    assert "RAW PERFORMANCE TIMING" in out
    assert "Not quantized" in out
    assert "--json PATH" in out
    assert ".wav" in out


def test_command_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([])

    assert exit_info.value.code == 2
    assert "required" in capsys.readouterr().err


def test_audio_argument_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["transcribe"])

    assert exit_info.value.code == 2
    assert "audio" in capsys.readouterr().err


# --- valid invocation ------------------------------------------------------------------------


def test_prints_raw_event_table(audio_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    transcriber = FakeTranscriber()

    code = run(["transcribe", str(audio_file)], transcriber)

    assert code == EXIT_OK
    assert transcriber.calls == [audio_file]
    out = capsys.readouterr().out
    assert out.startswith("RAW PERFORMANCE TIMING")
    assert "0.499" in out and "0.975" in out and "0.476" in out  # onset, offset, duration
    assert "G3" in out and "E4" in out
    assert "2 notes detected." in out


def test_progress_goes_to_stderr_not_stdout(
    audio_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run(["transcribe", str(audio_file)], FakeTranscriber())

    captured = capsys.readouterr()
    assert "Transcribing" in captured.err
    assert "Transcribing" not in captured.out


def test_writes_json_that_round_trips(
    audio_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    json_path = tmp_path / "outputs" / "nested" / "clip.events.json"  # parents are created

    code = run(["transcribe", str(audio_file), "--json", str(json_path)], FakeTranscriber())

    assert code == EXIT_OK
    document = json.loads(json_path.read_text())
    assert document["timing"] == "raw-performance-seconds"
    assert document["source"] == str(audio_file)
    assert events_from_dict(document) == EVENTS
    assert "Wrote 2 events" in capsys.readouterr().err


def test_no_json_file_without_flag(audio_file: Path, tmp_path: Path) -> None:
    run(["transcribe", str(audio_file)], FakeTranscriber())

    assert sorted(p.name for p in tmp_path.iterdir()) == ["clip.wav"]


def test_no_notes_detected(audio_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = run(["transcribe", str(audio_file)], FakeTranscriber(events=[]))

    assert code == EXIT_OK
    assert "No notes detected." in capsys.readouterr().out


# --- input problems --------------------------------------------------------------------------


def test_missing_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    transcriber = FakeTranscriber()

    code = run(["transcribe", str(tmp_path / "missing.wav")], transcriber)

    assert code == EXIT_BAD_INPUT
    assert "audio file not found" in capsys.readouterr().err
    assert transcriber.calls == []  # validated before the (slow) model is involved


def test_file_format_is_decided_by_the_backend_not_the_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Review IMPORTANT 9: the CLI used to reject files using Basic Pitch's format list even when
    # another transcriber was plugged in. Now a backend that reads .aiff gets .aiff files...
    take = tmp_path / "take.aiff"
    take.write_bytes(b"FORM fake aiff")
    accepting = FakeTranscriber()

    assert run(["transcribe", str(take)], accepting) == EXIT_OK
    assert accepting.calls == [take]

    # ...and a backend that can't read a file reports it, which the CLI maps to exit code 2.
    notes = tmp_path / "notes.txt"
    notes.write_text("not audio")
    rejecting = FakeTranscriber(error=UnsupportedAudioError("unsupported audio type '.txt'"))

    assert run(["transcribe", str(notes)], rejecting) == EXIT_BAD_INPUT
    assert "unsupported audio type '.txt'" in capsys.readouterr().err


def test_empty_file_is_rejected_before_the_backend_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "empty.wav"
    empty.touch()
    transcriber = FakeTranscriber()

    assert run(["transcribe", str(empty)], transcriber) == EXIT_BAD_INPUT
    assert "empty" in capsys.readouterr().err
    assert transcriber.calls == []


def test_directory_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = run(["transcribe", str(tmp_path)], FakeTranscriber())

    assert code == EXIT_BAD_INPUT
    assert "not a file" in capsys.readouterr().err


def test_transcriber_is_not_built_for_invalid_input(tmp_path: Path) -> None:
    built = []

    def factory(pitch_range: tuple[int, int] | None) -> FakeTranscriber:
        built.append(True)
        return FakeTranscriber()

    main(["transcribe", str(tmp_path / "missing.wav")], make_transcriber=factory)

    assert built == []


# --- mocked transcriber failures -------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (UnsupportedAudioError("could not decode audio file"), EXIT_BAD_INPUT),
        (TranscriptionError("model exploded"), EXIT_FAILURE),
        (
            BackendUnavailableError("Install the optional extra: uv sync --extra basic-pitch"),
            EXIT_FAILURE,
        ),
    ],
)
def test_transcriber_errors_become_exit_codes(
    audio_file: Path, capsys: pytest.CaptureFixture[str], error: Exception, code: int
) -> None:
    assert run(["transcribe", str(audio_file)], FakeTranscriber(error=error)) == code

    captured = capsys.readouterr()
    assert f"guitar-transcribe: error: {error}" in captured.err
    assert "RAW PERFORMANCE TIMING" not in captured.out


def test_backend_missing_during_construction(
    audio_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def factory(pitch_range: tuple[int, int] | None) -> FakeTranscriber:
        raise BackendUnavailableError("Basic Pitch is not installed. uv sync --extra basic-pitch")

    assert main(["transcribe", str(audio_file)], make_transcriber=factory) == EXIT_FAILURE
    assert "--extra basic-pitch" in capsys.readouterr().err


def test_unwritable_json_path(
    audio_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")  # a file where a directory is needed

    code = run(
        ["transcribe", str(audio_file), "--json", str(blocker / "out.json")], FakeTranscriber()
    )

    assert code == EXIT_FAILURE
    assert "error" in capsys.readouterr().err


# --- output formatting ---------------------------------------------------------------------


def test_table_layout() -> None:
    lines = format_events_table(EVENTS).splitlines()

    assert lines[0].startswith("RAW PERFORMANCE TIMING: seconds from the start of the recording")
    assert lines[2].split() == ["onset_s", "offset_s", "duration_s", "midi", "note", "velocity"]
    assert lines[3].split() == ["0.499", "0.975", "0.476", "55", "G3", "0.86"]
    assert lines[4].split() == ["1.498", "1.962", "0.464", "64", "E4", "0.77"]
    assert lines[-1] == "2 notes detected."


def test_table_columns_are_aligned() -> None:
    header, *rows = format_events_table(EVENTS).splitlines()[2:4]

    assert len({len(header), *(len(row) for row in rows)}) == 1


def test_missing_velocity_is_shown_as_dash() -> None:
    event = PerformanceEvent(onset_seconds=0.0, offset_seconds=0.25, pitch_midi=61)

    row = format_events_table([event]).splitlines()[3]

    assert row.split() == ["0.000", "0.250", "0.250", "61", "C#4", "-"]


def test_singular_note_count() -> None:
    assert format_events_table(EVENTS[:1]).splitlines()[-1] == "1 note detected."


def test_table_never_uses_note_values() -> None:
    text = format_events_table(EVENTS).lower()

    for word in ("quarter", "eighth", "beat", "bpm", "measure"):
        assert word not in text.replace("not quantized to beats", "")


# --- packaging ----------------------------------------------------------------------------


def test_console_script_is_registered() -> None:
    [script] = entry_points(group="console_scripts", name="guitar-transcribe")

    assert script.value == "guitar_transcription.cli:main"


# --- notation output (--musicxml) -----------------------------------------------------------

# At 120 BPM: G3 quarter, quarter rest, E4 half -> one 4/4 measure.
NOTATION_EVENTS = [
    PerformanceEvent(onset_seconds=0.01, offset_seconds=0.49, pitch_midi=55, velocity=0.8),
    PerformanceEvent(onset_seconds=1.02, offset_seconds=1.98, pitch_midi=64, velocity=0.7),
]


def test_help_describes_notation_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["transcribe", "--help"])

    out = " ".join(capsys.readouterr().out.split())
    for flag in ("--musicxml PATH", "--tempo BPM", "--time-signature N/D", "--grid"):
        assert flag in out
    assert "quarter notes per minute" in out


def test_writes_musicxml(
    audio_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from music21 import converter, meter, tempo

    out_path = tmp_path / "outputs" / "clip.musicxml"
    argv = ["transcribe", str(audio_file), "--tempo", "120", "--time-signature", "4/4"]

    code = run([*argv, "--musicxml", str(out_path)], FakeTranscriber(NOTATION_EVENTS))

    assert code == EXIT_OK
    score = converter.parse(out_path)
    assert [p.nameWithOctave for p in score.pitches] == ["G3", "E4"]
    assert [(e.isRest, float(e.quarterLength)) for e in score.recurse().notesAndRests] == [
        (False, 1.0),
        (True, 1.0),
        (False, 2.0),
    ]
    assert score.recurse().getElementsByClass(meter.TimeSignature)[0].ratioString == "4/4"
    assert score.recurse().getElementsByClass(tempo.MetronomeMark)[0].number == 120
    assert score.metadata.bestTitle == "clip"
    captured = capsys.readouterr()
    assert captured.out.startswith("RAW PERFORMANCE TIMING")  # raw table is still printed
    assert "1.020" in captured.out  # raw, not quantized, timing in the table
    assert "Wrote MusicXML" in captured.err and "1 measure(s) of 4/4" in captured.err


def test_musicxml_with_grid_and_other_meter(audio_file: Path, tmp_path: Path) -> None:
    from music21 import converter, meter

    out_path = tmp_path / "clip.musicxml"
    argv = ["transcribe", str(audio_file), "--musicxml", str(out_path), "--tempo", "90"]

    code = run([*argv, "--time-signature", "3/4", "--grid", "eighth"], FakeTranscriber())

    assert code == EXIT_OK
    score = converter.parse(out_path)
    assert score.recurse().getElementsByClass(meter.TimeSignature)[0].ratioString == "3/4"


def test_json_and_musicxml_together(audio_file: Path, tmp_path: Path) -> None:
    json_path, xml_path = tmp_path / "e.json", tmp_path / "n.musicxml"
    argv = ["transcribe", str(audio_file), "--json", str(json_path), "--musicxml", str(xml_path)]

    assert run([*argv, "--tempo", "120", "--time-signature", "4/4"], FakeTranscriber()) == EXIT_OK
    assert events_from_dict(json.loads(json_path.read_text())) == EVENTS  # raw timing in JSON
    assert xml_path.exists()


def test_overlaps_are_reported_when_shortened(
    audio_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ringing = [
        PerformanceEvent(onset_seconds=0.0, offset_seconds=2.0, pitch_midi=40),
        PerformanceEvent(onset_seconds=0.5, offset_seconds=1.0, pitch_midi=64),
    ]
    argv = ["transcribe", str(audio_file), "--musicxml", str(tmp_path / "n.musicxml")]

    assert run([*argv, "--tempo", "120", "--time-signature", "4/4"], FakeTranscriber(ringing)) == 0
    assert "shortened overlapping notes" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--musicxml", "o.musicxml"], "requires --time-signature"),
        (["--musicxml", "o.musicxml", "--tempo", "120"], "requires --time-signature"),
        (["--musicxml", "o.musicxml", "--auto-tempo"], "requires --time-signature"),
        (["--musicxml", "o.musicxml", "--time-signature", "4/4"], "--tempo BPM or --auto-tempo"),
        (
            [
                "--musicxml",
                "o.musicxml",
                "--time-signature",
                "4/4",
                "--tempo",
                "90",
                "--auto-tempo",
            ],
            "not allowed with",
        ),
        (["--auto-tempo"], "--auto-tempo only apply to notation output"),
        (["--tempo", "120"], "--tempo only apply to notation output"),
        (["--time-signature", "3/4", "--grid", "eighth"], "add --musicxml"),
        (["--musicxml", "o.musicxml", "--tempo", "0", "--time-signature", "4/4"], "positive"),
        (
            ["--musicxml", "o.musicxml", "--tempo", "fast", "--time-signature", "4/4"],
            "not a number",
        ),
        (["--musicxml", "o.musicxml", "--tempo", "120", "--time-signature", "4"], "invalid time"),
        (["--musicxml", "o.musicxml", "--tempo", "120", "--time-signature", "4/3"], "power of two"),
        (
            [
                "--musicxml",
                "o.musicxml",
                "--tempo",
                "120",
                "--time-signature",
                "4/4",
                "--grid",
                "32nd",
            ],
            "invalid choice",
        ),
        (
            [
                "--musicxml",
                "o.musicxml",
                "--tempo",
                "120",
                "--time-signature",
                "3/8",
                "--grid",
                "quarter",
            ],
            "does not divide",
        ),
    ],
)
def test_notation_usage_errors_exit_2_before_transcribing(
    audio_file: Path, capsys: pytest.CaptureFixture[str], extra: list[str], message: str
) -> None:
    transcriber = FakeTranscriber()

    with pytest.raises(SystemExit) as exit_info:
        run(["transcribe", str(audio_file), *extra], transcriber)

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
    assert transcriber.calls == []


# --- automatic tempo (--auto-tempo) ----------------------------------------------------------


class FakeTempoEstimator:
    def __init__(self, bpm: float = 99.37, error: Exception | None = None) -> None:
        self.bpm = bpm
        self.error = error
        self.calls: list[Path] = []

    def estimate(self, audio_path: object) -> TempoEstimate:
        self.calls.append(Path(str(audio_path)))
        if self.error:
            raise self.error
        return TempoEstimate(self.bpm, tuple(0.2 + i * 60 / self.bpm for i in range(8)))


def run_with_tempo(
    argv: list[str], estimator: FakeTempoEstimator, transcriber: FakeTranscriber | None = None
) -> int:
    return main(
        argv,
        make_transcriber=lambda _pitch_range: transcriber or FakeTranscriber(NOTATION_EVENTS),
        make_tempo_estimator=lambda: estimator,
    )


def test_auto_tempo_estimates_and_uses_the_tempo(
    audio_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from music21 import converter, tempo

    out_path = tmp_path / "n.musicxml"
    estimator = FakeTempoEstimator(99.3712)
    argv = ["transcribe", str(audio_file), "--auto-tempo", "--time-signature", "4/4"]

    assert run_with_tempo([*argv, "--musicxml", str(out_path)], estimator) == EXIT_OK

    assert estimator.calls == [audio_file]
    [mark] = converter.parse(out_path).recurse().getElementsByClass(tempo.MetronomeMark)
    assert mark.number == 99.37  # rounded to 0.01 BPM
    err = capsys.readouterr().err
    assert "Estimated tempo: 99.37 BPM from 8 detected beats" in err
    assert "--tempo 49.69" in err and "--tempo 198.74" in err  # half/double alternatives
    assert "quarter = 99.37" in err


def test_explicit_tempo_never_builds_an_estimator(audio_file: Path, tmp_path: Path) -> None:
    built = []

    def factory() -> FakeTempoEstimator:
        built.append(True)
        return FakeTempoEstimator()

    argv = ["transcribe", str(audio_file), "--tempo", "120", "--time-signature", "4/4"]
    code = main(
        [*argv, "--musicxml", str(tmp_path / "n.musicxml")],
        make_transcriber=lambda _pitch_range: FakeTranscriber(NOTATION_EVENTS),
        make_tempo_estimator=factory,
    )

    assert code == EXIT_OK
    assert built == []


def test_auto_tempo_does_not_change_raw_timing(
    audio_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    json_path = tmp_path / "e.json"
    argv = ["transcribe", str(audio_file), "--json", str(json_path), "--auto-tempo"]

    code = run_with_tempo(
        [*argv, "--time-signature", "4/4", "--musicxml", str(tmp_path / "n.musicxml")],
        FakeTempoEstimator(),
    )

    assert code == EXIT_OK
    assert events_from_dict(json.loads(json_path.read_text())) == NOTATION_EVENTS
    assert "1.020" in capsys.readouterr().out


def test_auto_tempo_failure_exits_1(
    audio_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    estimator = FakeTempoEstimator(error=TempoEstimationError("need at least 4 beats"))
    argv = ["transcribe", str(audio_file), "--auto-tempo", "--time-signature", "4/4"]

    code = run_with_tempo([*argv, "--musicxml", str(tmp_path / "n.musicxml")], estimator)

    assert code == EXIT_FAILURE
    assert "need at least 4 beats" in capsys.readouterr().err
    assert not (tmp_path / "n.musicxml").exists()


def test_auto_tempo_warns_about_pulse_in_non_quarter_meters(
    audio_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["transcribe", str(audio_file), "--auto-tempo", "--time-signature", "6/8"]

    assert (
        run_with_tempo([*argv, "--musicxml", str(tmp_path / "n.musicxml")], FakeTempoEstimator())
        == 0
    )
    assert "In 6/8 the pulse is often not a quarter note" in capsys.readouterr().err


# --- detection range ------------------------------------------------------------------------


def test_default_detection_range_is_a_standard_22_fret_guitar(
    audio_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ranges = []

    def factory(pitch_range: tuple[int, int] | None) -> FakeTranscriber:
        ranges.append(pitch_range)
        return FakeTranscriber()

    assert main(["transcribe", str(audio_file)], make_transcriber=factory) == EXIT_OK
    assert ranges == [(40, 86)]  # E2 (open low E) .. D6 (high E, fret 22)
    assert "notes E2-D6" in capsys.readouterr().err


def test_full_range_disables_the_limit(
    audio_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ranges = []

    def factory(pitch_range: tuple[int, int] | None) -> FakeTranscriber:
        ranges.append(pitch_range)
        return FakeTranscriber()

    assert main(["transcribe", str(audio_file), "--full-range"], make_transcriber=factory) == 0
    assert ranges == [None]
    assert "full model range" in capsys.readouterr().err


# --- downbeat / pickup (review IMPORTANT 7) --------------------------------------------------

# Leading silence, then a pickup on beat 4 (1.5 s) before the downbeat at 2.0 s (quarter = 120).
PICKUP_EVENTS = [
    PerformanceEvent(onset_seconds=1.5, offset_seconds=1.98, pitch_midi=55),
    PerformanceEvent(onset_seconds=2.0, offset_seconds=2.98, pitch_midi=60),
]


def notation_args(audio_file: Path, out: Path, *extra: str) -> list[str]:
    return [
        "transcribe", str(audio_file), "--tempo", "120", "--time-signature", "4/4",
        "--musicxml", str(out), *extra,
    ]  # fmt: skip


def test_default_downbeat_is_the_first_note(
    audio_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from music21 import converter

    out = tmp_path / "n.musicxml"

    assert run(notation_args(audio_file, out), FakeTranscriber(PICKUP_EVENTS)) == EXIT_OK

    first = next(iter(converter.parse(out).recurse().notesAndRests))
    assert (first.isRest, first.nameWithOctave, first.measureNumber) == (False, "G3", 1)
    assert "first detected note" in capsys.readouterr().err


def test_downbeat_option_writes_a_pickup_bar(
    audio_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from music21 import converter

    out = tmp_path / "n.musicxml"

    code = run(notation_args(audio_file, out, "--downbeat", "2.0"), FakeTranscriber(PICKUP_EVENTS))

    assert code == EXIT_OK
    notes = list(converter.parse(out).recurse().notes)
    assert [(n.nameWithOctave, n.measureNumber, float(n.beat)) for n in notes] == [
        ("G3", 1, 4.0),  # pickup on beat 4 of measure 1 (after 3 beats of rest)
        ("C4", 2, 1.0),  # the stated downbeat is beat 1 of measure 2
    ]
    err = capsys.readouterr().err
    assert "2.000 s (--downbeat)" in err and "pickup in measure 1" in err


def test_downbeat_zero_restores_recording_start_as_beat_1(audio_file: Path, tmp_path: Path) -> None:
    from music21 import converter

    out = tmp_path / "n.musicxml"

    assert (
        run(notation_args(audio_file, out, "--downbeat", "0"), FakeTranscriber(PICKUP_EVENTS)) == 0
    )
    first = next(iter(converter.parse(out).recurse().notesAndRests))
    assert first.isRest and float(first.quarterLength) == 3.0  # 1.5 s of silence = 3 beats


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--downbeat", "1.0"], "--downbeat only apply to notation output"),
        (
            [
                "--musicxml",
                "o.musicxml",
                "--tempo",
                "120",
                "--time-signature",
                "4/4",
                "--downbeat",
                "-1",
            ],
            ">= 0",
        ),
        (
            [
                "--musicxml",
                "o.musicxml",
                "--tempo",
                "120",
                "--time-signature",
                "4/4",
                "--downbeat",
                "x",
            ],
            "not a number",
        ),
    ],
)
def test_downbeat_usage_errors(
    audio_file: Path, capsys: pytest.CaptureFixture[str], extra: list[str], message: str
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        run(["transcribe", str(audio_file), *extra], FakeTranscriber())

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


# --- guitar options (review IMPORTANT 3) -----------------------------------------------------


def detection_range_for(audio_file: Path, *options: str) -> tuple[int, int] | None:
    ranges: list[tuple[int, int] | None] = []

    def factory(pitch_range: tuple[int, int] | None) -> FakeTranscriber:
        ranges.append(pitch_range)
        return FakeTranscriber()

    assert main(["transcribe", str(audio_file), *options], make_transcriber=factory) == EXIT_OK
    return ranges[0]


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ((), (40, 86)),  # standard, 22 frets: E2..D6
        (("--tuning", "drop-d"), (38, 86)),  # low D is now detectable
        (("--tuning", "D2,A2,D3,G3,B3,E4"), (38, 86)),
        (("--tuning", "B1,E2,A2,D3,G3,B3,E4"), (35, 86)),  # 7-string
        (("--capo", "2"), (42, 86)),  # open low E is impossible under a capo
        (("--max-fret", "24"), (40, 88)),
        (("--tuning", "open-g", "--capo", "5", "--max-fret", "20"), (43, 82)),
        (("--tuning", "drop-d", "--full-range"), None),
    ],
)
def test_detection_range_follows_the_guitar(
    audio_file: Path, options: tuple[str, ...], expected: tuple[int, int] | None
) -> None:
    assert detection_range_for(audio_file, *options) == expected


def test_guitar_is_reported(audio_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    detection_range_for(audio_file, "--tuning", "drop-d", "--capo", "3")

    err = capsys.readouterr().err
    assert "guitar: D2 A2 D3 G3 B3 E4, capo 3, 22 frets; notes F2-D6" in err


def test_json_records_the_guitar(audio_file: Path, tmp_path: Path) -> None:
    from guitar_transcription.domain.serialization import guitar_from_dict

    out = tmp_path / "e.json"
    argv = ["transcribe", str(audio_file), "--json", str(out), "--tuning", "drop-d", "--capo", "2"]

    assert run(argv, FakeTranscriber()) == EXIT_OK
    guitar = guitar_from_dict(json.loads(out.read_text()))
    assert guitar is not None
    assert (guitar.open_strings[5], guitar.capo, guitar.max_fret) == (38, 2, 22)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--tuning", "EADGBE"], "not a pitch name"),
        (["--tuning", "drop-q"], "not a pitch name"),
        (["--capo", "-1"], "must be >= 0"),
        (["--capo", "two"], "not a whole number"),
        (["--max-fret", "0"], "must be >= 1"),
        (["--capo", "22"], "invalid guitar: capo must be in 0..21"),
        (["--capo", "12", "--max-fret", "12"], "invalid guitar"),
        (["--tuning", "G9"], "invalid guitar: highest string at max_fret"),
    ],
)
def test_guitar_usage_errors_exit_2_before_transcribing(
    audio_file: Path, capsys: pytest.CaptureFixture[str], options: list[str], message: str
) -> None:
    transcriber = FakeTranscriber()

    with pytest.raises(SystemExit) as exit_info:
        run(["transcribe", str(audio_file), *options], transcriber)

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
    assert transcriber.calls == []


def test_help_documents_guitar_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["transcribe", "--help"])

    out = " ".join(capsys.readouterr().out.split())
    for text in (
        "--tuning NOTES|NAME",
        "--capo FRET",
        "--max-fret N",
        "drop-d",
        "E2 A2 D3 G3 B3 E4",
    ):
        assert text in out
