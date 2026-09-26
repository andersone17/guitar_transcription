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
    return main(argv, make_transcriber=lambda: transcriber)


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


def test_unsupported_input_type(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("not audio")
    transcriber = FakeTranscriber()

    code = run(["transcribe", str(notes)], transcriber)

    assert code == EXIT_BAD_INPUT
    assert "unsupported audio type '.txt'" in capsys.readouterr().err
    assert transcriber.calls == []


def test_directory_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = run(["transcribe", str(tmp_path)], FakeTranscriber())

    assert code == EXIT_BAD_INPUT
    assert "not a file" in capsys.readouterr().err


def test_transcriber_is_not_built_for_invalid_input(tmp_path: Path) -> None:
    built = []

    def factory() -> FakeTranscriber:
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
    def factory() -> FakeTranscriber:
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
        (["--musicxml", "o.musicxml"], "requires --tempo and --time-signature"),
        (["--musicxml", "o.musicxml", "--tempo", "120"], "requires --time-signature"),
        (["--musicxml", "o.musicxml", "--time-signature", "4/4"], "requires --tempo"),
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
