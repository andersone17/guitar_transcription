"""MusicXML export tests: output is parsed back (music21, ElementTree) to check meaning."""

import copy
import xml.etree.ElementTree as ET
from fractions import Fraction as F
from pathlib import Path

import pytest
from music21 import chord, clef, converter, meter, note, stream, tempo

from guitar_transcription.domain import PerformanceEvent
from guitar_transcription.notation.musicxml import write_musicxml
from guitar_transcription.rhythm import (
    NoteValue,
    QuantizedEvent,
    QuantizedPerformance,
    TimeSignature,
    quantize,
    to_single_voice,
)

FOUR_FOUR = TimeSignature(4, 4)


def ev(onset: float, offset: float, pitch: int) -> PerformanceEvent:
    return PerformanceEvent(
        onset_seconds=onset, offset_seconds=offset, pitch_midi=pitch, velocity=0.8
    )


def export(
    tmp_path: Path,
    events: list[PerformanceEvent],
    signature: TimeSignature = FOUR_FOUR,
    bpm: float = 120.0,
    title: str | None = "Test clip",
) -> Path:
    quantized = to_single_voice(quantize(events, quarter_note_bpm=bpm, time_signature=signature))
    return write_musicxml(quantized, tmp_path / "out.musicxml", title=title)


def parse(path: Path) -> stream.Score:
    score = converter.parse(path)
    assert isinstance(score, stream.Score)
    return score


def sounding(score: stream.Score) -> list[tuple[str, float]]:
    """(pitch or chord or 'rest', total quarterLength) with ties merged back together."""
    merged = score.stripTies()
    result = []
    for element in merged.recurse().notesAndRests:
        if isinstance(element, chord.Chord):
            label = "+".join(p.nameWithOctave for p in element.pitches)
        elif isinstance(element, note.Note):
            label = element.nameWithOctave
        else:
            label = "rest"
        result.append((label, float(element.quarterLength)))
    return result


# At 120 BPM: quarter = 0.5 s. A C-major melody line in 4/4:
# q G3, q rest, q E4, q F4 | h G4, e A4, e rest, q B4 |
MELODY = [
    ev(0.00, 0.49, 55),
    ev(1.01, 1.50, 64),
    ev(1.49, 2.01, 65),
    ev(2.00, 3.00, 67),
    ev(3.00, 3.24, 69),
    ev(3.51, 4.00, 71),
]


# --- file and structure ---------------------------------------------------------------------


def test_writes_well_formed_partwise_musicxml(tmp_path: Path) -> None:
    path = export(tmp_path, MELODY)

    assert path == tmp_path / "out.musicxml"
    root = ET.parse(path).getroot()
    assert root.tag == "score-partwise"
    assert root.find("part-list/score-part/part-name").text == "Guitar"  # type: ignore[union-attr]
    assert len(root.findall("part")) == 1


def test_expected_pitches(tmp_path: Path) -> None:
    pitches = [p.nameWithOctave for p in parse(export(tmp_path, MELODY)).stripTies().pitches]

    assert pitches == ["G3", "E4", "F4", "G4", "A4", "B4"]


def test_expected_durations_and_rests(tmp_path: Path) -> None:
    assert sounding(parse(export(tmp_path, MELODY))) == [
        ("G3", 1.0),
        ("rest", 1.0),
        ("E4", 1.0),
        ("F4", 1.0),
        ("G4", 2.0),
        ("A4", 0.5),
        ("rest", 0.5),
        ("B4", 1.0),
    ]


def test_expected_time_signature(tmp_path: Path) -> None:
    score = parse(export(tmp_path, MELODY, signature=TimeSignature(3, 4)))

    [signature] = score.recurse().getElementsByClass(meter.TimeSignature)
    assert signature.ratioString == "3/4"


def test_expected_tempo(tmp_path: Path) -> None:
    path = export(tmp_path, MELODY, bpm=96)

    [mark] = parse(path).recurse().getElementsByClass(tempo.MetronomeMark)
    assert mark.number == 96
    assert mark.referent.type == "quarter"
    assert ET.parse(path).getroot().find(".//sound").get("tempo") == "96"  # type: ignore[union-attr]


def test_fractional_tempo(tmp_path: Path) -> None:
    [mark] = (
        parse(export(tmp_path, MELODY, bpm=92.5)).recurse().getElementsByClass(tempo.MetronomeMark)
    )

    assert mark.number == 92.5


def test_expected_measure_count(tmp_path: Path) -> None:
    score = parse(export(tmp_path, MELODY))

    assert len(score.parts[0].getElementsByClass(stream.Measure)) == 2


def test_measure_count_in_three_four_and_trailing_fill(tmp_path: Path) -> None:
    # 5 quarters in 3/4 -> 2 measures; the last beat of measure 2 is filled with a rest.
    events = [ev(i * 0.5, (i + 1) * 0.5, 60 + i) for i in range(5)]
    score = parse(export(tmp_path, events, signature=TimeSignature(3, 4)))

    measures = score.parts[0].getElementsByClass(stream.Measure)
    assert len(measures) == 2
    assert [float(m.duration.quarterLength) for m in measures] == [3.0, 3.0]
    assert sounding(score)[-1] == ("rest", 1.0)


def test_six_eight(tmp_path: Path) -> None:
    # Six eighths at quarter = 120 fill one 6/8 bar (1.5 s).
    events = [ev(i * 0.25, (i + 1) * 0.25, 60 + i) for i in range(6)]
    score = parse(export(tmp_path, events, signature=TimeSignature(6, 8)))

    assert len(score.parts[0].getElementsByClass(stream.Measure)) == 1
    assert [d for _, d in sounding(score)] == [0.5] * 6


def test_guitar_clef_writes_sounding_pitch(tmp_path: Path) -> None:
    path = export(tmp_path, [ev(0.0, 2.0, 40)])  # low E string, E2

    [staff_clef] = parse(path).recurse().getElementsByClass(clef.Clef)
    assert isinstance(staff_clef, clef.Treble8vbClef)
    root = ET.parse(path).getroot()
    assert root.find(".//clef/clef-octave-change").text == "-1"  # type: ignore[union-attr]
    assert root.find(".//note/pitch/octave").text == "2"  # type: ignore[union-attr]


# --- spelling (music21 does it; we check the result) -----------------------------------------


def test_note_across_barline_is_tied_with_correct_total(tmp_path: Path) -> None:
    # Half note starting on beat 4 of measure 1.
    score = parse(export(tmp_path, [ev(1.5, 2.5, 64)]))

    pieces = [n for n in score.recurse().notes]
    assert [(p.measureNumber, float(p.quarterLength), p.tie.type) for p in pieces] == [
        (1, 1.0, "start"),
        (2, 1.0, "stop"),
    ]
    assert sounding(score)[1] == ("E4", 2.0)


def test_duration_needing_tie_within_measure(tmp_path: Path) -> None:
    score = parse(export(tmp_path, [ev(0.0, 0.625, 64)]))  # 5 sixteenths

    assert sounding(score)[0] == ("E4", 1.25)
    assert [float(n.quarterLength) for n in score.recurse().notes] == [1.0, 0.25]


def test_dotted_values(tmp_path: Path) -> None:
    score = parse(
        export(tmp_path, [ev(0.0, 0.75, 60), ev(0.75, 1.0, 62)])
    )  # dotted quarter, eighth

    first, second = list(score.recurse().notes)
    assert (first.duration.type, first.duration.dots) == ("quarter", 1)
    assert (second.duration.type, second.duration.dots) == ("eighth", 0)


def test_leading_rest_when_first_note_is_late(tmp_path: Path) -> None:
    assert sounding(parse(export(tmp_path, [ev(1.0, 1.5, 60)])))[:2] == [("rest", 2.0), ("C4", 1.0)]


def test_chord_from_simultaneous_onsets(tmp_path: Path) -> None:
    strum = [ev(0.00, 1.0, 43), ev(0.01, 1.0, 47), ev(0.02, 1.0, 50)]

    assert sounding(parse(export(tmp_path, strum)))[0] == ("G2+B2+D3", 2.0)


def test_empty_performance_gives_one_measure_of_rest(tmp_path: Path) -> None:
    score = parse(export(tmp_path, []))

    assert len(score.parts[0].getElementsByClass(stream.Measure)) == 1
    assert sounding(score) == [("rest", 4.0)]


# --- metadata --------------------------------------------------------------------------------


def test_title_and_honest_credit(tmp_path: Path) -> None:
    root = ET.parse(export(tmp_path, MELODY, title="g_major")).getroot()

    assert root.find("movement-title").text == "g_major"  # type: ignore[union-attr]
    creators = {c.get("type"): c.text for c in root.findall("identification/creator")}
    assert creators == {"transcriber": "guitar-transcription"}  # no fake "Music21" composer


def test_untitled(tmp_path: Path) -> None:
    root = ET.parse(export(tmp_path, MELODY, title=None)).getroot()

    assert {c.text for c in root.findall("identification/creator")} == {"guitar-transcription"}


# --- separation of concerns ------------------------------------------------------------------


def test_source_events_are_not_mutated(tmp_path: Path) -> None:
    raw = [ev(0.013, 0.49, 55), ev(1.01, 1.50, 64)]
    before = copy.deepcopy(raw)
    quantized = to_single_voice(quantize(raw, quarter_note_bpm=120, time_signature=FOUR_FOUR))
    quantized_before = copy.deepcopy(quantized)

    write_musicxml(quantized, tmp_path / "out.musicxml")

    assert raw == before
    assert quantized == quantized_before
    assert quantized.events[0].source.onset_seconds == 0.013


def test_rejects_overlapping_notes_instead_of_trimming_them(tmp_path: Path) -> None:
    source = ev(0.0, 2.0, 40)
    overlapping = QuantizedPerformance(
        (QuantizedEvent(source, F(0), F(4)), QuantizedEvent(ev(0.5, 1.0, 64), F(1), F(1))),
        120.0,
        FOUR_FOUR,
        NoteValue.SIXTEENTH,
    )

    with pytest.raises(ValueError, match="to_single_voice"):
        write_musicxml(overlapping, tmp_path / "out.musicxml")
    assert not (tmp_path / "out.musicxml").exists()


def test_exported_rhythm_matches_rhythm_layer(tmp_path: Path) -> None:
    # Notation must render exactly what rhythm decided: same onsets and durations.
    quantized = to_single_voice(quantize(MELODY, quarter_note_bpm=120, time_signature=FOUR_FOUR))
    score = parse(write_musicxml(quantized, tmp_path / "out.musicxml"))

    exported = [
        (
            F(n.getOffsetInHierarchy(score)).limit_denominator(64),
            F(n.quarterLength).limit_denominator(64),
        )
        for n in score.stripTies().recurse().notes
    ]
    assert exported == [(e.onset_quarters, e.duration_quarters) for e in quantized.events]
