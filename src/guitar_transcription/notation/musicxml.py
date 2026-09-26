"""MusicXML export of a quantized performance, via music21 (the only module that imports it).

This module *renders*; it makes no timing decisions. Every onset, duration, and rest comes from the
``rhythm`` output. What happens here is notation spelling, delegated to music21's ``makeNotation``:
splitting notes at barlines into ties, choosing note symbols and dots, beaming, and filling the last
measure with rests. MusicXML is the interchange format, and engraving is left to MuseScore etc.

Conventions: one part named "Guitar", treble clef 8vb (guitar sounds an octave below written), and
pitches written at sounding pitch, which is how MusicXML represents octave-transposing clefs.
Tempo is written as quarter = ``quarter_note_bpm``.

TAB is not written yet. When it is, the data is already reachable: each ``QuantizedEvent.source``
carries ``string``/``fret`` (and, later, technique), which map onto music21's ``StringIndication``,
``FretIndication``, ``HammerOn``, ``PullOff``, and ``FretBend`` (MusicXML ``<technical>``). The note
construction in ``_sounding_element`` is where they attach. A TAB staff would be a second staff
built from the same quantized events plus the ``GuitarConfig`` (tuning; capo-relative frets).
"""

from collections.abc import Sequence
from fractions import Fraction
from itertools import groupby
from os import PathLike
from pathlib import Path

from music21 import chord, clef, instrument, metadata, meter, note, stream, tempo

from guitar_transcription.rhythm import QuantizedEvent, QuantizedPerformance, is_single_voice

SOFTWARE_NAME = "guitar-transcription"


def write_musicxml(
    performance: QuantizedPerformance,
    path: str | PathLike[str],
    *,
    title: str | None = None,
) -> Path:
    """Write ``performance`` as a MusicXML file and return its path.

    ``performance`` must be single-voice (see ``rhythm.to_single_voice``). Overlapping notes are
    rejected rather than silently trimmed, because trimming is a rhythm decision.
    """
    if not is_single_voice(performance):
        raise ValueError(
            "MusicXML export needs a single voice: notes overlap or chord tones differ in length. "
            "Apply rhythm.to_single_voice() first."
        )
    score = build_score(performance, title=title)
    output = Path(path)
    score.write("musicxml", fp=output)
    return output


def build_score(performance: QuantizedPerformance, *, title: str | None = None) -> stream.Score:
    """Build the music21 score (exposed for inspection/tests; ``write_musicxml`` is the API)."""
    part = stream.Part()
    part.insert(0, instrument.Guitar())
    part.insert(0, clef.Treble8vbClef())
    part.insert(0, meter.TimeSignature(str(performance.time_signature)))
    part.insert(0, _metronome_mark(performance.quarter_note_bpm))

    for rest in performance.rests():
        part.insert(
            _offset(rest.onset_quarters), note.Rest(quarterLength=_offset(rest.duration_quarters))
        )
    ordered = sorted(performance.events, key=lambda e: e.onset_quarters)
    for onset, group in groupby(ordered, key=lambda e: e.onset_quarters):
        part.insert(_offset(onset), _sounding_element(list(group)))
    if not performance.events:
        measure = performance.time_signature.measure_quarters
        part.insert(0, note.Rest(quarterLength=_offset(measure)))

    score = stream.Score()
    score.metadata = _metadata(title)
    score.insert(0, part)
    # Measures, barline ties, note types/dots, beams, and trailing rests: spelling only.
    notated = score.makeNotation()
    if not isinstance(notated, stream.Score):  # music21 returns an untyped copy of the score
        raise TypeError(f"music21 makeNotation returned {type(notated).__name__}, not a Score")
    return notated


def _sounding_element(events: Sequence[QuantizedEvent]) -> note.Note | chord.Chord:
    """One note, or a chord for simultaneous events (single-voice input shares one duration)."""
    duration = _offset(events[0].duration_quarters)
    if len(events) == 1:
        element: note.Note | chord.Chord = note.Note(events[0].pitch_midi)
    else:
        element = chord.Chord([event.pitch_midi for event in events])
    element.quarterLength = duration
    return element


def _metronome_mark(quarter_note_bpm: float) -> tempo.MetronomeMark:
    number = int(quarter_note_bpm) if float(quarter_note_bpm).is_integer() else quarter_note_bpm
    return tempo.MetronomeMark(number=number, referent=note.Note(type="quarter"))


def _metadata(title: str | None) -> metadata.Metadata:
    # Always name a contributor: otherwise music21 credits "Music21" as the composer.
    md = metadata.Metadata(title=title) if title else metadata.Metadata()
    md.add("transcriber", metadata.Contributor(role="transcriber", name=SOFTWARE_NAME))
    return md


def _offset(quarters: Fraction) -> Fraction | float:
    """music21 wants floats for binary fractions and ``Fraction`` only for tuplets."""
    return float(quarters) if (quarters.denominator & (quarters.denominator - 1)) == 0 else quarters
