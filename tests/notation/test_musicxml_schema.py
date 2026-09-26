"""Validate generated MusicXML against the official W3C MusicXML 4.0 schema. Opt-in.

The schema (~380 KB) isn't committed and lxml isn't a project dependency, so this runs only when
both are provided:

    mkdir -p /tmp/musicxml-xsd && cd /tmp/musicxml-xsd
    for f in musicxml.xsd xlink.xsd xml.xsd; do
      curl -sfLO "https://raw.githubusercontent.com/w3c/musicxml/v4.0/schema/$f"; done
    sed -i 's#http://www.musicxml.org/xsd/##' musicxml.xsd
    cd - && MUSICXML_XSD=/tmp/musicxml-xsd/musicxml.xsd \\
        uv run --with lxml pytest -m integration tests/notation/test_musicxml_schema.py
"""

import os
from pathlib import Path
from typing import Any

import pytest

from guitar_transcription.domain import PerformanceEvent
from guitar_transcription.notation.musicxml import write_musicxml
from guitar_transcription.rhythm import TimeSignature, quantize, to_single_voice

pytestmark = pytest.mark.integration


def ev(onset: float, offset: float, pitch: int) -> PerformanceEvent:
    return PerformanceEvent(onset_seconds=onset, offset_seconds=offset, pitch_midi=pitch)


CASES = {
    # ties across a barline, a chord, a dotted value, rests, and a note needing a tie in-bar
    "4/4": (
        [ev(0, 0.49, 55), ev(1.0, 1.62, 64), ev(1.5, 2.5, 67), ev(2.5, 2.9, 43)]
        + [ev(2.51, 3.1, 47), ev(3.3, 3.68, 61)],
        TimeSignature(4, 4),
        120.0,
    ),
    "3/4": ([ev(i * 0.5, i * 0.5 + 0.4, 60 + i) for i in range(5)], TimeSignature(3, 4), 120.0),
    "6/8 fractional tempo": (
        [ev(i * 0.3, i * 0.3 + 0.3, 50 + i) for i in range(9)],
        TimeSignature(6, 8),
        92.5,
    ),
    "empty": ([], TimeSignature(4, 4), 120.0),
}


@pytest.fixture(scope="module")
def schema() -> Any:
    etree = pytest.importorskip("lxml.etree", reason="needs lxml (uv run --with lxml ...)")
    xsd = os.environ.get("MUSICXML_XSD")
    if not xsd or not Path(xsd).is_file():
        pytest.skip("set MUSICXML_XSD to a local musicxml.xsd (see module docstring)")
    return etree.XMLSchema(etree.parse(xsd))


@pytest.mark.parametrize("case", sorted(CASES))
def test_output_validates_against_musicxml_4_schema(schema: Any, tmp_path: Path, case: str) -> None:
    from lxml import etree

    events, signature, bpm = CASES[case]
    quantized = to_single_voice(
        quantize(events, quarter_note_bpm=bpm, time_signature=signature, downbeat_seconds=0.0)
    )
    path = write_musicxml(quantized, tmp_path / "out.musicxml", title=case)

    document = etree.parse(str(path))
    assert schema.validate(document), schema.error_log
