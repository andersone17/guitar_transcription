"""JSON-compatible (de)serialization of raw ``PerformanceEvent`` lists.

The document is self-describing so later stages (rhythm, evaluation) can load it and refuse files
they don't understand. Every event field is written, including ``None`` for unknowns, so "not
inferred" stays explicit. Derived values (pitch name, duration) are not stored.

An optional ``guitar`` object records the instrument (``open_strings`` in string-1-first order,
``capo``, ``max_fret``): once events carry string/fret, the numbers only mean something relative
to the tuning and capo they were played with.
"""

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any

from guitar_transcription.domain.events import PerformanceEvent, PickDirection
from guitar_transcription.domain.guitar_config import GuitarConfig

FORMAT = "guitar-transcription/performance-events"
VERSION = 1
TIMING = "raw-performance-seconds"
"""Times are seconds from the start of the recording, as performed; never quantized to beats."""

# Declaration order, so files read naturally (onset, offset, pitch, ...).
_FIELD_NAMES = tuple(field.name for field in dataclasses.fields(PerformanceEvent))


def events_to_dict(
    events: Iterable[PerformanceEvent],
    *,
    source: str | None = None,
    guitar: GuitarConfig | None = None,
) -> dict[str, Any]:
    """Build a JSON-ready document; ``source`` says where the events came from (e.g. a path)."""
    document: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "timing": TIMING,
        "source": source,
    }
    if guitar is not None:
        document["guitar"] = {
            "open_strings": list(guitar.open_strings),
            "capo": guitar.capo,
            "max_fret": guitar.max_fret,
        }
    document["events"] = [{name: getattr(event, name) for name in _FIELD_NAMES} for event in events]
    return document


def guitar_from_dict(document: Mapping[str, Any]) -> GuitarConfig | None:
    """The document's ``guitar`` object as a validated ``GuitarConfig``, or ``None`` if absent."""
    raw = document.get("guitar")
    if raw is None:
        return None
    try:
        return GuitarConfig(tuple(raw["open_strings"]), capo=raw["capo"], max_fret=raw["max_fret"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid guitar object: {error}") from error


def events_from_dict(document: Mapping[str, Any]) -> list[PerformanceEvent]:
    """Rebuild events from ``events_to_dict`` output; every event is re-validated.

    Raises ``ValueError`` for an unknown format/version or malformed events.
    """
    if document.get("format") != FORMAT or document.get("version") != VERSION:
        raise ValueError(
            f"not a {FORMAT} v{VERSION} document "
            f"(format={document.get('format')!r}, version={document.get('version')!r})"
        )
    if document.get("timing") != TIMING:
        raise ValueError(f"expected timing {TIMING!r}, got {document.get('timing')!r}")
    events = []
    for index, raw in enumerate(document.get("events", [])):
        unknown = set(raw) - set(_FIELD_NAMES)
        if unknown:
            raise ValueError(f"event {index}: unknown fields {sorted(unknown)}")
        fields = dict(raw)
        if fields.get("pick_direction") is not None:
            fields["pick_direction"] = PickDirection(fields["pick_direction"])
        try:
            events.append(PerformanceEvent(**fields))
        except (TypeError, ValueError) as error:
            raise ValueError(f"event {index}: {error}") from error
    return events
