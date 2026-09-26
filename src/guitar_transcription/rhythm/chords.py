"""Grouping near-simultaneous raw onsets into chords before they are snapped to the grid.

A guitar strum sounds its strings one after another, typically 20-100 ms apart from first to last.
Snapping each note on its own splits such a strum across grid lines (e.g. a 100 ms strum at 120 BPM
on a sixteenth grid became a four-note chord plus a two-note chord). So onsets are grouped first,
and each group is snapped as one.

Grouping is deliberately conservative, because merging real melody notes into a fake chord is worse
than leaving a slow roll as an arpeggio. The key cue is the gap between *consecutive* onsets: a
strum's strings follow each other within ~10-30 ms even when the whole strum takes 100 ms, while
arpeggio and melody notes are about a grid step apart (e.g. ~125 ms for sixteenths at 120 BPM),
even when they ring on. A note joins the current group only if:

1. it starts within ``max_gap_seconds`` of the previous note in the group;
2. its onset is within ``window_seconds`` of the group's **first** onset, so a slow roll can't chain
   into one long "chord";
3. every note already in the group is still sounding when it starts (a strum rings together; a fast
   legato or hammer-on line moves from note to note);
4. its pitch isn't already in the group.

Raw onsets are never changed; this only decides which notes share a notated onset.
"""

from collections.abc import Iterable

from guitar_transcription.domain import PerformanceEvent

DEFAULT_CHORD_WINDOW_SECONDS = 0.1
"""Longest strum (first to last string) treated as one chord; typical strums span 20-100 ms."""

MAX_STRING_GAP_SECONDS = 0.05
"""Longest gap between consecutive strings of one strum; melodic notes are further apart."""

_TOLERANCE = 1e-9  # so a limit of exactly 0.08 s isn't missed through float error (0.2 - 0.12)


def group_simultaneous(
    events: Iterable[PerformanceEvent],
    window_seconds: float,
    max_gap_seconds: float = MAX_STRING_GAP_SECONDS,
) -> list[list[PerformanceEvent]]:
    """Split events (in onset order) into groups that should share one notated onset.

    ``window_seconds == 0`` disables grouping, except for exactly simultaneous onsets (which would
    snap to the same grid line anyway).
    """
    ordered = sorted(events, key=lambda e: (e.onset_seconds, e.pitch_midi))
    groups: list[list[PerformanceEvent]] = []
    for event in ordered:
        current = groups[-1] if groups else None
        if current is not None and _joins(current, event, window_seconds, max_gap_seconds):
            current.append(event)
        else:
            groups.append([event])
    return groups


def _joins(
    group: list[PerformanceEvent],
    event: PerformanceEvent,
    window_seconds: float,
    max_gap_seconds: float,
) -> bool:
    return (
        event.onset_seconds - group[-1].onset_seconds <= max_gap_seconds + _TOLERANCE
        and event.onset_seconds - group[0].onset_seconds <= window_seconds + _TOLERANCE
        and all(member.offset_seconds > event.onset_seconds for member in group)
        and all(member.pitch_midi != event.pitch_midi for member in group)
    )
