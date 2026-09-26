"""Reduce quantized events to a single voice, the Stage 1 notation simplification.

Real transcriptions overlap: guitar strings ring on after the next note starts, and chord tones
end at different times. One staff voice can't show that, so something must shorten notes. That
is a timing decision, so it lives here in ``rhythm`` rather than in the notation exporter.
"""

from dataclasses import replace
from itertools import groupby

from guitar_transcription.rhythm.quantized import QuantizedEvent, QuantizedPerformance


def to_single_voice(performance: QuantizedPerformance) -> QuantizedPerformance:
    """Return a copy in which notes never overlap, except as chords with a shared duration.

    - Notes starting together form a chord. The chord lasts as long as its longest note, but never
      past the next onset.
    - A later note cuts off whatever is still sounding (let-ring is not represented).
    - A repeated pitch within one chord is kept once (the first, i.e. from the earliest raw onset).

    Source events are carried over untouched, so raw timing stays available.
    """
    ordered = sorted(performance.events, key=lambda e: (e.onset_quarters, e.pitch_midi))
    groups = [list(group) for _, group in groupby(ordered, key=lambda e: e.onset_quarters)]

    voiced: list[QuantizedEvent] = []
    for index, group in enumerate(groups):
        onset = group[0].onset_quarters
        duration = max(event.duration_quarters for event in group)
        if index + 1 < len(groups):
            duration = min(duration, groups[index + 1][0].onset_quarters - onset)
        seen_pitches: set[int] = set()
        for event in sorted(group, key=lambda e: (e.pitch_midi, e.source.onset_seconds)):
            if event.pitch_midi in seen_pitches:
                continue
            seen_pitches.add(event.pitch_midi)
            voiced.append(replace(event, duration_quarters=duration))

    return replace(performance, events=tuple(voiced))


def is_single_voice(performance: QuantizedPerformance) -> bool:
    """True if every set of simultaneous notes shares one duration and ends by the next onset."""
    ordered = sorted(performance.events, key=lambda e: e.onset_quarters)
    groups = [list(group) for _, group in groupby(ordered, key=lambda e: e.onset_quarters)]
    for index, group in enumerate(groups):
        if len({event.duration_quarters for event in group}) > 1:
            return False
        if len({event.pitch_midi for event in group}) < len(group):
            return False
        if (
            index + 1 < len(groups)
            and group[0].offset_quarters > groups[index + 1][0].onset_quarters
        ):
            return False
    return True
