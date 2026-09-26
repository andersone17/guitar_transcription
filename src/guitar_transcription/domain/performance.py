"""``Performance``: the events of one recording, bound to the instrument they were played on."""

from collections.abc import Iterable
from dataclasses import dataclass

from guitar_transcription.domain.events import PerformanceEvent
from guitar_transcription.domain.guitar_config import GuitarConfig


def _sort_key(event: PerformanceEvent) -> tuple[float, int]:
    return (event.onset_seconds, event.pitch_midi)


# init=False: the custom __init__ accepts any iterable and normalizes it to a sorted tuple.
@dataclass(frozen=True, slots=True, init=False)
class Performance:
    """Time-ordered ``PerformanceEvent``s plus the ``GuitarConfig`` they refer to.

    Events are always stored sorted by (onset, pitch), so consumers can rely on the order.
    Any event with a known string/fret is checked against the config: the position must be
    playable and must actually sound ``pitch_midi``.
    """

    events: tuple[PerformanceEvent, ...]
    config: GuitarConfig

    def __init__(self, events: Iterable[PerformanceEvent], config: GuitarConfig) -> None:
        ordered = tuple(sorted(events, key=_sort_key))
        for event in ordered:
            _check_position(event, config)
        object.__setattr__(self, "events", ordered)
        object.__setattr__(self, "config", config)

    def __len__(self) -> int:
        return len(self.events)

    @property
    def end_seconds(self) -> float:
        """Latest offset of any event; 0.0 for an empty performance."""
        return max((event.offset_seconds for event in self.events), default=0.0)


def _check_position(event: PerformanceEvent, config: GuitarConfig) -> None:
    if event.string is None or event.fret is None:
        return
    try:
        sounded = config.pitch_at(event.string, event.fret)
    except ValueError as error:
        raise ValueError(f"event at {event.onset_seconds}s: {error}") from None
    if sounded != event.pitch_midi:
        raise ValueError(
            f"event at {event.onset_seconds}s: string {event.string} fret {event.fret} sounds "
            f"MIDI {sounded}, but pitch_midi is {event.pitch_midi}"
        )
