import pytest

from guitar_transcription.domain import (
    STANDARD_GUITAR,
    STANDARD_TUNING,
    FretboardPosition,
    GuitarConfig,
)
from guitar_transcription.guitar import candidate_positions, pitch_for_position, pitch_range

P = FretboardPosition

STANDARD_24 = GuitarConfig(STANDARD_TUNING, max_fret=24)
DROP_D = GuitarConfig((64, 59, 55, 50, 45, 38))  # E4 B3 G3 D3 A2 D2
OPEN_G = GuitarConfig((62, 59, 55, 50, 43, 38))  # D4 B3 G3 D3 G2 D2
SEVEN_STRING = GuitarConfig((*STANDARD_TUNING, 35), max_fret=24)  # + low B1
NASHVILLE = GuitarConfig((64, 59, 67, 62, 57, 52))  # re-entrant: G/D/A/E an octave up
CAPO_2 = GuitarConfig(STANDARD_TUNING, capo=2)

CONFIGS = [STANDARD_GUITAR, STANDARD_24, DROP_D, OPEN_G, SEVEN_STRING, NASHVILLE, CAPO_2]


# --- pitch_for_position ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("string", "pitch"), [(1, 64), (2, 59), (3, 55), (4, 50), (5, 45), (6, 40)]
)
def test_every_open_string_in_standard_tuning(string: int, pitch: int) -> None:
    assert pitch_for_position(STANDARD_GUITAR, string, 0) == pitch


def test_string_1_is_highest_and_string_6_is_lowest_in_standard_tuning() -> None:
    open_pitches = [pitch_for_position(STANDARD_GUITAR, s, 0) for s in range(1, 7)]

    assert open_pitches == sorted(open_pitches, reverse=True)
    assert max(open_pitches) == pitch_for_position(STANDARD_GUITAR, 1, 0)
    assert min(open_pitches) == pitch_for_position(STANDARD_GUITAR, 6, 0)


@pytest.mark.parametrize(
    ("string", "fret", "pitch"),
    [(1, 12, 76), (6, 5, 45), (6, 12, 52), (3, 4, 59), (1, 22, 86)],
)
def test_fretted_positions(string: int, fret: int, pitch: int) -> None:
    assert pitch_for_position(STANDARD_GUITAR, string, fret) == pitch


def test_drop_d_lowers_only_string_6() -> None:
    assert pitch_for_position(DROP_D, 6, 0) == 38  # D2
    assert pitch_for_position(DROP_D, 6, 2) == 40  # E2 now needs fret 2
    for string in range(1, 6):
        assert pitch_for_position(DROP_D, string, 0) == STANDARD_TUNING[string - 1]


def test_capo_fret_is_physical() -> None:
    assert pitch_for_position(CAPO_2, 6, 2) == 42  # capo'd "open" low string = F#2
    assert pitch_for_position(CAPO_2, 1, 5) == 69  # same as without capo: physical fret 5


def test_max_fret_boundary_is_inclusive() -> None:
    assert pitch_for_position(STANDARD_GUITAR, 1, 22) == 86
    with pytest.raises(ValueError, match=r"fret must be in 0\.\.22, got 23"):
        pitch_for_position(STANDARD_GUITAR, 1, 23)


@pytest.mark.parametrize("string", [0, -1, 7, 100])
def test_invalid_string_numbers(string: int) -> None:
    with pytest.raises(ValueError, match=r"string must be in 1\.\.6"):
        pitch_for_position(STANDARD_GUITAR, string, 0)


def test_seventh_string_exists_only_on_seven_string() -> None:
    assert pitch_for_position(SEVEN_STRING, 7, 0) == 35
    with pytest.raises(ValueError, match="string"):
        pitch_for_position(STANDARD_GUITAR, 7, 0)


@pytest.mark.parametrize(
    ("config", "fret", "message"),
    [
        (STANDARD_GUITAR, -1, r"fret must be in 0\.\.22"),
        (STANDARD_GUITAR, 23, r"fret must be in 0\.\.22"),
        (CAPO_2, 0, r"fret must be in 2\.\.22"),  # behind the capo
        (CAPO_2, 1, r"fret must be in 2\.\.22"),
    ],
)
def test_invalid_fret_values(config: GuitarConfig, fret: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        pitch_for_position(config, 1, fret)


@pytest.mark.parametrize(("string", "fret"), [(1.0, 0), (1, 5.0), (True, 0), (1, None)])
def test_non_int_string_or_fret_is_a_type_error(string: object, fret: object) -> None:
    with pytest.raises(TypeError, match="must be an int"):
        pitch_for_position(STANDARD_GUITAR, string, fret)  # type: ignore[arg-type]


# --- candidate_positions -----------------------------------------------------------------


def test_e4_has_six_positions_on_24_fret_standard() -> None:
    assert candidate_positions(STANDARD_24, 64) == (
        P(1, 0),
        P(2, 5),
        P(3, 9),
        P(4, 14),
        P(5, 19),
        P(6, 24),
    )


def test_max_fret_removes_positions() -> None:
    # Same pitch on a 22-fret neck: low E string would need fret 24.
    assert candidate_positions(STANDARD_GUITAR, 64) == (
        P(1, 0),
        P(2, 5),
        P(3, 9),
        P(4, 14),
        P(5, 19),
    )


def test_candidates_are_ordered_by_string_number() -> None:
    for pitch in range(40, 87):
        strings = [c.string for c in candidate_positions(STANDARD_GUITAR, pitch)]
        assert strings == sorted(strings)


@pytest.mark.parametrize(
    ("pitch", "expected"),
    [
        (40, (P(6, 0),)),  # lowest note: only the open low E
        (45, (P(5, 0), P(6, 5))),  # A2
        (59, (P(2, 0), P(3, 4), P(4, 9), P(5, 14), P(6, 19))),  # B3
        (86, (P(1, 22),)),  # highest note on a 22-fret neck
    ],
)
def test_notes_playable_at_multiple_or_single_positions(
    pitch: int, expected: tuple[FretboardPosition, ...]
) -> None:
    assert candidate_positions(STANDARD_GUITAR, pitch) == expected


@pytest.mark.parametrize(("string", "pitch"), list(enumerate(STANDARD_TUNING, start=1)))
def test_every_open_string_is_a_candidate_for_its_pitch(string: int, pitch: int) -> None:
    assert P(string, 0) in candidate_positions(STANDARD_GUITAR, pitch)


@pytest.mark.parametrize("pitch", [0, 20, 39])
def test_below_instrument_range_returns_empty(pitch: int) -> None:
    assert candidate_positions(STANDARD_GUITAR, pitch) == ()


@pytest.mark.parametrize("pitch", [87, 100, 127])
def test_above_max_fret_returns_empty(pitch: int) -> None:
    assert candidate_positions(STANDARD_GUITAR, pitch) == ()


def test_drop_d_extends_range_down_and_moves_low_positions() -> None:
    assert candidate_positions(DROP_D, 38) == (P(6, 0),)
    assert candidate_positions(STANDARD_GUITAR, 38) == ()
    assert candidate_positions(DROP_D, 40) == (P(6, 2),)  # E2 is no longer an open string
    assert candidate_positions(DROP_D, 45) == (P(5, 0), P(6, 7))


def test_capo_excludes_frets_behind_it_and_shifts_open_pitches() -> None:
    assert candidate_positions(CAPO_2, 40) == ()  # open low E is impossible under the capo
    assert candidate_positions(CAPO_2, 41) == ()  # so is fret 1
    assert candidate_positions(CAPO_2, 42) == (P(6, 2),)  # capo'd open string, +2 semitones
    for pitch in range(128):
        assert all(c.fret >= 2 for c in candidate_positions(CAPO_2, pitch))


def test_capo_keeps_open_string_pitch_plus_capo_as_candidate() -> None:
    for string, open_pitch in enumerate(STANDARD_TUNING, start=1):
        assert P(string, 2) in candidate_positions(CAPO_2, open_pitch + 2)


def test_custom_open_g_tuning() -> None:
    assert candidate_positions(OPEN_G, 43) == (P(5, 0), P(6, 5))  # G2
    assert (
        candidate_positions(OPEN_G, 62)  # D4; string 6 would need fret 24 on a 22-fret neck
        == (P(1, 0), P(2, 3), P(3, 7), P(4, 12), P(5, 19))
    )


def test_reentrant_tuning_orders_by_string_not_pitch() -> None:
    # Nashville: string 3 (G4) is higher than string 1 (E4); G4 is playable open on string 3.
    candidates = candidate_positions(NASHVILLE, 67)

    assert candidates[0] == P(1, 3)
    assert P(3, 0) in candidates


def test_seven_string_low_b() -> None:
    assert candidate_positions(SEVEN_STRING, 35) == (P(7, 0),)
    assert P(7, 5) in candidate_positions(SEVEN_STRING, 40)


def test_single_string_instrument() -> None:
    one_string = GuitarConfig((40,), max_fret=5)

    assert candidate_positions(one_string, 43) == (P(1, 3),)
    assert candidate_positions(one_string, 46) == ()


def test_unison_strings_give_one_candidate_per_string() -> None:
    doubled = GuitarConfig((52, 52))

    assert candidate_positions(doubled, 55) == (P(1, 3), P(2, 3))


@pytest.mark.parametrize("pitch", [-1, 128])
def test_invalid_midi_pitch_raises(pitch: int) -> None:
    with pytest.raises(ValueError, match="pitch_midi"):
        candidate_positions(STANDARD_GUITAR, pitch)


def test_non_int_pitch_raises() -> None:
    with pytest.raises(TypeError, match="pitch_midi"):
        candidate_positions(STANDARD_GUITAR, 64.0)  # type: ignore[arg-type]


# --- invariants over whole instruments ---------------------------------------------------


@pytest.mark.parametrize("config", CONFIGS)
def test_every_candidate_sounds_the_requested_pitch(config: GuitarConfig) -> None:
    for pitch in range(128):
        for candidate in candidate_positions(config, pitch):
            assert pitch_for_position(config, candidate.string, candidate.fret) == pitch


@pytest.mark.parametrize("config", CONFIGS)
def test_every_playable_position_is_found(config: GuitarConfig) -> None:
    for string in range(1, config.num_strings + 1):
        for fret in range(config.min_fret, config.max_fret + 1):
            pitch = pitch_for_position(config, string, fret)
            assert P(string, fret) in candidate_positions(config, pitch)


@pytest.mark.parametrize("config", CONFIGS)
def test_candidates_are_deterministic(config: GuitarConfig) -> None:
    for pitch in range(128):
        assert candidate_positions(config, pitch) == candidate_positions(config, pitch)


# --- pitch_range -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (STANDARD_GUITAR, (40, 86)),
        (STANDARD_24, (40, 88)),
        (DROP_D, (38, 86)),
        (CAPO_2, (42, 86)),
        (SEVEN_STRING, (35, 88)),
        (NASHVILLE, (52, 89)),  # highest open string is string 3 (G4), not string 1
    ],
)
def test_pitch_range(config: GuitarConfig, expected: tuple[int, int]) -> None:
    assert pitch_range(config) == expected


@pytest.mark.parametrize("config", CONFIGS)
def test_pitch_range_bounds_candidates(config: GuitarConfig) -> None:
    low, high = pitch_range(config)

    assert candidate_positions(config, low)
    assert candidate_positions(config, high)
    assert candidate_positions(config, low - 1) == ()
    assert candidate_positions(config, high + 1) == ()
