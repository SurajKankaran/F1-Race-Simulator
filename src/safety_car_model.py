"""Safety car (SC) and virtual safety car (VSC) modelling.

Three building blocks for the Monte Carlo strategy simulation:

1. ``extract_safety_car_periods`` - find the SC/VSC laps of one race.
2. ``historical_sc_probability``  - how often, and when, a circuit sees them.
3. ``simulate_random_sc_event``   - draw a random SC scenario for a race.

Typical use::

    info = historical_sc_probability("Baku")
    event = simulate_random_sc_event(info["sc_probability"], 51, info)
"""

import logging
import unicodedata

import numpy as np
import pandas as pd

from src.data_loader import (
    DataLoadError, get_lap_data, get_race_list, load_race_session,
)

logger = logging.getLogger(__name__)

# FastF1 TrackStatus codes: "4" = Safety Car, "6" = VSC deployed,
# "7" = VSC ending. A lap's TrackStatus lists every code seen during it.
_SC_CODE = "4"
_VSC_CODES = "[67]"

_PERIOD_COLUMNS = ["kind", "start_lap", "end_lap", "n_laps"]

# Fallback P(at least one SC or VSC in a race) for circuits with too little
# history (new venues, cancelled seasons). Reasoning for 0.65:
#  * In the SC + VSC era (2014-2017) roughly 60-75% of races were affected
#    by at least one neutralisation, so "about two races in three" is the
#    pooled all-circuit rate.
#  * Circuits differ hugely (Singapore is near-certain, Bahrain is rare), so
#    the pooled rate is the fair prior when nothing circuit-specific is known.
#  * Recent new venues are mostly street tracks, which lean higher, but not
#    every new track is one, so a street-circuit value would be too bold.
DEFAULT_SC_PROBABILITY = 0.65

# Fewer usable races than this cannot support a probability: with two races
# the estimate can only be 0%, 50% or 100%.
MIN_RACES_FOR_HISTORY = 3

# Typical length of a neutralisation in laps (counting the deployment lap).
DEFAULT_SC_DURATION_LAPS = 4.0

# Sampling settings for simulate_random_sc_event().
MAX_START_LAP_CENTRES = 3    # average lap of the 1st, 2nd and 3rd SC of a race
START_LAP_SPREAD = 0.10      # std dev of the start lap, share of race length
DURATION_SPREAD_LAPS = 1.0   # std dev of the sampled duration, in laps
MIN_SC_LAP = 1
MIN_RACE_LENGTH = 3
_MAX_DRAWS = 25              # rejection-sampling attempts before clipping

# Shared generator used when the caller does not supply one.
_RNG = np.random.default_rng()


def extract_safety_car_periods(laps_df, min_driver_fraction=0.5):
    """Find the laps of a race that ran under a Safety Car or VSC.

    Each driver crosses the line at a different moment, so one SC period
    shows up in slightly different lap numbers for different drivers. A lap
    number therefore counts as neutralised when at least
    ``min_driver_fraction`` of the drivers show an SC or VSC code in it. This
    usually lands on the race leader's lap and occasionally one lap earlier.
    Consecutive neutralised laps are merged into one period. Red-flag laps
    (code 5) are not counted.

    Args:
        laps_df: Lap table from ``data_loader.get_lap_data`` (needs the
            ``LapNumber`` and ``TrackStatus`` columns; all drivers' laps).
        min_driver_fraction: Share of drivers (0 < x <= 1) that must show the
            status on a lap number for the lap to count. Default 0.5.

    Returns:
        DataFrame with one row per period, ordered by start lap, and the
        columns ``kind`` ("SC" if any Safety Car lap is inside, else
        "VSC"), ``start_lap``, ``end_lap`` (inclusive) and ``n_laps``.
        Empty (same columns) when the race had no SC or VSC.

    Raises:
        ValueError: If a required column is missing, ``min_driver_fraction``
            is out of range, or TrackStatus is empty for every lap (the data
            is missing, which must not be mistaken for a clean race).
    """
    missing = {"LapNumber", "TrackStatus"} - set(laps_df.columns)
    if missing:
        raise ValueError(f"laps_df is missing column(s): {sorted(missing)}")
    if not 0.0 < min_driver_fraction <= 1.0:
        raise ValueError("min_driver_fraction must be in (0, 1].")

    status = laps_df["TrackStatus"].fillna("").astype(str)
    if not (status != "").any():
        raise ValueError(
            "TrackStatus is empty for every lap; no track status data."
        )

    flags = pd.DataFrame({
        "lap": pd.to_numeric(laps_df["LapNumber"], errors="coerce"),
        "sc": status.str.contains(_SC_CODE, regex=False),
        "vsc": status.str.contains(_VSC_CODES),
    }).dropna(subset=["lap"])
    flags["lap"] = flags["lap"].astype(int)

    # Fraction of drivers showing each status, per lap number.
    share = flags.groupby("lap")[["sc", "vsc"]].mean()
    flagged_laps = share.index[(share >= min_driver_fraction).any(axis=1)]

    # Merge runs of consecutive flagged laps into periods.
    periods = []
    for lap in flagged_laps:
        if periods and lap == periods[-1]["end_lap"] + 1:
            periods[-1]["end_lap"] = int(lap)
        else:
            periods.append({"start_lap": int(lap), "end_lap": int(lap)})

    for period in periods:
        window = share.loc[period["start_lap"]:period["end_lap"]]
        has_sc = window["sc"].max() >= window["vsc"].max()
        period["kind"] = "SC" if has_sc else "VSC"
        period["n_laps"] = period["end_lap"] - period["start_lap"] + 1

    result = pd.DataFrame(periods, columns=_PERIOD_COLUMNS)
    return result.astype(
        {"start_lap": int, "end_lap": int, "n_laps": int}
    )


def _normalise(text):
    """Lower-case ``text`` and strip accents ("Sao Paulo" == "São Paulo")."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return plain.lower().strip()


def _circuit_ids(races):
    """Label each race in ``races`` with the circuit it was held at.

    FastF1's calendar spells some venues differently from year to year
    (Monaco is "Monte Carlo" in 2019 but "Monaco" in 2022; Singapore is
    "Marina Bay" in 2022), and some events are renamed (Brazilian GP -> Sao
    Paulo GP). Two races are therefore treated as the same circuit when they
    share a Location OR an EventName.

    Args:
        races: DataFrame with ``Location`` and ``EventName`` columns.

    Returns:
        List of ints, one circuit id per row of ``races``.
    """
    parent = list(range(len(races)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_seen = {}
    for i, row in enumerate(races.itertuples(index=False)):
        for key in (("loc", _normalise(row.Location)),
                    ("name", _normalise(row.EventName))):
            if key in first_seen:
                parent[find(i)] = find(first_seen[key])
            else:
                first_seen[key] = i
    return [find(i) for i in range(len(races))]


def _find_circuit_races(track_name, years):
    """Find every race held at one circuit during the given seasons.

    A race matches when ``track_name`` appears (ignoring case and accents)
    in its event name, location or country. If the name matches several
    different circuits (e.g. "Italy" -> Monza, Imola, Mugello) the one with
    the most races is used and a warning lists the others.

    Args:
        track_name: Circuit or Grand Prix name, e.g. "Baku" or "Monza".
        years: Seasons to search.

    Returns:
        Tuple ``(races, circuit)``: a DataFrame (Year, RoundNumber,
        EventName, Location; possibly empty) and the circuit's label (its
        latest Location) or None if nothing matched.
    """
    query = _normalise(track_name)
    frames = []
    for year in years:
        try:
            calendar = get_race_list(year)
        except (DataLoadError, ValueError) as exc:
            logger.warning("No race calendar for %s (%s); skipping.",
                           year, exc)
            continue
        hit = pd.Series(False, index=calendar.index)
        for column in ("EventName", "Location", "Country"):
            hit |= (calendar[column].astype(str).map(_normalise)
                    .str.contains(query, regex=False))
        found = calendar.loc[hit, ["RoundNumber", "EventName", "Location"]]
        frames.append(found.assign(Year=year))

    columns = ["Year", "RoundNumber", "EventName", "Location"]
    if not frames or all(f.empty for f in frames):
        return pd.DataFrame(columns=columns), None

    races = pd.concat(frames, ignore_index=True)[columns]
    races = races.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)
    races["circuit"] = _circuit_ids(races)

    groups = races.groupby("circuit")
    sizes = groups.size()
    latest = groups["Year"].max()
    chosen = max(sizes.index, key=lambda g: (sizes[g], latest[g]))
    if len(sizes) > 1:
        options = ", ".join(
            f"{groups['Location'].last()[g]} ({sizes[g]})" for g in sizes.index
        )
        logger.warning(
            "%r matches several circuits: %s. Using the one with the most "
            "races; pass a more specific name to choose another.",
            track_name, options,
        )

    races = races[races["circuit"] == chosen].drop(columns="circuit")
    return races.reset_index(drop=True), races["Location"].iloc[-1]


def _default_result(track_name, circuit, labels, default_probability):
    """Build the fallback result returned when history is insufficient."""
    return {
        "track_name": track_name,
        "circuit": circuit,
        "sc_probability": float(default_probability),
        "avg_sc_start_laps": [],
        "sc_start_weights": [],
        "avg_sc_duration_laps": DEFAULT_SC_DURATION_LAPS,
        "sample_size": len(labels),
        "races_used": labels,
        "used_default": True,
    }


def _summarise(track_name, circuit, analysed):
    """Turn per-race SC periods into the summary dictionary.

    Args:
        track_name: Name the caller asked for.
        circuit: Label of the circuit that name resolved to.
        analysed: List of ``(race_label, periods_df)`` pairs, where each
            ``periods_df`` comes from ``extract_safety_car_periods``.

    Returns:
        The dictionary documented in ``historical_sc_probability``.
    """
    labels = [label for label, _ in analysed]
    starts = [periods["start_lap"].tolist() for _, periods in analysed]
    durations = [n for _, periods in analysed for n in periods["n_laps"]]

    probability = sum(1 for s in starts if s) / len(starts)

    # Average start lap of the 1st, 2nd, ... SC/VSC period of each race.
    # counts[k] = number of races that had a (k+1)th period.
    centres, counts = [], []
    for k in range(MAX_START_LAP_CENTRES):
        kth_starts = [s[k] for s in starts if len(s) > k]
        if not kth_starts:
            break
        centres.append(int(round(float(np.mean(kth_starts)))))
        counts.append(len(kth_starts))
    weights = [round(c / sum(counts), 3) for c in counts]

    avg_duration = (round(float(np.mean(durations)), 1) if durations
                    else DEFAULT_SC_DURATION_LAPS)
    return {
        "track_name": track_name,
        "circuit": circuit,
        "sc_probability": round(probability, 3),
        "avg_sc_start_laps": centres,
        "sc_start_weights": weights,
        "avg_sc_duration_laps": avg_duration,
        "sample_size": len(analysed),
        "races_used": labels,
        "used_default": False,
    }


def historical_sc_probability(
    track_name,
    years=range(2019, 2024),
    default_probability=DEFAULT_SC_PROBABILITY,
    min_races=MIN_RACES_FOR_HISTORY,
):
    """Estimate how often, and when, a circuit sees a Safety Car or VSC.

    Finds the races held at the circuit in ``years``, downloads each with
    ``data_loader``, extracts the SC/VSC periods and summarises them. Races
    that fail to load are skipped with a warning. Data is cached by FastF1,
    so only the first call for a circuit is slow. Call
    ``data_loader.setup_cache()`` first.

    If fewer than ``min_races`` usable races are found (new circuits,
    cancelled seasons, no internet) the history is too thin to trust, so the
    default probability is returned instead and ``used_default`` is True.

    Args:
        track_name: Circuit or Grand Prix name ("Baku", "Monza", "Bahrain").
            Use the full GP name ("Bahrain Grand Prix") to exclude other
            events at the same venue (the 2020 Sakhir GP).
        years: Seasons to sample. Default 2019-2023.
        default_probability: Fallback probability (0-1) for thin history.
        min_races: Minimum usable races needed to trust the history.

    Returns:
        Dictionary with:

        - ``track_name`` (str): the name passed in.
        - ``circuit`` (str | None): the circuit it resolved to.
        - ``sc_probability`` (float): share of sampled races with at least
          one SC or VSC (or ``default_probability``).
        - ``avg_sc_start_laps`` (list[int]): average start lap of the 1st,
          2nd (and 3rd) SC/VSC period of a race, e.g. ``[12, 34]``. Empty
          when there is no history.
        - ``sc_start_weights`` (list[float]): how often each of those
          periods occurred (sums to 1); used to pick between them.
        - ``avg_sc_duration_laps`` (float): mean period length in laps.
        - ``sample_size`` (int): number of races analysed.
        - ``races_used`` (list[str]): the races behind the numbers.
        - ``used_default`` (bool): True if the default was substituted.

    Raises:
        ValueError: If an argument is invalid.
    """
    if not isinstance(track_name, str) or not track_name.strip():
        raise ValueError("track_name must be a non-empty string.")
    if not 0.0 <= default_probability <= 1.0:
        raise ValueError("default_probability must be between 0 and 1.")
    if min_races < 1:
        raise ValueError("min_races must be at least 1.")
    years = sorted({int(year) for year in years})

    races, circuit = _find_circuit_races(track_name, years)
    if races.empty:
        logger.warning("No races found for %r in %s-%s.",
                       track_name, years[0] if years else "?",
                       years[-1] if years else "?")

    analysed = []
    for row in races.itertuples(index=False):
        label = f"{row.Year} {row.EventName}"
        logger.info("Analysing %s ...", label)
        try:
            session = load_race_session(int(row.Year), int(row.RoundNumber))
            periods = extract_safety_car_periods(get_lap_data(session))
        except (DataLoadError, ValueError) as exc:
            logger.warning("Skipping %s: %s", label, exc)
            continue
        analysed.append((label, periods))

    if len(analysed) < min_races:
        logger.warning(
            "Only %d usable race(s) for %r (need %d); using the default SC "
            "probability of %.2f.",
            len(analysed), track_name, min_races, default_probability,
        )
        labels = [label for label, _ in analysed]
        return _default_result(track_name, circuit, labels,
                               default_probability)
    return _summarise(track_name, circuit, analysed)


def _sample_start_lap(generator, race_length, centres, weights):
    """Draw the lap on which a simulated SC starts.

    Args:
        generator: ``numpy.random.Generator`` to draw from.
        race_length: Total laps of the simulated race.
        centres: Historical average start laps. If empty, the start lap is
            uniform over the race.
        weights: Relative frequency of each centre (None = equal).

    Returns:
        The start lap, an int between MIN_SC_LAP and ``race_length - 1``.
    """
    first, last = MIN_SC_LAP, race_length - 1
    if not centres:
        return int(generator.integers(first, last + 1))

    probs = None
    if weights is not None and len(weights) == len(centres):
        probs = np.asarray(weights, dtype=float)
        probs = probs / probs.sum() if probs.sum() > 0 else None
    centre = float(generator.choice(centres, p=probs))

    # Normal distribution around the centre, kept inside the race by
    # redrawing (a clipped normal would pile up on the boundary laps).
    sigma = max(1.0, START_LAP_SPREAD * race_length)
    for _ in range(_MAX_DRAWS):
        lap = int(round(generator.normal(centre, sigma)))
        if first <= lap <= last:
            return lap
    return int(np.clip(round(centre), first, last))


def simulate_random_sc_event(probability, race_length, history=None,
                             rng=None):
    """Randomly decide whether a simulated race has an SC and when.

    First a coin is flipped with ``probability``. If an SC occurs, its start
    lap is drawn from a normal distribution centred on one of the historical
    average start laps (chosen by how often each occurred), and its length
    from a normal distribution around the historical average duration.
    Meant to be called once per scenario in a Monte Carlo loop.

    Args:
        probability: Chance (0-1) that the race has an SC/VSC, e.g.
            ``info["sc_probability"]`` or a value picked by the user.
        race_length: Total laps of the simulated race (at least 3).
        history: Optional dictionary from ``historical_sc_probability``. Its
            ``avg_sc_start_laps``, ``sc_start_weights`` and
            ``avg_sc_duration_laps`` shape the event. Without it the start
            lap is uniform over the race and the duration averages
            ``DEFAULT_SC_DURATION_LAPS``.
        rng: Optional ``numpy.random.Generator`` (or int seed) for
            reproducible runs. Create one Generator and reuse it across
            calls; an int seed restarts the sequence on every call.

    Returns:
        Dictionary with ``sc_occurs`` (bool), ``sc_start_lap`` (int or
        None), ``sc_end_lap`` (int or None, inclusive) and
        ``sc_duration_laps`` (int, 0 when there is no SC).

    Raises:
        ValueError: If ``probability`` is outside 0-1 or the race is too
            short.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"probability must be between 0 and 1, got {probability!r}."
        )
    race_length = int(race_length)
    if race_length < MIN_RACE_LENGTH:
        raise ValueError(
            f"race_length must be at least {MIN_RACE_LENGTH} laps."
        )
    generator = _RNG if rng is None else np.random.default_rng(rng)

    if generator.random() >= probability:
        return {"sc_occurs": False, "sc_start_lap": None,
                "sc_end_lap": None, "sc_duration_laps": 0}

    history = history or {}
    centres = history.get("avg_sc_start_laps")
    start_lap = _sample_start_lap(
        generator, race_length,
        [] if centres is None else list(centres),
        history.get("sc_start_weights"),
    )
    mean_duration = (history.get("avg_sc_duration_laps")
                     or DEFAULT_SC_DURATION_LAPS)
    duration = max(1, int(round(
        generator.normal(mean_duration, DURATION_SPREAD_LAPS)
    )))
    end_lap = min(start_lap + duration - 1, race_length)
    return {"sc_occurs": True, "sc_start_lap": start_lap,
            "sc_end_lap": end_lap,
            "sc_duration_laps": end_lap - start_lap + 1}
