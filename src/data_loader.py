"""FastF1 data loading and lap-data cleaning.

This module is the entry point of the data pipeline. Typical use::

    from src.data_loader import (
        get_lap_data, get_race_list, load_race_session, setup_cache)

    setup_cache()                        # once, before loading anything
    get_race_list(2023)                  # look up races to choose from
    session = load_race_session(2023, "Bahrain")
    laps = get_lap_data(session)         # clean DataFrame for modelling
"""

import logging
from pathlib import Path

import fastf1
import pandas as pd
import requests
from fastf1.exceptions import DataNotLoadedError, FastF1CriticalError

logger = logging.getLogger(__name__)

# Project root = the folder that contains ``src/`` (this file lives in src/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# FastF1 has lap-level timing data from the 2018 season onwards.
MIN_SUPPORTED_YEAR = 2018

# Columns copied from FastF1's raw ``session.laps`` table.
LAP_COLUMNS = [
    "Driver", "LapNumber", "LapTime", "Compound", "TyreLife",
    "Stint", "PitInTime", "PitOutTime", "TrackStatus",
]

# Columns returned by get_race_list().
SCHEDULE_COLUMNS = [
    "RoundNumber", "EventName", "Location", "Country", "EventDate",
]

# Errors that mean "the data could not be fetched" (as opposed to a bug).
_NETWORK_ERRORS = (
    requests.exceptions.RequestException,  # no internet, timeout, ...
    FastF1CriticalError,                   # e.g. API rate limit exceeded
)


class DataLoadError(RuntimeError):
    """Raised when F1 data (or the local cache) cannot be prepared."""


def _validate_year(year):
    """Check that ``year`` is a season FastF1 has lap timing data for.

    Args:
        year: Season as an int (numpy integers and numeric strings work too).

    Returns:
        The year as a plain ``int``.

    Raises:
        ValueError: If ``year`` is not an integer >= MIN_SUPPORTED_YEAR.
    """
    try:
        year = int(year)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"year must be an integer, got {year!r}.") from exc
    if year < MIN_SUPPORTED_YEAR:
        raise ValueError(
            f"FastF1 has lap timing data from {MIN_SUPPORTED_YEAR} onwards; "
            f"got year={year}."
        )
    return year


def setup_cache(cache_dir="data/"):
    """Enable FastF1's on-disk cache so data is downloaded only once.

    Call this before loading any session. Relative paths are resolved
    against the project root (not the current working directory), so the
    cache lands in ``f1-strategy-simulator/data/`` whether the code runs
    from a notebook, from ``app.py`` or from a terminal. FastF1 refuses to
    start when the folder does not exist, so it is created here if needed.

    Args:
        cache_dir: Cache folder (str or Path). Defaults to ``"data/"``.

    Returns:
        The absolute ``Path`` of the cache folder now in use.

    Raises:
        DataLoadError: If the folder cannot be created or enabled
            (e.g. missing write permission).
    """
    path = Path(cache_dir).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    try:
        path.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(path))
    except OSError as exc:  # includes NotADirectoryError / PermissionError
        raise DataLoadError(
            f"Could not set up the FastF1 cache at '{path}': {exc}"
        ) from exc

    logger.info("FastF1 cache enabled at %s", path)
    return path


def get_race_list(year, only_completed=True):
    """List the Grand Prix weekends of a season so one can be chosen.

    Args:
        year: Season to list, 2018 or later.
        only_completed: If True (default), hide races that have not taken
            place yet, because there is no data to load for them.

    Returns:
        A DataFrame with one row per race in calendar order and the
        columns RoundNumber, EventName, Location, Country, EventDate.
        Pass ``EventName`` or ``RoundNumber`` to ``load_race_session``.

    Raises:
        ValueError: If ``year`` is not a supported season.
        DataLoadError: If the calendar cannot be downloaded.
    """
    year = _validate_year(year)

    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except (ValueError, *_NETWORK_ERRORS) as exc:
        raise DataLoadError(
            f"Could not load the {year} race calendar ({exc}). "
            "Check your internet connection and try again."
        ) from exc

    races = pd.DataFrame(schedule)[SCHEDULE_COLUMNS]
    if only_completed:
        races = races[races["EventDate"] < pd.Timestamp.now().normalize()]
    return races.reset_index(drop=True)


def load_race_session(year, gp_name, session_type="R", telemetry=False):
    """Download (or read from cache) one session and load its lap data.

    Args:
        year: Season, 2018 or later (e.g. ``2023``).
        gp_name: Grand Prix name (``"Bahrain"``, ``"Monza"``, ...) or round
            number (``1``, ``2``, ...). FastF1 fuzzy-matches names, so a
            typo can silently select another race; the ``RoundNumber``
            from ``get_race_list`` is always exact.
        session_type: FastF1 session code. ``"R"`` = race (default),
            ``"S"`` = sprint.
        telemetry: Also load car telemetry. Not needed for this project and
            much slower to download, so it is off by default.

    Returns:
        A loaded ``fastf1.core.Session`` with laps, weather and race
        control messages available.

    Raises:
        ValueError: If ``year`` is not a supported season.
        DataLoadError: If the session cannot be found, the download fails,
            or FastF1 returns no lap data.
    """
    year = _validate_year(year)
    label = f"{year} {gp_name} (session '{session_type}')"

    try:
        session = fastf1.get_session(year, gp_name, session_type)
    except (ValueError, *_NETWORK_ERRORS) as exc:
        raise DataLoadError(
            f"Could not find {label}: {exc}. Check the Grand Prix name, "
            "the session code and your internet connection."
        ) from exc

    event_name = session.event["EventName"]
    if not session.f1_api_support:
        raise DataLoadError(f"No lap timing data exists for {event_name}.")

    try:
        session.load(
            laps=True, telemetry=telemetry, weather=True, messages=True
        )
    except _NETWORK_ERRORS as exc:
        raise DataLoadError(
            f"Network/API problem while loading {label}: {exc}"
        ) from exc

    # FastF1 catches most download errors inside load() and only logs a
    # warning, so a failed download surfaces here, when laps are accessed.
    try:
        laps = session.laps
    except DataNotLoadedError as exc:
        raise DataLoadError(
            f"FastF1 could not load lap data for {event_name} {year}. "
            "Likely causes: no internet connection, the F1 data servers "
            "are unreachable, or the race has not taken place / been "
            "published yet. See the FastF1 log messages above for details."
        ) from exc
    if laps.empty:
        raise DataLoadError(f"{event_name} {year} contains no lap data.")

    return session


def get_lap_data(session):
    """Convert a loaded session into a clean, model-ready lap table.

    No lap is ever dropped. Laps that later modules may want to exclude
    (or treat specially) are marked with boolean flag columns instead.

    Args:
        session: A ``fastf1.core.Session`` returned by
            ``load_race_session``.

    Returns:
        A DataFrame with one row per lap for every driver, sorted by
        Driver then LapNumber, with these columns:

        - Driver (str): three-letter code, e.g. ``"VER"``.
        - LapNumber (int): race lap, starting at 1.
        - LapTime (float): lap time in seconds; NaN when not recorded.
        - Compound (str): upper-case tyre compound; ``"UNKNOWN"`` when
          the data has none.
        - TyreLife (float): laps completed on this set of tyres,
          including laps from earlier sessions for used sets.
        - Stint (Int64): stint number of the driver (1 = first set).
        - PitInTime / PitOutTime (Timedelta): session time at pit entry /
          exit; NaT when the driver did not pit on that lap.
        - TrackStatus (str): every track-status code seen during the lap
          (e.g. ``"1"``, ``"12"``, ``"124"``); ``""`` if unknown.
        - MissingLapTime (bool): True when LapTime is NaN.
        - IsInLap (bool): True when the lap ends with a pit entry.
        - IsOutLap (bool): True when the lap starts with a pit exit.

    Raises:
        DataLoadError: If the session has no lap data or FastF1 no longer
            provides the expected columns.
    """
    try:
        raw_laps = session.laps
    except DataNotLoadedError as exc:
        raise DataLoadError(
            "The session has no lap data. Pass the object returned by "
            "load_race_session()."
        ) from exc
    if raw_laps.empty:
        raise DataLoadError("The session contains no laps.")

    absent = [col for col in LAP_COLUMNS if col not in raw_laps.columns]
    if absent:
        raise DataLoadError(
            f"FastF1 lap data lacks the expected columns {absent}. Your "
            "FastF1 version may differ from requirements.txt."
        )

    # Work on a plain DataFrame copy; never modify FastF1's session object.
    laps = pd.DataFrame(raw_laps)[LAP_COLUMNS].copy()

    # Lap time: Timedelta -> float seconds (NaT becomes NaN).
    laps["LapTime"] = pd.to_timedelta(
        laps["LapTime"], errors="coerce"
    ).dt.total_seconds()

    # Tidy dtypes. Stint is nullable so a missing value stays missing.
    laps["LapNumber"] = laps["LapNumber"].astype(int)
    laps["Stint"] = laps["Stint"].astype("Int64")
    laps["TyreLife"] = laps["TyreLife"].astype(float)

    # Text columns: consistent case, no missing values to trip up .str.
    # FastF1 uses an empty string when a driver has no tyre data.
    laps["Compound"] = (
        laps["Compound"]
        .astype("string")
        .str.strip()
        .str.upper()
        .fillna("UNKNOWN")
        .replace("", "UNKNOWN")
        .astype(object)
    )
    laps["TrackStatus"] = laps["TrackStatus"].fillna("").astype(str)

    # Flag laps instead of dropping them; later modules decide what to do.
    laps["MissingLapTime"] = laps["LapTime"].isna()
    laps["IsInLap"] = laps["PitInTime"].notna()
    laps["IsOutLap"] = laps["PitOutTime"].notna()

    laps = laps.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)

    logger.info(
        "Prepared %d laps for %d drivers; %d laps have no lap time "
        "(flagged with MissingLapTime, not dropped).",
        len(laps), laps["Driver"].nunique(), int(laps["MissingLapTime"].sum()),
    )
    return laps
