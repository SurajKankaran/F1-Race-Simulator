"""Race strategy simulation engine (pit stops, tyre wear, safety cars).

A strategy is a list of stints. The engine walks through the race lap by
lap, times each lap with the degradation model of the compound in use, adds
a time penalty for every pit stop and returns the total race time. On top of
that it can drop a Safety Car into the race (which makes nearby pit stops
cheaper) and run Monte Carlo experiments over random Safety Car scenarios.

Degradation models (from Part 2)
--------------------------------
``degradation_models`` maps each compound name ("SOFT", "MEDIUM", "HARD")
to a model that returns the lap time in seconds for a tyre age, where age 1
is the first lap on a fresh set. Any of these work:

* a callable ``model(tyre_age)`` - a function, lambda or ``np.poly1d``;
* an object with a ``.predict()`` method (scikit-learn style);
* polynomial coefficients, highest power first (``np.polyfit`` order).

Typical use::

    strategy = Strategy.from_pit_laps("1-Stop", ["MEDIUM", "HARD"], [22], 57)
    seconds = simulate_strategy(strategy, models, race_length=57)

Run ``python -m src.strategy_simulator`` from the project root for a worked
example on a real race.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_loader import get_lap_data, load_race_session, setup_cache
from src.safety_car_model import (
    DEFAULT_SC_DURATION_LAPS, historical_sc_probability,
    simulate_random_sc_event,
)

logger = logging.getLogger(__name__)

# Time lost by a green-flag pit stop compared with staying out: the trip
# through the pit lane at the speed limit plus the stop itself (2-3 s), minus
# the time the car would have covered on track meanwhile. That comes to
# roughly 20-25 s at most circuits (about 18-28 s from the shortest pit lanes
# to the longest), so 22 s is a sensible default. It is circuit-specific,
# which is why every function takes it as a parameter.
DEFAULT_PIT_STOP_LOSS_S = 22.0

# Time lost by a stop made under a Safety Car: the field crawls, so the same
# pit-lane trip costs far less against the cars on track. 8 s is generous;
# measured figures are often higher (roughly 10-15 s), so raise it for a more
# cautious estimate.
DEFAULT_SC_PIT_LOSS_S = 8.0

# How many laps, counting the deployment lap, a stop still counts as "under
# the SC" when the caller gives only the lap the SC was deployed on.
DEFAULT_SC_WINDOW_LAPS = max(1, int(round(DEFAULT_SC_DURATION_LAPS)))

# Time gained per lap as fuel burns off. Only used by the stand-in
# degradation fit in the worked example (see _baseline_degradation_models).
FUEL_EFFECT_S_PER_LAP = 0.06


def _as_lap(value, label):
    """Convert ``value`` to an int lap number, rejecting fractions.

    Args:
        value: Number-like value (ints, numpy ints, whole floats).
        label: Name used in the error message.

    Returns:
        The value as a plain ``int``.

    Raises:
        ValueError: If ``value`` is not a whole number.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must be a whole number, got {value!r}."
        ) from exc
    if not number.is_integer():
        raise ValueError(f"{label} must be a whole number, got {value!r}.")
    return int(number)


def _validate_stints(stints):
    """Check a list of stint dicts and return a normalised copy.

    Args:
        stints: Iterable of dicts with ``compound``, ``start_lap`` and
            ``end_lap``. Stints must be consecutive and start on lap 1.

    Returns:
        New list of dicts with upper-case compounds and int laps.

    Raises:
        ValueError: If the stints are malformed, overlap or leave gaps.
    """
    if isinstance(stints, (str, bytes, Mapping)) or not hasattr(
            stints, "__iter__"):
        raise ValueError("stints must be a list of stint dicts.")
    stints = list(stints)
    if not stints:
        raise ValueError("A strategy needs at least one stint.")

    clean, expected_start = [], 1
    for number, stint in enumerate(stints, start=1):
        try:
            compound = str(stint["compound"]).strip().upper()
            raw_start, raw_end = stint["start_lap"], stint["end_lap"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Stint {number} must be a dict with 'compound', "
                "'start_lap' and 'end_lap'."
            ) from exc
        start = _as_lap(raw_start, f"Stint {number} start_lap")
        end = _as_lap(raw_end, f"Stint {number} end_lap")
        if not compound:
            raise ValueError(f"Stint {number} has an empty compound.")
        if start != expected_start:
            raise ValueError(
                f"Stint {number} starts on lap {start} but should start on "
                f"lap {expected_start}: stints must be consecutive from "
                "lap 1 with no gaps or overlaps."
            )
        if end < start:
            raise ValueError(
                f"Stint {number} ends on lap {end}, before it starts "
                f"(lap {start})."
            )
        clean.append(
            {"compound": compound, "start_lap": start, "end_lap": end}
        )
        expected_start = end + 1
    return clean


@dataclass
class Strategy:
    """A pit-stop strategy: a name and an ordered list of stints.

    Each stint is a dict such as
    ``{"compound": "MEDIUM", "start_lap": 1, "end_lap": 20}``. A stint
    ending before the final lap means the car pits at the end of that lap.
    Stints must be consecutive, start on lap 1 and end on the last lap of
    the race. The list is validated and normalised on creation. Dry F1 races
    require two different compounds; that is not enforced here so that
    single-compound (0-stop) plans can still be simulated for comparison.

    Attributes:
        name: Label used in result tables, e.g. "1-Stop Medium-Hard".
        stints: The validated list of stint dicts.
    """

    name: str
    stints: list

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Strategy name must be a non-empty string.")
        self.name = self.name.strip()
        self.stints = _validate_stints(self.stints)

    @classmethod
    def from_pit_laps(cls, name, compounds, pit_laps, race_length):
        """Build a strategy from compounds and the laps to pit on.

        Args:
            name: Strategy label.
            compounds: Compounds in the order they are used.
            pit_laps: Laps at the end of which the car pits, increasing.
                Needs one fewer entry than ``compounds``.
            race_length: Total laps of the race.

        Returns:
            The corresponding ``Strategy``.

        Raises:
            ValueError: If the counts do not match or the laps are invalid.
        """
        compounds = list(compounds)
        pit_laps = [_as_lap(lap, "pit lap") for lap in pit_laps]
        if len(compounds) != len(pit_laps) + 1:
            raise ValueError(
                f"{len(compounds)} compound(s) need {len(compounds) - 1} "
                f"pit lap(s), got {len(pit_laps)}."
            )
        bounds = [0, *pit_laps, _as_lap(race_length, "race_length")]
        stints = [
            {"compound": compound, "start_lap": first + 1, "end_lap": last}
            for compound, first, last in zip(compounds, bounds, bounds[1:])
        ]
        return cls(name, stints)

    @property
    def n_stops(self):
        """Number of pit stops (stints minus one)."""
        return len(self.stints) - 1

    @property
    def pit_laps(self):
        """Laps at the end of which the car pits."""
        return [stint["end_lap"] for stint in self.stints[:-1]]

    @property
    def race_laps(self):
        """Total laps covered by the strategy."""
        return self.stints[-1]["end_lap"]

    @property
    def compounds(self):
        """Compounds in the order they are used."""
        return [stint["compound"] for stint in self.stints]

    def describe(self):
        """One-line summary, e.g. ``MEDIUM L1-20 -> HARD L21-57``."""
        return " -> ".join(
            f"{s['compound']} L{s['start_lap']}-{s['end_lap']}"
            for s in self.stints
        )


def _as_strategy(strategy, name="Strategy"):
    """Accept a ``Strategy`` or a plain list of stint dicts.

    Args:
        strategy: A ``Strategy`` or a list of stint dicts.
        name: Name given to a strategy built from a plain list.

    Returns:
        A ``Strategy``.

    Raises:
        TypeError: If ``strategy`` is neither.
    """
    if isinstance(strategy, Strategy):
        return strategy
    if isinstance(strategy, (list, tuple)):
        return Strategy(name, list(strategy))
    raise TypeError(
        "strategy must be a Strategy or a list of stint dicts, got "
        f"{type(strategy).__name__}."
    )


def _as_strategy_dict(strategies):
    """Turn strategies into an ordered ``{name: Strategy}`` dict.

    Args:
        strategies: A dict ``{name: Strategy or stint list}`` or an iterable
            of ``Strategy`` objects / stint lists (the latter are named
            "Strategy 1", "Strategy 2", ...).

    Returns:
        Ordered dict of ``Strategy`` objects.

    Raises:
        ValueError: If names repeat or nothing was given.
    """
    if isinstance(strategies, Mapping):
        pairs = [(str(name), _as_strategy(value, str(name)))
                 for name, value in strategies.items()]
    else:
        pairs = []
        for number, item in enumerate(strategies, start=1):
            strategy = _as_strategy(item, f"Strategy {number}")
            pairs.append((strategy.name, strategy))
    if not pairs:
        raise ValueError("Give at least one strategy.")
    names = [name for name, _ in pairs]
    if len(set(names)) != len(names):
        raise ValueError(f"Strategy names must be unique, got {names}.")
    return dict(pairs)


def _check_race_length(strategy, race_length):
    """Validate ``race_length`` and that the strategy covers it exactly."""
    race_length = _as_lap(race_length, "race_length")
    if race_length < 1:
        raise ValueError("race_length must be at least 1 lap.")
    if strategy.race_laps != race_length:
        raise ValueError(
            f"Strategy {strategy.name!r} covers {strategy.race_laps} laps "
            f"but race_length is {race_length}."
        )
    return race_length


def _check_seconds(value, label):
    """Return ``value`` as a finite, non-negative float of seconds."""
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number, got {value!r}.") from exc
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError(
            f"{label} must be a non-negative number of seconds, "
            f"got {value!r}."
        )
    return seconds


def _check_pit_losses(pit_stop_loss, sc_pit_loss):
    """Validate the two pit-loss values; a SC stop cannot cost more."""
    pit_stop_loss = _check_seconds(pit_stop_loss, "pit_stop_loss_seconds")
    sc_pit_loss = _check_seconds(sc_pit_loss, "sc_pit_loss_seconds")
    if sc_pit_loss > pit_stop_loss:
        raise ValueError(
            f"sc_pit_loss_seconds ({sc_pit_loss:g}) cannot exceed "
            f"pit_stop_loss_seconds ({pit_stop_loss:g}); are they swapped?"
        )
    return pit_stop_loss, sc_pit_loss


def _predict_curve(model, ages):
    """Predict lap times for several tyre ages with any supported model.

    Args:
        model: Callable, object with ``.predict()``, or polynomial
            coefficients (see the module docstring).
        ages: Tyre ages (1 = first lap on a fresh set).

    Returns:
        1-D float array of lap times in seconds, one per age.

    Raises:
        TypeError: If ``model`` is not a supported kind of model.
    """
    ages = np.asarray(ages, dtype=float)

    if hasattr(model, "predict"):
        attempts = [
            lambda a: model.predict(a.reshape(-1, 1)),   # scikit-learn
            lambda a: model.predict(a),                  # numpy style
            lambda a: [model.predict(float(x)) for x in a],
        ]
    elif callable(model):
        attempts = [
            lambda a: model(a),                          # vectorised
            lambda a: [model(float(x)) for x in a],      # scalar only
        ]
    else:
        coefficients = model.get("coefficients") if isinstance(
            model, Mapping) else model
        try:
            return np.polyval(np.asarray(coefficients, dtype=float), ages)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Unsupported degradation model of type "
                f"{type(model).__name__}: expected a callable, an object "
                "with .predict(), or polynomial coefficients."
            ) from exc

    for attempt in attempts:
        try:
            curve = np.asarray(attempt(ages), dtype=float).reshape(-1)
        except (TypeError, ValueError, AttributeError, IndexError):
            continue
        if curve.shape == ages.shape:
            return curve
    raise TypeError(
        f"Could not get lap-time predictions from {type(model).__name__}."
    )


def predict_lap_time(model, tyre_age):
    """Predict one lap time (seconds) from a degradation model.

    Handy for sanity-checking a model before simulating with it.

    Args:
        model: Callable, object with ``.predict()``, or polynomial
            coefficients (see the module docstring).
        tyre_age: Laps completed on the set, counting the current one.

    Returns:
        Predicted lap time in seconds as a float.
    """
    return float(_predict_curve(model, [tyre_age])[0])


def _lap_time_tables(degradation_models, strategies):
    """Predict, once, the lap time for every tyre age the strategies need.

    Args:
        degradation_models: Dict mapping compound -> model.
        strategies: ``Strategy`` objects that will be simulated.

    Returns:
        Dict mapping compound -> list of lap times, where entry ``i`` is
        the lap time at tyre age ``i + 1``.

    Raises:
        TypeError: If ``degradation_models`` is not a dict.
        ValueError: If a compound has no model, or a model predicts a
            non-finite or non-positive lap time.
    """
    if not isinstance(degradation_models, Mapping):
        raise TypeError(
            "degradation_models must be a dict mapping compound names to "
            "models."
        )
    models = {str(key).strip().upper(): value
              for key, value in degradation_models.items()}

    longest = {}      # compound -> longest stint using it
    for strategy in strategies:
        for stint in strategy.stints:
            length = stint["end_lap"] - stint["start_lap"] + 1
            longest[stint["compound"]] = max(
                longest.get(stint["compound"], 0), length)

    tables = {}
    for compound, length in longest.items():
        if compound not in models:
            raise ValueError(
                f"No degradation model for compound {compound!r}. "
                f"Available: {sorted(models)}."
            )
        curve = _predict_curve(models[compound], np.arange(1, length + 1))
        bad = ~np.isfinite(curve) | (curve <= 0)
        if bad.any():
            age = int(np.flatnonzero(bad)[0]) + 1
            raise ValueError(
                f"The {compound} degradation model predicted an invalid lap "
                f"time ({curve[age - 1]!r}) at tyre age {age}; check the "
                "model, especially if it is extrapolated to long stints."
            )
        tables[compound] = curve.tolist()
    return tables


def _race_time(strategy, tables, pit_stop_loss, sc_window=None,
               sc_pit_loss=0.0):
    """Total race time: the sum of every lap plus the pit-stop penalties.

    Args:
        strategy: The ``Strategy`` to run.
        tables: Lap-time tables from ``_lap_time_tables``.
        pit_stop_loss: Seconds lost by a normal (green-flag) stop.
        sc_window: Optional ``(first_lap, last_lap)`` of the Safety Car
            period. A stop made on any lap in it costs ``sc_pit_loss``.
        sc_pit_loss: Seconds lost by a stop made under the Safety Car.

    Returns:
        Total predicted race time in seconds.
    """
    total = 0.0
    last_stint = len(strategy.stints) - 1
    for index, stint in enumerate(strategy.stints):
        lap_times = tables[stint["compound"]]
        for lap in range(stint["start_lap"], stint["end_lap"] + 1):
            tyre_age = lap - stint["start_lap"] + 1
            total += lap_times[tyre_age - 1]

        if index < last_stint:                 # the stint ends: pit stop
            under_sc = (sc_window is not None
                        and sc_window[0] <= stint["end_lap"] <= sc_window[1])
            total += sc_pit_loss if under_sc else pit_stop_loss
    return total


def simulate_strategy(strategy, degradation_models, race_length,
                      pit_stop_loss_seconds=DEFAULT_PIT_STOP_LOSS_S):
    """Predict the total race time of a strategy with no Safety Car.

    Walks through the race lap by lap. Each lap is timed with the
    degradation model of the compound in use at that lap's tyre age, and
    ``pit_stop_loss_seconds`` is added whenever a stint ends. The lap-time
    models are used as given; if they are fuel-corrected, the fuel-burn gain
    is missing from the total, but it is the same for every strategy so
    differences and win rates are unaffected.

    Args:
        strategy: A ``Strategy`` or a list of stint dicts such as
            ``[{"compound": "MEDIUM", "start_lap": 1, "end_lap": 20},
            {"compound": "HARD", "start_lap": 21, "end_lap": 57}]``.
        degradation_models: Dict mapping compound -> model (see the module
            docstring). Predictions beyond the tyre ages a model was fitted
            on are extrapolations.
        race_length: Total laps; must equal the strategy's last lap.
        pit_stop_loss_seconds: Time lost per pit stop. About 20-25 s is
            realistic at most circuits, but it varies with pit-lane length,
            so set it per circuit. Default 22 s.

    Returns:
        Total predicted race time in seconds.

    Raises:
        ValueError: If the strategy does not match ``race_length``, a
            compound has no model, or an input is invalid.
    """
    strategy = _as_strategy(strategy)
    _check_race_length(strategy, race_length)
    pit_stop_loss = _check_seconds(pit_stop_loss_seconds,
                                   "pit_stop_loss_seconds")
    tables = _lap_time_tables(degradation_models, [strategy])
    return _race_time(strategy, tables, pit_stop_loss)


def _sc_window(sc_lap, sc_duration_laps, race_length):
    """Return ``(first_lap, last_lap)`` of a Safety Car period."""
    sc_lap = _as_lap(sc_lap, "sc_lap")
    sc_duration = _as_lap(sc_duration_laps, "sc_duration_laps")
    if not 1 <= sc_lap <= race_length:
        raise ValueError(
            f"sc_lap must be between 1 and {race_length}, got {sc_lap}."
        )
    if sc_duration < 1:
        raise ValueError("sc_duration_laps must be at least 1.")
    return sc_lap, min(race_length, sc_lap + sc_duration - 1)


def simulate_strategy_with_safety_car(
    strategy,
    degradation_models,
    race_length,
    sc_lap,
    pit_stop_loss_seconds=DEFAULT_PIT_STOP_LOSS_S,
    sc_pit_loss_seconds=DEFAULT_SC_PIT_LOSS_S,
    sc_duration_laps=DEFAULT_SC_WINDOW_LAPS,
):
    """Predict the race time when a Safety Car comes out on ``sc_lap``.

    Runs the same lap-by-lap simulation as ``simulate_strategy``, but any
    stop made during the Safety Car period (from ``sc_lap`` for
    ``sc_duration_laps`` laps) costs ``sc_pit_loss_seconds`` instead of the
    full loss. Strategies whose stops happen to line up with the Safety Car
    are rewarded; the others are unchanged. The slow laps behind the Safety
    Car are not modelled: they cost every strategy roughly the same, so the
    ranking is unaffected.

    Args:
        strategy: A ``Strategy`` or a list of stint dicts.
        degradation_models: Dict mapping compound -> model.
        race_length: Total laps; must equal the strategy's last lap.
        sc_lap: Lap on which the Safety Car is deployed.
        pit_stop_loss_seconds: Time lost by a green-flag stop.
        sc_pit_loss_seconds: Time lost by a stop under the Safety Car
            (must not exceed ``pit_stop_loss_seconds``). Default 8 s, which
            is generous; measured values are often 10-15 s.
        sc_duration_laps: Laps the Safety Car period lasts, counting the
            deployment lap. Default 4. Cut off at the end of the race.

    Returns:
        Total predicted race time in seconds.

    Raises:
        ValueError: If an input is invalid (see ``simulate_strategy``), or
            ``sc_lap`` is outside the race.
    """
    strategy = _as_strategy(strategy)
    race_length = _check_race_length(strategy, race_length)
    pit_stop_loss, sc_pit_loss = _check_pit_losses(
        pit_stop_loss_seconds, sc_pit_loss_seconds)
    window = _sc_window(sc_lap, sc_duration_laps, race_length)
    tables = _lap_time_tables(degradation_models, [strategy])
    return _race_time(strategy, tables, pit_stop_loss, window, sc_pit_loss)


def _unpack_sc_data(sc_probability_data):
    """Split ``sc_probability_data`` into ``(probability, history)``."""
    if sc_probability_data is None:
        return 0.0, None
    if isinstance(sc_probability_data, Mapping):
        if "sc_probability" not in sc_probability_data:
            raise ValueError(
                "sc_probability_data needs an 'sc_probability' key; pass "
                "the dict from historical_sc_probability()."
            )
        return float(sc_probability_data["sc_probability"]), \
            sc_probability_data
    return float(sc_probability_data), None


def _summarise_times(names, strategies, times):
    """Build the per-strategy summary table from simulated race times.

    Args:
        names: Strategy names, in column order of ``times``.
        strategies: Dict name -> ``Strategy``.
        times: Array of shape ``(n_simulations, n_strategies)``.

    Returns:
        DataFrame indexed by strategy name (see ``run_monte_carlo``).
    """
    n_sims = times.shape[0]
    # The fastest strategy wins each simulation; exact ties share the win so
    # the win rates always add up to 100%.
    fastest = np.isclose(times, times.min(axis=1, keepdims=True),
                         rtol=0.0, atol=1e-9)
    credit = fastest / fastest.sum(axis=1, keepdims=True)
    std = (times.std(axis=0, ddof=1) if n_sims > 1
           else np.zeros(len(names)))
    return pd.DataFrame(
        {
            "stops": [strategies[name].n_stops for name in names],
            "mean_time_s": times.mean(axis=0),
            "median_time_s": np.median(times, axis=0),
            "std_time_s": std,
            "best_time_s": times.min(axis=0),
            "worst_time_s": times.max(axis=0),
            "p05_time_s": np.percentile(times, 5, axis=0),
            "p95_time_s": np.percentile(times, 95, axis=0),
            "win_rate_pct": credit.mean(axis=0) * 100.0,
        },
        index=pd.Index(names, name="strategy"),
    )


def run_monte_carlo(
    strategies,
    degradation_models,
    race_length,
    sc_probability_data=None,
    n_simulations=1000,
    pit_stop_loss_seconds=DEFAULT_PIT_STOP_LOSS_S,
    sc_pit_loss_seconds=DEFAULT_SC_PIT_LOSS_S,
    rng=None,
):
    """Race the strategies against random Safety Car scenarios.

    In each simulated race ``simulate_random_sc_event`` (Part 3) decides
    whether a Safety Car comes out and on which lap. Every strategy then
    runs under that same scenario, so differences between strategies are not
    blurred by luck. The finishing time of each strategy in each run is
    recorded.

    Args:
        strategies: Strategies to test: a list of ``Strategy`` objects or
            stint lists, or a dict ``{name: strategy}``.
        degradation_models: Dict mapping compound -> model.
        race_length: Total laps of the race.
        sc_probability_data: The dict from ``historical_sc_probability``
            (its probability, typical start laps and duration shape the
            scenarios), a plain probability between 0 and 1, or None for a
            race with no Safety Car.
        n_simulations: Number of simulated races.
        pit_stop_loss_seconds: Time lost by a green-flag stop.
        sc_pit_loss_seconds: Time lost by a stop under the Safety Car.
        rng: Optional ``numpy.random.Generator`` or int seed, for
            reproducible results.

    Returns:
        Dict with:

        - ``summary``: DataFrame indexed by strategy with ``stops``,
          ``mean_time_s``, ``median_time_s``, ``std_time_s``,
          ``best_time_s``, ``worst_time_s``, ``p05_time_s``,
          ``p95_time_s`` and ``win_rate_pct`` (share of simulations in
          which the strategy was fastest).
        - ``times``: DataFrame of every finishing time in seconds, one row
          per simulation and one column per strategy (the full
          distribution, ready for histograms or box plots).
        - ``scenarios``: DataFrame with one row per simulation
          (``sc_occurs``, ``sc_start_lap``, ``sc_end_lap``,
          ``sc_duration_laps``).
        - ``n_simulations`` and ``sc_rate`` (share of races with a SC).

    Raises:
        ValueError: If an input is invalid.
    """
    strategy_map = _as_strategy_dict(strategies)
    n_simulations = _as_lap(n_simulations, "n_simulations")
    if n_simulations < 1:
        raise ValueError("n_simulations must be at least 1.")
    for strategy in strategy_map.values():
        race_length = _check_race_length(strategy, race_length)
    pit_stop_loss, sc_pit_loss = _check_pit_losses(
        pit_stop_loss_seconds, sc_pit_loss_seconds)
    probability, history = _unpack_sc_data(sc_probability_data)

    tables = _lap_time_tables(degradation_models, strategy_map.values())
    generator = np.random.default_rng(rng)
    names = list(strategy_map)

    logger.info("Running %d simulated races for %d strategies ...",
                n_simulations, len(names))
    times = np.empty((n_simulations, len(names)))
    events = []
    for run in range(n_simulations):
        event = simulate_random_sc_event(
            probability, race_length, history, generator)
        window = ((event["sc_start_lap"], event["sc_end_lap"])
                  if event["sc_occurs"] else None)
        for column, name in enumerate(names):
            times[run, column] = _race_time(
                strategy_map[name], tables, pit_stop_loss, window,
                sc_pit_loss)
        events.append(event)

    scenarios = pd.DataFrame(events).astype(
        {"sc_start_lap": "Int64", "sc_end_lap": "Int64"})
    return {
        "summary": _summarise_times(names, strategy_map, times),
        "times": pd.DataFrame(times, columns=names),
        "scenarios": scenarios,
        "n_simulations": n_simulations,
        "sc_rate": float(scenarios["sc_occurs"].mean()),
    }


class MonteCarloResult:
    """Read-only, dict-like wrapper around a ``run_monte_carlo`` result.

    Supports the same ``result["times"]`` / ``"scenarios" in result`` /
    ``len(result)`` access as the plain dict ``run_monte_carlo`` returns.
    ``compare_strategies`` stores one of these in ``table.attrs
    ["monte_carlo"]`` instead of a plain dict, because pandas compares
    ``.attrs`` (element by element) whenever DataFrames are combined, e.g.
    ``pd.concat`` or ``df.loc[new_label] = row``. Comparing two DataFrames
    with ``==`` returns another DataFrame rather than True/False, which
    pandas cannot use as a yes/no answer and raises over; this class
    deliberately does not define ``==``, so it falls back to comparison by
    identity like any ordinary object, and that comparison never raises.
    Access the contents the same way either way: ``result["times"]``.
    """

    def __init__(self, data):
        self._data = dict(data)

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def get(self, key, default=None):
        """Same as ``dict.get``: ``self[key]`` if present, else default."""
        return self._data.get(key, default)

    def __repr__(self):
        return f"MonteCarloResult(keys={list(self._data)})"


def compare_strategies(
    strategies_list,
    degradation_models,
    race_length,
    sc_probability_data=None,
    pit_stop_loss_seconds=DEFAULT_PIT_STOP_LOSS_S,
    sc_pit_loss_seconds=DEFAULT_SC_PIT_LOSS_S,
    n_simulations=1000,
    rng=None,
):
    """Compare named strategies side by side in one clean table.

    Built for 2-4 strategies such as "1-Stop Medium-Hard" and "2-Stop
    Soft-Medium-Soft". Combines the deterministic prediction with a Monte
    Carlo run (see ``run_monte_carlo``).

    Args:
        strategies_list: Strategies to compare: a list of named
            ``Strategy`` objects, or a dict ``{name: strategy or stints}``.
            At least two are needed.
        degradation_models: Dict mapping compound -> model.
        race_length: Total laps of the race.
        sc_probability_data: Dict from ``historical_sc_probability``, a
            probability, or None for no Safety Car.
        pit_stop_loss_seconds: Time lost by a green-flag stop.
        sc_pit_loss_seconds: Time lost by a stop under the Safety Car.
        n_simulations: Number of Monte Carlo races.
        rng: Optional ``numpy.random.Generator`` or int seed.

    Returns:
        DataFrame indexed by strategy, best first (by Monte Carlo average),
        with columns:

        - ``stops``: number of pit stops.
        - ``predicted_time_s``: total time with no Safety Car.
        - ``delta_vs_best_s``: gap to the best ``predicted_time_s``.
        - ``mc_avg_time_s``: average finishing time over the Monte Carlo
          races (includes random Safety Cars).
        - ``mc_delta_vs_best_s``: gap to the best ``mc_avg_time_s``.
        - ``win_rate_pct``: share of Monte Carlo races the strategy won.

        The full ``run_monte_carlo`` result (time distributions and
        scenarios) is stored in ``table.attrs["monte_carlo"]`` as a
        ``MonteCarloResult`` (dict-like: ``table.attrs["monte_carlo"]
        ["times"]``).

    Raises:
        ValueError: If fewer than two strategies are given or an input is
            invalid.
    """
    strategy_map = _as_strategy_dict(strategies_list)
    if len(strategy_map) < 2:
        raise ValueError("compare_strategies needs at least two strategies.")

    monte_carlo = run_monte_carlo(
        strategy_map, degradation_models, race_length, sc_probability_data,
        n_simulations, pit_stop_loss_seconds, sc_pit_loss_seconds, rng)
    predicted = pd.Series({
        name: simulate_strategy(strategy, degradation_models, race_length,
                                pit_stop_loss_seconds)
        for name, strategy in strategy_map.items()
    })
    summary = monte_carlo["summary"]
    mean_time = summary["mean_time_s"]

    table = pd.DataFrame({
        "stops": summary["stops"],
        "predicted_time_s": predicted,
        "delta_vs_best_s": predicted - predicted.min(),
        "mc_avg_time_s": mean_time,
        "mc_delta_vs_best_s": mean_time - mean_time.min(),
        "win_rate_pct": summary["win_rate_pct"],
    })
    table.index.name = "strategy"
    table = table.sort_values("mc_avg_time_s").round({
        "predicted_time_s": 3, "delta_vs_best_s": 3, "mc_avg_time_s": 3,
        "mc_delta_vs_best_s": 3, "win_rate_pct": 1,
    })
    table.attrs["monte_carlo"] = MonteCarloResult(monte_carlo)
    return table


def format_race_time(seconds):
    """Format seconds as ``H:MM:SS.s`` (5636.7 -> ``"1:33:56.7"``)."""
    total = round(float(seconds), 1)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours)}:{int(minutes):02d}:{secs:04.1f}"


# --------------------------------------------------------------------------
# Worked example
# --------------------------------------------------------------------------
def _baseline_degradation_models(laps_df, min_laps=30):
    """Stand-in for the Part 2 models: one straight-line fit per compound.

    Part 2 (tyre degradation curve fitting) has not been built yet, so the
    worked example fits a simple line of lap time against tyre age for each
    dry compound from the race's own clean laps. Replace it with the Part 2
    models when they exist; the engine accepts any model in the shapes
    listed in the module docstring.

    Laps are cleaned by dropping lap 1, in/out laps, laps without a time,
    laps under a Safety Car, VSC or red flag, and outliers (traffic,
    errors). Lap times are fuel-corrected (fuel burn makes cars quicker as
    the race goes on, which would otherwise hide tyre wear), so absolute
    totals are a few percent slow; differences between strategies are
    unaffected.

    Args:
        laps_df: Lap table from ``data_loader.get_lap_data``.
        min_laps: Minimum clean laps needed to fit a compound.

    Returns:
        Dict mapping compound -> ``np.poly1d`` (slope, intercept).
    """
    track_ok = ~laps_df["TrackStatus"].str.contains("[4567]")
    clean = laps_df[
        track_ok & ~laps_df["MissingLapTime"] & ~laps_df["IsInLap"]
        & ~laps_df["IsOutLap"] & (laps_df["LapNumber"] > 1)
        & laps_df["TyreLife"].notna()
    ]
    corrected = clean["LapTime"] + FUEL_EFFECT_S_PER_LAP * (
        clean["LapNumber"] - 1)

    models = {}
    for compound in ("SOFT", "MEDIUM", "HARD"):
        rows = (clean["Compound"] == compound).to_numpy()
        age = clean.loc[rows, "TyreLife"].to_numpy(dtype=float)
        lap_time = corrected[rows].to_numpy(dtype=float)
        if len(age) < min_laps:
            logger.warning("Only %d clean %s laps; no model fitted.",
                           len(age), compound)
            continue
        fit = np.polyfit(age, lap_time, 1)
        residual = lap_time - np.polyval(fit, age)
        mad = np.median(np.abs(residual - np.median(residual)))
        keep = np.abs(residual) <= 2.5 * max(1.4826 * mad, 1e-9)
        if keep.sum() >= min_laps:                 # refit without outliers
            fit = np.polyfit(age[keep], lap_time[keep], 1)
        models[compound] = np.poly1d(fit)
    return models


def _example_strategies(models, race_length):
    """Illustrative 1-stop and 2-stop candidates with equal-length stints.

    Args:
        models: Fitted degradation models (needs two dry compounds).
        race_length: Total laps of the race.

    Returns:
        List of two ``Strategy`` objects.

    Raises:
        ValueError: If fewer than two dry compounds have a model.
    """
    dry = [c for c in ("SOFT", "MEDIUM", "HARD") if c in models]
    if len(dry) < 2:
        raise ValueError(
            "Need degradation models for at least two dry compounds; the "
            "race has too few clean laps on them."
        )
    one_stop = [c for c in ("MEDIUM", "HARD") if c in models]
    if len(one_stop) < 2:
        one_stop = dry[:2]
    two_stop = ["SOFT", "MEDIUM", "SOFT"]
    if not set(two_stop) <= set(models):
        two_stop = [dry[0], dry[1], dry[0]]

    strategies = []
    for compounds in (one_stop, two_stop):
        stints = len(compounds)
        pit_laps = [round(race_length * k / stints) for k in range(1, stints)]
        name = f"{stints - 1}-Stop " + "-".join(c.title() for c in compounds)
        strategies.append(
            Strategy.from_pit_laps(name, compounds, pit_laps, race_length))
    return strategies


def run_worked_example(
    year=2023,
    gp_name="Bahrain",
    n_simulations=2000,
    pit_stop_loss_seconds=DEFAULT_PIT_STOP_LOSS_S,
    sc_pit_loss_seconds=DEFAULT_SC_PIT_LOSS_S,
):
    """Compare a 1-stop and a 2-stop strategy for a real race and print it.

    Pipeline: load the race (Part 1), fit tyre degradation from its laps
    (stand-in for Part 2), get the circuit's Safety Car history (Part 3),
    then compare the strategies with ``compare_strategies``. The first run
    downloads several races and is slow; FastF1 caches them afterwards.

    Args:
        year: Season of the race to analyse.
        gp_name: Grand Prix name or round number.
        n_simulations: Monte Carlo races to simulate.
        pit_stop_loss_seconds: Time lost by a green-flag stop.
        sc_pit_loss_seconds: Time lost by a stop under the Safety Car.

    Returns:
        The comparison DataFrame from ``compare_strategies``.

    Raises:
        DataLoadError: If the race cannot be loaded (see ``data_loader``).
        ValueError: If the race has too little data to fit two compounds.
    """
    setup_cache()
    session = load_race_session(year, gp_name)
    laps = get_lap_data(session)
    event_name = session.event["EventName"]
    race_length = int(laps["LapNumber"].max())

    models = _baseline_degradation_models(laps)
    strategies = _example_strategies(models, race_length)
    # The full GP name keeps the sample to one layout (Bahrain 2020 also
    # staged a Sakhir GP on a different track).
    sc_data = historical_sc_probability(event_name)
    table = compare_strategies(
        strategies, models, race_length, sc_data,
        pit_stop_loss_seconds=pit_stop_loss_seconds,
        sc_pit_loss_seconds=sc_pit_loss_seconds,
        n_simulations=n_simulations, rng=42)
    monte_carlo = table.attrs["monte_carlo"]

    slopes = " | ".join(f"{c} {m.coeffs[0]:+.3f} s/lap"
                        for c, m in models.items())
    history = ("default value, too little history" if sc_data["used_default"]
               else f"{sc_data['sample_size']} past races")
    print(f"\nWorked example: {year} {event_name} ({race_length} laps)")
    print("=" * 64)
    print(f"Tyre degradation (stand-in fit for Part 2): {slopes}")
    print(f"Safety car data: P(SC/VSC) = {sc_data['sc_probability']:.0%} "
          f"({history}); typical start laps "
          f"{sc_data['avg_sc_start_laps'] or 'n/a'}")
    print(f"Pit loss: {pit_stop_loss_seconds:g} s green flag, "
          f"{sc_pit_loss_seconds:g} s under a safety car")
    print("\nCandidates:")
    for strategy in strategies:
        print(f"  {strategy.name}: {strategy.describe()}")
    print(f"\nResults over {n_simulations:,} simulated races "
          f"(safety car in {monte_carlo['sc_rate']:.0%} of them):")
    print(table.to_string())

    print("\nPredicted race time with no safety car:")
    for name, seconds in table["predicted_time_s"].items():
        print(f"  {name}: {format_race_time(seconds)}")

    best, runner_up = table.index[0], table.index[1]
    print(f"\n{best} is fastest on average, "
          f"{table.loc[runner_up, 'mc_delta_vs_best_s']:.1f} s ahead of "
          f"{runner_up}, and wins {table.loc[best, 'win_rate_pct']:.0f}% of "
          "the simulated races.")
    if table["predicted_time_s"].idxmin() != best:
        print("Safety car risk changes the answer: without one, "
              f"{table['predicted_time_s'].idxmin()} would be faster.")
    return table


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s")
    logging.getLogger("src").setLevel(logging.INFO)
    run_worked_example()
