"""F1 Race Strategy Simulator - interactive dashboard.

Pick a real Formula 1 race and compare custom tyre-stop strategies (0-3
stops) to see which would likely have finished fastest. Tyre wear is
fitted from the race's own lap data, and a Monte Carlo simulation -
drawing thousands of random safety-car scenarios from this circuit's
history - estimates each strategy's average finishing time and win rate.

Run with: streamlit run app.py
"""

import datetime as dt

import pandas as pd
import streamlit as st

from src.data_loader import (
    DataLoadError, MIN_SUPPORTED_YEAR, get_lap_data, get_race_list,
    load_race_session, setup_cache,
)
from src.degradation_model import (
    degradation_fit_summary, fit_degradation_models,
)
from src.safety_car_model import historical_sc_probability
from src.strategy_simulator import (
    DEFAULT_PIT_STOP_LOSS_S, Strategy, compare_strategies, format_race_time,
)
from src.visualizer import (
    plot_degradation_curves, plot_monte_carlo_distribution,
    plot_strategy_comparison, plot_stint_timeline, plot_win_rate_pie,
)

st.set_page_config(
    page_title="F1 Strategy Simulator", page_icon="\U0001F3CE",
    layout="wide")

MAX_STOPS = 3
STRATEGY_COUNT_CHOICES = [2, 3]
# (default stops, compound rotation offset) per strategy slot, so the
# out-of-the-box strategies differ from each other rather than all
# matching.
DEFAULT_PLANS = [(1, 0), (2, 1), (2, 2)]
SC_HISTORY_SEASONS = 6      # how many recent seasons to sample per circuit


# ---------------------------------------------------------------------
# Cached data loading - re-running with the same race reuses this instead
# of hitting FastF1 again.
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Fetching the season calendar...")
def cached_race_list(year):
    """Completed races for a season, as a plain (picklable) DataFrame."""
    return get_race_list(year)


@st.cache_data(show_spinner=(
    "Loading race data, fitting tyre degradation, and estimating safety "
    "car risk - this can take a little while on the first run for a "
    "race..."))
def load_race_bundle(year, round_number):
    """Everything the dashboard needs for one race, computed once.

    Loads the session (Part 1), fits a degradation curve per compound
    from its own laps (Part 2), and estimates this circuit's safety car
    probability from its recent history (Part 3).

    Args:
        year: Season of the race.
        round_number: Round number within that season (exact - unlike a
            name, it is never fuzzy-matched to the wrong race).

    Returns:
        Dict with ``event_name``, ``race_length``, ``models``
        (degradation curves), ``fit_summary`` (fit diagnostics) and
        ``sc_data`` (from ``historical_sc_probability``). The lap table
        itself is not kept, to keep the cached payload small.

    Raises:
        DataLoadError: If the race cannot be loaded.
    """
    setup_cache()
    session = load_race_session(int(year), int(round_number))
    laps = get_lap_data(session)
    event_name = session.event["EventName"]
    race_length = int(laps["LapNumber"].max())

    models = fit_degradation_models(laps)
    fit_summary = degradation_fit_summary(laps, models)
    # historical_sc_probability's own default (2019-2023) drifts further
    # out of date every season; use a rolling recent window instead.
    years = range(
        max(MIN_SUPPORTED_YEAR, year - SC_HISTORY_SEASONS + 1), year + 1)
    sc_data = historical_sc_probability(event_name, years=years)

    return {
        "event_name": event_name,
        "race_length": race_length,
        "models": models,
        "fit_summary": fit_summary,
        "sc_data": sc_data,
    }


# ---------------------------------------------------------------------
# Small presentation helpers (UI-facing text/formatting only - the
# modelling itself lives in src/).
# ---------------------------------------------------------------------
def format_comparison_table(table):
    """Comparison table with friendly headers and h:mm:ss time columns."""
    display = pd.DataFrame({
        "Stops": table["stops"],
        "Predicted time (no SC)": table["predicted_time_s"].map(
            format_race_time),
        "Avg. time (Monte Carlo)": table["mc_avg_time_s"].map(
            format_race_time),
        "Gap to fastest": table["mc_delta_vs_best_s"].map(
            lambda s: "-" if s == 0 else f"+{s:.1f}s"),
        "Win rate": table["win_rate_pct"].map(lambda p: f"{p:.1f}%"),
    })
    display.index.name = "Strategy"
    return display


def build_recommendation(table, sc_data, strategies_by_name):
    """Plain-language explanation of the recommended strategy.

    Grounded only in numbers actually computed for this comparison (win
    rate, the green-flag-vs-Monte-Carlo ranking, and whether the winning
    strategy's pit stops land near this circuit's typical safety-car
    laps) rather than a generic canned explanation.

    Args:
        table: The DataFrame from ``compare_strategies``.
        sc_data: The dict from ``historical_sc_probability``.
        strategies_by_name: Dict mapping strategy name -> ``Strategy``.

    Returns:
        A markdown string.
    """
    best = table.index[0]
    win_rate = table.loc[best, "win_rate_pct"]
    n_races = table.attrs["monte_carlo"]["n_simulations"]
    green_best = table["predicted_time_s"].idxmin()
    sc_prob = sc_data["sc_probability"]
    sc_starts = sc_data.get("avg_sc_start_laps") or []
    std = table.attrs["monte_carlo"]["summary"].loc[best, "std_time_s"]

    headline = f"**Recommended strategy: {best}**"
    if len(table) > 1:
        runner_up = table.index[1]
        runner_up_rate = table.loc[runner_up, "win_rate_pct"]
        margin = win_rate - runner_up_rate
        if margin < 12:
            headline += (
                f" - a close call, winning {win_rate:.0f}% of races to "
                f"{runner_up}'s {runner_up_rate:.0f}%.")
        else:
            headline += (
                f" - wins **{win_rate:.0f}%** of {n_races:,} simulated "
                "races.")
    else:
        headline += f" - wins **{win_rate:.0f}%** of {n_races:,} races."

    if green_best == best:
        best_time = format_race_time(table.loc[best, "predicted_time_s"])
        pace = (
            "It is also the fastest strategy on raw pace with no safety "
            f"car ({best_time}), so it wins on both counts.")
    else:
        gap = table.loc[best, "delta_vs_best_s"]
        best_pits = strategies_by_name[best].pit_laps
        near_sc = bool(sc_starts) and any(
            abs(p - s) <= 4 for p in best_pits for s in sc_starts)
        pace = (
            f"On raw pace alone it is {gap:.1f}s slower than {green_best} "
            f"- but with a **{sc_prob:.0%} chance of a safety car** at "
            "this circuit")
        if near_sc:
            pit_text = ", ".join(f"lap {p}" for p in best_pits)
            pace += (
                f", and its pit stop ({pit_text}) landing close to where "
                "one has typically appeared, it can pit cheaply under "
                "caution often enough to win more often overall.")
        else:
            pace += (
                ", it comes out ahead often enough across randomised "
                "safety-car scenarios to win more often overall.")

    consistency = (
        f"Its finishing time varies by about \u00b1{std:.1f}s (one "
        "standard deviation) across the simulated races.")
    return f"{headline}\n\n{pace}\n\n{consistency}"


def strategy_forms_signature(strategy_forms, race_key, pit_loss,
                             n_simulations):
    """Hashable snapshot of the strategy builder's current state.

    Computed the same way both when results are saved and when they are
    displayed, so the two can be compared to detect stale results (built
    strategy objects and raw form input aren't directly comparable, so
    this is derived from the form values in both places).
    """
    forms = tuple(
        (f["name"], tuple(f["compounds"]), tuple(f["pit_laps"]))
        for f in strategy_forms)
    return (race_key, pit_loss, n_simulations, forms)


def build_strategy_or_none(name, compounds, pit_laps, race_length):
    """Try to build a Strategy; return (Strategy, None) or (None, error)."""
    try:
        return Strategy.from_pit_laps(
            name.strip() or "Untitled", compounds, pit_laps, race_length
        ), None
    except ValueError as exc:
        return None, str(exc)


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Simulation settings")
    pit_loss = st.slider(
        "Pit stop time loss (s)", min_value=10.0, max_value=35.0,
        value=DEFAULT_PIT_STOP_LOSS_S, step=0.5,
        help=(
            "Time lost making a green-flag pit stop. Real pit lanes "
            "cost roughly 18-28s depending on the circuit."))
    n_simulations = st.slider(
        "Monte Carlo simulations", min_value=100, max_value=5000,
        value=1000, step=100,
        help=(
            "More simulations give a steadier win-rate estimate, at the "
            "cost of a slower run."))

    with st.expander("\u2139\ufe0f How this works"):
        st.markdown(
            "**Tyre wear.** Every lap, a car's tyres wear down and lap "
            "times get slower. This tool looks at the actual lap times "
            "from the chosen race to learn how much slower each tyre "
            "compound gets, lap by lap.\n\n"
            "**Pit stops.** Changing tyres costs time - the car slows "
            "down, enters the pits, and rejoins the race. A fixed time "
            "loss per stop is used (adjustable above).\n\n"
            "**Safety cars.** When a Safety Car comes out, the whole "
            "field slows down together, so a pit stop made during that "
            "period costs much less time than normal. This tool looks "
            "at how often a Safety Car has appeared at this circuit in "
            "past races, then uses that history to randomly decide if "
            "and when one happens in each simulated race.\n\n"
            "**Monte Carlo simulation.** Nobody can know in advance "
            "whether a Safety Car will appear, so the race is simulated "
            "thousands of times, each with a fresh random roll of the "
            "dice on the Safety Car, and the fastest strategy in each "
            "simulated race is recorded. The **win rate** is simply the "
            "share of those simulated races each strategy won.")

# ---------------------------------------------------------------------
# Header + race selection (Part 1)
# ---------------------------------------------------------------------
st.title("\U0001F3CE F1 Race Strategy Simulator")
st.caption("Pit-stop strategy analysis for real Formula 1 races")

current_year = dt.date.today().year
col_year, col_race = st.columns([1, 3])
with col_year:
    year = st.selectbox(
        "Season", list(range(current_year, MIN_SUPPORTED_YEAR - 1, -1)))

races = cached_race_list(year)
if races.empty:
    st.warning(
        f"No completed races found for {year} yet - pick a different "
        "season.")
    st.stop()

with col_race:
    race_row = st.selectbox(
        "Grand Prix", list(races.itertuples()),
        format_func=lambda r: (
            f"Round {r.RoundNumber}: {r.EventName} ({r.EventDate.date()})"))
round_number = int(race_row.RoundNumber)
race_key = f"{year}_{round_number}"        # namespaces widget keys below

# ---------------------------------------------------------------------
# Load + fit (Parts 1-3), with a spinner (built into @st.cache_data above)
# ---------------------------------------------------------------------
try:
    bundle = load_race_bundle(year, round_number)
except (DataLoadError, ValueError) as exc:
    st.error(f"Couldn't load this race: {exc}")
    st.stop()

race_length = bundle["race_length"]
models = bundle["models"]
sc_data = bundle["sc_data"]
available_compounds = list(models)      # already alphabetical (see Part 2)

st.subheader(f"{bundle['event_name']} {year} \u2014 {race_length} laps")

# ---------------------------------------------------------------------
# Degradation chart + safety car metric (Part 5 function 1)
# ---------------------------------------------------------------------
chart_col, metric_col = st.columns([2, 1])
with chart_col:
    if models:
        st.plotly_chart(
            plot_degradation_curves(models), use_container_width=True)
    else:
        st.warning(
            "Not enough clean laps in this race to fit a degradation "
            "curve for any compound.")

with metric_col:
    st.metric("Safety car probability", f"{sc_data['sc_probability']:.0%}")
    if sc_data["used_default"]:
        st.caption(
            "Not enough race history at this circuit - using a "
            "league-wide default.")
    else:
        st.caption(
            f"Based on {sc_data['sample_size']} past race(s) at this "
            "circuit.")
    if sc_data["avg_sc_start_laps"]:
        laps_text = ", ".join(
            f"lap {lap}" for lap in sc_data["avg_sc_start_laps"])
        st.caption(f"Typically appears around {laps_text}.")

if not bundle["fit_summary"].empty:
    with st.expander("Degradation fit details"):
        st.dataframe(
            bundle["fit_summary"].round(3), use_container_width=True)

if not available_compounds:
    st.error(
        "No tyre compound has enough clean laps in this race to build "
        "a strategy. Try a different race.")
    st.stop()

# ---------------------------------------------------------------------
# Strategy builder (Part 4's Strategy, via simple UI controls)
# ---------------------------------------------------------------------
st.divider()
st.subheader("Build strategies to compare")

n_strategies = st.radio(
    "Strategies to compare", STRATEGY_COUNT_CHOICES, index=1,
    horizontal=True, key=f"n_strat_{race_key}")
max_feasible_stops = min(MAX_STOPS, max(0, race_length - 1))

strategy_forms = []
for i, col in enumerate(st.columns(n_strategies)):
    with col:
        st.markdown(f"**Strategy {i + 1}**")
        default_stops, offset = DEFAULT_PLANS[i % len(DEFAULT_PLANS)]
        default_stops = min(default_stops, max_feasible_stops)

        name = st.text_input(
            "Name", value=f"{default_stops}-Stop {i + 1}",
            key=f"name_{race_key}_{i}")
        stops = st.selectbox(
            "Pit stops", list(range(max_feasible_stops + 1)),
            index=default_stops, key=f"stops_{race_key}_{i}")

        compounds, pit_laps, prev_lap = [], [], 0
        for stint in range(stops + 1):
            default_compound = available_compounds[
                (offset + stint) % len(available_compounds)]
            compound = st.selectbox(
                f"Stint {stint + 1} tyre", available_compounds,
                index=available_compounds.index(default_compound),
                key=f"cmp_{race_key}_{i}_{stint}")
            compounds.append(compound)
            if stint < stops:
                low, high = prev_lap + 1, race_length - 1
                default_lap = min(max(
                    round(race_length * (stint + 1) / (stops + 1)), low),
                    high)
                lap = st.slider(
                    f"Pit at end of lap (stint {stint + 1})", low, high,
                    default_lap, key=f"lap_{race_key}_{i}_{stint}")
                pit_laps.append(lap)
                prev_lap = lap
        strategy_forms.append(
            {"name": name, "compounds": compounds, "pit_laps": pit_laps})

# ---------------------------------------------------------------------
# Run simulation (Part 4's compare_strategies, which runs
# run_monte_carlo internally) + results (Part 5's remaining charts)
# ---------------------------------------------------------------------
st.divider()
run_clicked = st.button("\U0001F3C1 Run Simulation", type="primary")

if run_clicked:
    built, errors = [], []
    for form in strategy_forms:
        strategy, error = build_strategy_or_none(
            form["name"], form["compounds"], form["pit_laps"], race_length)
        (built if strategy else errors).append(strategy or error)
    names = [s.name for s in built]
    if len(set(names)) != len(names):
        errors.append("Strategy names must be different from each other.")

    if errors:
        for error in errors:
            st.error(error)
    else:
        with st.spinner(f"Running {n_simulations:,} Monte Carlo "
                        "simulations..."):
            result_table = compare_strategies(
                built, models, race_length, sc_data,
                pit_stop_loss_seconds=pit_loss,
                n_simulations=n_simulations)
        st.session_state["results"] = {
            "table": result_table,
            "strategies": {s.name: s for s in built},
            "signature": strategy_forms_signature(
                strategy_forms, race_key, pit_loss, n_simulations),
        }

if "results" in st.session_state:
    results = st.session_state["results"]
    table = results["table"]
    strategies_by_name = results["strategies"]
    current_signature = strategy_forms_signature(
        strategy_forms, race_key, pit_loss, n_simulations)

    if results["signature"][0] != race_key:
        st.info(
            "This result is for a different race. Click **Run "
            "Simulation** to analyse the current selection.")
    elif results["signature"] != current_signature:
        st.info(
            "Settings have changed since this result. Click **Run "
            "Simulation** again to refresh it.")

    st.subheader("Results")
    st.markdown(build_recommendation(table, sc_data, strategies_by_name))

    st.dataframe(format_comparison_table(table), use_container_width=True)

    tab_compare, tab_timeline, tab_winrate, tab_spread = st.tabs(
        ["Comparison", "Stint timelines", "Win rate", "Time spread"])
    with tab_compare:
        st.plotly_chart(
            plot_strategy_comparison(table, metric="mc_avg_time_s"),
            use_container_width=True)
    with tab_timeline:
        st.plotly_chart(
            plot_stint_timeline(list(strategies_by_name.values())),
            use_container_width=True)
    with tab_winrate:
        st.plotly_chart(plot_win_rate_pie(table), use_container_width=True)
    with tab_spread:
        st.plotly_chart(
            plot_monte_carlo_distribution(table), use_container_width=True)
else:
    st.info(
        "Configure your strategies above, then click **Run Simulation**.")
