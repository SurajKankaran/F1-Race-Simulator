"""Plotly charts for the F1 strategy simulator dashboard.

Every function returns a ``plotly.graph_objects.Figure`` (all five use the
same library, so they behave the same way). Show a figure with
``fig.show()`` in a notebook or ``st.plotly_chart(fig,
use_container_width=True)`` in Streamlit.

Colours
-------
Tyre compounds always get the same colour, taken from ``COMPOUND_COLORS``:
Soft = red, Medium = yellow, Hard = white/grey, Intermediate = green,
Wet = blue. The hard tyre is drawn light grey rather than pure white so it
stays visible on a white background. Charts of whole strategies (which mix
compounds) use a separate palette so a strategy is never mistaken for a
tyre, and small coloured dots after a strategy's name show which compounds
it uses. Purple marks the fastest strategy, as on F1 timing screens.

Typical use::

    fig = plot_stint_timeline(strategy)
    fig.show()
"""

import re
from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.strategy_simulator import (
    Strategy, format_race_time, predict_lap_time,
)

# Real F1 tyre colours. Hard is light grey (see the module docstring).
COMPOUND_COLORS = {
    "SOFT": "#E53935",
    "MEDIUM": "#FDD835",
    "HARD": "#B0B0B0",
    "INTERMEDIATE": "#43A047",
    "WET": "#1E88E5",
}
OTHER_COMPOUND_COLOR = "#455A64"      # any compound not listed above
_COMPOUND_ALIASES = {"INTER": "INTERMEDIATE", "INTERS": "INTERMEDIATE",
                     "INTERMEDIATES": "INTERMEDIATE"}

FASTEST_COLOR = "#8E24AA"             # purple: fastest, as on timing screens
OTHER_BAR_COLOR = "#90A4AE"           # neutral blue-grey for the rest
# One colour per strategy, chosen to look unlike the tyre colours.
STRATEGY_COLORS = ["#00897B", "#EF6C00", "#5C6BC0", "#D81B60", "#6D4C41",
                   "#546E7A"]
_OUTLINE = "#2B2B2B"                  # thin outline so pale fills stay visible

_COMPOUND_WORD = re.compile(
    r"\b(soft|medium|hard|intermediate|inters?|wet)s?\b", re.IGNORECASE)
_TICK_STEPS = (0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600)
_DEFAULT_MAX_TYRE_AGE = 40


def _normalise_compound(compound):
    """Upper-case a compound name and expand short forms ("inter")."""
    name = str(compound).strip().upper()
    return _COMPOUND_ALIASES.get(name, name)


def compound_color(compound):
    """Return the hex colour used for a tyre compound in every chart.

    Args:
        compound: Compound name, any case ("SOFT", "medium", "inter").

    Returns:
        Hex colour string; a dark slate for compounds without a set colour.
    """
    return COMPOUND_COLORS.get(
        _normalise_compound(compound), OTHER_COMPOUND_COLOR)


def _ordered_compounds(compounds):
    """Sort compounds soft to wet, unknown ones last (alphabetically)."""
    known = list(COMPOUND_COLORS)
    return sorted(compounds, key=lambda c: (
        known.index(c) if c in known else len(known), c))


def _text_on(color):
    """Pick black or white text, whichever is readable on ``color``."""
    red, green, blue = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "#111111" if luminance > 150 else "#FFFFFF"


def _rgba(color, alpha):
    """Convert a ``#RRGGBB`` colour to an ``rgba(...)`` string."""
    red, green, blue = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def _label_with_chips(name):
    """Append a coloured dot per compound found in a strategy name.

    "1-Stop Medium-Hard" gets a yellow and a grey dot. Names without
    compound words are returned unchanged.
    """
    compounds = [match.group(1) for match in _COMPOUND_WORD.finditer(name)]
    dots = "".join(
        f"<span style='color:{compound_color(c)}'>\u25cf</span>"
        for c in compounds)
    return f"{name} {dots}" if dots else name


def _strategy_colors(names):
    """Assign each strategy name a colour from ``STRATEGY_COLORS``."""
    return {name: STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
            for i, name in enumerate(names)}


def _time_ticks(low, high, target=6):
    """Round-number tick positions and H:MM:SS labels for a seconds axis.

    Args:
        low: Lowest value on the axis, in seconds.
        high: Highest value on the axis, in seconds.
        target: Rough maximum number of ticks wanted.

    Returns:
        Tuple ``(positions, labels)`` of equal-length lists.
    """
    span = max(high - low, 1e-9)
    step = next((s for s in _TICK_STEPS if span / s <= target),
                _TICK_STEPS[-1])
    first = np.ceil(low / step) * step
    positions = np.arange(first, high + step * 1e-6, step)
    if len(positions) < 2:
        positions = np.array([low, high])
    labels = [format_race_time(p) if step < 1 else format_race_time(p)[:-2]
              for p in positions]
    return positions.tolist(), labels


def _apply_layout(fig, title, subtitle=None, height=460):
    """Apply the shared look (white template, left-aligned title)."""
    heading = f"<b>{title}</b>"
    if subtitle:
        heading += (f"<br><span style='font-size:13px;color:#666666'>"
                    f"{subtitle}</span>")
    fig.update_layout(
        template="plotly_white",
        title=dict(text=heading, x=0.01, xanchor="left"),
        height=height,
        margin=dict(l=70, r=30, t=95, b=60),
        font=dict(size=13),
    )
    return fig


def _age_limit(max_tyre_age, compound):
    """Longest tyre age to draw for one compound."""
    if isinstance(max_tyre_age, Mapping):
        limits = {_normalise_compound(k): v for k, v in max_tyre_age.items()}
        limit = limits.get(compound, _DEFAULT_MAX_TYRE_AGE)
    else:
        limit = max_tyre_age
    if not float(limit).is_integer() or limit < 2:
        raise ValueError(
            f"max_tyre_age must be a whole number of at least 2, got "
            f"{limit!r}.")
    return int(limit)


def plot_degradation_curves(degradation_models,
                            max_tyre_age=_DEFAULT_MAX_TYRE_AGE):
    """Overlay the tyre degradation curve of every compound.

    Each curve shows the predicted lap time (y) against tyre life (x) in
    the compound's colour. The legend gives the average change in lap time
    per lap over the plotted range, so a steeper line and a bigger number
    mean faster wear.

    Args:
        degradation_models: Dict mapping compound -> model, in any form
            ``strategy_simulator`` accepts (callable, object with
            ``.predict()``, or polynomial coefficients).
        max_tyre_age: Oldest tyre age to draw, as one number for all
            compounds or a dict per compound. Curves drawn beyond the ages a
            model was fitted on are extrapolations. Default 40.

    Returns:
        A Plotly ``Figure``.

    Raises:
        ValueError: If ``degradation_models`` is empty or not a dict, or
            ``max_tyre_age`` is invalid.
    """
    if not isinstance(degradation_models, Mapping) or not degradation_models:
        raise ValueError(
            "degradation_models must be a non-empty dict mapping compound "
            "names to models.")
    models = {_normalise_compound(k): v for k, v in degradation_models.items()}

    fig = go.Figure()
    for compound in _ordered_compounds(models):
        ages = np.arange(1, _age_limit(max_tyre_age, compound) + 1)
        lap_times = np.array(
            [predict_lap_time(models[compound], age) for age in ages])
        slope = (lap_times[-1] - lap_times[0]) / (ages[-1] - ages[0])
        fig.add_trace(go.Scatter(
            x=ages, y=lap_times, mode="lines",
            name=f"{compound.title()} (avg {slope:+.3f} s/lap)",
            line=dict(color=compound_color(compound), width=4),
            hovertemplate="%{y:.2f} s<extra></extra>",
        ))

    _apply_layout(
        fig, "Tyre degradation by compound",
        "Predicted lap time against tyre life; a steeper line means faster "
        "wear")
    fig.update_layout(hovermode="x unified", legend=dict(title="Compound"))
    fig.update_xaxes(title="Tyre life (laps)", dtick=5)
    fig.update_yaxes(title="Predicted lap time (s)", tickformat=".1f")
    return fig


def plot_strategy_comparison(comparison_df, metric="predicted_time_s"):
    """Bar chart of race time per strategy, fastest highlighted in purple.

    Bars are sorted fastest first. Because strategies usually differ by
    seconds in a race lasting about 90 minutes, the time axis is truncated
    (and says so) so the gaps are visible; each bar is labelled with its
    time and its gap to the fastest.

    Args:
        comparison_df: DataFrame from ``compare_strategies`` (one row per
            strategy, indexed by strategy name).
        metric: Column to plot. ``"predicted_time_s"`` (default) is the
            prediction without a safety car; ``"mc_avg_time_s"`` is the
            Monte Carlo average.

    Returns:
        A Plotly ``Figure``.

    Raises:
        ValueError: If ``comparison_df`` is empty, has repeated strategy
            names, or ``metric`` is missing or not numeric.
    """
    if not isinstance(comparison_df, pd.DataFrame) or comparison_df.empty:
        raise ValueError(
            "comparison_df must be the non-empty DataFrame returned by "
            "compare_strategies().")
    if metric not in comparison_df.columns:
        raise ValueError(
            f"comparison_df has no {metric!r} column; available columns: "
            f"{list(comparison_df.columns)}.")
    if comparison_df.index.duplicated().any():
        raise ValueError("Strategy names (the index) must be unique.")
    values = pd.to_numeric(comparison_df[metric], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Column {metric!r} must be numeric, with no gaps.")

    values = values.sort_values(kind="stable")
    names = [str(name) for name in values.index]
    seconds = values.to_numpy(dtype=float)
    best = seconds.min()
    is_best = np.isclose(seconds, best, rtol=0.0, atol=1e-9)
    gaps = seconds - best
    colors = [FASTEST_COLOR if flag else OTHER_BAR_COLOR for flag in is_best]
    labels = [f"{format_race_time(s)}   " + ("fastest" if flag
                                             else f"+{gap:.1f} s")
              for s, flag, gap in zip(seconds, is_best, gaps)]

    span = seconds.max() - best
    axis_low = best - max(2.0 * span, 20.0)
    axis_high = seconds.max() + max(0.08 * (seconds.max() - axis_low), 1.0)
    ticks, tick_labels = _time_ticks(axis_low, axis_high)

    fig = go.Figure(go.Bar(
        x=seconds, y=names, orientation="h",
        marker=dict(color=colors),
        text=labels, textposition="inside", insidetextanchor="end",
        textfont=dict(color=[_text_on(c) for c in colors], size=13),
        customdata=[[format_race_time(s), float(g)]
                    for s, g in zip(seconds, gaps)],
        hovertemplate=("%{y}<br>%{customdata[0]}<br>"
                       "+%{customdata[1]:.1f} s vs fastest<extra></extra>"),
    ))
    title = {"predicted_time_s": "Predicted race time (no safety car)",
             "mc_avg_time_s": "Average race time over the Monte Carlo runs",
             }.get(metric, f"Race time by strategy ({metric})")
    _apply_layout(fig, title,
                  "Fastest strategy in purple; the time axis is truncated to "
                  "show the gaps", height=max(280, 120 + 70 * len(names)))
    fig.update_xaxes(range=[axis_low, axis_high], tickmode="array",
                     tickvals=ticks, ticktext=tick_labels,
                     title="Total race time (h:mm:ss)")
    fig.update_yaxes(autorange="reversed", tickmode="array", tickvals=names,
                     ticktext=[_label_with_chips(n) for n in names])
    return fig


def _monte_carlo_results(results):
    """Return the Monte Carlo results, unwrapping a comparison table.

    Accepts the plain dict ``run_monte_carlo`` returns or the dict-like
    ``MonteCarloResult`` ``compare_strategies`` stores in ``table.attrs``;
    duck-typed (``in`` / ``[]``) rather than an ``isinstance`` check so
    either works.
    """
    if isinstance(results, pd.DataFrame) and "monte_carlo" in results.attrs:
        results = results.attrs["monte_carlo"]
    try:
        ok = "summary" in results and "times" in results
    except TypeError:
        ok = False
    if not ok:
        raise ValueError(
            "Pass the dict returned by run_monte_carlo() (or the table "
            "returned by compare_strategies()).")
    return results


def plot_monte_carlo_distribution(monte_carlo_results, kind="violin"):
    """Show the spread of finishing times of every strategy.

    A narrow shape means a consistent strategy, a wide one means its result
    depends heavily on the safety car. Under each violin the label gives
    the standard deviation of the finishing time (sigma); the dashed line
    is the mean.

    Args:
        monte_carlo_results: Dict from ``run_monte_carlo`` (or the table
            from ``compare_strategies``, which carries it).
        kind: ``"violin"`` (default) or ``"histogram"`` (overlaid
            histograms with a dashed line at each mean).

    Returns:
        A Plotly ``Figure``.

    Raises:
        ValueError: If the results are not from ``run_monte_carlo`` or
            ``kind`` is not recognised.
    """
    if kind not in ("violin", "histogram"):
        raise ValueError("kind must be 'violin' or 'histogram'.")
    times = _monte_carlo_results(monte_carlo_results)["times"]
    names = [str(name) for name in times.columns]
    colors = _strategy_colors(names)
    all_times = times.to_numpy(dtype=float)
    low, high = float(all_times.min()), float(all_times.max())
    stds = {name: (float(np.std(times[col], ddof=1)) if len(times) > 1
                   else 0.0) for name, col in zip(names, times.columns)}
    n_runs = len(times)

    fig = go.Figure()
    if kind == "violin":
        # One bandwidth for every violin keeps the shapes comparable. Times
        # often pile up on a few exact values (no safety car, or one
        # discounted stop), which the default smoothing collapses into a
        # needle.
        bandwidth = max((high - low) / 12.0, 1.0)
        for name, column in zip(names, times.columns):
            values = times[column].to_numpy(dtype=float)
            fig.add_trace(go.Violin(
                y=values, x=[name] * len(values), name=name,
                meanline_visible=True, bandwidth=bandwidth,
                points="all" if np.ptp(values) == 0 else False,
                spanmode="hard", scalemode="width",
                line=dict(color=colors[name]),
                fillcolor=_rgba(colors[name], 0.45), showlegend=False,
                hoveron="violins+points",
            ))
        fig.update_xaxes(
            tickmode="array", tickvals=names,
            ticktext=[f"{_label_with_chips(n)}<br><span style="
                      f"'font-size:11px;color:#666666'>\u03c3 = "
                      f"{stds[n]:.1f} s</span>" for n in names])
        pad = max((high - low) * 0.08, 1.0)
        ticks, tick_labels = _time_ticks(low - pad, high + pad)
        fig.update_yaxes(range=[low - pad, high + pad], tickmode="array",
                         tickvals=ticks, ticktext=tick_labels,
                         title="Finishing time (h:mm:ss)")
        fig.update_layout(violinmode="group")
        title = "Spread of finishing times"
        subtitle = (f"{n_runs:,} simulated races; narrower = more "
                    "consistent, \u03c3 = standard deviation, dashed line "
                    "= mean")
    else:
        bins = dict(start=low, end=high + 1e-6,
                    size=max((high - low) / 40.0, 0.25))
        for name, column in zip(names, times.columns):
            fig.add_trace(go.Histogram(
                x=times[column].to_numpy(dtype=float), name=name,
                xbins=bins, histnorm="percent", opacity=0.6,
                marker=dict(color=colors[name],
                            line=dict(color="white", width=0.5)),
            ))
            fig.add_vline(x=float(times[column].mean()),
                          line=dict(color=colors[name], width=2,
                                    dash="dash"))
        pad = max((high - low) * 0.08, 1.0)
        ticks, tick_labels = _time_ticks(low - pad, high + pad)
        fig.update_xaxes(range=[low - pad, high + pad], tickmode="array",
                         tickvals=ticks, ticktext=tick_labels,
                         title="Finishing time (h:mm:ss)")
        fig.update_yaxes(title="Share of simulated races (%)")
        fig.update_layout(barmode="overlay", legend=dict(title="Strategy"))
        title = "Distribution of finishing times"
        subtitle = (f"{n_runs:,} simulated races; dashed lines mark each "
                    "strategy's mean")
    return _apply_layout(fig, title, subtitle, height=520)


def plot_win_rate_pie(monte_carlo_results):
    """Donut chart of the share of simulated races each strategy won.

    Args:
        monte_carlo_results: Dict from ``run_monte_carlo`` (or the table
            from ``compare_strategies``, which carries it).

    Returns:
        A Plotly ``Figure``. Strategies that never won stay in the legend
        with 0.0%.

    Raises:
        ValueError: If the results are not from ``run_monte_carlo``.
    """
    results = _monte_carlo_results(monte_carlo_results)
    summary = results["summary"]
    names = [str(name) for name in summary.index]
    rates = summary["win_rate_pct"].to_numpy(dtype=float)
    colors = _strategy_colors(names)
    n_runs = len(results["times"])

    fig = go.Figure(go.Pie(
        labels=[f"{n}  ({r:.1f}%)" for n, r in zip(names, rates)],
        values=rates, hole=0.55, sort=False, direction="clockwise",
        marker=dict(colors=[colors[n] for n in names],
                    line=dict(color="white", width=2)),
        texttemplate="%{percent:.1%}", textposition="inside",
        textfont=dict(color="white", size=14),
        hovertemplate="%{label}<br>fastest in %{value:.1f}% of races"
                      "<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>Win rate</b><br>{n_runs:,} races",
                       x=0.5, y=0.5, showarrow=False, font=dict(size=15))
    _apply_layout(fig, "Win rate by strategy",
                  "Share of simulated races in which each strategy "
                  "finished first", height=460)
    fig.update_layout(legend=dict(title="Strategy"))
    return fig


def _stint_rows(strategy):
    """Turn the ``strategy`` argument into a list of ``Strategy`` objects."""
    if isinstance(strategy, Strategy):
        return [strategy]
    if isinstance(strategy, Mapping):
        if "compound" in strategy:
            raise ValueError(
                "Pass a list of stints, not a single stint dict.")
        return [s if isinstance(s, Strategy) else Strategy(str(name), s)
                for name, s in strategy.items()]
    items = list(strategy)
    if not items:
        raise ValueError("Give at least one strategy.")
    if all(isinstance(item, Mapping) for item in items):
        return [Strategy("Strategy", items)]           # one list of stints
    return [item if isinstance(item, Strategy)
            else Strategy(f"Strategy {number}", item)
            for number, item in enumerate(items, start=1)]


def _unique_labels(names):
    """Make row labels unique by numbering repeats ("A", "A (2)")."""
    seen, labels = {}, []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        labels.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return labels


def plot_stint_timeline(strategy):
    """Gantt-style tyre chart: which compound is used on which laps.

    Draws one row per strategy with a bar per stint in the compound's
    colour, labelled with the compound and its length, and a diamond at
    every pit stop, like the stint charts on F1 broadcasts.

    Args:
        strategy: A ``Strategy``, a list of stint dicts (one strategy), or
            several strategies as a list of ``Strategy`` objects or a dict
            ``{name: strategy}`` to stack them for comparison.

    Returns:
        A Plotly ``Figure``.

    Raises:
        ValueError: If nothing is given or a strategy is invalid.
    """
    strategies = _stint_rows(strategy)
    labels = _unique_labels([s.name for s in strategies])
    race_laps = max(s.race_laps for s in strategies)

    fig = go.Figure()
    # Legend entries in soft-to-wet order (bars only add them in the order
    # they first appear); clicking one hides every stint of that compound.
    used = {_normalise_compound(st["compound"])
            for strat in strategies for st in strat.stints}
    for compound in _ordered_compounds(used):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=compound.title(),
            legendgroup=compound,
            marker=dict(symbol="square", size=13,
                        color=compound_color(compound),
                        line=dict(color=_OUTLINE, width=1.5)),
        ))
    pit_x, pit_y, pit_text = [], [], []
    for label, strat in zip(labels, strategies):
        for stint in strat.stints:
            compound = stint["compound"]
            color = compound_color(compound)
            first, last = stint["start_lap"], stint["end_lap"]
            n_laps = last - first + 1
            share = n_laps / race_laps
            if share < 0.06:
                text = compound[0]
            elif share < 0.13:
                text = compound.title()
            else:
                text = f"<b>{compound.title()}</b><br>{n_laps} laps"
            fig.add_trace(go.Bar(
                x=[n_laps], y=[label], base=[first - 0.5], orientation="h",
                width=0.62, name=_normalise_compound(compound).title(),
                legendgroup=_normalise_compound(compound), showlegend=False,
                marker=dict(color=color, line=dict(color=_OUTLINE, width=1.5)),
                text=[text], textposition="inside",
                insidetextanchor="middle", constraintext="none",
                textfont=dict(color=_text_on(color), size=13),
                hovertemplate=(f"{label}<br>{compound.title()}: laps "
                               f"{first}-{last} ({n_laps} laps)"
                               "<extra></extra>"),
            ))
        for stint in strat.stints[:-1]:
            pit_x.append(stint["end_lap"] + 0.5)
            pit_y.append(label)
            pit_text.append(f"{label}<br>Pit stop at the end of lap "
                            f"{stint['end_lap']}")
    if pit_x:
        fig.add_trace(go.Scatter(
            x=pit_x, y=pit_y, mode="markers", name="Pit stop",
            marker=dict(symbol="diamond", size=11, color=_OUTLINE,
                        line=dict(color="white", width=1.5)),
            hovertext=pit_text, hoverinfo="text",
        ))

    title = ("Stint timeline: " + labels[0] if len(strategies) == 1
             else "Tyre strategies")
    top, bottom = 95, 125                # bottom leaves room for the legend
    plot_height = 62 * len(strategies) + 30
    _apply_layout(fig, title, "Compound used on each lap; diamonds mark "
                  "pit stops", height=top + bottom + plot_height)
    fig.update_layout(
        barmode="overlay", margin=dict(l=70, r=30, t=top, b=bottom),
        legend=dict(orientation="h", yanchor="top",
                    y=-70.0 / plot_height, x=0.0))
    fig.update_xaxes(range=[0.5, race_laps + 0.5], title="Lap",
                     dtick=5 if race_laps <= 80 else 10)
    # type="category" is required: the legend-proxy traces above carry
    # y=[None], and without an explicit type Plotly would infer a numeric
    # axis from them and every bar would vanish. categoryarray fixes the
    # row order (first strategy on top) independently of trace order.
    fig.update_yaxes(type="category", categoryarray=list(reversed(labels)),
                     showgrid=False)
    return fig
