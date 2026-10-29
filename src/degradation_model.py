"""Tyre degradation curve fitting from real race lap data.

Fits one degradation curve per compound: predicted lap time as a function
of tyre age (laps completed on the current set, 1 = first lap on a fresh
set). The fitted models plug directly into ``strategy_simulator`` and
``visualizer``, which both accept a ``numpy.poly1d`` (or a callable, or an
object with ``.predict()``) - this module returns ``numpy.poly1d`` objects.

Method: a straight line (lap time vs tyre age) fitted per compound on laps
considered "clean" (raced at normal pace, unaffected by a pit stop or a
track-status neutralisation), after correcting for the fuel already burned
off, then refit once with outliers removed. This is a first-order model -
real tyre wear can curve late in a stint ("falling off a cliff"), which a
straight line will not capture, and a curve is extrapolated, with growing
uncertainty, for any tyre age longer than what was actually raced on it.

Typical use::

    from src.data_loader import get_lap_data, load_race_session, setup_cache
    from src.degradation_model import fit_degradation_models

    setup_cache()
    laps = get_lap_data(load_race_session(2023, "Bahrain"))
    models = fit_degradation_models(laps)   # {"MEDIUM": poly1d, ...}
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# TrackStatus codes that make a lap unrepresentative of normal tyre wear:
# 4 = Safety Car, 5 = Red Flag, 6 = VSC deployed, 7 = VSC ending (see
# safety_car_model.py for the full code table).
_NEUTRALISED_STATUS = "[4567]"

# Time a car gains per lap purely from burning off fuel, independent of
# tyre wear. A commonly cited approximation is about 0.03 s per kg of
# fuel; for a roughly 100 kg starting fuel load burned evenly across a
# race, that works out to somewhere around 0.05-0.08 s per lap of race
# distance, and 0.06 sits in the middle of that range. This module ADDS
# it back (fuel_effect x laps completed) so later, lighter-fuel laps
# become comparable to earlier ones - without it, the fuel effect reads
# as tyres getting FASTER with age, masking real wear.
DEFAULT_FUEL_EFFECT_S_PER_LAP = 0.06

# Minimum clean laps needed before a compound gets a fitted curve; below
# this a straight-line fit is unreliable.
DEFAULT_MIN_LAPS = 30

# Robust outlier rejection: laps whose residual from the first fit exceeds
# this many scaled-MAD (median absolute deviation) are dropped and the
# line is refit once. Traffic and small timing errors are the main source
# of outliers here, not the tyre-wear signal itself.
_OUTLIER_MAD_THRESHOLD = 2.5
_MAD_TO_STD = 1.4826    # scales MAD to be comparable to a standard deviation


def clean_laps_for_fitting(laps_df):
    """Keep only laps that reflect normal, green-flag tyre wear.

    Drops: laps with no recorded time, in/out laps (a pit stop distorts
    them), laps with no tyre-age data, the first lap of the race (rolling
    start and first-lap traffic), and any lap touched by a Safety Car,
    Virtual Safety Car or red flag - cars run well off normal pace under
    all three, which would bias a degradation fit.

    Args:
        laps_df: Lap table from ``data_loader.get_lap_data``.

    Returns:
        A filtered copy of ``laps_df`` with the same columns.

    Raises:
        ValueError: If a required column is missing.
    """
    required = {"LapNumber", "LapTime", "Compound", "TyreLife",
                "TrackStatus", "MissingLapTime", "IsInLap", "IsOutLap"}
    missing = required - set(laps_df.columns)
    if missing:
        raise ValueError(f"laps_df is missing column(s): {sorted(missing)}")

    neutralised = laps_df["TrackStatus"].fillna("").astype(str).str.contains(
        _NEUTRALISED_STATUS)
    keep = (
        ~laps_df["MissingLapTime"] & ~laps_df["IsInLap"]
        & ~laps_df["IsOutLap"] & ~neutralised
        & (laps_df["LapNumber"] > 1) & laps_df["TyreLife"].notna()
    )
    return laps_df.loc[keep].copy()


def fuel_corrected_lap_time(
        laps_df, fuel_effect_s_per_lap=DEFAULT_FUEL_EFFECT_S_PER_LAP):
    """Lap times with the estimated fuel-burn effect removed.

    Cars get faster through a race simply because they carry less fuel,
    which would otherwise look like negative tyre degradation. This adds
    back ``fuel_effect_s_per_lap`` seconds for every lap already
    completed, so laps from different points in the race become
    comparable on tyre wear alone.

    Args:
        laps_df: Lap table with ``LapNumber`` and ``LapTime`` (seconds)
            columns.
        fuel_effect_s_per_lap: Seconds gained per lap of race distance
            from a lighter fuel load. Default 0.06 (a commonly cited
            range is about 0.05-0.08 s/lap).

    Returns:
        A ``pandas.Series`` of corrected lap times, aligned to
        ``laps_df``'s index.

    Raises:
        ValueError: If ``fuel_effect_s_per_lap`` is negative.
    """
    if fuel_effect_s_per_lap < 0:
        raise ValueError("fuel_effect_s_per_lap must not be negative.")
    return laps_df["LapTime"] + fuel_effect_s_per_lap * (
        laps_df["LapNumber"] - 1)


def _fit_one_compound(age, lap_time, min_laps):
    """Fit one outlier-robust straight line; None if there's too little
    data."""
    if len(age) < min_laps:
        return None
    fit = np.polyfit(age, lap_time, 1)
    residual = lap_time - np.polyval(fit, age)
    mad = np.median(np.abs(residual - np.median(residual)))
    scale = max(_MAD_TO_STD * mad, 1e-9)
    keep = np.abs(residual) <= _OUTLIER_MAD_THRESHOLD * scale
    if keep.sum() >= min_laps:                 # refit without outliers
        fit = np.polyfit(age[keep], lap_time[keep], 1)
    return np.poly1d(fit)


def fit_degradation_models(
    laps_df,
    min_laps=DEFAULT_MIN_LAPS,
    fuel_effect_s_per_lap=DEFAULT_FUEL_EFFECT_S_PER_LAP,
):
    """Fit a tyre degradation curve for every compound with enough data.

    A straight line (lap time vs tyre age) is fitted per compound on
    clean, fuel-corrected laps (see ``clean_laps_for_fitting`` and
    ``fuel_corrected_lap_time``), then refit once with outliers removed
    (laps more than 2.5 scaled-MAD from the first fit - mostly traffic).
    Compounds are discovered from the data, so a wet race fits
    INTERMEDIATE/WET curves the same way a dry race fits
    SOFT/MEDIUM/HARD.

    The returned models plug directly into
    ``strategy_simulator.simulate_strategy`` and
    ``visualizer.plot_degradation_curves``, which both accept a
    ``numpy.poly1d`` directly.

    Args:
        laps_df: Lap table from ``data_loader.get_lap_data``, for one
            session (usually a race).
        min_laps: Minimum clean laps a compound needs before it gets a
            fitted curve. Default 30.
        fuel_effect_s_per_lap: Passed to ``fuel_corrected_lap_time``.

    Returns:
        Dict mapping compound name -> ``numpy.poly1d``. A compound with
        fewer than ``min_laps`` clean laps is left out (a warning is
        logged for each), so the dict may have anywhere from zero to
        five entries.

    Raises:
        ValueError: If ``laps_df`` is missing a required column or
            ``min_laps`` is not positive.
    """
    if min_laps < 1:
        raise ValueError("min_laps must be at least 1.")
    clean = clean_laps_for_fitting(laps_df)
    corrected = fuel_corrected_lap_time(clean, fuel_effect_s_per_lap)

    models = {}
    for compound in sorted(clean["Compound"].unique()):
        if compound == "UNKNOWN":
            continue
        rows = (clean["Compound"] == compound).to_numpy()
        age = clean.loc[rows, "TyreLife"].to_numpy(dtype=float)
        lap_time = corrected[rows].to_numpy(dtype=float)
        model = _fit_one_compound(age, lap_time, min_laps)
        if model is None:
            logger.warning(
                "Only %d clean %s lap(s) (need %d); no degradation curve "
                "fitted for this compound.", len(age), compound, min_laps)
            continue
        models[compound] = model
    return models


def degradation_fit_summary(
        laps_df, models,
        fuel_effect_s_per_lap=DEFAULT_FUEL_EFFECT_S_PER_LAP):
    """Small diagnostics table describing how each curve was fitted.

    For display alongside the fitted curves (e.g. in a dashboard); the
    simulator itself only needs the models, not this summary.

    Args:
        laps_df: The same lap table passed to ``fit_degradation_models``.
        models: The dict ``fit_degradation_models`` returned.
        fuel_effect_s_per_lap: Must match the value used to fit
            ``models``, so the lap counts and average lap time line up
            with the curve.

    Returns:
        DataFrame indexed by compound with ``n_laps`` (clean laps used),
        ``avg_lap_time_s`` (mean fuel-corrected lap time) and
        ``slope_s_per_lap`` (predicted change in lap time per extra lap
        of tyre age - the degradation rate). Empty (with these columns)
        if ``models`` is empty.
    """
    clean = clean_laps_for_fitting(laps_df)
    corrected = fuel_corrected_lap_time(clean, fuel_effect_s_per_lap)
    columns = ["n_laps", "avg_lap_time_s", "slope_s_per_lap"]
    if not models:
        return pd.DataFrame(columns=columns)

    rows = {}
    for compound, model in models.items():
        mask = (clean["Compound"] == compound).to_numpy()
        rows[compound] = {
            "n_laps": int(mask.sum()),
            "avg_lap_time_s": (float(corrected[mask].mean()) if mask.any()
                               else float("nan")),
            "slope_s_per_lap": float(model.coeffs[0]),
        }
    return pd.DataFrame.from_dict(rows, orient="index")[columns]
