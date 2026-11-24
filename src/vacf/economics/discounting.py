"""Discounting of costs and health effects.

Costs and DALYs are discounted at a constant annual rate back to a reference year. Both are
discounted at the same rate, which is standard practice and what the economic input
spreadsheet assumes (3% per year).

Note on double counting: the per-death YLL and per-survivor post-TB DALY values in the input
spreadsheet are already present values of a discounted lifetime stream, discounted to the
year of the event. Applying :func:`discount_series` to them additionally discounts from the
year the event occurs back to the reference year. That is correct and intended, but it means
the input table's discount rate and the rate used here must match.
"""

import numpy as np
import pandas as pd

DEFAULT_DISCOUNT_RATE = 0.03


def discount_factors(
    years, rate: float = DEFAULT_DISCOUNT_RATE, reference_year: int | float = None
) -> pd.Series:
    """Discount factor for each year, relative to ``reference_year``.

    Parameters
    ----------
    years
        Years at which quantities occur.
    rate
        Annual discount rate. Zero gives undiscounted results, which is a common
        sensitivity analysis.
    reference_year
        Year to which values are discounted. Defaults to the first year supplied.
    """
    years = pd.Index(years, name="year")
    if reference_year is None:
        reference_year = float(years.min())
    if rate < 0:
        raise ValueError("Discount rate must not be negative")
    return pd.Series(1.0 / (1.0 + rate) ** (years.values - reference_year), index=years)


def discount_series(
    series: pd.Series, rate: float = DEFAULT_DISCOUNT_RATE, reference_year: int | float = None
) -> pd.Series:
    """Apply discounting to a year-indexed series of costs or health effects."""
    return series * discount_factors(series.index, rate, reference_year)


def discounted_total(
    series: pd.Series,
    rate: float = DEFAULT_DISCOUNT_RATE,
    reference_year: int | float = None,
    start_year: int | float = None,
    end_year: int | float = None,
) -> float:
    """Sum a year-indexed series over an analytic horizon, with discounting applied.

    The horizon bounds are inclusive. Restricting the horizon before discounting matters:
    truncating afterwards would discount years that are then discarded, which is harmless,
    but doing it in this order makes the horizon explicit in the call.
    """
    windowed = series
    if start_year is not None:
        windowed = windowed[windowed.index >= start_year]
    if end_year is not None:
        windowed = windowed[windowed.index <= end_year]
    if windowed.empty:
        raise ValueError(
            f"No data within horizon [{start_year}, {end_year}] for series spanning "
            f"[{series.index.min()}, {series.index.max()}]"
        )
    return float(np.sum(discount_series(windowed, rate, reference_year)))
