"""Epidemiological calibration targets from the summer3 calibration notebook."""

from __future__ import annotations

import pandas as pd

from tb_macro.constants import CALENDAR_YEAR_MIDPOINT


def _at_midyear(values: dict[float, float]) -> pd.Series:
    """Place calendar-year observations at mid-year in model time."""
    series = pd.Series(values)
    series.index = series.index.astype(float) + CALENDAR_YEAR_MIDPOINT
    return series


# Vietnam NTP notification counts.
NOTIF_TARGET = _at_midyear(
    {
        2011: 100518,
        2012: 103906,
        2013: 102196,
        2014: 102087,
        2015: 100780,
        2016: 102097,
        2017: 102725,
        2018: 99658,
        2019: 102503,
        2020: 99582,
        2021: 77657,
        2022: 102479,
        2023: 104517,
    }
)

# Marks et al., IJTLD. Adults aged 15 and over, percent ever infected.
LATENT_TARGET = _at_midyear({2016.0: 36.8})

# Second national prevalence survey, per 100 000 adults.
PULM_PREV_TARGET = _at_midyear({2017.0: 322.0})

# High-infectious share of that survey's prevalent cases.
INF_PREV_TARGET = _at_midyear({2017.0: 79.0 / 322.0})

# Nguyen et al., EID. Two survey rounds, used as a ratio.
PREV_DECLINE_TARGET = _at_midyear({2007.0: 199.0, 2017.0: 125.0})
