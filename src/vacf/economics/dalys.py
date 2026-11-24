"""DALY calculation.

Three components, following the input spreadsheet:

* **YLLs** — years of life lost per TB death, age-specific.
* **Post-TB DALYs** — lifetime disability accruing to survivors of TB treatment,
  age-specific (Menzies et al. 2021).
* **YLDs** — disability weight applied to person-time spent in symptomatic TB and on
  treatment.

The age-specific inputs are supplied in five-year bands, while the model uses far fewer age
groups. Rather than accept the resulting loss of precision, the per-death YLL for a model
age group is computed as the population-weighted mean of the five-year-band values across
the single years of age that the group spans, using the UN single-age population
distribution the demography module already loads. This keeps DALYs accurate even when the
model is run with coarse age bands.
"""

import numpy as np
import pandas as pd

from vacf.economics import contract
from vacf.economics.discounting import DEFAULT_DISCOUNT_RATE, discount_series

MAX_AGE = 120


def age_group_bounds(age_groups: list[str], max_age: int = MAX_AGE) -> list[tuple]:
    """Lower and upper bound (inclusive) of each model age group."""
    lowers = [int(a) for a in age_groups]
    if lowers != sorted(lowers):
        raise ValueError(f"Age groups must be in ascending order, got {age_groups}")
    bounds = []
    for i, lower in enumerate(lowers):
        upper = lowers[i + 1] - 1 if i < len(lowers) - 1 else max_age
        bounds.append((lower, upper, age_groups[i]))
    return bounds


def expand_to_single_ages(daly_table: pd.DataFrame, max_age: int = MAX_AGE) -> pd.DataFrame:
    """Expand a banded DALY table to one row per single year of age.

    Every year of age takes the value of the band containing it. The final band is open
    ended and extends to ``max_age``.
    """
    rows = []
    lowers = list(daly_table.index)
    for i, lower in enumerate(lowers):
        upper = lowers[i + 1] - 1 if i < len(lowers) - 1 else max_age
        for age in range(lower, upper + 1):
            rows.append({"age": age, **daly_table.loc[lower].to_dict()})
    return pd.DataFrame(rows).set_index("age").sort_index()


def age_weighted_daly_values(
    daly_table: pd.DataFrame,
    pop_by_age: pd.Series,
    age_groups: list[str],
    max_age: int = MAX_AGE,
) -> pd.DataFrame:
    """Per-event YLL and post-TB DALY values, remapped onto the model's age groups.

    Parameters
    ----------
    daly_table
        Banded table from :func:`vacf.economics.inputs.load_age_daly_table`.
    pop_by_age
        Population by single year of age, from
        :func:`vacf.economics.indicators.single_age_population`. A single reference year is used
        rather than weighting per simulated year; that is accurate as long as the
        within-band age distribution is reasonably stable over the analytic horizon.
    age_groups
        Model age group lower bounds.

    Returns
    -------
    pd.DataFrame
        Indexed by model age group label, with columns ``yll`` and ``post_tb_daly``.
    """
    by_single_age = expand_to_single_ages(daly_table, max_age)
    if pop_by_age.empty:
        raise ValueError("pop_by_age is empty; cannot weight DALY values")

    rows = {}
    for lower, upper, label in age_group_bounds(age_groups, max_age):
        ages = [a for a in range(lower, upper + 1) if a in pop_by_age.index]
        if not ages:
            raise ValueError(f"No population data for ages {lower}-{upper}")
        weights = pop_by_age.loc[ages]
        weights = weights / weights.sum()
        values = by_single_age.loc[ages]
        rows[label] = (values.T * weights.values).T.sum()

    return pd.DataFrame(rows).T[["yll", "post_tb_daly"]]


def compute_daly_streams(
    outputs_df: pd.DataFrame,
    age_groups: list[str],
    daly_values: pd.DataFrame,
    yld_weight: float,
) -> pd.DataFrame:
    """Undiscounted DALYs per year, split into components.

    Parameters
    ----------
    outputs_df
        Derived outputs for one scenario, indexed by year.
    daly_values
        Per-event YLL and post-TB DALY values by model age group, from
        :func:`age_weighted_daly_values`.
    yld_weight
        Disability weight for symptomatic TB and TB on treatment.

    Returns
    -------
    pd.DataFrame
        Year-indexed, with columns ``ylls``, ``post_tb_dalys``, ``ylds`` and ``total``.
    """
    ylls = pd.Series(0.0, index=outputs_df.index)
    post_tb = pd.Series(0.0, index=outputs_df.index)
    ylds = pd.Series(0.0, index=outputs_df.index)

    for age in age_groups:
        ylls += outputs_df[contract.TB_DEATHS_BY_AGE.format(age=age)] * daly_values.loc[age, "yll"]
        post_tb += (
            outputs_df[contract.TREATMENT_COMPLETIONS_BY_AGE.format(age=age)]
            * daly_values.loc[age, "post_tb_daly"]
        )
        # Person-time in disabling states, weighted. Both symptomatic disease and treatment
        # carry the same weight in the supplied inputs.
        ylds += yld_weight * (
            outputs_df[contract.SYMPTOMATIC_PERSON_TIME.format(age=age)]
            + outputs_df[contract.TREATMENT_PERSON_TIME.format(age=age)]
        )

    streams = pd.DataFrame({"ylls": ylls, "post_tb_dalys": post_tb, "ylds": ylds})
    streams["total"] = streams.sum(axis=1)
    return streams


def total_dalys(
    outputs_df: pd.DataFrame,
    age_groups: list[str],
    daly_values: pd.DataFrame,
    yld_weight: float,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
    reference_year: int = None,
    start_year: int = None,
    end_year: int = None,
) -> pd.Series:
    """Discounted DALYs by component over the analytic horizon."""
    streams = compute_daly_streams(outputs_df, age_groups, daly_values, yld_weight)
    if start_year is not None:
        streams = streams[streams.index >= start_year]
    if end_year is not None:
        streams = streams[streams.index <= end_year]
    if reference_year is None:
        reference_year = float(streams.index.min())
    return streams.apply(lambda col: np.sum(discount_series(col, discount_rate, reference_year)))
