"""Economic indicators derived from a solved model run.

Everything here is post-processing on ``vacf`` output. No model code is modified, so
``src/vacf`` stays byte-identical to upstream and the golden parity tests continue to
cover it unchanged.

One quantity needs care. The model's ACF flow counts people *detected*, and the screening
rate it uses is a hazard, built as ``peak = -log(1 - coverage)``. Costs in the input
spreadsheet are per person *screened*. Those differ: a hazard is memoryless, so someone who
screens negative stays eligible and is screened again, and at 80% coverage the hazard
(1.609) exceeds the throughput (0.800) by a factor of 2.01. Using the hazard would inflate
the dominant provider cost accordingly. :func:`screening_throughput` therefore rebuilds the
same cosine campaign shape with ``peak = coverage``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from jax import numpy as jnp
from jax import vmap

from vacf.constants import (
    AGES,
    AGE_STRATA,
    ISO3,
    N_TIMES,
    START_TIME,
)
from vacf.curves import get_cos_multicurve, get_scale_data
from vacf.inputs import get_country_pop, get_single_age_pop_from_ungroups
from vacf.model import PMAP
from vacf.outputs import age_deaths, age_treatment, flow_by_age

from vacf.economics import contract


def eligible_ages(params: dict[str, Any]) -> tuple[str, ...]:
    """Model age groups eligible for screening, from the run's ``acf_min_age``.

    Read from parameters rather than fixed at import, so that the eligibility floor used
    for costing can never drift from the one the model actually screened.
    """
    floor = float(params["acf_min_age"])
    return tuple(str(age) for age in AGE_STRATA if age >= floor)


def model_years() -> np.ndarray:
    """Annual output grid of the solved model."""
    return np.arange(START_TIME, START_TIME + N_TIMES, dtype=float)


def screening_throughput(times: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """People screened per person-year, over the ACF campaign.

    The same cosine ramp the model uses for its screening hazard, but with the plateau set
    to ``acf_coverage`` rather than ``-log(1 - acf_coverage)``, so that the integral over
    the campaign is the number of people screened rather than the number of screening
    events.

    Args:
        times: Years at which to evaluate the campaign.
        params: Model parameters, supplying the ACF campaign shape.

    Returns:
        Throughput at each requested year, zero outside the campaign window.

    Notes:
    -----
    Active case finding is delivered over a campaign beginning in
    {{ACF_START}} and lasting {{ACF_DURATION}} years, ramping on and
    off over {{ACF_SCALING_TIME}} years. The number of people screened
    each year is the campaign coverage applied to the eligible
    population, which is everyone aged {{ACF_MIN_AGE}} years and over
    who is not already receiving treatment.
    """
    start = params["acf_start"]
    scale = params["acf_scaling_time"]
    duration = params["acf_duration"]
    coverage = params["acf_coverage"]
    x = get_scale_data(
        jnp.array([start, start + scale, start + duration, start + duration + scale])
    )
    y = get_scale_data(jnp.array([0.0, coverage, coverage, 0.0]))
    # One vectorised call: per-year calls re-trace lax.switch each time (~15 s per run).
    curve = vmap(lambda t: get_cos_multicurve(t, x, y))
    return np.asarray(curve(jnp.asarray(times, dtype=jnp.float64)))


def _sum_compartments(result: Any, predicate) -> dict[str, np.ndarray]:
    """Sum compartment occupancy by age, over compartments matching ``predicate``."""
    values = result["compartments"].values
    data = np.asarray(values.data if hasattr(values, "data") else values)
    out = {age: np.zeros(data.shape[0]) for age in AGES}
    for row, traits in enumerate(PMAP.to_dicts()):
        if predicate(traits):
            out[traits["age"]] = out[traits["age"]] + data[:, row]
    return out


def symptomatic_person_time(result: Any) -> dict[str, np.ndarray]:
    """Person-time in symptomatic active TB, by age.

    Notes:
    -----
    Years lived with disability are accrued by people occupying
    symptomatic active tuberculosis states and by people receiving
    treatment, weighted by the disability weight for active
    tuberculosis.
    """
    return _sum_compartments(
        result,
        lambda t: t["disease_state"] == "active" and t.get("clinical") == "clin",
    )


def eligible_population(result: Any, params: dict[str, Any]) -> np.ndarray:
    """Population eligible for screening, excluding those already on treatment.

    Excluding people on treatment matches the basis on which the screening unit costs were
    derived.
    """
    on_treatment = age_treatment(result)
    population = _sum_compartments(result, lambda t: True)
    return sum(
        np.asarray(population[age]) - np.asarray(on_treatment[age]) for age in eligible_ages(params)
    )


def single_age_population(year: int) -> pd.Series:
    """UN single-year-of-age population for the modelled country, indexed by age.

    Used to weight age-specific DALY values within model age bands: the supplied DALY table
    is in five-year bands while the model's bands are wider and uneven, so a flat average
    across a band would misstate years of life lost.
    """
    single = get_single_age_pop_from_ungroups(get_country_pop(ISO3))
    rows = single[single["Time"] == year]
    if rows.empty:
        span = (int(single["Time"].min()), int(single["Time"].max()))
        raise ValueError(f"No single-age population for {year}; data span {span}")
    return rows.set_index("Age")["Pop"]


def economic_indicators(result: Any, params: dict[str, Any]) -> pd.DataFrame:
    """Year-indexed frame of every quantity the economic layer consumes.

    Args:
        result: Solved model output.
        params: The parameters the model was solved with, for the campaign shape.

    Returns:
        A frame indexed by calendar year, with the columns named in :mod:`vacf.economics.contract`.
    """
    years = model_years()
    columns: dict[str, np.ndarray] = {}

    columns[contract.N_SCREENED] = screening_throughput(years, params) * eligible_population(
        result, params
    )

    # The model keeps active and passive detection as separate flows, so the split needs no
    # apportionment assumption.
    acf_by_age = flow_by_age(result, "acf")
    passive_by_age = flow_by_age(result, "detection")
    columns[contract.TREATMENT_STARTS_ACTIVE] = sum(np.asarray(v) for v in acf_by_age.values())
    columns[contract.TREATMENT_STARTS_PASSIVE] = sum(np.asarray(v) for v in passive_by_age.values())

    deaths = age_deaths(result)
    completions = flow_by_age(result, "success")
    symptomatic = symptomatic_person_time(result)
    on_treatment = age_treatment(result)

    for age in AGES:
        columns[contract.TB_DEATHS_BY_AGE.format(age=age)] = np.asarray(deaths[age])
        columns[contract.TREATMENT_COMPLETIONS_BY_AGE.format(age=age)] = np.asarray(
            completions[age]
        )
        columns[contract.SYMPTOMATIC_PERSON_TIME.format(age=age)] = np.asarray(symptomatic[age])
        columns[contract.TREATMENT_PERSON_TIME.format(age=age)] = np.asarray(on_treatment[age])

    frame = pd.DataFrame(columns, index=pd.Index(years, name="year"))
    contract.validate(frame, list(AGES))
    return frame
