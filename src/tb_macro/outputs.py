"""Inspect-notebook indicators, computed the way ``outputs.py`` on the branch does."""

from __future__ import annotations

from typing import Any

from jax import numpy as jnp

from tb_macro.constants import AGES, DISEASE_STATES, INFECTED_STATES, PREV_STATES
from tb_macro.model import PMAP, get_compiled
from tb_macro.parameters import BASE_PARAMS


def _data(trace: Any) -> Any:
    values = trace.values
    return values.data if hasattr(values, "data") else values


def flow_total(result: Any, name: str) -> Any:
    """Sum a flow's per-edge instantaneous mass. Shape ``(time,)``."""
    return jnp.sum(_data(result[name]), axis=-1)


def _rows_for_age(src: Any, age: str) -> Any:
    rows = [
        i
        for i, comp in enumerate(src)
        if "age" in PMAP.to_dicts()[int(comp)] and PMAP.to_dicts()[int(comp)]["age"] == age
    ]
    return jnp.asarray(rows, dtype=jnp.int32)


def flow_by_age(result: Any, name: str) -> dict[str, Any]:
    """Sum flow mass by the source compartment's age."""
    compiled = get_compiled()
    src = compiled.flows[name].src_idx
    data = _data(result[name])
    out: dict[str, Any] = {}
    for age in AGES:
        rows = _rows_for_age(src, age)
        out[age] = jnp.sum(data[:, rows], axis=-1) if rows.size else jnp.zeros(data.shape[0])
    return out


def compartments_by_label(result: Any) -> dict[str, Any]:
    """summer3 labels: stratum values joined in stratification order."""
    data = _data(result["compartments"])
    order = ("disease_state", "age", "infectious", "clinical")
    labelled: dict[str, Any] = {}
    for row, traits in enumerate(PMAP.to_dicts()):
        name = "_".join(traits[key] for key in order if key in traits)
        labelled[name] = data[:, row]
    return labelled


def _sum_states_by_age(result: Any, states: tuple[str, ...]) -> dict[str, Any]:
    data = _data(result["compartments"])
    out = {age: jnp.zeros(data.shape[0]) for age in AGES}
    for row, traits in enumerate(PMAP.to_dicts()):
        if traits["disease_state"] not in states:
            continue
        out[traits["age"]] = out[traits["age"]] + data[:, row]
    return out


def age_population(result: Any) -> dict[str, Any]:
    """Total population by modelled age."""
    return _sum_states_by_age(result, DISEASE_STATES)


def age_prevalence(result: Any) -> dict[str, Any]:
    """``active`` plus ``treatment``, by age."""
    return _sum_states_by_age(result, PREV_STATES)


def age_latent(result: Any) -> dict[str, Any]:
    """Everyone not ``mtb_naive``, by age."""
    return _sum_states_by_age(result, INFECTED_STATES)


def age_pulmonary_prevalence(result: Any) -> dict[str, Any]:
    """High infectious, plus ``prop_lowinf_bactpos`` of low, plus on treatment.

    The original multiplies by ``BASE_PARAMS["prop_lowinf_bactpos"]``, not the
    run's parameter. None of the parity fixtures change that value.
    """
    data = _data(result["compartments"])
    high = {age: jnp.zeros(data.shape[0]) for age in AGES}
    low = {age: jnp.zeros(data.shape[0]) for age in AGES}
    on_rx = {age: jnp.zeros(data.shape[0]) for age in AGES}
    frac = BASE_PARAMS["prop_lowinf_bactpos"]
    for row, traits in enumerate(PMAP.to_dicts()):
        age = traits["age"]
        column = data[:, row]
        if traits.get("infectious") == "high":
            high[age] = high[age] + column
        elif traits.get("infectious") == "low":
            low[age] = low[age] + column
        if traits["disease_state"] == "treatment":
            on_rx[age] = on_rx[age] + column
    return {age: high[age] + low[age] * frac + on_rx[age] for age in AGES}


def age_incidence(result: Any) -> dict[str, Any]:
    """Progression into active disease, by source age."""
    return flow_by_age(result, "progression")


def age_notifications(result: Any) -> dict[str, Any]:
    """Routine detection plus ACF, by source age."""
    routine = flow_by_age(result, "detection")
    acf = flow_by_age(result, "acf")
    return {age: routine[age] + acf[age] for age in AGES}


def age_deaths(result: Any) -> dict[str, Any]:
    """Community TB deaths plus deaths on treatment, by source age."""
    community = flow_by_age(result, "tb_mortality")
    treated = flow_by_age(result, "rx_death")
    return {age: community[age] + treated[age] for age in AGES}


def disease_state_totals(result: Any) -> dict[str, Any]:
    """Compartment sums by disease state, for the notebook plot."""
    data = _data(result["compartments"])
    totals: dict[str, Any] = {}
    for row, traits in enumerate(PMAP.to_dicts()):
        state = traits["disease_state"]
        column = data[:, row]
        totals[state] = column if state not in totals else totals[state] + column
    return totals


INDICATORS: tuple[tuple[str, Any], ...] = (
    ("incidence", age_incidence),
    ("prevalence", age_prevalence),
    ("pulmonary", age_pulmonary_prevalence),
    ("notifications", age_notifications),
    ("deaths", age_deaths),
    ("population", age_population),
)
