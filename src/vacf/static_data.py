"""Arrays the vector field closes over. Loaded once, from the same CSVs as summer3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from jax import numpy as jnp

from vacf.constants import (
    AGE_STRATA,
    DATA_PATH,
    ISO3,
    POP_PER_AGE,
    START_TIME,
)
from vacf.curves import get_scale_data
from vacf.demography import (
    frame_matrix,
    inflate_oldest_death_rates,
    prepare_pop_data_for_entries,
)
from vacf.inputs import load_demography, load_fertility, load_who_outcomes

# Touch DATA_PATH so a missing checkout fails at import with a clear path.
assert DATA_PATH.is_dir(), DATA_PATH


@dataclass(frozen=True, slots=True)
class StaticInputs:
    """Host constants. ``t`` in the solver is still the index, not a year."""

    age_weights: Any
    weight_ends: Any
    pops: Any
    pop_ends: Any
    fert: Any
    fert_ends: Any
    death_times: Any
    death_rates: Any
    tsr_x: Any
    tsr_y: Any
    unsucc_x: Any
    unsucc_y: Any
    entry_times: Any
    entry_rates: Any


def _ends(index: Any) -> Any:
    values = np.asarray(index, dtype=np.float64)
    return jnp.asarray(values[[0, -1]])


def load_static() -> StaticInputs:
    """Build the arrays ``add_flows_to_model`` closes over."""
    group_popsize, death_rates, age_weights = load_demography(ISO3)
    fert = load_fertility(ISO3)
    tsr, death_unsucc, _who_mort = load_who_outcomes(ISO3)
    death_rates = inflate_oldest_death_rates(death_rates)

    if list(group_popsize.columns) != list(AGE_STRATA):
        raise ValueError(f"population columns {list(group_popsize.columns)} != {AGE_STRATA}")
    weight_cols = [int(c) for c in age_weights.columns]
    if weight_cols[0] != 0 or any(b - a != 1 for a, b in zip(weight_cols, weight_cols[1:])):
        raise ValueError("age-weight columns are not consecutive single years from 0")

    death_cols = [np.asarray(death_rates[age], dtype=np.float64) for age in AGE_STRATA]
    stacked = np.stack(death_cols, axis=1)
    start_pop = POP_PER_AGE * len(AGE_STRATA)
    entry_times, entry_rates = prepare_pop_data_for_entries(group_popsize, START_TIME, start_pop)

    tsr_x = np.asarray(tsr.index, dtype=np.float64)
    tsr_y = np.asarray(tsr.to_numpy(), dtype=np.float64)
    unsucc_x = np.asarray(death_unsucc.index, dtype=np.float64)
    unsucc_y = np.asarray(death_unsucc.to_numpy(), dtype=np.float64)
    return StaticInputs(
        age_weights=jnp.asarray(frame_matrix(age_weights)),
        weight_ends=_ends(age_weights.index),
        pops=jnp.asarray(frame_matrix(group_popsize)),
        pop_ends=_ends(group_popsize.index),
        fert=jnp.asarray(frame_matrix(fert)),
        fert_ends=_ends(fert.index),
        death_times=jnp.asarray(np.asarray(death_rates.index, dtype=np.float64)),
        death_rates=jnp.asarray(stacked),
        tsr_x=get_scale_data(jnp.asarray(tsr_x)),
        tsr_y=get_scale_data(jnp.asarray(tsr_y)),
        unsucc_x=get_scale_data(jnp.asarray(unsucc_x)),
        unsucc_y=get_scale_data(jnp.asarray(unsucc_y)),
        entry_times=entry_times,
        entry_rates=entry_rates,
    )
