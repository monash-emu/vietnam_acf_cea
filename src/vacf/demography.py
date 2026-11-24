"""Host-side demography helpers copied from ``tweak_cases_calib``.

The summer3 flow builders are not copied. Only the numeric prep the vector
field closes over.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from jax import numpy as jnp

from vacf.constants import AGE_STRATA, TOP_AGE_BRACKET_INFLATION


def inflate_oldest_death_rates(death_rates: pd.DataFrame) -> pd.DataFrame:
    """Multiply the open-ended oldest age group's hazard by two.

    The original does this before building death flows. See
    ``demography.inflate_oldest_death_rates`` on ``tweak_cases_calib``.
    """
    out = death_rates.copy()
    out[AGE_STRATA[-1]] *= TOP_AGE_BRACKET_INFLATION
    return out


def prepare_pop_data_for_entries(
    group_popsize: pd.DataFrame,
    start_time: float,
    start_pop: float,
) -> tuple[Any, Any]:
    """Year-to-year total-population increments, including the model start.

    The returned rates can be negative. summer3 applies them as an absolute
    entry, and summer4 ``EntryFlow`` does the same (no sign check).
    """
    total_pop_size = group_popsize.sum(axis=1)
    total_pop_size[start_time] = start_pop
    non_dec_data_w_start = total_pop_size.sort_index()
    pop_increments = non_dec_data_w_start.diff()
    time_increments = non_dec_data_w_start.index.diff()
    entry_birth_rates = (pop_increments / time_increments).dropna()
    times = jnp.array(entry_birth_rates.index)
    rates = jnp.array(entry_birth_rates)
    return times, rates


def birth_rate_at(model_time: Any, times: Any, rates: Any, start_time: float) -> Any:
    """Step lookup. ``model_time`` is the solver index, as in summer3."""
    time = model_time + start_time
    idx = jnp.clip(jnp.searchsorted(times, time), 0, rates.shape[0] - 1)
    return rates[idx]


def smooth_birth_rate_at(model_time: Any, times: Any, rates: Any, start_time: float) -> Any:
    """C0 alternative to :func:`birth_rate_at`. Default entry rate for ``run``.

    The step lookup jumps on every 1 January, and an adaptive solver rejects a
    step at each of them. With the spline mixing matrix this is what is left of
    the rejected-step budget. See ``plans/mixing-spline.md`` for the measurement.

    The rates are already a finite difference of the population series, so a
    linear blend between them is a smoother estimate of the same derivative,
    not a new assumption. ``match_golden=True`` keeps the step lookup so parity
    goldens stay on the summer3 vector field.
    """
    time = model_time + start_time
    return jnp.interp(time, times, rates, left=rates[0], right=rates[-1])


def frame_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Numeric body of a DataFrame, column order preserved."""
    return np.ascontiguousarray(frame.to_numpy(dtype=np.float64))
