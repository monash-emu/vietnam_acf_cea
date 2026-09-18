"""Queries that match the original notebook's infection target."""

from __future__ import annotations

from typing import Any

from jax import numpy as jnp

from tb_macro.constants import AGES, TARGET_SLICE
from tb_macro.model import PMAP, get_compiled


def _data(trace: Any) -> Any:
    values = trace.values
    return values.data if hasattr(values, "data") else values


def flow_total(result: Any, name: str) -> Any:
    """Sum a flow's per-edge instantaneous mass over edges. Shape ``(time,)``."""
    return jnp.sum(_data(result[name]), axis=-1)


def flow_by_age(result: Any, name: str) -> dict[str, Any]:
    """Sum flow mass by the source compartment's age."""
    compiled = get_compiled()
    src = compiled.flows[name].src_idx
    data = _data(result[name])
    out: dict[str, Any] = {}
    for age in AGES:
        rows = jnp.asarray(
            [i for i, comp in enumerate(src) if PMAP.to_dicts()[int(comp)]["age"] == age],
            dtype=jnp.int32,
        )
        out[age] = jnp.sum(data[:, rows], axis=-1) if rows.size else jnp.zeros(data.shape[0])
    return out


def compartments_by_label(result: Any) -> dict[str, Any]:
    """Map summer3 compartment strings to column vectors."""
    data = _data(result["compartments"])
    order = ("disease_state", "age", "infectious", "clinical")
    labelled: dict[str, Any] = {}
    for row, traits in enumerate(PMAP.to_dicts()):
        name = "_".join(traits[key] for key in order if key in traits)
        labelled[name] = data[:, row]
    return labelled


def infection_target(result: Any) -> Any:
    """Rolling 7-year sum of ``infect_mtb_naive``, at the original slice."""
    rolled = result["infect_mtb_naive"].rolling(7, how="sum")
    total = jnp.sum(_data(rolled), axis=-1)
    return total[TARGET_SLICE]


def disease_state_totals(result: Any) -> dict[str, Any]:
    """Compartment sums by disease state, for the notebook plot."""
    data = _data(result["compartments"])
    totals: dict[str, Any] = {}
    for row, traits in enumerate(PMAP.to_dicts()):
        state = traits["disease_state"]
        column = data[:, row]
        totals[state] = column if state not in totals else totals[state] + column
    return totals
