"""Year-frozen contact matrix. Same values as ``get_norm_c_matrix``.

Demographic pieces are built once per integer year. ``materialize`` folds the
mixing scalars and the spectral radius into an ``(n_years, n_groups, n_groups)``
table. ``matrix`` on that result is an index, with the same clip-then-truncate
rule as ``get_year_index``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from jax import numpy as jnp, vmap

from vacf.constants import AGE_STRATA, MAX_AGE
from vacf.contact.aggregate import (
    band_ids,
    components,
    fertility_pairs,
    pair_layout,
    unit_years,
    year_index_np,
)
from vacf.mixing import normalise_by_spectral_radius


def year_index_jax(time: Any, start: float, stop: float) -> Any:
    """Clip ``time`` to ``[start, stop]`` and truncate toward zero."""
    clamped = jnp.clip(jnp.asarray(time, dtype=jnp.float64), start, stop)
    return (clamped - start).astype(jnp.int32)


def _combine_lag(
    offset: Any,
    background: Any,
    parent: Any,
    pop: Any,
    lagged: Any,
    bg: Any,
    pc: Any,
    a: Any,
    lags: Any,
) -> Any:
    kernel = jnp.exp(-lags / a) / a
    mixing = offset + bg * background + pc * parent
    mixing = mixing + jnp.tensordot(lagged, kernel, axes=([2], [0]))
    return normalise_by_spectral_radius(mixing * pop[None, :])


def _combine_fixed_a(
    offset: Any,
    background: Any,
    parent: Any,
    pop: Any,
    bg: Any,
    pc: Any,
) -> Any:
    mixing = offset + bg * background + pc * parent
    return normalise_by_spectral_radius(mixing * pop[None, :])


def _require(fixed: bool, passed: Any, name: str) -> Any:
    if fixed:
        if passed is not None:
            raise ValueError(f"{name} was fixed at prepare; do not pass it again")
        return 0.0
    if passed is None:
        raise ValueError(f"pass {name}, or fix it in prepare_stepwise")
    return passed


@dataclass(frozen=True, slots=True)
class StepwiseContact:
    """Integer-year components, and the normalised table once scalars are folded.

    ``bg_fixed`` means that scalar already sits in ``offset`` and the matching
    array (``background`` or ``parent``) is zeros. ``a_fixed`` means the lag
    tensor has been contracted and ``lagged`` is ``None``.
    """

    year0: float
    year1: float
    offset: Any
    background: Any
    parent: Any
    pop: Any
    lagged: Any
    lags: Any
    bg_fixed: bool
    a_fixed: bool
    pc_fixed: bool
    table: Any = None

    def matrix(
        self,
        time: Any,
        bg_mixing: Any = None,
        a_spread: Any = None,
        pc_strength: Any = None,
    ) -> Any:
        """Normalised contact matrix at ``time``.

        On a materialised contact this indexes the table. Otherwise it builds
        one year, which is the path ``jax.grad`` can see the scalars through.
        """
        index = year_index_jax(time, self.year0, self.year1)
        if (
            self.table is not None
            and bg_mixing is None
            and a_spread is None
            and pc_strength is None
        ):
            return self.table[index]
        bg = _require(self.bg_fixed, bg_mixing, "bg_mixing")
        pc = _require(self.pc_fixed, pc_strength, "pc_strength")
        if self.a_fixed:
            if a_spread is not None:
                raise ValueError("a_spread was fixed at prepare; do not pass it again")
            return _combine_fixed_a(
                self.offset[index],
                self.background[index],
                self.parent[index],
                self.pop[index],
                bg,
                pc,
            )
        if a_spread is None:
            raise ValueError("pass a_spread, or fix it in prepare_stepwise")
        return _combine_lag(
            self.offset[index],
            self.background[index],
            self.parent[index],
            self.pop[index],
            self.lagged[index],
            bg,
            pc,
            a_spread,
            self.lags,
        )

    def materialize(
        self,
        bg_mixing: Any = None,
        a_spread: Any = None,
        pc_strength: Any = None,
    ) -> StepwiseContact:
        """Fold scalars into a normalised year table.

        The returned contact's ``matrix(time)`` only indexes that table.
        ``table`` has shape ``(n_years, n_groups, n_groups)``.
        """
        if (
            self.table is not None
            and bg_mixing is None
            and a_spread is None
            and pc_strength is None
        ):
            return self
        bg = _require(self.bg_fixed, bg_mixing, "bg_mixing")
        pc = _require(self.pc_fixed, pc_strength, "pc_strength")
        if self.a_fixed:
            if a_spread is not None:
                raise ValueError("a_spread was fixed at prepare; do not pass it again")
            table = vmap(_combine_fixed_a, in_axes=(0, 0, 0, 0, None, None))(
                self.offset, self.background, self.parent, self.pop, bg, pc
            )
        else:
            if a_spread is None:
                raise ValueError("pass a_spread, or fix it in prepare_stepwise")
            table = vmap(_combine_lag, in_axes=(0, 0, 0, 0, 0, None, None, None, None))(
                self.offset,
                self.background,
                self.parent,
                self.pop,
                self.lagged,
                bg,
                pc,
                a_spread,
                self.lags,
            )
        return replace(
            self,
            table=table,
            bg_fixed=True,
            a_fixed=True,
            pc_fixed=True,
        )


def prepare_stepwise(
    weights: Any,
    weight_ends: Any,
    pops: Any,
    pop_ends: Any,
    fert: Any,
    fert_ends: Any,
    *,
    bg_mixing: float | None = None,
    a_spread: float | None = None,
    pc_strength: float | None = None,
    age_strata: list[int] | None = None,
    max_age: int | None = None,
    years: Any = None,
) -> StepwiseContact:
    """Build one record per integer year.

    The default years are the weight table. Pass ``years`` as consecutive unit
    steps to cover a longer span: each series is still clipped with
    ``year_index``, so a year before the weight table uses the first weight
    row and that year's fertility.

    A scalar passed here is fixed: it is folded into ``offset`` and is not a
    tracer. Leave it as ``None`` to pass it to ``matrix`` or ``materialize``.
    When all three are fixed the normalised table is built immediately.
    """
    breaks = list(AGE_STRATA if age_strata is None else age_strata)
    last_age = MAX_AGE if max_age is None else max_age
    n_ages = last_age + 1
    n_groups = len(breaks)
    weight_np = np.asarray(weights, dtype=np.float64)
    pop_np = np.asarray(pops, dtype=np.float64)
    fert_np = np.asarray(fert, dtype=np.float64)
    if weight_np.ndim != 2 or weight_np.shape[1] != n_ages:
        raise ValueError(f"weights must be (n_years, {n_ages})")
    if fert_np.shape[1] != n_ages:
        raise ValueError(f"fertility must have {n_ages} age columns")
    if pop_np.ndim != 2 or pop_np.shape[1] != n_groups:
        raise ValueError(f"pops must be (n_years, {n_groups})")

    if years is None:
        years = unit_years(weight_ends, weight_np.shape[0], "weights")
    else:
        years = np.asarray(years, dtype=np.float64)
        if years.ndim != 1 or years.size < 2 or not np.allclose(np.diff(years), 1.0):
            raise ValueError("years must be consecutive unit steps")
    band = band_ids(breaks, last_age)
    _, gap, child = pair_layout(last_age)
    n_years = years.shape[0]
    background = np.zeros((n_years, n_groups, n_groups), dtype=np.float64)
    parent = np.zeros((n_years, n_groups, n_groups), dtype=np.float64)
    lagged = np.zeros((n_years, n_groups, n_groups, n_ages), dtype=np.float64)
    pop = np.zeros((n_years, n_groups), dtype=np.float64)
    for i, time in enumerate(years):
        w_index = int(year_index_np(weight_ends, time))
        p_index = int(year_index_np(pop_ends, time))
        fertility = fertility_pairs(fert_np, fert_ends, float(time), child, gap)
        background[i], parent[i], lagged[i] = components(
            weight_np[w_index], fertility, band, gap, n_groups
        )
        pop[i] = pop_np[p_index]

    offset = np.zeros((n_years, n_groups, n_groups), dtype=np.float64)
    bg_fixed = bg_mixing is not None
    pc_fixed = pc_strength is not None
    a_fixed = a_spread is not None
    if bg_fixed:
        offset = offset + float(bg_mixing) * background
        background = np.zeros_like(background)
    if pc_fixed:
        offset = offset + float(pc_strength) * parent
        parent = np.zeros_like(parent)
    lagged_jax: Any = None
    if a_fixed:
        kernel = np.exp(-np.arange(n_ages, dtype=np.float64) / float(a_spread)) / float(a_spread)
        offset = offset + np.tensordot(lagged, kernel, axes=([-1], [0]))
    else:
        lagged_jax = jnp.asarray(lagged)

    contact = StepwiseContact(
        year0=float(years[0]),
        year1=float(years[-1]),
        offset=jnp.asarray(offset),
        background=jnp.asarray(background),
        parent=jnp.asarray(parent),
        pop=jnp.asarray(pop),
        lagged=lagged_jax,
        lags=jnp.arange(n_ages, dtype=jnp.float64),
        bg_fixed=bg_fixed,
        a_fixed=a_fixed,
        pc_fixed=pc_fixed,
    )
    if bg_fixed and a_fixed and pc_fixed:
        return contact.materialize()
    return contact
