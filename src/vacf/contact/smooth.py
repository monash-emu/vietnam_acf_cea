"""PCHIP contact matrix. Demographic tables are samples, not steps.

Annual weights, fertility and populations are interpolated, renormalised, and
aggregated at four knots per year. The step evaluates those curves and only
then divides by the spectral radius, so the radius stays 1 at fractional times.
``exact_matrix`` aggregates the interpolated inputs directly and is the
definition the stored curves are checked against.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from jax import numpy as jnp

from vacf.constants import AGE_STRATA, MAX_AGE
from vacf.contact.aggregate import (
    band_ids,
    components,
    fertility_pairs,
    pair_layout,
    renormalise_bands,
    renormalise_rows,
    unit_years,
    year_index_np,
)
from vacf.contact.pchip import eval_jax, eval_np, pchip_slopes
from vacf.mixing import normalise_by_spectral_radius


def _require(fixed: bool, passed: Any, name: str) -> Any:
    if fixed:
        if passed is not None:
            raise ValueError(f"{name} was fixed at prepare; do not pass it again")
        return 0.0
    if passed is None:
        raise ValueError(f"pass {name}, or fix it in prepare_smooth")
    return passed


def _on_integer(time: float) -> bool:
    return abs(time - round(time)) < 1e-8


@dataclass(frozen=True, slots=True)
class _Inputs:
    """PCHIP of the measured series, kept so ``exact_matrix`` can re-aggregate."""

    weight_x: np.ndarray
    weight_y: np.ndarray
    weight_m: np.ndarray
    fert_x: np.ndarray
    fert_y: np.ndarray
    fert_m: np.ndarray
    pop_x: np.ndarray
    pop_y: np.ndarray
    pop_m: np.ndarray
    fert_ends: np.ndarray
    band: np.ndarray
    gap: np.ndarray
    child: np.ndarray


@dataclass(frozen=True, slots=True)
class SmoothContact:
    """8×8 curves of the aggregated kernel, plus the input PCHIPs.

    Fixed scalars are folded into ``offset`` values and slopes. Hermite
    interpolation is linear, so folding ``a_spread`` by contracting the lag
    tensor's values and slopes matches dotting the interpolated tensor.
    """

    knot_x: Any
    offset_y: Any
    offset_m: Any
    background_y: Any
    background_m: Any
    parent_y: Any
    parent_m: Any
    pop_y: Any
    pop_m: Any
    lagged_y: Any
    lagged_m: Any
    lags: Any
    bg_fixed: bool
    a_fixed: bool
    pc_fixed: bool
    n_ages: int
    n_groups: int
    inputs: _Inputs

    def matrix(
        self,
        time: Any,
        bg_mixing: Any = None,
        a_spread: Any = None,
        pc_strength: Any = None,
    ) -> Any:
        """Normalised contact matrix. The eigenproblem runs at ``time``."""
        bg = _require(self.bg_fixed, bg_mixing, "bg_mixing")
        pc = _require(self.pc_fixed, pc_strength, "pc_strength")
        mixing = eval_jax(time, self.knot_x, self.offset_y, self.offset_m)
        if not self.bg_fixed:
            background = eval_jax(time, self.knot_x, self.background_y, self.background_m)
            mixing = mixing + bg * background
        if not self.pc_fixed:
            parent = eval_jax(time, self.knot_x, self.parent_y, self.parent_m)
            mixing = mixing + pc * parent
        if not self.a_fixed:
            if a_spread is None:
                raise ValueError("pass a_spread, or fix it in prepare_smooth")
            lagged = eval_jax(time, self.knot_x, self.lagged_y, self.lagged_m)
            kernel = jnp.exp(-self.lags / a_spread) / a_spread
            mixing = mixing + jnp.tensordot(lagged, kernel, axes=([2], [0]))
        elif a_spread is not None:
            raise ValueError("a_spread was fixed at prepare; do not pass it again")
        pop = eval_jax(time, self.knot_x, self.pop_y, self.pop_m)
        return normalise_by_spectral_radius(mixing * pop[None, :])

    def materialize(
        self,
        bg_mixing: Any = None,
        a_spread: Any = None,
        pc_strength: Any = None,
    ) -> SmoothContact:
        """Fold any scalars still free into the stored curves.

        After this, ``matrix(time)`` does not read the lag tensor. The radius
        is still computed at ``time``.
        """
        if self.bg_fixed and bg_mixing is not None:
            raise ValueError("bg_mixing was fixed at prepare; do not pass it again")
        if self.pc_fixed and pc_strength is not None:
            raise ValueError("pc_strength was fixed at prepare; do not pass it again")
        if self.a_fixed and a_spread is not None:
            raise ValueError("a_spread was fixed at prepare; do not pass it again")
        if bg_mixing is None and a_spread is None and pc_strength is None:
            if self.bg_fixed and self.a_fixed and self.pc_fixed:
                return self
            raise ValueError("pass the scalars that are still free")

        # Stay in JAX. A calibration folds the scalars inside its own jit, so
        # ``float(bg_mixing)`` here would make the fast path unusable under
        # ``jit`` and undifferentiable in the three scalars.
        offset_y: Any = jnp.asarray(self.offset_y)
        offset_m: Any = jnp.asarray(self.offset_m)
        background_y: Any = self.background_y
        background_m: Any = self.background_m
        parent_y: Any = self.parent_y
        parent_m: Any = self.parent_m
        lagged_y: Any = self.lagged_y
        lagged_m: Any = self.lagged_m
        bg_fixed = self.bg_fixed
        pc_fixed = self.pc_fixed
        a_fixed = self.a_fixed
        if bg_mixing is not None:
            offset_y = offset_y + bg_mixing * self.background_y
            offset_m = offset_m + bg_mixing * self.background_m
            background_y = None
            background_m = None
            bg_fixed = True
        if pc_strength is not None:
            offset_y = offset_y + pc_strength * self.parent_y
            offset_m = offset_m + pc_strength * self.parent_m
            parent_y = None
            parent_m = None
            pc_fixed = True
        if a_spread is not None:
            kernel = jnp.exp(-self.lags / a_spread) / a_spread
            offset_y = offset_y + jnp.tensordot(self.lagged_y, kernel, axes=([-1], [0]))
            offset_m = offset_m + jnp.tensordot(self.lagged_m, kernel, axes=([-1], [0]))
            lagged_y = None
            lagged_m = None
            a_fixed = True
        return replace(
            self,
            offset_y=offset_y,
            offset_m=offset_m,
            background_y=background_y,
            background_m=background_m,
            parent_y=parent_y,
            parent_m=parent_m,
            lagged_y=lagged_y,
            lagged_m=lagged_m,
            bg_fixed=bg_fixed,
            a_fixed=a_fixed,
            pc_fixed=pc_fixed,
        )

    def exact_matrix(
        self,
        time: float,
        bg_mixing: float,
        a_spread: float,
        pc_strength: float,
    ) -> Any:
        """Aggregate interpolated inputs at ``time``, then normalise.

        This is the definition. ``matrix`` interpolates curves that were
        sampled from it. At an integer measurement year the two agree with
        ``get_norm_c_matrix``; between years this is the reference for the
        stored-knot error.
        """
        t = float(time)
        weights = renormalise_bands(
            eval_np(t, self.inputs.weight_x, self.inputs.weight_y, self.inputs.weight_m),
            self.inputs.band,
            self.n_groups,
        )
        births = t - np.arange(self.n_ages, dtype=np.float64)
        rows = renormalise_rows(
            eval_np(births, self.inputs.fert_x, self.inputs.fert_y, self.inputs.fert_m)
        )
        fertility = rows[self.inputs.child, self.inputs.gap]
        pop = eval_np(t, self.inputs.pop_x, self.inputs.pop_y, self.inputs.pop_m)
        background, parent, lagged = components(
            weights, fertility, self.inputs.band, self.inputs.gap, self.n_groups
        )
        kernel = np.exp(-np.arange(self.n_ages) / a_spread) / a_spread
        mixing = (
            bg_mixing * background
            + np.tensordot(lagged, kernel, axes=([-1], [0]))
            + pc_strength * parent
        )
        scaled = mixing * np.asarray(pop, dtype=np.float64)[None, :]
        return normalise_by_spectral_radius(jnp.asarray(scaled))


def _measured_components(
    time: float,
    weight_np: np.ndarray,
    weight_ends: Any,
    pop_np: np.ndarray,
    pop_ends: Any,
    fert_np: np.ndarray,
    fert_ends: Any,
    child: np.ndarray,
    gap: np.ndarray,
    band: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w_index = int(year_index_np(weight_ends, time))
    p_index = int(year_index_np(pop_ends, time))
    fertility = fertility_pairs(fert_np, fert_ends, time, child, gap)
    background, parent, lagged = components(weight_np[w_index], fertility, band, gap, n_groups)
    return background, parent, lagged, pop_np[p_index]


def prepare_smooth(
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
    knots_per_year: int = 4,
) -> SmoothContact:
    """Fit PCHIPs and sample the aggregated kernel.

    Scalars passed here are fixed, as in :func:`prepare_stepwise`.
    ``knots_per_year`` is how often the aggregated curves are sampled. Four
    is the default; raise it if ``exact_matrix`` disagrees by more than
    ``1e-3`` relative.
    """
    if knots_per_year < 1:
        raise ValueError("knots_per_year must be at least 1")
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

    weight_x = unit_years(weight_ends, weight_np.shape[0], "weights")
    fert_x = unit_years(fert_ends, fert_np.shape[0], "fertility")
    pop_x = unit_years(pop_ends, pop_np.shape[0], "pops")
    band = band_ids(breaks, last_age)
    _, gap, child = pair_layout(last_age)
    inputs = _Inputs(
        weight_x=weight_x,
        weight_y=weight_np,
        weight_m=pchip_slopes(weight_x, weight_np),
        fert_x=fert_x,
        fert_y=fert_np,
        fert_m=pchip_slopes(fert_x, fert_np),
        pop_x=pop_x,
        pop_y=pop_np,
        pop_m=pchip_slopes(pop_x, pop_np),
        fert_ends=np.asarray(fert_ends, dtype=np.float64).reshape(-1),
        band=band,
        gap=gap,
        child=child,
    )

    n_intervals = int(round(weight_x[-1] - weight_x[0]))
    knot_x = np.linspace(weight_x[0], weight_x[-1], n_intervals * knots_per_year + 1)
    n_knots = knot_x.shape[0]
    background = np.zeros((n_knots, n_groups, n_groups), dtype=np.float64)
    parent = np.zeros((n_knots, n_groups, n_groups), dtype=np.float64)
    lagged = np.zeros((n_knots, n_groups, n_groups, n_ages), dtype=np.float64)
    pop = np.zeros((n_knots, n_groups), dtype=np.float64)
    for i, time in enumerate(knot_x):
        t = float(time)
        if _on_integer(t):
            background[i], parent[i], lagged[i], pop[i] = _measured_components(
                t,
                weight_np,
                weight_ends,
                pop_np,
                pop_ends,
                fert_np,
                fert_ends,
                child,
                gap,
                band,
                n_groups,
            )
        else:
            weights_t = renormalise_bands(
                eval_np(t, inputs.weight_x, inputs.weight_y, inputs.weight_m),
                band,
                n_groups,
            )
            births = t - np.arange(n_ages, dtype=np.float64)
            rows = renormalise_rows(eval_np(births, inputs.fert_x, inputs.fert_y, inputs.fert_m))
            fertility = rows[child, gap]
            background[i], parent[i], lagged[i] = components(
                weights_t, fertility, band, gap, n_groups
            )
            pop[i] = eval_np(t, inputs.pop_x, inputs.pop_y, inputs.pop_m)

    background_m = pchip_slopes(knot_x, background)
    parent_m = pchip_slopes(knot_x, parent)
    lagged_m = pchip_slopes(knot_x, lagged)
    pop_m = pchip_slopes(knot_x, pop)
    offset_y = np.zeros_like(background)
    offset_m = np.zeros_like(background_m)
    bg_fixed = bg_mixing is not None
    pc_fixed = pc_strength is not None
    a_fixed = a_spread is not None
    background_y: Any = background
    parent_y: Any = parent
    lagged_y: Any = lagged
    if bg_fixed:
        offset_y = offset_y + float(bg_mixing) * background
        offset_m = offset_m + float(bg_mixing) * background_m
        background_y = None
        background_m = None
    if pc_fixed:
        offset_y = offset_y + float(pc_strength) * parent
        offset_m = offset_m + float(pc_strength) * parent_m
        parent_y = None
        parent_m = None
    if a_fixed:
        kernel = np.exp(-np.arange(n_ages) / float(a_spread)) / float(a_spread)
        offset_y = offset_y + np.tensordot(lagged, kernel, axes=([-1], [0]))
        offset_m = offset_m + np.tensordot(lagged_m, kernel, axes=([-1], [0]))
        lagged_y = None
        lagged_m = None

    def _jax(values: Any) -> Any:
        if values is None:
            return None
        return jnp.asarray(values)

    return SmoothContact(
        knot_x=jnp.asarray(knot_x),
        offset_y=jnp.asarray(offset_y),
        offset_m=jnp.asarray(offset_m),
        background_y=_jax(background_y),
        background_m=_jax(background_m),
        parent_y=_jax(parent_y),
        parent_m=_jax(parent_m),
        pop_y=jnp.asarray(pop),
        pop_m=jnp.asarray(pop_m),
        lagged_y=_jax(lagged_y),
        lagged_m=_jax(lagged_m),
        lags=jnp.arange(n_ages, dtype=jnp.float64),
        bg_fixed=bg_fixed,
        a_fixed=a_fixed,
        pc_fixed=pc_fixed,
        n_ages=n_ages,
        n_groups=n_groups,
        inputs=inputs,
    )
