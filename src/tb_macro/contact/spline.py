"""Contact matrix as a cubic curve through the normalised annual matrices.

``prepare_stepwise`` is the baseline's own vector field: it jumps on every
1 January, so an adaptive stepper rejects a step at each of them. ``prepare_smooth``
removes the jump but keeps an 8x8 eigenproblem inside the step.

This builder does neither. The demographic tables are annual, so the matrix is
built once per calendar year and normalised there, and the *normalised* matrices
are joined by a Catmull-Rom cubic. A solver stage is then a gather and a Horner
cubic: no kernel, no projection, no eigenproblem.

The design note says not to interpolate a normalised matrix, because a blend of
two matrices of spectral radius 1 does not have spectral radius 1. That is true,
and here it is worth 3e-5. It is removed anyway: the dominant left and right
vectors are stored as curves of their own, and the stage divides by the Rayleigh
quotient of the interpolated matrix. The radius is then 1 to 1e-9 at any time,
and the correction costs two 8x8 matrix-vector products.

What is given up against ``prepare_smooth`` is that the curve runs through the
aggregate of the measured years rather than through the aggregate of interpolated
inputs. Those two differ by about 1e-3 on entries of size 1e-1. Neither is more
authoritative: the underlying series are annual, and nothing in the data says what
happens in July.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from jax import numpy as jnp, tree_util

from tb_macro.constants import AGE_STRATA, MAX_AGE
from tb_macro.contact.aggregate import (
    band_ids,
    components,
    fertility_pairs,
    pair_layout,
    unit_years,
    year_index_np,
)
from tb_macro.contact.perron import DEFAULT_SQUARINGS, perron_pair, rayleigh_radius


def catmull_slopes(values: Any) -> Any:
    """Central-difference slopes on unit knots, one-sided at the two ends.

    Linear in ``values``, so a cotangent on a mixing scalar passes through
    unchanged and the fit does not need its own derivative rule. Unlike PCHIP
    there is no limiter, which a normalised contact matrix does not need: every
    entry is bounded away from zero while ``bg_mixing`` is positive.
    """
    n_knots = values.shape[0]
    if n_knots < 2:
        raise ValueError(f"a spline needs at least two knots, got {n_knots}")
    index = np.arange(n_knots)
    above = np.minimum(index + 1, n_knots - 1)
    below = np.maximum(index - 1, 0)
    span = (above - below).astype(np.float64).reshape((-1,) + (1,) * (values.ndim - 1))
    return (values[above] - values[below]) / span


def _neighbours(index: Any, n_knots: int) -> tuple[Any, Any, Any, Any]:
    """The four knots a Catmull-Rom segment on ``[index, index + 1]`` reads."""
    top = n_knots - 1
    return (
        jnp.clip(index - 1, 0, top),
        index,
        jnp.clip(index + 1, 0, top),
        jnp.clip(index + 2, 0, top),
    )


def _segment(time: Any, year0: float, n_knots: int) -> tuple[Any, Any]:
    """Left knot index and the fraction into that unit segment.

    Outside the knots the fraction saturates, so the curve is flat there and its
    time derivative is zero. That is the clip ``get_year_index`` already applies.
    """
    offset = jnp.clip(jnp.asarray(time, dtype=jnp.float64) - year0, 0.0, float(n_knots - 1))
    index = jnp.clip(jnp.floor(offset).astype(jnp.int32), 0, n_knots - 2)
    return index, offset - index


def _hermite(frac: Any, y0: Any, y1: Any, m0: Any, m1: Any) -> Any:
    extra = (1,) * (y0.ndim - jnp.ndim(frac))
    s = jnp.reshape(frac, jnp.shape(frac) + extra)
    s2 = s * s
    s3 = s2 * s
    return (
        (2.0 * s3 - 3.0 * s2 + 1.0) * y0
        + (s3 - 2.0 * s2 + s) * m0
        + (-2.0 * s3 + 3.0 * s2) * y1
        + (s3 - s2) * m1
    )


def evaluate(time: Any, year0: float, values: Any, slopes: Any) -> Any:
    """Cubic through ``values`` with ``slopes``, on unit knots from ``year0``."""
    index, frac = _segment(time, year0, values.shape[0])
    return _hermite(frac, values[index], values[index + 1], slopes[index], slopes[index + 1])


@dataclass(frozen=True, slots=True)
class SplineCurve:
    """A normalised matrix curve, and the Perron curves that keep its radius at 1."""

    year0: float
    value: Any
    slope: Any
    left: Any
    left_slope: Any
    right: Any
    right_slope: Any

    def at(self, time: Any) -> Any:
        """Normalised contact matrix at ``time``. Two gathers and a cubic."""
        matrix = evaluate(time, self.year0, self.value, self.slope)
        if self.left is None:
            return matrix
        left = evaluate(time, self.year0, self.left, self.left_slope)
        right = evaluate(time, self.year0, self.right, self.right_slope)
        return matrix / rayleigh_radius(matrix, left, right)


# A curve is the thing a calibration passes from the draw into the solve, so it
# has to survive ``jit`` boundaries as one argument. ``year0`` is static: a new
# start year is a different table and should recompile.
tree_util.register_dataclass(
    SplineCurve,
    data_fields=["value", "slope", "left", "left_slope", "right", "right_slope"],
    meta_fields=["year0"],
)


def _fit(year0: float, matrices: Any, pin_radius: bool, squarings: int) -> SplineCurve:
    """Normalise each knot, then fit the curve. Everything here is traceable."""
    left, right = perron_pair(matrices, squarings)
    radius = rayleigh_radius(matrices, left, right)
    value = matrices / radius[..., None, None]
    if not pin_radius:
        return SplineCurve(year0, value, catmull_slopes(value), None, None, None, None)
    return SplineCurve(
        year0=year0,
        value=value,
        slope=catmull_slopes(value),
        left=left,
        left_slope=catmull_slopes(left),
        right=right,
        right_slope=catmull_slopes(right),
    )


def _require(fixed: bool, passed: Any, name: str) -> Any:
    if fixed:
        if passed is not None:
            raise ValueError(f"{name} was fixed at prepare; do not pass it again")
        return 0.0
    if passed is None:
        raise ValueError(f"pass {name}, or fix it in prepare_spline")
    return passed


@dataclass(frozen=True, slots=True)
class SplineContact:
    """Annual components, and the fitted curve once the mixing scalars are known.

    ``bg_fixed`` means ``bg_mixing`` already sits in ``offset`` and ``background``
    is zeros; ``a_fixed`` means the lag tensor has been contracted and ``lagged``
    is ``None``. Those are the pieces that can be folded once per calibration.
    ``materialize`` folds what is left, once per parameter set, and is where the
    Perron problem lives.
    """

    year0: float
    offset: Any
    background: Any
    parent: Any
    pop: Any
    lagged: Any
    lags: Any
    bg_fixed: bool
    a_fixed: bool
    pc_fixed: bool
    pin_radius: bool
    squarings: int
    curve: SplineCurve | None = None

    @property
    def n_knots(self) -> int:
        """One knot per calendar year."""
        return int(self.offset.shape[0])

    def _raw(self, bg: Any, pc: Any, a_spread: Any, index: Any = None) -> Any:
        """Unnormalised population-scaled matrices, for every knot or a few."""
        take = (lambda arr: arr) if index is None else (lambda arr: arr[index])
        mixing = take(self.offset) + bg * take(self.background) + pc * take(self.parent)
        if not self.a_fixed:
            kernel = jnp.exp(-self.lags / a_spread) / a_spread
            mixing = mixing + jnp.tensordot(take(self.lagged), kernel, axes=([-1], [0]))
        return mixing * take(self.pop)[..., None, :]

    def knot_matrices(
        self,
        bg_mixing: Any = None,
        a_spread: Any = None,
        pc_strength: Any = None,
    ) -> Any:
        """Population-scaled matrices at every knot, before normalisation.

        ``(n_knots, n_groups, n_groups)``. This is what :meth:`materialize`
        divides by the spectral radius, exposed so the radius itself can be
        checked against another solver.
        """
        bg = _require(self.bg_fixed, bg_mixing, "bg_mixing")
        pc = _require(self.pc_fixed, pc_strength, "pc_strength")
        if self.a_fixed and a_spread is not None:
            raise ValueError("a_spread was fixed at prepare; do not pass it again")
        if not self.a_fixed and a_spread is None:
            raise ValueError("pass a_spread, or fix it in prepare_spline")
        return self._raw(bg, pc, a_spread)

    def materialize(
        self,
        bg_mixing: Any = None,
        a_spread: Any = None,
        pc_strength: Any = None,
    ) -> SplineContact:
        """Fold the scalars and fit the curve. Once per parameter set.

        The result's ``matrix(time)`` is a gather and a cubic. ``bg_mixing``,
        ``a_spread`` and ``pc_strength`` may be tracers: the fit is
        differentiable in all three.
        """
        free = (bg_mixing, a_spread, pc_strength)
        if self.curve is not None and all(value is None for value in free):
            return self
        bg = _require(self.bg_fixed, bg_mixing, "bg_mixing")
        pc = _require(self.pc_fixed, pc_strength, "pc_strength")
        if self.a_fixed and a_spread is not None:
            raise ValueError("a_spread was fixed at prepare; do not pass it again")
        if not self.a_fixed and a_spread is None:
            raise ValueError("pass a_spread, or fix it in prepare_spline")
        curve = _fit(
            self.year0,
            self._raw(bg, pc, a_spread),
            self.pin_radius,
            self.squarings,
        )
        return replace(self, curve=curve)

    def matrix(
        self,
        time: Any,
        bg_mixing: Any = None,
        a_spread: Any = None,
        pc_strength: Any = None,
    ) -> Any:
        """Normalised contact matrix at ``time``.

        On a materialised contact this is the curve. Otherwise it fits the four
        knots the segment reads, which is the same function at a much higher
        per-call cost. Prefer :meth:`materialize` once per parameter set.
        """
        free = (bg_mixing, a_spread, pc_strength)
        if self.curve is not None and all(value is None for value in free):
            return self.curve.at(time)
        bg = _require(self.bg_fixed, bg_mixing, "bg_mixing")
        pc = _require(self.pc_fixed, pc_strength, "pc_strength")
        if self.a_fixed and a_spread is not None:
            raise ValueError("a_spread was fixed at prepare; do not pass it again")
        if not self.a_fixed and a_spread is None:
            raise ValueError("pass a_spread, or fix it in prepare_spline")
        return self._local(time, bg, pc, a_spread)

    def _local(self, time: Any, bg: Any, pc: Any, a_spread: Any) -> Any:
        """Catmull-Rom is local: one segment reads four knots and nothing else."""
        n_knots = self.n_knots
        index, frac = _segment(time, self.year0, n_knots)
        picks = jnp.stack(_neighbours(index, n_knots))
        raw = self._raw(bg, pc, a_spread, picks)
        left, right = perron_pair(raw, self.squarings)
        value = raw / rayleigh_radius(raw, left, right)[..., None, None]
        span = jnp.asarray(picks[2] - picks[0], dtype=jnp.float64)
        later = jnp.asarray(picks[3] - picks[1], dtype=jnp.float64)
        matrix = _hermite(
            frac,
            value[1],
            value[2],
            (value[2] - value[0]) / span,
            (value[3] - value[1]) / later,
        )
        if not self.pin_radius:
            return matrix
        vec_left = _hermite(
            frac, left[1], left[2], (left[2] - left[0]) / span, (left[3] - left[1]) / later
        )
        vec_right = _hermite(
            frac, right[1], right[2], (right[2] - right[0]) / span, (right[3] - right[1]) / later
        )
        return matrix / rayleigh_radius(matrix, vec_left, vec_right)


def prepare_spline(
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
    pin_radius: bool = True,
    squarings: int = DEFAULT_SQUARINGS,
) -> SplineContact:
    """Build one knot per calendar year. Once per process.

    The default years are the weight table. Pass ``years`` as consecutive unit
    steps to cover a longer span; each series is clipped the way
    ``get_year_index`` clips it, so a year before the weight table uses the first
    weight row and that year's fertility.

    A scalar passed here is fixed for the whole calibration: it is folded into
    ``offset`` and stops being a tracer. Leave it ``None`` to fold it per
    parameter set in :meth:`SplineContact.materialize`. When all three are fixed
    the curve is fitted immediately.

    ``pin_radius`` keeps the spectral radius at 1 between knots at the cost of
    two matrix-vector products per stage. With it off the radius drifts by about
    3e-5 within a year.
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
    n_knots = years.shape[0]
    background = np.zeros((n_knots, n_groups, n_groups), dtype=np.float64)
    parent = np.zeros((n_knots, n_groups, n_groups), dtype=np.float64)
    lagged = np.zeros((n_knots, n_groups, n_groups, n_ages), dtype=np.float64)
    pop = np.zeros((n_knots, n_groups), dtype=np.float64)
    for i, time in enumerate(years):
        w_index = int(year_index_np(weight_ends, time))
        fertility = fertility_pairs(fert_np, fert_ends, float(time), child, gap)
        background[i], parent[i], lagged[i] = components(
            weight_np[w_index], fertility, band, gap, n_groups
        )
        pop[i] = pop_np[int(year_index_np(pop_ends, time))]

    offset = np.zeros((n_knots, n_groups, n_groups), dtype=np.float64)
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

    contact = SplineContact(
        year0=float(years[0]),
        offset=jnp.asarray(offset),
        background=jnp.asarray(background),
        parent=jnp.asarray(parent),
        pop=jnp.asarray(pop),
        lagged=lagged_jax,
        lags=jnp.arange(n_ages, dtype=jnp.float64),
        bg_fixed=bg_fixed,
        a_fixed=a_fixed,
        pc_fixed=pc_fixed,
        pin_radius=pin_radius,
        squarings=squarings,
    )
    if bg_fixed and a_fixed and pc_fixed:
        return contact.materialize()
    return contact
