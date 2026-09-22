"""Calendar curves copied from ``tweak_cases_calib`` ``utils.py``.

The triangular seed is a rate tree in ``model._seed_peak_rate`` (``clip`` /
``abs``). Detection and the WHO outcome curves still call
:func:`get_cos_multicurve` via ``defer`` — there is no piecewise half-cosine
rate node. ``jnp.maximum(..., 0)`` replaces ``jnp.clip(..., a_min=)``; JAX 0.11
dropped the keyword.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any

from jax import lax, numpy as jnp

InterpolatorScaleData = namedtuple("InterpolatorScaleData", ["points", "ranges", "bounds"])


def get_triang_vals(time: Any, peak_time: Any, peak_height: Any, width: Any) -> Any:
    """Triangular pulse, zero outside ``peak_time ± width``."""
    time_from_peak = jnp.absolute(time - peak_time)
    return jnp.maximum(peak_height * (1.0 - time_from_peak / width), 0.0)


def binary_search_sum_ge(x: Any, points: Any) -> Any:
    """``(x >= points).sum()`` via binary search. Copied from the original.

    ``jnp.searchsorted(points, x, side="right")`` is the same integer and
    reads better, but it was measured and is not faster here. Two of these
    per stage are about 2% of a solve, inside the run-to-run spread.
    """

    def cond(state: tuple[Any, Any]) -> Any:
        low, high = state
        return (high - low) > 1

    def body(state: tuple[Any, Any]) -> tuple[Any, Any]:
        low, high = state
        midpoint = (0.5 * (low + high)).astype(int)
        update_upper = x < points[midpoint]
        low = jnp.where(update_upper, low, midpoint)
        high = jnp.where(update_upper, midpoint, high)
        return (low, high)

    low, high = lax.while_loop(cond, body, (-1, len(points) - 1))
    return lax.cond(x < points[high], lambda: low, lambda: high) + 1


def _get_cos_curve_at_x(x: Any, x_data: Any, y_data: Any) -> Any:
    """Half-cosine blend on the segment containing ``x``."""
    idx = binary_search_sum_ge(x, x_data.points) - 1
    offset = x - x_data.points[idx]
    relx = offset / x_data.ranges[idx]
    rely = 0.5 + 0.5 * -jnp.cos(relx * jnp.pi)
    return y_data.points[idx] + (rely * y_data.ranges[idx])


def get_cos_multicurve(t: Any, x_data: Any, y_data: Any) -> Any:
    """Piecewise half-cosine through the knots, clamped outside them."""
    bounds_state = sum(t > x_data.bounds)
    branches = [
        lambda _, __, ___: y_data.bounds[0],
        _get_cos_curve_at_x,
        lambda _, __, ___: y_data.bounds[1],
    ]
    return lax.switch(bounds_state, branches, t, x_data, y_data)


def get_scale_data(points: Any) -> Any:
    """Precompute diffs and end values for :func:`get_cos_multicurve`."""
    ranges = jnp.diff(points)
    return InterpolatorScaleData(points, ranges, jnp.array([points[0], points[-1]]))
