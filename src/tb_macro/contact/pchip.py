"""Fritsch–Carlson cubic Hermite interpolation.

Slopes are fit on the host. The step evaluates the cubic with Horner so
``jax.grad`` sees a C1 function of time. Knot positions are data and carry no
tangent; knot values and slopes do, because ``SmoothContact.materialize``
folds the mixing scalars into them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from jax import custom_jvp, numpy as jnp


def pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch–Carlson slopes along the first axis of ``y``.

    Args:
        x: Knot positions, shape ``(n,)``, strictly increasing
        y: Values, shape ``(n, ...)``

    Returns:
        Slopes ``dy/dx`` with the same shape as ``y``
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_knots = x.shape[0]
    if n_knots < 2:
        raise ValueError(f"PCHIP needs at least two knots, got {n_knots}")
    if y.shape[0] != n_knots:
        raise ValueError(f"y has {y.shape[0]} knots, x has {n_knots}")
    widths = np.diff(x)
    if np.any(widths <= 0.0):
        raise ValueError("PCHIP knots must be strictly increasing")
    delta = np.diff(y, axis=0) / widths.reshape((-1,) + (1,) * (y.ndim - 1))
    slopes = np.zeros_like(y)
    if n_knots == 2:
        slopes[0] = delta[0]
        slopes[1] = delta[0]
        return slopes

    left = delta[:-1]
    right = delta[1:]
    width_left = widths[:-1].reshape((-1,) + (1,) * (left.ndim - 1))
    width_right = widths[1:].reshape((-1,) + (1,) * (right.ndim - 1))
    same_sign = left * right > 0.0
    weight_right = 2.0 * width_right + width_left
    weight_left = width_right + 2.0 * width_left
    # ``where`` still evaluates both sides, so the division is masked.
    left_term = np.divide(weight_right, left, out=np.ones_like(left), where=same_sign)
    right_term = np.divide(weight_left, right, out=np.ones_like(right), where=same_sign)
    harmonic = (weight_right + weight_left) / (left_term + right_term)
    slopes[1:-1] = np.where(same_sign, harmonic, 0.0)
    slopes[0] = _edge_slope(widths[0], widths[1], delta[0], delta[1])
    slopes[-1] = _edge_slope(widths[-1], widths[-2], delta[-1], delta[-2])
    return slopes


def _edge_slope(
    width_near: float,
    width_far: float,
    delta_near: np.ndarray,
    delta_far: np.ndarray,
) -> np.ndarray:
    """One-sided three-point slope, flattened so the cubic stays monotone."""
    slope = ((2.0 * width_near + width_far) * delta_near - width_near * delta_far) / (
        width_near + width_far
    )
    slope = np.where(np.sign(slope) != np.sign(delta_near), 0.0, slope)
    too_steep = (np.sign(delta_near) != np.sign(delta_far)) & (
        np.abs(slope) > 3.0 * np.abs(delta_near)
    )
    return np.where(too_steep, 3.0 * delta_near, slope)


def eval_np(t: Any, x: np.ndarray, y: np.ndarray, slopes: np.ndarray) -> np.ndarray:
    """Evaluate a fitted cubic at ``t``.

    ``t`` is a scalar or a 1-d array of queries. Outside ``x`` the value is the
    nearest end knot, matching a clipped lookup.
    """
    queries = np.asarray(t, dtype=np.float64)
    scalar = queries.ndim == 0
    queries = np.atleast_1d(queries)
    clipped = np.clip(queries, x[0], x[-1])
    index = np.searchsorted(x, clipped, side="right") - 1
    index = np.clip(index, 0, x.shape[0] - 2)
    x0 = x[index]
    width = x[index + 1] - x0
    fraction = (clipped - x0) / width
    y0 = y[index]
    y1 = y[index + 1]
    m0 = slopes[index]
    m1 = slopes[index + 1]
    extra = y0.ndim - 1
    if extra:
        shape = (queries.shape[0],) + (1,) * extra
        fraction = fraction.reshape(shape)
        width = width.reshape(shape)
    values = _hermite(fraction, width, y0, y1, m0, m1)
    if scalar:
        return values[0]
    return values


def _hermite(s: Any, h: Any, y0: Any, y1: Any, m0: Any, m1: Any) -> Any:
    s2 = s * s
    s3 = s2 * s
    h00 = (2.0 * s3) - (3.0 * s2) + 1.0
    h10 = s3 - (2.0 * s2) + s
    h01 = (-2.0 * s3) + (3.0 * s2)
    h11 = s3 - s2
    return (h00 * y0) + (h10 * h * m0) + (h01 * y1) + (h11 * h * m1)


def _hermite_ds(s: Any, h: Any, y0: Any, y1: Any, m0: Any, m1: Any) -> Any:
    """Derivative of :func:`_hermite` with respect to the fraction ``s``."""
    s2 = s * s
    dh00 = (6.0 * s2) - (6.0 * s)
    dh10 = (3.0 * s2) - (4.0 * s) + 1.0
    dh01 = (-6.0 * s2) + (6.0 * s)
    dh11 = (3.0 * s2) - (2.0 * s)
    return (dh00 * y0) + (dh10 * h * m0) + (dh01 * y1) + (dh11 * h * m1)


def _locate(t: Any, x: Any) -> tuple[Any, Any, Any]:
    """Segment index, fraction into it, and its width. Clipped at both ends."""
    clipped = jnp.clip(t, x[0], x[-1])
    index = jnp.searchsorted(x, clipped, side="right") - 1
    index = jnp.clip(index, 0, x.shape[0] - 2)
    x0 = x[index]
    width = x[index + 1] - x0
    return index, (clipped - x0) / width, width


def _value_and_slope(t: Any, x: Any, y: Any, slopes: Any) -> tuple[Any, Any]:
    index, fraction, width = _locate(t, x)
    value = _hermite(fraction, width, y[index], y[index + 1], slopes[index], slopes[index + 1])
    dp_ds = _hermite_ds(fraction, width, y[index], y[index + 1], slopes[index], slopes[index + 1])
    dp_dt = dp_ds / width
    # Flat outside the table. Inside, including the end knots, use the cubic.
    dp_dt = jnp.where((t >= x[0]) & (t <= x[-1]), dp_dt, 0.0)
    return value, dp_dt


@custom_jvp
def eval_jax(t: Any, x: Any, y: Any, slopes: Any) -> Any:
    """Evaluate a fitted cubic at scalar ``t``.

    ``x`` is knot positions and is not differentiable. ``y`` and ``slopes``
    are: ``SmoothContact.materialize`` folds the mixing scalars into them, so
    a cotangent on ``bg_mixing`` reaches this function through the knot data
    and nowhere else.
    """
    value, _ = _value_and_slope(t, x, y, slopes)
    return value


@eval_jax.defjvp
def _eval_jax_jvp(primals: Any, tangents: Any) -> tuple[Any, Any]:
    t, x, y, slopes = primals
    dt, _, dy, dslopes = tangents
    index, fraction, width = _locate(t, x)
    value = _hermite(fraction, width, y[index], y[index + 1], slopes[index], slopes[index + 1])
    dp_ds = _hermite_ds(fraction, width, y[index], y[index + 1], slopes[index], slopes[index + 1])
    dp_dt = jnp.where((t >= x[0]) & (t <= x[-1]), dp_ds / width, 0.0)
    # The cubic is linear in the knot values and slopes, so their tangent is
    # the same cubic evaluated on the tangents.
    from_knots = _hermite(
        fraction, width, dy[index], dy[index + 1], dslopes[index], dslopes[index + 1]
    )
    return value, dp_dt * dt + from_knots
