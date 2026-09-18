"""Time functions copied from the original tb_macro.utils.

# GLUE(TM4): replaced by WP13 — triangular seed and tanh scale-up as rate
# trees (abs, clip, tanh) once those nodes exist.
"""

from __future__ import annotations

from typing import Any

from jax import numpy as jnp


def get_triang_vals(time: Any, peak_time: Any, peak_height: Any, width: Any) -> Any:
    """Triangular pulse: peak_height at peak_time, 0 outside ±width."""
    time_from_peak = jnp.absolute(time - peak_time)
    # Vendored original uses jnp.clip(..., a_min=0.0). JAX 0.11 dropped a_min;
    # maximum is the same function.
    return jnp.maximum(peak_height * (1.0 - time_from_peak / width), 0.0)


def tanh_based_scaleup(
    t: Any,
    shape: Any,
    inflection_time: Any,
    start_asymptote: Any,
    end_asymptote: Any,
) -> Any:
    """Smooth step from start_asymptote to end_asymptote."""
    rng = end_asymptote - start_asymptote
    return (jnp.tanh(shape * (t - inflection_time)) / 2.0 + 0.5) * rng + start_asymptote
