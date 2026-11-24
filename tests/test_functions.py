"""Copied curve helpers match the formulas in the vendored utils."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
from jax import numpy as jnp

from vacf.curves import (
    binary_search_sum_ge,
    get_cos_multicurve,
    get_scale_data,
    get_triang_vals,
)


def _original_utils() -> Any:
    """Load the vendored summer3 helper for comparison.

    The snapshot under ``reference/snapshot`` is a verbatim copy of the originating
    project and keeps its own package name; renaming inside it would falsify the record
    and break golden regeneration. Its ``src`` directory goes on ``sys.path`` so its
    internal imports resolve against the vendored tree rather than against this package,
    which keeps the reference genuinely independent.
    """
    snapshot_src = (
        Path(__file__).resolve().parents[1] / "reference" / "snapshot" / "src" / "tb_macro"
    )
    path = snapshot_src / "tb_macro" / "utils.py"
    sys.path.insert(0, str(snapshot_src))
    try:
        spec = importlib.util.spec_from_file_location("vacf_snapshot_utils", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(snapshot_src))


def test_triangle_matches_formula() -> None:
    times = np.linspace(1800.0, 2100.0, 100)
    for time in times:
        raw = 0.01 * (1.0 - abs(time - 1830.0) / 10.0)
        np.testing.assert_allclose(get_triang_vals(time, 1830.0, 0.01, 10.0), max(raw, 0.0))


def test_cosine_matches_vendored_helper() -> None:
    original = _original_utils()
    knots = np.array([1957.0, 1986.0, 2007.0, 2014.0, 2021.0, 2021.25, 2021.75, 2022.0])
    values = np.array([0.0, 0.3, 0.5, 0.86, 0.86, 0.43, 0.43, 0.86])
    x_data = get_scale_data(knots)
    y_data = get_scale_data(values)
    original_x = original.get_scale_data(knots)
    original_y = original.get_scale_data(values)
    for time in np.linspace(1800.0, 2100.0, 80):
        got = float(get_cos_multicurve(time, x_data, y_data))
        exp = float(original.get_cos_multicurve(time, original_x, original_y))
        np.testing.assert_allclose(got, exp, rtol=0.0, atol=0.0)
    # The search helper is part of the cosine segment picker.
    assert int(binary_search_sum_ge(2000.0, jnp.asarray(knots))) == int(np.sum(2000.0 >= knots))
