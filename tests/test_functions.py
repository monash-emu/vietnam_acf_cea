"""Time functions match the vendored originals."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from tb_macro.functions import get_triang_vals, tanh_based_scaleup


def _original() -> Any:
    path = Path(__file__).resolve().parents[1] / "reference" / "tb_macro_original" / "utils.py"
    spec = importlib.util.spec_from_file_location("tb_macro_original_utils", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_time_functions_match_originals() -> None:
    original = _original()
    source = (
        Path(__file__).resolve().parents[1] / "reference" / "tb_macro_original" / "utils.py"
    ).read_text()
    assert "a_min=0.0" in source
    times = np.linspace(0.0, 199.0, 100)
    for t in times:
        raw = 0.01 * (1.0 - abs(t - 30.0) / 10.0)
        np.testing.assert_allclose(get_triang_vals(t, 30.0, 0.01, 10.0), max(raw, 0.0))
        got_s = tanh_based_scaleup(t, 0.12804084477668393, 40.0, 0.7117933876877114, 1.0)
        exp_s = original.tanh_based_scaleup(t, 0.12804084477668393, 40.0, 0.7117933876877114, 1.0)
        np.testing.assert_allclose(got_s, exp_s, rtol=0.0, atol=0.0)
