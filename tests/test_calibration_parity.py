"""Calibration log-density parity against summer3wip goldens."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference"))

from calibration_golden import (  # noqa: E402
    GOLDEN_DIR,
    MANIFEST_PATH,
    PARAMS_PATH,
    TERMS_PATH,
    load_manifest,
    load_params,
    load_terms,
)
from tb_macro.calibration import (
    TERM_NAMES as CALIB_TERM_NAMES,
    build_mixing_cases,
    compile_case,
    make_log_likelihood,
)

# Log-density terms can differ slightly when the adaptive solver stops on a
# different step; this is tighter than the stepwise/baseline smoke test.
TERM_RTOL = 1.0e-4
TERM_ATOL = 1.0e-3


def test_calibration_goldens_present() -> None:
    manifest = load_manifest()
    assert MANIFEST_PATH.is_file()
    assert TERMS_PATH.is_file()
    assert PARAMS_PATH.is_file()
    draws = manifest["draws"]
    assert len(draws) >= 3
    terms = load_terms()
    assert list(terms["draw"]) == draws
    params = load_params()
    assert set(params) == set(draws)


@pytest.mark.slow
def test_calibration_terms_match_summer3() -> None:
    expected = load_terms().set_index("draw")
    params_by_draw = load_params()
    case = next(item for item in build_mixing_cases() if item.name == "baseline")
    compiled = compile_case(case)
    evaluate = make_log_likelihood(compiled, mixing=case.mixing, fold=case.fold)
    errors: list[str] = []
    for draw, params in params_by_draw.items():
        got = evaluate(params)
        assert bool(got["ok"]), f"{draw} solver did not succeed"
        row = expected.loc[draw]
        for name in CALIB_TERM_NAMES:
            value = float(got[name])
            target = float(row[name])
            scale = max(abs(target), 1.0)
            rel = abs(value - target) / scale
            errors.append(f"{draw} {name} rel={rel:.3e}")
            np.testing.assert_allclose(value, target, rtol=TERM_RTOL, atol=TERM_ATOL)
        total = float(got["total"])
        target_total = float(row["total"])
        errors.append(
            f"{draw} total rel={abs(total - target_total) / max(abs(target_total), 1.0):.3e}"
        )
        np.testing.assert_allclose(total, target_total, rtol=TERM_RTOL, atol=TERM_ATOL)
        recomputed = float(sum(float(got[name]) for name in CALIB_TERM_NAMES))
        np.testing.assert_allclose(total, recomputed, rtol=0.0, atol=1.0e-9)
    assert errors
    assert GOLDEN_DIR.is_dir()
