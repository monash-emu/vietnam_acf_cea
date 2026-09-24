"""Parity against summer3wip goldens for the Vietnam model.

Default ``run`` / ``get_compiled`` is the corrected model. Golden matching
is opt-in via ``match_golden=True``.
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from tb_macro.constants import AGES, FLOW_TOTALS, PARITY_ATOL, PARITY_RTOL
from tb_macro.model import get_compiled, run
from tb_macro.outputs import (
    INDICATORS,
    compartments_by_label,
    flow_total,
)
from tb_macro.parameters import fixtures

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"
RTOL = PARITY_RTOL
ATOL = PARITY_ATOL


def _max_rel(got: np.ndarray, exp: np.ndarray) -> float:
    scale = np.maximum(np.abs(exp), ATOL)
    return float(np.max(np.abs(got - exp) / scale))


def test_default_repairs_the_three_bugs() -> None:
    """Deaths return to age 0, and progression is not split a second time."""
    compiled = get_compiled()
    for name in ("nontb_deaths_65", "tb_mortality", "rx_death"):
        dest = np.asarray(compiled.flows[name].dest_idx)
        np.testing.assert_array_equal(dest, np.zeros(dest.shape, dtype=np.int32))
    weight = np.asarray(compiled.flows["progression"].weight)
    np.testing.assert_allclose(weight, 1.0)


def test_default_uses_spline_contact_and_smooth_entry() -> None:
    """Corrected path folds a spline curve; entry_rate_at is the C0 lookup."""
    from tb_macro.demography import smooth_birth_rate_at
    from tb_macro.model import entry_rate_at, prepare_run_params
    from tb_macro.parameters import BASE_PARAMS

    assert entry_rate_at is smooth_birth_rate_at
    prepared = prepare_run_params(dict(BASE_PARAMS))
    assert "curve" in prepared
    assert prepare_run_params(dict(BASE_PARAMS), match_golden=True).get("curve") is None


def test_spline_contact_is_concrete_at_import() -> None:
    """Import-time prepare keeps demographic tables out of jax tracers.

    A lazy first touch under ``@jit`` stored DynamicJaxprTracers on the module
    contact and then parallel NUTS failed at pmap lowering.
    """
    from jax import core

    from tb_macro.model import _SPLINE_CONTACT

    for name in ("offset", "background", "parent", "pop", "lags"):
        value = getattr(_SPLINE_CONTACT, name)
        assert not isinstance(value, core.Tracer), name


def test_match_golden_copies_summer3_destination_index() -> None:
    """Edge i of a many-to-one flow lands in compartment i, not age 0."""
    compiled = get_compiled(match_golden=True)
    for name in ("nontb_deaths_0", "tb_mortality", "rx_death"):
        dest = np.asarray(compiled.flows[name].dest_idx)
        np.testing.assert_array_equal(dest, np.arange(dest.size))


def test_match_golden_progression_edge_weights_are_half() -> None:
    """summer3 scatters 1/2, then multiplies by the infectiousness split."""
    weight = np.asarray(get_compiled(match_golden=True).flows["progression"].weight)
    assert weight.size > 0
    np.testing.assert_allclose(weight, 0.5)


def test_summer4_import_and_goldens_present() -> None:
    assert version("summer4") == "0.2.0a4"
    for name in fixtures():
        for filename in (
            "compartments.parquet",
            "flows.parquet",
            "indicators.parquet",
            "params.yaml",
        ):
            assert (GOLDEN / name / filename).is_file(), f"{name}/{filename}"


def test_parity() -> None:
    errors: list[str] = []
    for name, params in fixtures().items():
        result = run(params, match_golden=True)
        folder = GOLDEN / name
        compartments = pd.read_parquet(folder / "compartments.parquet")
        got = compartments_by_label(result)
        missing = [col for col in compartments.columns if col != "time" and col not in got]
        extra = [col for col in got if col not in compartments.columns]
        assert not missing, missing
        assert not extra, extra
        for col in got:
            exp = compartments[col].to_numpy()
            arr = np.asarray(got[col])
            errors.append(f"{name} compartment {col} {_max_rel(arr, exp):.3e}")
            np.testing.assert_allclose(arr, exp, rtol=RTOL, atol=ATOL)

        flows = pd.read_parquet(folder / "flows.parquet")
        for col in FLOW_TOTALS:
            arr = np.asarray(flow_total(result, col))
            exp = flows[col].to_numpy()
            errors.append(f"{name} flow {col} {_max_rel(arr, exp):.3e}")
            np.testing.assert_allclose(arr, exp, rtol=RTOL, atol=ATOL)

        indicators = pd.read_parquet(folder / "indicators.parquet")
        for label, fn in INDICATORS:
            by_age = fn(result)
            for age in AGES:
                column = f"{label}_{age}"
                arr = np.asarray(by_age[age])
                exp = indicators[column].to_numpy()
                errors.append(f"{name} {column} {_max_rel(arr, exp):.3e}")
                np.testing.assert_allclose(arr, exp, rtol=RTOL, atol=ATOL)
    assert errors
