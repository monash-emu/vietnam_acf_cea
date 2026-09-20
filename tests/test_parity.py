"""Parity against summer3wip goldens for the Vietnam model."""

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


def test_progression_edge_weights_are_half() -> None:
    """summer3 scatters 1/2, then multiplies by the infectiousness split."""
    weight = np.asarray(get_compiled().flows["progression"].weight)
    assert weight.size > 0
    np.testing.assert_allclose(weight, 0.5)


def test_parity_copies_summer3_destination_index() -> None:
    """Edge i of a many-to-one flow lands in compartment i, not age 0."""
    compiled = get_compiled(release=False)
    for name in ("nontb_deaths_0", "tb_mortality", "rx_death"):
        dest = np.asarray(compiled.flows[name].dest_idx)
        np.testing.assert_array_equal(dest, np.arange(dest.size))


def test_release_repairs_the_three_bugs() -> None:
    """Deaths return to age 0, and progression is not split a second time."""
    compiled = get_compiled(release=True)
    for name in ("nontb_deaths_65", "tb_mortality", "rx_death"):
        dest = np.asarray(compiled.flows[name].dest_idx)
        np.testing.assert_array_equal(dest, np.zeros(dest.shape, dtype=np.int32))
    weight = np.asarray(compiled.flows["progression"].weight)
    np.testing.assert_allclose(weight, 1.0)


def test_summer4_import_and_goldens_present() -> None:
    assert version("summer4") == "0.2.0a3"
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
        result = run(params)
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
