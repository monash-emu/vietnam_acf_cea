"""Parity against summer3wip goldens."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from tb_macro.constants import AGES, FIXTURES, INFECT_COMPS
from tb_macro.model import get_compiled, run
from tb_macro.outputs import compartments_by_label, flow_by_age, flow_total, infection_target

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"
RTOL = 1e-6
ATOL = 1e-9


def _max_rel(got: np.ndarray, exp: np.ndarray) -> float:
    scale = np.maximum(np.abs(exp), ATOL)
    return float(np.max(np.abs(got - exp) / scale))


def test_progression_edge_weights_are_even() -> None:
    weight = np.asarray(get_compiled().flows["progression"].weight)
    assert weight.size > 0
    np.testing.assert_allclose(weight, 0.5)


def test_summer4_import_and_goldens_present() -> None:
    assert version("summer4") == "0.2.0a1"
    for name in FIXTURES:
        for filename in ("compartments.parquet", "flows.parquet", "target.parquet", "params.yaml"):
            assert (GOLDEN / name / filename).is_file(), filename


def test_parity() -> None:
    errors: list[str] = []
    for name, params in FIXTURES.items():
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
        for col in flows.columns:
            if col == "time":
                continue
            if "__age_" in col:
                flow_name, _, age = col.partition("__age_")
                arr = np.asarray(flow_by_age(result, flow_name)[age])
            else:
                arr = np.asarray(flow_total(result, col))
            exp = flows[col].to_numpy()
            errors.append(f"{name} flow {col} {_max_rel(arr, exp):.3e}")
            np.testing.assert_allclose(arr, exp, rtol=RTOL, atol=ATOL)

        target = pd.read_parquet(folder / "target.parquet")
        got_target = np.asarray(infection_target(result))
        errors.append(f"{name} target {_max_rel(got_target, target['value'].to_numpy()):.3e}")
        np.testing.assert_allclose(got_target, target["value"].to_numpy(), rtol=RTOL, atol=ATOL)

        age_cols = [c for c in flows.columns if c.startswith("infect_") and "__age_" in c]
        assert age_cols, "infect per-age columns missing"
        assert any(c.startswith("detection__age_") for c in flows.columns)
        for comp in INFECT_COMPS:
            assert f"infect_{comp}" in flows.columns
        for age in AGES:
            assert f"detection__age_{age}" in flows.columns
    # Printed on failure via the assertion message above; keep the list so a
    # passing run still exercises the error helper.
    assert errors
