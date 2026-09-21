"""The calibration likelihood selects the adult compartments the targets use."""

from __future__ import annotations

import pytest
from jax import numpy as jnp

from tb_macro.calibration import (
    ADULT_ROWS,
    HIGH_ROWS,
    LATENT_ROWS,
    LOW_ROWS,
    TERM_NAMES,
    TREATMENT_ROWS,
)
from tb_macro.inputs import get_norm_conmat
from tb_macro.targets import NOTIF_TARGET


def test_likelihood_has_the_seven_summer3_terms() -> None:
    assert TERM_NAMES == (
        "latent",
        "notifications",
        "deaths",
        "prevalence",
        "decline",
        "infectiousness",
        "mixing",
    )


def test_targets_are_at_mid_year() -> None:
    assert float(NOTIF_TARGET.index[0]) == 2011.5


def test_synthetic_contact_matrix_is_normalised() -> None:
    matrix = get_norm_conmat("VNM")
    assert matrix.shape == (8, 8)
    radius = jnp.max(jnp.abs(jnp.linalg.eigvals(matrix)))
    assert abs(float(radius) - 1.0) < 1.0e-6


def test_adult_selections_are_nonempty_and_nested() -> None:
    adult = set(int(i) for i in ADULT_ROWS)
    latent = set(int(i) for i in LATENT_ROWS)
    high = set(int(i) for i in HIGH_ROWS)
    low = set(int(i) for i in LOW_ROWS)
    treated = set(int(i) for i in TREATMENT_ROWS)
    assert latent < adult
    assert high.isdisjoint(low)
    assert high < adult
    assert low < adult
    assert treated < adult
    assert treated.isdisjoint(high)


@pytest.mark.slow
def test_stepwise_likelihood_matches_the_baseline() -> None:
    """The folded year table is the baseline vector field, so the density agrees."""
    from tb_macro.calibration import build_mixing_cases, compile_case, make_log_likelihood
    from tb_macro.parameters import BASE_PARAMS, get_nuts_init_values

    params = dict(BASE_PARAMS)
    params.update(get_nuts_init_values())
    totals = {}
    for case in build_mixing_cases():
        if case.name not in {"baseline", "stepwise"}:
            continue
        evaluate = make_log_likelihood(compile_case(case), mixing=case.mixing, fold=case.fold)
        totals[case.name] = float(evaluate(params)["total"])
    assert abs(totals["baseline"] - totals["stepwise"]) < 1.0e-3
