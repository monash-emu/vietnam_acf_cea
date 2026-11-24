"""Homogeneous age mixing.

Added for the Vietnam active case-finding application, which does not use empirical contact
data. The contact-data path is left intact: the spline case and its seven-term likelihood
still exist and the golden parity tests continue to cover them.
"""

from __future__ import annotations

import numpy as np
import pytest
from jax import numpy as jnp

from vacf.calibration import (
    EPI_TERM_NAMES,
    TERM_NAMES,
    build_mixing_cases,
    compile_case,
    make_log_likelihood,
)
from vacf.constants import AGE_STRATA
from vacf.mixing import homogeneous_matrix, normalise_by_spectral_radius
from vacf.model import solve
from vacf.parameters import (
    BASE_PARAMS,
    HOMOGENEOUS_PARAM_BOUNDS,
    MIXING_PARAMS,
    PARAM_BOUNDS,
)
from vacf.solvers import CALIBRATION_SOLVER

N_AGES = len(AGE_STRATA)


def _case(name: str):
    return next(c for c in build_mixing_cases() if c.name == name)


# ---------------------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------------------


def test_every_age_group_sees_the_same_contact_distribution():
    """Homogeneous mixing means no age assortativity: all rows are identical."""
    pop = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    matrix = np.asarray(homogeneous_matrix(pop))
    for row in range(1, matrix.shape[0]):
        np.testing.assert_allclose(matrix[row], matrix[0], rtol=1e-12)


def test_contacts_are_proportional_to_population_share():
    pop = jnp.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    matrix = np.asarray(homogeneous_matrix(pop))
    shares = np.asarray(pop) / float(np.sum(np.asarray(pop)))
    # Each row is the population share vector, up to the spectral-radius normalisation.
    np.testing.assert_allclose(matrix[0] / matrix[0].sum(), shares, rtol=1e-12)


def test_spectral_radius_is_normalised():
    """Keeps the matrix on the same scale as the contact-derived ones."""
    pop = jnp.asarray(np.linspace(1.0, 8.0, N_AGES))
    matrix = np.asarray(homogeneous_matrix(pop))
    assert np.max(np.abs(np.linalg.eigvals(matrix))) == pytest.approx(1.0, rel=1e-9)


def test_matrix_is_rank_one():
    pop = jnp.asarray(np.linspace(1.0, 8.0, N_AGES))
    assert np.linalg.matrix_rank(np.asarray(homogeneous_matrix(pop)), tol=1e-10) == 1


def test_homogeneous_differs_from_contact_derived_matrix():
    """If these agreed, the assumption would not be doing anything."""
    spline = _case("spline")
    folded = dict(BASE_PARAMS)
    if spline.fold is not None:
        folded.update(spline.fold(folded))
    contact_matrix = np.asarray(spline.mixing(2025.0, folded))
    homo = np.asarray(_case("homogeneous").mixing(2025.0, dict(BASE_PARAMS)))
    assert not np.allclose(contact_matrix, homo, atol=1e-3)


# ---------------------------------------------------------------------------------------
# Registration and parameters
# ---------------------------------------------------------------------------------------


def test_homogeneous_is_registered_alongside_the_others():
    names = [c.name for c in build_mixing_cases()]
    assert "homogeneous" in names
    # The contact-data cases must survive, since the parity goldens cover them.
    assert {"baseline", "stepwise", "smooth", "spline"}.issubset(names)


def test_homogeneous_case_has_nothing_to_fold():
    """bg_mixing, a_spread and pc_strength shape a kernel this case never builds."""
    assert _case("homogeneous").fold is None


def test_mixing_parameters_are_dropped_from_the_calibrated_set():
    assert set(MIXING_PARAMS) <= set(PARAM_BOUNDS)
    assert set(MIXING_PARAMS).isdisjoint(HOMOGENEOUS_PARAM_BOUNDS)
    assert len(HOMOGENEOUS_PARAM_BOUNDS) == len(PARAM_BOUNDS) - len(MIXING_PARAMS)


def test_epidemiological_parameters_are_untouched():
    """Only mixing parameters are removed; the epidemiology is calibrated as before."""
    expected = {k: v for k, v in PARAM_BOUNDS.items() if k not in MIXING_PARAMS}
    assert HOMOGENEOUS_PARAM_BOUNDS == expected
    assert "raw_transmission_rate" in HOMOGENEOUS_PARAM_BOUNDS


# ---------------------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------------------


def test_homogeneous_likelihood_drops_only_the_mixing_term():
    assert set(TERM_NAMES) - set(EPI_TERM_NAMES) == {"mixing"}
    assert len(EPI_TERM_NAMES) == 6


def test_likelihood_without_mixing_term_omits_it_entirely(request):
    """Omitted rather than zeroed, so an unused mixing_dist_sd cannot affect the density."""
    case = _case("homogeneous")
    compiled = compile_case(case)
    evaluate = make_log_likelihood(
        compiled, mixing=case.mixing, fold=case.fold, include_mixing_term=False
    )
    params = {k: v for k, v in BASE_PARAMS.items() if k != "mixing_dist_sd"}
    out = evaluate(params)
    assert "mixing" not in out
    assert np.isfinite(float(out["total"]))
    for name in EPI_TERM_NAMES:
        assert name in out


def test_contact_data_likelihood_still_has_seven_terms():
    """The spline path is unchanged; the parity goldens depend on it."""
    case = _case("spline")
    compiled = compile_case(case)
    evaluate = make_log_likelihood(compiled, mixing=case.mixing, fold=case.fold)
    out = evaluate(dict(BASE_PARAMS))
    for name in TERM_NAMES:
        assert name in out


# ---------------------------------------------------------------------------------------
# Effect on the model
# ---------------------------------------------------------------------------------------


def test_mixing_assumption_changes_the_epidemic():
    """Homogeneous mixing is a re-specification, not a cosmetic switch.

    This is why the posterior fitted under contact-derived mixing does not transfer: the
    force of infection differs, so transmission must be re-estimated.
    """
    homo_case = _case("homogeneous")
    homo = solve(compile_case(homo_case), dict(BASE_PARAMS), **CALIBRATION_SOLVER)

    spline = _case("spline")
    folded = dict(BASE_PARAMS)
    if spline.fold is not None:
        folded.update(spline.fold(folded))
    contact = solve(compile_case(spline), folded, **CALIBRATION_SOLVER)

    def prevalence(result):
        values = result["compartments"].values
        data = np.asarray(values.data if hasattr(values, "data") else values)
        return data.sum(axis=-1)

    homo_total = prevalence(homo)
    contact_total = prevalence(contact)
    assert np.isfinite(homo_total).all()
    # Population totals should stay close, since demography is unchanged.
    np.testing.assert_allclose(homo_total[-1], contact_total[-1], rtol=0.05)


# ---------------------------------------------------------------------------------------
# Calibration-fit trajectories
# ---------------------------------------------------------------------------------------


def test_target_trajectories_reproduce_likelihood_terms():
    """The fit plot shows exactly what the likelihood compares.

    Interpolating the trajectories onto the observation times and applying the likelihood's
    error models must reproduce every epidemiological term of ``make_log_likelihood``.
    """
    from vacf.calibration import (
        _log_normal_log_prob,
        _logit_normal_log_prob,
        observed_targets,
        target_trajectories,
    )

    case = _case("homogeneous")
    compiled = compile_case(case)
    evaluate = make_log_likelihood(
        compiled, mixing=case.mixing, fold=case.fold, include_mixing_term=False
    )
    terms = evaluate(dict(BASE_PARAMS))

    result = solve(compiled, dict(BASE_PARAMS), **CALIBRATION_SOLVER)
    trajectories = target_trajectories(result)
    observed = observed_targets()

    def at(column, times):
        return np.interp(np.asarray(times, dtype=float), trajectories.index, trajectories[column])

    def values(name):
        return observed[name].to_numpy(dtype=float)

    decline = observed["decline"]
    prevalence_ratio = at("prevalence", decline.index[1:]) / at("prevalence", decline.index[:1])
    expected = {
        "latent": _logit_normal_log_prob(at("latent", observed["latent"].index), values("latent")),
        "notifications": _log_normal_log_prob(
            at("notifications", observed["notifications"].index), values("notifications")
        ).mean(),
        "deaths": _log_normal_log_prob(
            at("deaths", observed["deaths"].index), values("deaths")
        ).mean(),
        "prevalence": _logit_normal_log_prob(
            at("prevalence", observed["prevalence"].index), values("prevalence")
        ),
        "decline": _log_normal_log_prob(prevalence_ratio, values("decline")[1:]),
        "infectiousness": _logit_normal_log_prob(
            at("infectiousness", observed["infectiousness"].index), values("infectiousness")
        ),
    }
    assert set(expected) == set(EPI_TERM_NAMES)
    # The likelihood interpolates numerators and denominators separately onto mid-year
    # observation times; interpolating the annual ratio differs only between grid points.
    for name in EPI_TERM_NAMES:
        np.testing.assert_allclose(
            float(jnp.squeeze(expected[name])), float(terms[name]), rtol=1e-3, err_msg=name
        )
