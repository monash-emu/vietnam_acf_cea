"""The Vietnam ACF study specification and its analysis pipeline.

Pins what the analysis simulates, so a change to the specification is visible as a test
change rather than a silent shift in results.
"""

from __future__ import annotations

import numpy as np
import pytest

from vacf.economics import analysis, costing, study
from vacf.economics.costs import cost_items_from_spreadsheet


@pytest.fixture(scope="module")
def runs():
    """Forward runs at BASE_PARAMS. Calibration is exercised separately and is slow."""
    return analysis.run_all()


@pytest.fixture(scope="module")
def items():
    return cost_items_from_spreadsheet()


# ---------------------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------------------


def test_three_scenarios_with_baseline_first():
    assert [a.name for a in study.SCENARIOS] == ["baseline", "acf_xpert", "acf_cxr_xpert"]


def test_campaign_is_ten_years_at_eighty_percent_from_fifteen():
    assert study.ACF_DURATION == 10.0
    assert study.ACF_COVERAGE == 0.80
    assert study.ACF_MIN_AGE == 15.0
    for arm in (study.XPERT_ACF, study.CXR_XPERT_ACF):
        assert arm.params["acf_duration"] == 10.0
        assert arm.params["acf_coverage"] == 0.80
        assert arm.params["acf_min_age"] == 15.0


def test_baseline_does_no_screening():
    assert study.BASELINE.params["acf_coverage"] == 0.0


def test_sensitivities_match_the_source_appendix():
    """Pins the values carried over from the Kiribati methods appendix."""
    assert study.XPERT_CXR_SENSITIVITY == {
        "sympt_low": 1.0, "sympt_high": 1.0, "asympt_low": 1.0, "asympt_high": 1.0,
    }
    assert study.CXR_SENSITIVITY["sympt_low"] == 1.0
    assert study.CXR_SENSITIVITY["sympt_high"] == 1.0
    # Headline values, fixed within the appendix's uniform priors.
    assert study.CXR_SENSITIVITY["asympt_low"] == 0.2
    assert study.CXR_SENSITIVITY["asympt_high"] == 0.9
    for state, (low, high) in study.CXR_SENSITIVITY_RANGE.items():
        assert low <= study.CXR_SENSITIVITY[state] <= high, state


def test_cxr_matches_xpert_on_symptomatic_but_misses_asymptomatic():
    """The whole comparison rests on this: X-ray misses asymptomatic disease."""
    xpert = study.ALGORITHM_SENSITIVITIES["acf_xpert"]
    cxr = study.ALGORITHM_SENSITIVITIES["acf_cxr_xpert"]
    assert cxr["sympt_low"] == xpert["sympt_low"]
    assert cxr["sympt_high"] == xpert["sympt_high"]
    assert cxr["asympt_low"] < xpert["asympt_low"]
    assert cxr["asympt_high"] < xpert["asympt_high"]
    # Less-infectious asymptomatic disease is the hardest for X-ray to find.
    assert cxr["asympt_low"] < cxr["asympt_high"]


def test_confirmatory_step_leaves_front_line_sensitivity_unchanged():
    """Xpert confirmation has sensitivity one, so the serial product reduces to CXR."""
    assert study.ALGORITHM_SENSITIVITIES["acf_cxr_xpert"] == study.CXR_SENSITIVITY


def test_serial_sensitivity_is_the_product():
    combined = study.serial_sensitivity(
        {"sympt_low": 0.9, "sympt_high": 0.8, "asympt_low": 0.6, "asympt_high": 0.5},
        {"sympt_low": 1.0, "sympt_high": 0.5, "asympt_low": 0.5, "asympt_high": 0.4},
    )
    assert combined["sympt_high"] == pytest.approx(0.40)
    assert combined["asympt_low"] == pytest.approx(0.30)


def test_model_parameter_names_cover_every_state():
    params = study.as_model_params(study.CXR_SENSITIVITY)
    assert set(params) == {f"acf_sensitivity_{s}" for s in study.DISEASE_STATES}
    with pytest.raises(ValueError, match="Sensitivity not given"):
        study.as_model_params({"sympt_low": 1.0})


def test_every_scenario_has_a_cost_structure(items):
    for arm in study.SCENARIOS:
        structure = costing.cost_structure_for(arm.name, items)
        assert structure.components


def test_no_provisional_inputs_remain():
    """Sensitivities are sourced, so nothing should be flagged as placeholder."""
    assert study.PROVISIONAL_INPUTS == ()
    assert study.provisional_warning() is None
    assert not any(arm.provisional for arm in study.SCENARIOS)


def test_description_states_the_design():
    text = study.describe()
    for fragment in ("10 years", "2027-2037", "80%", "aged 15", "homogeneous"):
        assert fragment in text


# ---------------------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------------------


def test_run_all_produces_both_output_families(runs):
    assert set(runs) == {"baseline", "acf_xpert", "acf_cxr_xpert"}
    for run in runs.values():
        assert not run["epi"].empty
        assert not run["economic"].empty


def test_only_the_eligible_age_range_is_screened(runs):
    """Costing eligibility must be the 15-and-over groups, matching what the model screened."""
    from vacf.constants import AGE_STRATA
    from vacf.economics import indicators

    run = runs["acf_xpert"]
    assert indicators.eligible_ages(run["params"]) == tuple(
        str(a) for a in AGE_STRATA if a >= 15
    )
    # Under-15s are excluded, so screening volume must fall short of the whole population.
    peak = run["economic"].loc[2032, "n_screened"]
    whole_population = run["epi"].loc[2032, "population"]
    assert 0 < peak < whole_population


def test_screening_runs_only_during_the_campaign(runs):
    from vacf.economics import contract

    frame = runs["acf_xpert"]["economic"][contract.N_SCREENED]
    assert frame.loc[2026] == pytest.approx(0.0, abs=1e-6)
    assert frame.loc[2030] > 0
    assert frame.loc[2045] == pytest.approx(0.0, abs=1e-6)


def test_both_arms_avert_burden(runs):
    epi = analysis.epi_comparison(runs)
    baseline = study.BASELINE.label
    assert epi.loc[baseline, "cum_deaths_averted"] == pytest.approx(0.0, abs=1e-6)
    for arm in (study.XPERT_ACF, study.CXR_XPERT_ACF):
        assert epi.loc[arm.label, "cum_deaths_averted"] > 0
        assert epi.loc[arm.label, "cum_incidence_averted"] > 0


def test_more_sensitive_algorithm_averts_more(runs):
    """Front-line Xpert finds asymptomatic disease X-ray misses, so it averts more."""
    epi = analysis.epi_comparison(runs)
    assert (
        epi.loc[study.XPERT_ACF.label, "cum_deaths_averted"]
        > epi.loc[study.CXR_XPERT_ACF.label, "cum_deaths_averted"]
    )


def test_triage_is_cheaper_to_deliver(runs, items):
    """X-ray triage avoids giving everyone the expensive assay."""
    econ = analysis.economic_comparison(runs, items=items)
    assert (
        econ.loc[study.CXR_XPERT_ACF.label, "cost_provider"]
        < econ.loc[study.XPERT_ACF.label, "cost_provider"]
    )


def test_screening_volume_is_similar_across_arms(runs, items):
    """Both arms screen the same people; they differ in what the screen detects."""
    econ = analysis.economic_comparison(runs, items=items)
    xpert = econ.loc[study.XPERT_ACF.label, "n_screened"]
    cxr = econ.loc[study.CXR_XPERT_ACF.label, "n_screened"]
    assert cxr == pytest.approx(xpert, rel=0.01)


def test_economic_comparison_decomposes_totals(runs, items):
    econ = analysis.economic_comparison(runs, items=items)
    np.testing.assert_allclose(
        econ["cost"].values, (econ["cost_provider"] + econ["cost_patient"]).values, rtol=1e-9
    )
    np.testing.assert_allclose(
        econ["dalys"].values,
        econ[["ylls", "post_tb_dalys", "ylds"]].sum(axis=1).values,
        rtol=1e-9,
    )


def test_incremental_analysis_ranks_the_arms(runs, items):
    econ = analysis.economic_comparison(runs, items=items)
    inc = analysis.incremental(econ)
    assert set(inc["status"]) <= {"frontier", "dominated", "extended_dominated"}
    # Doing nothing should not be on the frontier when screening both saves money and
    # averts DALYs under these inputs.
    assert inc.loc[study.BASELINE.label, "status"] == "dominated"
