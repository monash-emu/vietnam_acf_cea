"""End-to-end test of the economic layer against real model runs.

The chain under test is: solve the model, derive economic indicators, cost them, convert
health outcomes to DALYs, and compare arms. Model runs are shared across tests via a
module-scoped fixture because each solve is expensive.

Nothing here modifies ``vacf``; these tests exist alongside the golden parity tests,
which continue to cover the model itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vacf.constants import ACF_MIN_AGE, AGES, AGE_STRATA
from vacf.model import run
from vacf.parameters import BASE_PARAMS, fixtures

from vacf.economics import contract, costing, indicators, scenarios
from vacf.economics.cea import acceptability_curve, efficiency_frontier, net_monetary_benefit
from vacf.economics.costs import compute_cost_streams, cost_items_from_spreadsheet, total_costs
from vacf.economics.dalys import age_weighted_daly_values, total_dalys
from vacf.economics.discounting import discount_factors, discounted_total
from vacf.economics.inputs import load_age_daly_table

COVERAGE = 0.8
HORIZON = (2027, 2050)


@pytest.fixture(scope="module")
def base_run():
    params = dict(BASE_PARAMS)
    return params, run(params)


@pytest.fixture(scope="module")
def acf_run():
    params = fixtures()["acf"]
    return params, run(params)


@pytest.fixture(scope="module")
def items():
    return cost_items_from_spreadsheet()


@pytest.fixture(scope="module")
def daly_values():
    table = load_age_daly_table()
    pop = indicators.single_age_population(2027)
    return age_weighted_daly_values(table, pop, list(AGES))


# ---------------------------------------------------------------------------------------
# Screening volume: people screened, not screening events
# ---------------------------------------------------------------------------------------


def test_screening_throughput_plateaus_at_coverage():
    params = fixtures()["acf"]
    years = indicators.model_years()
    tput = indicators.screening_throughput(years, params)
    mid = int(np.where(years == params["acf_start"] + 3.0)[0][0])
    assert tput[mid] == pytest.approx(params["acf_coverage"])


def test_screening_is_zero_outside_the_campaign():
    params = fixtures()["acf"]
    years = indicators.model_years()
    tput = indicators.screening_throughput(years, params)
    before = years < params["acf_start"]
    after = years > params["acf_start"] + params["acf_duration"] + params["acf_scaling_time"]
    assert np.allclose(tput[before], 0.0)
    assert np.allclose(tput[after], 0.0)


def test_throughput_is_not_the_model_hazard():
    """The model screens at a hazard; costing needs a throughput. They differ by 2x at 80%.

    Using the hazard would count the same person being screened repeatedly and inflate the
    dominant provider cost accordingly.
    """
    coverage = 0.8
    hazard_peak = -np.log(1.0 - coverage)
    assert hazard_peak == pytest.approx(1.6094, abs=1e-4)
    assert hazard_peak / coverage == pytest.approx(2.01, abs=0.01)


def test_screened_never_exceeds_eligible_population(acf_run):
    params, result = acf_run
    frame = indicators.economic_indicators(result, params)
    eligible = indicators.eligible_population(result, params)
    assert (frame[contract.N_SCREENED].values <= eligible + 1e-6).all()


def test_eligible_ages_follow_the_run_parameter(acf_run):
    """Costing eligibility must track whatever floor the model actually screened at."""
    params, _ = acf_run
    ages = indicators.eligible_ages(params)
    floor = float(params["acf_min_age"])
    assert all(int(age) >= floor for age in ages)
    excluded = [str(a) for a in AGE_STRATA if a < floor]
    assert excluded, "expected some age groups below the ACF minimum"
    assert set(ages).isdisjoint(excluded)


def test_changing_the_floor_changes_eligibility():
    assert indicators.eligible_ages({"acf_min_age": 15.0}) == ("15", "18", "40", "65")
    assert indicators.eligible_ages({"acf_min_age": 10.0}) == ("10", "15", "18", "40", "65")


def test_baseline_screens_nobody(base_run):
    params, result = base_run
    frame = indicators.economic_indicators(result, params)
    assert np.allclose(frame[contract.N_SCREENED].values, 0.0)


# ---------------------------------------------------------------------------------------
# The model/economics contract
# ---------------------------------------------------------------------------------------


def test_indicators_satisfy_the_contract(acf_run):
    params, result = acf_run
    frame = indicators.economic_indicators(result, params)
    contract.validate(frame, list(AGES))


def test_missing_column_is_reported_clearly(acf_run):
    params, result = acf_run
    frame = indicators.economic_indicators(result, params).drop(columns=[contract.N_SCREENED])
    with pytest.raises(KeyError, match=contract.N_SCREENED):
        contract.validate(frame, list(AGES))


def test_detection_routes_are_separate_and_nonnegative(acf_run):
    params, result = acf_run
    frame = indicators.economic_indicators(result, params)
    assert (frame[contract.TREATMENT_STARTS_ACTIVE] >= -1e-9).all()
    assert (frame[contract.TREATMENT_STARTS_PASSIVE] >= -1e-9).all()
    campaign = frame.loc[params["acf_start"] + 2]
    assert campaign[contract.TREATMENT_STARTS_ACTIVE] > 0


def test_baseline_has_no_active_detections(base_run):
    params, result = base_run
    frame = indicators.economic_indicators(result, params)
    assert np.allclose(frame[contract.TREATMENT_STARTS_ACTIVE].values, 0.0, atol=1e-9)
    assert (frame[contract.TREATMENT_STARTS_PASSIVE] > 0).any()


# ---------------------------------------------------------------------------------------
# Epidemiological direction of travel
# ---------------------------------------------------------------------------------------


def test_acf_reduces_deaths_relative_to_baseline(base_run, acf_run):
    base_params, base_result = base_run
    acf_params, acf_result = acf_run
    base_frame = indicators.economic_indicators(base_result, base_params)
    acf_frame = indicators.economic_indicators(acf_result, acf_params)
    death_cols = [contract.TB_DEATHS_BY_AGE.format(age=a) for a in AGES]
    base_deaths = base_frame[death_cols].sum(axis=1)
    acf_deaths = acf_frame[death_cols].sum(axis=1)
    assert acf_deaths.loc[2035] < base_deaths.loc[2035]


# ---------------------------------------------------------------------------------------
# Costing
# ---------------------------------------------------------------------------------------


def test_every_arm_has_a_costing(items):
    for arm in scenarios.ARMS_PENDING_SENSITIVITIES:
        structure = costing.acf_cost_structure(arm, items)
        names = {c.name for c in structure.components}
        assert {"screening", "active_diagnosis", "passive_diagnosis"}.issubset(names)


def test_baseline_costing_has_no_screening(items):
    structure = costing.baseline_cost_structure(items)
    assert "screening" not in {c.name for c in structure.components}


def test_patient_costs_are_assigned_by_detection_route(items):
    structure = costing.acf_cost_structure("acf_cxr_xpert", items)
    by_name = {c.name: c for c in structure.components}
    assert by_name["patient_cost_active"].quantity_output == contract.TREATMENT_STARTS_ACTIVE
    assert by_name["patient_cost_passive"].quantity_output == contract.TREATMENT_STARTS_PASSIVE
    assert by_name["patient_cost_passive"].unit_cost.mean > 2 * by_name["patient_cost_active"].unit_cost.mean


def test_routine_costs_persist_in_screening_arms(items):
    """Routine services continue during a campaign; omitting them understates arm costs."""
    structure = costing.acf_cost_structure("acf_xpert", items)
    assert "passive_diagnosis" in {c.name for c in structure.components}


def test_cxr_triage_is_cheaper_per_person_screened(items):
    """Triage exists to avoid giving everyone the expensive assay."""
    cost_of = lambda arm: next(
        c for c in costing.acf_cost_structure(arm, items).components if c.name == "screening"
    ).unit_cost.mean
    assert cost_of("acf_cxr_xpert") < cost_of("acf_xpert")
    assert cost_of("acf_cxr_pluslife") < cost_of("acf_pluslife")


def test_unknown_arm_rejected(items):
    with pytest.raises(ValueError, match="No costing defined"):
        costing.acf_cost_structure("acf_crystal_ball", items)


def test_costs_split_by_perspective(acf_run, items):
    params, result = acf_run
    frame = indicators.economic_indicators(result, params)
    structure = costing.acf_cost_structure("acf_cxr_xpert", items)
    table = compute_cost_streams(frame, structure)
    np.testing.assert_allclose(
        table["total"].values,
        (table["total_provider"] + table["total_patient"]).values,
        rtol=1e-12,
    )
    assert table.loc[2029, "screening"] > 0


def test_screening_dominates_provider_cost_during_campaign(acf_run, items):
    """A national campaign screening tens of millions should dominate provider spend."""
    params, result = acf_run
    frame = indicators.economic_indicators(result, params)
    table = compute_cost_streams(frame, costing.acf_cost_structure("acf_xpert", items))
    peak = table.loc[2029]
    assert peak["screening"] > peak["passive_diagnosis"]


# ---------------------------------------------------------------------------------------
# DALYs
# ---------------------------------------------------------------------------------------


def test_daly_values_remap_onto_model_age_groups(daly_values):
    assert list(daly_values.index) == list(AGES)
    assert daly_values["yll"].is_monotonic_decreasing
    assert (daly_values["post_tb_daly"] > 0).all()


def test_dalys_decompose_into_components(acf_run, daly_values):
    params, result = acf_run
    frame = indicators.economic_indicators(result, params)
    totals = total_dalys(
        frame, list(AGES), daly_values, yld_weight=costing.YLD_WEIGHT.mean,
        discount_rate=costing.DISCOUNT_RATE, reference_year=costing.REFERENCE_YEAR,
        start_year=HORIZON[0], end_year=HORIZON[1],
    )
    assert set(totals.index) == {"ylls", "post_tb_dalys", "ylds", "total"}
    assert totals["total"] == pytest.approx(
        totals[["ylls", "post_tb_dalys", "ylds"]].sum(), rel=1e-9
    )
    assert totals["total"] > 0


def test_discounting_reduces_later_value():
    factors = discount_factors([2027, 2037], rate=0.03, reference_year=2027)
    assert factors.loc[2027] == pytest.approx(1.0)
    assert factors.loc[2037] < 1.0


def test_horizon_restriction_changes_totals(acf_run, daly_values):
    params, result = acf_run
    frame = indicators.economic_indicators(result, params)
    short = total_dalys(frame, list(AGES), daly_values, costing.YLD_WEIGHT.mean,
                        start_year=2027, end_year=2035)
    long = total_dalys(frame, list(AGES), daly_values, costing.YLD_WEIGHT.mean,
                       start_year=2027, end_year=2060)
    assert long["total"] > short["total"]


# ---------------------------------------------------------------------------------------
# Cost-effectiveness
# ---------------------------------------------------------------------------------------


def _totals(run_pair, scenario, items, daly_values):
    params, result = run_pair
    frame = indicators.economic_indicators(result, params)
    structure = costing.cost_structure_for(scenario, items)
    costs = total_costs(
        frame, structure, discount_rate=costing.DISCOUNT_RATE,
        reference_year=costing.REFERENCE_YEAR, start_year=HORIZON[0], end_year=HORIZON[1],
    )
    dalys = total_dalys(
        frame, list(AGES), daly_values, costing.YLD_WEIGHT.mean,
        discount_rate=costing.DISCOUNT_RATE, reference_year=costing.REFERENCE_YEAR,
        start_year=HORIZON[0], end_year=HORIZON[1],
    )
    return {"cost": costs["total"], "dalys": dalys["total"]}


def test_full_chain_produces_an_incremental_comparison(base_run, acf_run, items, daly_values):
    """Model run through to an efficiency frontier, with no step done by hand."""
    results = pd.DataFrame(
        {
            "baseline": _totals(base_run, "baseline", items, daly_values),
            "acf_cxr_xpert": _totals(acf_run, "acf_cxr_xpert", items, daly_values),
        }
    ).T
    assert (results["cost"] > 0).all()
    assert (results["dalys"] > 0).all()

    # Screening averts DALYs; whether it also saves money is an empirical question, so the
    # test asserts only the direction the model structure guarantees.
    assert results.loc["acf_cxr_xpert", "dalys"] < results.loc["baseline", "dalys"]

    frontier = efficiency_frontier(results)
    assert set(frontier["status"]) <= {"frontier", "dominated", "extended_dominated"}
    assert frontier.loc["acf_cxr_xpert", "status"] == "frontier"


def test_net_benefit_prefers_cheaper_and_healthier():
    assert net_monetary_benefit(100.0, 50.0, 1000.0) > net_monetary_benefit(200.0, 60.0, 1000.0)


def test_acceptability_curve_rows_sum_to_one():
    rng = np.random.default_rng(0)
    draws = {
        "baseline": pd.DataFrame({"cost": rng.normal(1e9, 1e7, 200), "dalys": rng.normal(5e5, 1e4, 200)}),
        "acf": pd.DataFrame({"cost": rng.normal(2e9, 1e7, 200), "dalys": rng.normal(4e5, 1e4, 200)}),
    }
    curve = acceptability_curve(draws, thresholds=[0, 5000, 50000])
    assert np.allclose(curve.sum(axis=1), 1.0)
    assert curve.loc[0, "baseline"] > curve.loc[0, "acf"]


# ---------------------------------------------------------------------------------------
# Guard against placeholder diagnostic accuracy reaching results
# ---------------------------------------------------------------------------------------


def test_arms_refuse_to_build_without_measured_sensitivities():
    with pytest.raises(ValueError, match="No measured sensitivities"):
        scenarios.build_arms(coverage=COVERAGE)


def test_arms_can_be_inspected_explicitly():
    arms = scenarios.build_arms(coverage=COVERAGE, allow_provisional=True)
    assert len(arms) == len(scenarios.ARMS_PENDING_SENSITIVITIES)
    assert all(arm.provisional for arm in arms)


def test_supplied_sensitivities_make_an_arm_definitive():
    arms = scenarios.build_arms(
        coverage=COVERAGE,
        sensitivities={
            name: {
                "acf_sensitivity_sympt_low": 1.0,
                "acf_sensitivity_sympt_high": 1.0,
                "acf_sensitivity_asympt_low": 0.3,
                "acf_sensitivity_asympt_high": 0.8,
            }
            for name in scenarios.ARMS_PENDING_SENSITIVITIES
        },
    )
    assert not any(arm.provisional for arm in arms)
    assert all(arm.params["acf_coverage"] == COVERAGE for arm in arms)


def test_invalid_coverage_rejected():
    with pytest.raises(ValueError, match="coverage must be"):
        scenarios.acf_arm("acf_xpert", "Xpert", coverage=1.0)


# ---------------------------------------------------------------------------------------
# YLL discounting: the two-step calculation from the costing team's worked example
# ---------------------------------------------------------------------------------------


def test_worked_example_two_step_yll_discounting():
    """A 60-year-old dying in 2030, with the model's analysis starting in 2026.

    Step 1, already done in the supplied table: remaining life expectancy at age 60 of 18.2
    years becomes a discounted 14.1 via ``[1 - exp(-0.03 * LE)] / 0.03``.

    Step 2, done here: discount that back from the year of death to the reference year,
    ``14.1 / 1.03**4``.

    The table is therefore consumed as a per-death value and discounted once more from the
    year the death occurs. Applying only one of the two steps would misstate YLLs.
    """
    yll_at_60 = load_age_daly_table().loc[60, "yll"]
    assert yll_at_60 == pytest.approx(14.1, abs=0.05), "table should hold step-1 values"

    deaths_in_2030 = pd.DataFrame(
        {
            contract.TB_DEATHS_BY_AGE.format(age="40"): [0.0, 1.0, 0.0],
            contract.TREATMENT_COMPLETIONS_BY_AGE.format(age="40"): 0.0,
            contract.SYMPTOMATIC_PERSON_TIME.format(age="40"): 0.0,
            contract.TREATMENT_PERSON_TIME.format(age="40"): 0.0,
        },
        index=pd.Index([2029, 2030, 2031], name="year"),
    )
    values = pd.DataFrame({"yll": [yll_at_60], "post_tb_daly": [0.0]}, index=["40"])

    totals = total_dalys(
        deaths_in_2030, ["40"], values, yld_weight=0.0,
        discount_rate=0.03, reference_year=2026,
    )
    assert totals["ylls"] == pytest.approx(yll_at_60 / 1.03**4)
    assert totals["ylls"] == pytest.approx(12.549, abs=0.01)


def test_reference_year_leaves_icers_unchanged():
    """Discounting costs and DALYs to a different year rescales both identically."""
    frame = pd.DataFrame(
        {
            contract.TB_DEATHS_BY_AGE.format(age="40"): 10.0,
            contract.TREATMENT_COMPLETIONS_BY_AGE.format(age="40"): 0.0,
            contract.SYMPTOMATIC_PERSON_TIME.format(age="40"): 0.0,
            contract.TREATMENT_PERSON_TIME.format(age="40"): 0.0,
        },
        index=pd.Index(range(2027, 2041), name="year"),
    )
    values = pd.DataFrame({"yll": [14.0], "post_tb_daly": [0.0]}, index=["40"])
    a = total_dalys(frame, ["40"], values, 0.0, discount_rate=0.03, reference_year=2026)
    b = total_dalys(frame, ["40"], values, 0.0, discount_rate=0.03, reference_year=2027)
    assert b["total"] / a["total"] == pytest.approx(1.03)


# ---------------------------------------------------------------------------------------
# Campaign specification
# ---------------------------------------------------------------------------------------


def test_campaign_timing_is_settable_per_arm():
    arm = scenarios.acf_arm("acf_xpert", "Xpert", 0.6, start=2028, duration=3.0, scaling_time=0.5)
    assert arm.params["acf_start"] == 2028
    assert arm.params["acf_duration"] == 3.0
    assert arm.params["acf_scaling_time"] == 0.5


def test_campaign_timing_falls_back_to_defaults():
    """Unspecified campaign shape is inherited from BASE_PARAMS, not silently zeroed."""
    arm = scenarios.acf_arm("acf_xpert", "Xpert", 0.6)
    assert "acf_duration" not in arm.params
    resolved = scenarios.scenario_params(arm)
    assert resolved["acf_duration"] == BASE_PARAMS["acf_duration"]
    assert resolved["acf_start"] == BASE_PARAMS["acf_start"]


def test_invalid_campaign_timing_rejected():
    with pytest.raises(ValueError, match="duration must be positive"):
        scenarios.acf_arm("acf_xpert", "Xpert", 0.6, duration=0.0)
    with pytest.raises(ValueError, match="scaling_time must be positive"):
        scenarios.acf_arm("acf_xpert", "Xpert", 0.6, scaling_time=-1.0)


def test_coverage_is_annual_so_volume_scales_with_duration():
    """Coverage is screened-per-year, so a longer campaign screens proportionally more.

    This is the semantics that drives screening cost: a five-year campaign at 80% screens
    roughly four times the eligible population in total, because each annual round
    rescreens people.
    """
    years = indicators.model_years()
    short = indicators.screening_throughput(
        years, scenarios.scenario_params(
            scenarios.acf_arm("acf_xpert", "Xpert", 0.8, duration=2.0, scaling_time=0.01)
        )
    )
    long = indicators.screening_throughput(
        years, scenarios.scenario_params(
            scenarios.acf_arm("acf_xpert", "Xpert", 0.8, duration=6.0, scaling_time=0.01)
        )
    )
    assert long.sum() == pytest.approx(3.0 * short.sum(), rel=0.05)
