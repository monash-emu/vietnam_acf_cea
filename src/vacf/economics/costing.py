"""Cost structures for the Vietnam active case-finding analysis.

Maps rows of ``data/economics/cost_inputs_vnm.xlsx`` onto the model quantities they apply
to. The mapping is written out explicitly rather than parsed from the spreadsheet's prose
``applies_to`` column: free-text parsing fails silently and misattributes costs, and what is
being costed should be readable and reviewable alongside the methods.

Costs are assigned **by detection route**, not by scenario. The spreadsheet labels patient
costs "ACF scenarios" (578.91) and "No screening scenario" (1359.01), but the model keeps
active and passive detection as separate flows, so each case can be costed by how it was
actually found. Every arm therefore carries both: a screening campaign does not stop routine
services, and cases still found passively during a campaign incur the passive cost. Treating
them otherwise would credit screening with savings on cases it never found.
"""

from __future__ import annotations

from vacf.economics import contract
from vacf.economics.costs import CostComponent, CostStructure, cost_items_from_spreadsheet
from vacf.economics.inputs import UncertainQuantity, sd_from_ci

#: Disability weight for symptomatic TB and TB on treatment (GBD), from the DALY rows of
#: the spreadsheet. Held here because that row's value is prose rather than a number.
YLD_WEIGHT = UncertainQuantity(
    name="yld_weight_active_tb",
    mean=0.333,
    sd=sd_from_ci(0.224, 0.454),
    kind="beta",
)

#: Discount rate stated in the spreadsheet, applied to both costs and health effects.
DISCOUNT_RATE = 0.03

#: Reference year for discounting: the first year of the campaign.
REFERENCE_YEAR = 2027

SCREENING_KEY = "Screening, per person screened"
DIAGNOSIS_ACTIVE_KEY = "Diagnosis, per positive case"
DIAGNOSIS_PASSIVE_KEY = "Diagnosis, per diagnosed case"
TREATMENT_KEY = "Treatment, per person treated"
PATIENT_KEY = "Total patient cost, per person treated"

#: Spreadsheet ``applies_to`` text for each arm's screening cost.
SCREENING_ROWS = {
    "acf_xpert": "ACF_Xpert scenario (everyone except those in treatment)",
    "acf_cxr_xpert": "ACF_CXR_Xpert scenario (everyone except those in treatment)",
    "acf_pluslife": "ACF_Pluslife scenario (everyone except those in treatment)",
    "acf_cxr_pluslife": "ACF_CXR_Pluslife scenario (everyone except those in treatment)",
}

#: Xpert-based and Pluslife-based arms each share a per-positive diagnosis cost.
DIAGNOSIS_ROWS = {
    "acf_xpert": "ACF_Xpert scenario, ACF_CXR_Xpert scenario",
    "acf_cxr_xpert": "ACF_Xpert scenario, ACF_CXR_Xpert scenario",
    "acf_pluslife": "ACF_Pluslife scenario, ACF_CXR_Pluslife scenario",
    "acf_cxr_pluslife": "ACF_Pluslife scenario, ACF_CXR_Pluslife scenario",
}

PASSIVE_DIAGNOSIS_ROW = "No screening scenario"
TREATMENT_ROW = "All scenarios"
PATIENT_ACTIVE_ROW = "ACF scenarios"
PATIENT_PASSIVE_ROW = "No screening scenario"


def _lookup(items: dict, parameter: str, applies_to: str) -> UncertainQuantity:
    key = (parameter, applies_to)
    if key not in items:
        raise KeyError(
            f"Economic input spreadsheet has no row for {key}. Available: {sorted(items)}"
        )
    return items[key]


def routine_care_components(items: dict) -> list[CostComponent]:
    """Costs of passively detected cases, incurred in every arm including the comparator."""
    return [
        CostComponent(
            name="passive_diagnosis",
            unit_cost=_lookup(items, DIAGNOSIS_PASSIVE_KEY, PASSIVE_DIAGNOSIS_ROW),
            quantity_output=contract.TREATMENT_STARTS_PASSIVE,
            perspective="provider",
        ),
        CostComponent(
            name="treatment_passive",
            unit_cost=_lookup(items, TREATMENT_KEY, TREATMENT_ROW),
            quantity_output=contract.TREATMENT_STARTS_PASSIVE,
            perspective="provider",
        ),
        CostComponent(
            name="patient_cost_passive",
            unit_cost=_lookup(items, PATIENT_KEY, PATIENT_PASSIVE_ROW),
            quantity_output=contract.TREATMENT_STARTS_PASSIVE,
            perspective="patient",
        ),
    ]


def baseline_cost_structure(items: dict | None = None) -> CostStructure:
    """No-screening comparator: routine services only."""
    items = items if items is not None else cost_items_from_spreadsheet()
    return CostStructure(scenario="baseline", components=routine_care_components(items))


def acf_cost_structure(arm: str, items: dict | None = None) -> CostStructure:
    """Routine services plus the costs of an active case-finding campaign."""
    if arm not in SCREENING_ROWS:
        raise ValueError(f"No costing defined for arm '{arm}'; known: {sorted(SCREENING_ROWS)}")
    items = items if items is not None else cost_items_from_spreadsheet()

    return CostStructure(
        scenario=arm,
        components=routine_care_components(items)
        + [
            CostComponent(
                name="screening",
                unit_cost=_lookup(items, SCREENING_KEY, SCREENING_ROWS[arm]),
                quantity_output=contract.N_SCREENED,
                perspective="provider",
            ),
            CostComponent(
                name="active_diagnosis",
                unit_cost=_lookup(items, DIAGNOSIS_ACTIVE_KEY, DIAGNOSIS_ROWS[arm]),
                quantity_output=contract.TREATMENT_STARTS_ACTIVE,
                perspective="provider",
            ),
            CostComponent(
                name="treatment_active",
                unit_cost=_lookup(items, TREATMENT_KEY, TREATMENT_ROW),
                quantity_output=contract.TREATMENT_STARTS_ACTIVE,
                perspective="provider",
            ),
            CostComponent(
                name="patient_cost_active",
                unit_cost=_lookup(items, PATIENT_KEY, PATIENT_ACTIVE_ROW),
                quantity_output=contract.TREATMENT_STARTS_ACTIVE,
                perspective="patient",
            ),
        ],
    )


def cost_structure_for(scenario: str, items: dict | None = None) -> CostStructure:
    """Cost structure for a named scenario, comparator or ACF arm."""
    return (
        baseline_cost_structure(items)
        if scenario == "baseline"
        else acf_cost_structure(scenario, items)
    )
