"""The Vietnam active case-finding study: scenarios, campaign and inputs.

This is the single place to change what the analysis simulates. Everything else reads from
here, so the notebook and the tests cannot drift from the specification.

Three scenarios:

===================  ====================================================================
``baseline``         Routine services only; no active case finding.
``acf_xpert``        Xpert as the front-line screening test.
``acf_cxr_xpert``    Chest X-ray as the front-line test, with confirmatory Xpert for those
                     who screen X-ray positive.
===================  ====================================================================

Screening sensitivities follow the Kiribati methods appendix. Note what the values mean
there: the full algorithm (symptom screening, chest X-ray and Xpert) *defines* the model's
active TB, so it has sensitivity one in every disease state by construction, and the less
intensive approaches are scored by asking whether they would also have found each person
the full algorithm found. Carrying those values here assumes the Vietnamese model's active
disease is delimited the same way.
"""

from __future__ import annotations

from vacf.economics.scenarios import Arm, acf_arm

# ---------------------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------------------

#: First year of active case finding.
ACF_START: float = 2027.0

#: Campaign length in years.
ACF_DURATION: float = 10.0

#: Ramp-on and ramp-off time, in years.
ACF_SCALING_TIME: float = 1.0

#: Proportion of the eligible population screened **per year**. Over a ten-year campaign
#: this means roughly ten rounds, so total screening volume is about ten times the eligible
#: population and screening cost scales with duration.
ACF_COVERAGE: float = 0.80

#: Minimum age eligible for screening.
ACF_MIN_AGE: float = 15.0

#: Years over which costs and DALYs are accumulated. Extends beyond the campaign so that
#: averted disease and deaths occurring after screening ends are counted.
HORIZON: tuple[int, int] = (2027, 2050)


# ---------------------------------------------------------------------------------------
# Screening sensitivity
# ---------------------------------------------------------------------------------------

#: The four active disease states, cross-classified by symptom status and infectiousness.
DISEASE_STATES: tuple[str, ...] = ("sympt_low", "sympt_high", "asympt_low", "asympt_high")

#: Full algorithm: symptom screening, chest X-ray and Xpert with bacteriological
#: confirmation. One in every state by construction, because it is the reference standard
#: against which the other approaches were scored.
XPERT_CXR_SENSITIVITY: dict[str, float] = {
    "sympt_low": 1.0,
    "sympt_high": 1.0,
    "asympt_low": 1.0,
    "asympt_high": 1.0,
}

#: Chest X-ray, preceded by symptom screening. Symptomatic disease is detected with
#: certainty, since symptom screening is the first step and defines symptom status in the
#: model. The two asymptomatic values were the only materially uncertain screening
#: quantities: in Kiribati they were estimated under uniform priors whose bounds are the
#: exact binomial interval around the observed proportions. Those bounds are kept in
#: :data:`CXR_SENSITIVITY_RANGE`. The headline values (0.2 and 0.9) lie within them; they
#: are fixed rather than estimated because active case finding is switched off during
#: calibration here, so these parameters cannot be identified from Vietnamese data.
CXR_SENSITIVITY: dict[str, float] = {
    "sympt_low": 1.0,
    "sympt_high": 1.0,
    "asympt_low": 0.2,
    "asympt_high": 0.9,
}

#: Uniform ranges from the Kiribati priors, for one-way sensitivity analysis.
CXR_SENSITIVITY_RANGE: dict[str, tuple[float, float]] = {
    "asympt_low": (0.07, 0.52),
    "asympt_high": (0.72, 0.93),
}


def serial_sensitivity(first: dict[str, float], second: dict[str, float]) -> dict[str, float]:
    """Sensitivity of a two-stage algorithm where both stages must be positive.

    Args:
        first: Front-line test sensitivity, keyed by disease state.
        second: Confirmatory test sensitivity, same keys.

    Returns:
        Combined sensitivity by disease state.

    Notes:
    -----
    Where screening uses a front-line test followed by a
    confirmatory test, a person with tuberculosis is detected only if
    both tests are positive, so the sensitivity of the algorithm is
    the product of the sensitivities of its components. Because the
    confirmatory step here has a sensitivity of one in every disease
    state by construction, the sensitivity of the chest X-ray
    algorithm reduces to that of chest X-ray alone.
    """
    missing = set(first) ^ set(second)
    if missing:
        raise ValueError(f"Sensitivity dictionaries disagree on states: {sorted(missing)}")
    return {state: first[state] * second[state] for state in first}


def as_model_params(sensitivity: dict[str, float]) -> dict[str, float]:
    """Translate a per-state sensitivity into the model's parameter names."""
    unknown = set(sensitivity) - set(DISEASE_STATES)
    if unknown:
        raise ValueError(f"Unknown disease states: {sorted(unknown)}")
    missing = set(DISEASE_STATES) - set(sensitivity)
    if missing:
        raise ValueError(f"Sensitivity not given for: {sorted(missing)}")
    return {f"acf_sensitivity_{state}": value for state, value in sensitivity.items()}


ALGORITHM_SENSITIVITIES: dict[str, dict[str, float]] = {
    "acf_xpert": XPERT_CXR_SENSITIVITY,
    # Reduces to CXR alone, since the confirmatory step is the reference standard.
    "acf_cxr_xpert": serial_sensitivity(CXR_SENSITIVITY, XPERT_CXR_SENSITIVITY),
}

#: Inputs still awaiting measured values. Empty: the sensitivities are now sourced from the
#: Kiribati methods appendix. The residual assumption is that they transfer to Vietnam,
#: which is stated in the module docstring rather than flagged as provisional.
PROVISIONAL_INPUTS: tuple[str, ...] = ()


# ---------------------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------------------

#: Routine services only. ``acf_coverage`` of zero switches the campaign off entirely.
BASELINE = Arm(
    name="baseline",
    label="No active case finding",
    params={"acf_coverage": 0.0, "acf_min_age": ACF_MIN_AGE},
)


def _campaign(name: str, label: str) -> Arm:
    arm = acf_arm(
        name,
        label,
        coverage=ACF_COVERAGE,
        sensitivities=as_model_params(ALGORITHM_SENSITIVITIES[name]),
        start=ACF_START,
        duration=ACF_DURATION,
        scaling_time=ACF_SCALING_TIME,
    )
    return Arm(
        name=arm.name,
        label=arm.label,
        params=arm.params | {"acf_min_age": ACF_MIN_AGE},
        provisional=False,
    )


XPERT_ACF = _campaign("acf_xpert", "Xpert front-line")
CXR_XPERT_ACF = _campaign("acf_cxr_xpert", "CXR front-line, Xpert confirmation")

#: The scenarios the analysis compares, baseline first.
SCENARIOS: tuple[Arm, ...] = (BASELINE, XPERT_ACF, CXR_XPERT_ACF)


def sensitivity_table() -> dict[str, dict[str, float]]:
    """Sensitivity by algorithm and disease state, for the methods table."""
    return {
        "Xpert front-line": dict(XPERT_CXR_SENSITIVITY),
        "CXR front-line, Xpert confirmation": dict(
            ALGORITHM_SENSITIVITIES["acf_cxr_xpert"]
        ),
    }


def describe() -> str:
    """One-paragraph description of what is being simulated, for the notebook header."""
    end = ACF_START + ACF_DURATION
    return (
        f"Active case finding over {ACF_DURATION:.0f} years ({ACF_START:.0f}-{end:.0f}), "
        f"screening {ACF_COVERAGE:.0%} of the population aged {ACF_MIN_AGE:.0f} and over "
        f"each year. Costs and DALYs accumulated over {HORIZON[0]}-{HORIZON[1]}. "
        f"Mixing is homogeneous by age."
    )


def provisional_warning() -> str | None:
    """Warning text when any diagnostic accuracy input is still a placeholder."""
    if not PROVISIONAL_INPUTS:
        return None
    return (
        "PROVISIONAL INPUTS IN USE: "
        + ", ".join(PROVISIONAL_INPUTS)
        + ". Diagnostic sensitivities have not been confirmed against measured values."
    )
