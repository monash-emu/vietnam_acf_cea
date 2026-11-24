"""Active case-finding arms for the cost-effectiveness analysis.

Each arm is the model's ACF campaign with the diagnostic sensitivities of one algorithm.
Sensitivity is specified for each of the four active disease states, cross-classified by
symptom status and infectiousness, so an arm is a parameter override — no structural
change. See :mod:`vacf.economics.study` for the values this analysis uses.

An arm built without explicit sensitivities falls back to the model defaults, which are
the upstream blended values retained so the ``acf`` golden reproduces. Such an arm is
marked provisional, and :func:`build_arms` refuses to produce a multi-arm comparison from
defaults alone, so arms that are epidemiologically indistinguishable cannot reach a results
table by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vacf.parameters import BASE_PARAMS

#: Placeholder, pending the measured sensitivities for each algorithm.
_PLACEHOLDER = {
    name: value for name, value in BASE_PARAMS.items() if name.startswith("acf_sensitivity_")
}

ARMS_PENDING_SENSITIVITIES: frozenset[str] = frozenset(
    {"acf_xpert", "acf_cxr_xpert", "acf_pluslife", "acf_cxr_pluslife"}
)


@dataclass(frozen=True)
class Arm:
    """A named scenario: a label, a set of parameter overrides, and its costing key.

    Attributes:
        name: Scenario identifier, matching the cost structure key in ``vacf.economics.costing``.
        label: Human-readable label for figures and tables.
        params: Overrides applied on top of the calibrated parameters.
        provisional: Whether the arm's diagnostic accuracy is still a placeholder.
    """

    name: str
    label: str
    params: dict[str, float] = field(default_factory=dict)
    provisional: bool = False


#: The comparator: routine services only. ``BASE_PARAMS`` leaves ``acf_coverage`` at zero.
BASELINE = Arm(name="baseline", label="No active case finding", params={"acf_coverage": 0.0})


def acf_arm(
    name: str,
    label: str,
    coverage: float,
    sensitivities: dict | None = None,
    *,
    start: float | None = None,
    duration: float | None = None,
    scaling_time: float | None = None,
) -> Arm:
    """Build one ACF arm.

    Args:
        name: Arm identifier; must match a cost structure in ``vacf.economics.costing``.
        label: Human-readable label.
        coverage: Proportion of the eligible population screened **per year**, not over the
            campaign as a whole. A five-year campaign at 0.8 therefore screens roughly four
            times the eligible population in total, because people are rescreened in each
            annual round. Screening cost scales accordingly.
        sensitivities: ``acf_sensitivity_{sympt,asympt}_{low,high}`` for the algorithm.
            Omitted means the model defaults are used and the arm is marked provisional.
        start: Campaign start year. Defaults to ``BASE_PARAMS["acf_start"]``.
        duration: Campaign length in years. Defaults to ``BASE_PARAMS["acf_duration"]``.
        scaling_time: Ramp-on and ramp-off time in years. Defaults to
            ``BASE_PARAMS["acf_scaling_time"]``.

    Returns:
        The arm.

    Notes:
    -----
    Each active case-finding arm applies a distinct diagnostic
    algorithm, represented by its sensitivity for high-infectious and
    low-infectious tuberculosis, to the same screening campaign. Arms
    are compared against routine services alone.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"Arm '{name}': coverage must be in (0, 1), got {coverage}")
    if duration is not None and duration <= 0:
        raise ValueError(f"Arm '{name}': duration must be positive, got {duration}")
    if scaling_time is not None and scaling_time <= 0:
        raise ValueError(f"Arm '{name}': scaling_time must be positive, got {scaling_time}")

    campaign = {"acf_coverage": coverage}
    for key, value in (
        ("acf_start", start),
        ("acf_duration", duration),
        ("acf_scaling_time", scaling_time),
    ):
        if value is not None:
            campaign[key] = value

    return Arm(
        name=name,
        label=label,
        params=campaign | (sensitivities or _PLACEHOLDER),
        provisional=sensitivities is None,
    )


ALGORITHM_LABELS = {
    "acf_xpert": "Xpert",
    "acf_cxr_xpert": "CXR triage with Xpert confirmation",
    "acf_pluslife": "Pluslife tongue swab",
    "acf_cxr_pluslife": "CXR triage with Pluslife confirmation",
}


def build_arms(
    coverage: float = 0.8,
    sensitivities: dict[str, dict[str, float]] | None = None,
    allow_provisional: bool = False,
    *,
    start: float | None = None,
    duration: float | None = None,
    scaling_time: float | None = None,
) -> list[Arm]:
    """All four ACF arms at a given coverage.

    Args:
        coverage: Annual screened proportion of the eligible population.
        sensitivities: Arm name to its sensitivity pair. When omitted, every arm falls back
            to the placeholder values and is indistinguishable from the others.
        allow_provisional: Permit placeholder sensitivities. Off by default.

    Raises:
        ValueError: If any arm would use placeholder sensitivities and ``allow_provisional``
            is not set.
    """
    sensitivities = sensitivities or {}
    missing = sorted(ARMS_PENDING_SENSITIVITIES - set(sensitivities))
    if missing and not allow_provisional:
        raise ValueError(
            f"No measured sensitivities supplied for {missing}. Every arm would reuse the "
            f"model defaults {_PLACEHOLDER}, making the arms epidemiologically identical "
            "and any comparison between them meaningless. Pass allow_provisional=True only "
            "to inspect structure."
        )
    return [
        acf_arm(
            name,
            ALGORITHM_LABELS[name],
            coverage,
            sensitivities.get(name),
            start=start,
            duration=duration,
            scaling_time=scaling_time,
        )
        for name in sorted(ARMS_PENDING_SENSITIVITIES)
    ]


def scenario_params(arm: Arm, calibrated: dict[str, float] | None = None) -> dict[str, float]:
    """Full parameter set for an arm: calibrated values with the arm's overrides applied."""
    return dict(calibrated if calibrated is not None else BASE_PARAMS) | arm.params


def homogeneous_case():
    """The mixing assumption this application uses.

    Age mixing is homogeneous, so no empirical contact data informs the age pattern of
    transmission. This is a deliberate departure from the contact-data-derived matrix the
    upstream model was calibrated with, and it means the upstream posterior does not
    transfer: the model must be recalibrated with
    :data:`vacf.parameters.HOMOGENEOUS_PARAM_BOUNDS`.
    """
    from vacf.calibration import build_mixing_cases

    return next(case for case in build_mixing_cases() if case.name == "homogeneous")


def run_arm(arm: Arm, calibrated: dict[str, float] | None = None, **solver):
    """Solve the model for one arm under homogeneous mixing.

    Args:
        arm: The scenario to run.
        calibrated: Calibrated parameter values; ``BASE_PARAMS`` when omitted, which is a
            forward run rather than a fitted one.
        **solver: Overrides passed through to the solver.

    Returns:
        The solved model output, ready for ``vacf.economics.indicators``.
    """
    from vacf.calibration import compile_case
    from vacf.model import prepare_run_params, solve

    params = prepare_run_params(scenario_params(arm, calibrated))
    return solve(compile_case(homogeneous_case()), params, **solver)
