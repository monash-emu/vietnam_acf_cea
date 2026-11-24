"""Vietnam TB model from ``tweak_cases_calib``, on summer4.

Solver ``t`` is the summer3 index ``0 … 300``. Calendar year is ``t + 1800``.
``acf_coverage`` is 0 in ``BASE_PARAMS``, so the ACF flow is identically zero
on the default run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from jax import numpy as jnp

from summer4 import (
    Compartments,
    Dest,
    EntryFlow,
    Everything,
    FieldRef,
    FlowMass,
    FlowModel,
    Multiply,
    Overwrite,
    Param,
    Property,
    PropertyData,
    PropertyMap,
    SavePlan,
    SaveRequest,
    Time,
    TraitChain,
    TransitionFlow,
    clip,
    defer,
)
from summer4.data import Data
from summer4.epi import FOIKind, ForceOfInfection, MixingMatrix
from vacf.constants import (
    AGE_STRATA,
    AGES,
    CLIN_STRATA,
    DISEASE_STATES,
    END_TIME,
    INF_STRATA,
    INFECT_COMPS,
    N_TIMES,
    POP_PER_AGE,
    START_TIME,
    YOUNG_END_AGE,
)
from vacf.contact import prepare_spline
from vacf.curves import get_cos_multicurve, get_scale_data
from vacf.demography import birth_rate_at, smooth_birth_rate_at
from vacf.mixing import get_norm_c_matrix
from vacf.static_data import StaticInputs, load_static

# Sentinel for ``compile_model``: omit the argument to get the mode default.
_DEFAULT = object()

STATE = Property("disease_state", DISEASE_STATES)
AGE = Property("age", AGES)
INFECTIOUS = Property("infectious", INF_STRATA)
CLINICAL = Property("clinical", CLIN_STRATA)

PMAP: PropertyMap = (
    PropertyMap.from_property(STATE)
    .stratify(AGE)
    .stratify(INFECTIOUS, STATE["active"])
    .stratify(CLINICAL, STATE["active"])
)

_COMPILED: dict[bool, Any] = {}
# Flows whose declared destination is one compartment. summer3's
# many-to-one branch stores parent_indices = arange(n_edges), so edge i
# lands in compartment i. ``match_golden`` copies that. See
# reference/summer3-bugs.md.
_INDEX_BUG_FLOWS: tuple[str, ...] = tuple(f"nontb_deaths_{age}" for age in AGES) + (
    "tb_mortality",
    "rx_death",
)
_AGE_BREAKS = jnp.array(AGE_STRATA, dtype=jnp.float64)
# Loaded at import, never inside ``derived_fn``. A lazy load under jit leaks tracers.
STATIC: StaticInputs = load_static()
_DEATH_RATE = Data.table(
    np.asarray(STATIC.death_times, dtype=np.float64),
    np.asarray(STATIC.death_rates, dtype=np.float64),
    over=AGE,
)

# Default corrected path: folded spline contact + continuous (C0) entry rate.
# ``match_golden`` keeps get_norm_c_matrix and the step entry lookup.
# Built at import — never inside a jax transform. A lazy first touch under
# ``@jit`` / ``pmap`` would store DynamicJaxprTracers on this object and then
# fail at pmap lowering with "No constant handler for type: DynamicJaxprTracer".
_CONTACT_YEARS = np.arange(START_TIME, START_TIME + N_TIMES + 1.0)
_SPLINE_CONTACT = prepare_spline(
    STATIC.age_weights,
    STATIC.weight_ends,
    STATIC.pops,
    STATIC.pop_ends,
    STATIC.fert,
    STATIC.fert_ends,
    years=_CONTACT_YEARS,
)
# Patchable by timing harnesses (``reference/time_mixing.py --entry step``).
entry_rate_at = smooth_birth_rate_at


def _spline_contact() -> Any:
    """Prepared spline for the solver span. Always the import-time instance."""
    return _SPLINE_CONTACT


def fold_spline_contact(params: dict[str, Any]) -> dict[str, Any]:
    """Fold mixing scalars into a ``SplineCurve``. Once per parameter draw."""
    ready = _spline_contact().materialize(
        params["bg_mixing"], params["a_spread"], params["pc_strength"]
    )
    if ready.curve is None:
        raise ValueError("prepare_spline did not fit a curve")
    return {"curve": ready.curve}


def spline_mixing(calendar: Any, params: dict[str, Any]) -> Any:
    """Normalised contact matrix from a folded curve.

    Calibration and ``run`` always fold first. The free-in-stage fallback is
    for probes that omit ``curve``; it must not be the NUTS path.
    """
    curve = params["curve"] if "curve" in params else None
    if curve is not None:
        return curve.at(calendar)
    return _spline_contact().matrix(
        calendar, params["bg_mixing"], params["a_spread"], params["pc_strength"]
    )


def baseline_mixing(calendar: Any, params: dict[str, Any]) -> Any:
    """Year-frozen ``get_norm_c_matrix`` (golden / comparison path)."""
    return get_norm_c_matrix(
        STATIC.age_weights,
        STATIC.weight_ends,
        STATIC.pops,
        STATIC.pop_ends,
        STATIC.fert,
        STATIC.fert_ends,
        calendar,
        params["bg_mixing"],
        params["a_spread"],
        params["pc_strength"],
    )


def prepare_run_params(params: dict[str, Any], *, match_golden: bool = False) -> dict[str, Any]:
    """Attach a folded spline curve on the default path when the caller has not."""
    if match_golden or params.get("curve") is not None:
        return params
    prepared = dict(params)
    prepared.update(fold_spline_contact(prepared))
    return prepared


def _calendar() -> Any:
    """Solver index ``Time()`` plus the calendar origin."""
    return Time() + START_TIME


def _age_selector(ages: list[int]) -> Any:
    selected: Any = AGE[str(ages[0])]
    for age in ages[1:]:
        selected = selected | AGE[str(age)]
    return selected


_BAND_0 = _age_selector([age for age in AGE_STRATA if age < 5])
_BAND_5 = _age_selector([age for age in AGE_STRATA if 5 <= age < YOUNG_END_AGE])
_BAND_15 = _age_selector([age for age in AGE_STRATA if age >= YOUNG_END_AGE])
_YOUNG = _age_selector([age for age in AGE_STRATA if age < YOUNG_END_AGE])


def _outcome_rates(dur: Any, prop_neg: Any, tsr: Any, death_rate: Any) -> tuple[Any, Any, Any]:
    """Competing hazards. Copied from ``health_system.compute_outcome_props``."""
    prop_nat = 1.0 - jnp.exp(-dur * death_rate)
    req_death = (1.0 - tsr) * prop_neg
    prop_rx_death = jnp.maximum(req_death - prop_nat, 0.0)
    prop_total_death = prop_rx_death + prop_nat
    relapse = jnp.maximum(1.0 - tsr - prop_total_death, 0.0)
    success = jnp.maximum(1.0 - relapse - prop_total_death, 0.0)
    return success / dur, relapse / dur, prop_rx_death / dur


def _interp_deaths(calendar: Any, times: Any, rates: Any) -> Any:
    """Per-age death hazards for treatment outcome coupling.

    Nontb death *flows* use ``Data.table(...).interp`` (``_DEATH_RATE``). Treatment
    success / relapse / rx_death still need the same vector inside ``derived_fn``
    because they combine it with TSR and unsuccessful-outcome curves.
    """
    cols = [
        jnp.interp(
            calendar,
            times,
            rates[:, i],
            left=rates[0, i],
            right=rates[-1, i],
        )
        for i in range(rates.shape[1])
    ]
    return jnp.stack(cols)


def _acf_screen(calendar: Any, params: dict[str, Any]) -> Any:
    """Cosine campaign. Zero throughout calibration when coverage is 0."""
    coverage = params["acf_coverage"]
    peak = -jnp.log(1.0 - coverage)
    t_start = params["acf_start"]
    scale = params["acf_scaling_time"]
    duration = params["acf_duration"]
    times = get_scale_data(
        jnp.array(
            [
                t_start,
                t_start + scale,
                t_start + duration,
                t_start + duration + scale,
            ]
        )
    )
    vals = get_scale_data(jnp.array([0.0, peak, peak, 0.0]))
    return get_cos_multicurve(calendar, times, vals)


def force_of_infection(
    params: dict[str, Any],
    *,
    y: Any,
    t: Any,
    mixing: Any = _DEFAULT,
    entry: Any = _DEFAULT,
) -> dict[str, Any]:
    """Per-step mixing matrix, death rates, treatment hazards, ACF.

    Infection pressure itself is ``ForceOfInfection`` on the flow graph.
    Default ``mixing`` is the folded spline (:func:`spline_mixing`). Pass
    :func:`baseline_mixing` or ``None`` for ``get_norm_c_matrix`` (golden path).
    Default ``entry`` is :data:`entry_rate_at` (smooth); the golden path passes
    the step lookup ``birth_rate_at``.
    Treatment hazards and ACF stay here: they combine several time series into
    per-compartment rates that are not yet single rate-tree expressions.
    """
    static = STATIC
    calendar = t + START_TIME
    pop = PropertyData(PMAP, y)
    if mixing is _DEFAULT:
        mixing = spline_mixing
    if mixing is None:
        mixing_matrix = get_norm_c_matrix(
            static.age_weights,
            static.weight_ends,
            static.pops,
            static.pop_ends,
            static.fert,
            static.fert_ends,
            calendar,
            params["bg_mixing"],
            params["a_spread"],
            params["pc_strength"],
        )
    else:
        mixing_matrix = mixing(calendar, params)
    deaths = _interp_deaths(calendar, static.death_times, static.death_rates)
    tsr = get_cos_multicurve(calendar, static.tsr_x, static.tsr_y)
    unsucc = get_cos_multicurve(calendar, static.unsucc_x, static.unsucc_y)
    success, relapse, rx_death = _outcome_rates(params["rx_duration"], unsucc, tsr, deaths)
    screen = _acf_screen(calendar, params)
    # Screening sensitivity varies by symptom status as well as infectiousness, and the
    # four values are not a product of the two dimensions: in the source data the
    # symptomatic-to-asymptomatic ratio is 1/0.25 for less-infectious disease but 1/0.85
    # for more-infectious disease. They are therefore assembled as a sum of two masked
    # terms rather than a single outer product.
    #
    # No prop_lowinf_bactpos factor appears here. The sensitivities are defined against a
    # full-algorithm reference standard that already embeds bacteriological
    # confirmability, so applying it again would count that twice.
    symptomatic = screen * jnp.array(
        [params["acf_sensitivity_sympt_low"], params["acf_sensitivity_sympt_high"]]
    )
    asymptomatic = screen * jnp.array(
        [params["acf_sensitivity_asympt_low"], params["acf_sensitivity_asympt_high"]]
    )
    clin_only = jnp.array([1.0, 0.0])  # CLIN_STRATA order: ("clin", "subclin")
    subclin_only = jnp.array([0.0, 1.0])
    acf_sensitivity = pop.broadcast_over(CLINICAL, clin_only) * pop.broadcast_over(
        INFECTIOUS, symptomatic
    ) + pop.broadcast_over(CLINICAL, subclin_only) * pop.broadcast_over(
        INFECTIOUS, asymptomatic
    )
    # Eligibility floor is a parameter, so a scenario can change who is screened without
    # editing the model and without invalidating goldens generated at a different floor.
    age_mask = jnp.where(_AGE_BREAKS >= params["acf_min_age"], 1.0, 0.0)
    acf = (acf_sensitivity * pop.broadcast_over(AGE, age_mask)).keep(STATE["active"], 0.0)

    if entry is _DEFAULT:
        entry = entry_rate_at
    derived = dict(params)
    derived["mixing"] = mixing_matrix
    derived["success_rate"] = pop.broadcast_over(AGE, success)
    derived["relapse_rate"] = pop.broadcast_over(AGE, relapse)
    derived["rx_death_rate"] = pop.broadcast_over(AGE, rx_death)
    derived["acf_rate"] = acf
    derived["entry_rate"] = entry(t, static.entry_times, static.entry_rates, START_TIME)
    return derived


def _detection_rate(
    calendar: Any,
    current: Any,
    rel_2007: Any,
    rel_1986: Any,
    rel_2021: Any,
) -> Any:
    """Cosine detection scale-up. 1986 is current × rel_2007 × rel_1986."""
    r_2007 = current * rel_2007
    r_1986 = r_2007 * rel_1986
    r_covid = current * rel_2021
    times = get_scale_data(
        jnp.array([1957.0, 1986.0, 2007.0, 2014.0, 2021.0, 2021.25, 2021.75, 2022.0])
    )
    vals = get_scale_data(
        jnp.array([0.0, r_1986, r_2007, current, current, r_covid, r_covid, current])
    )
    return get_cos_multicurve(calendar, times, vals)


def _latency_adjust(prefix: str) -> tuple[Overwrite, ...]:
    return (
        Overwrite(Param(f"{prefix}_age0"), where=_BAND_0),
        Overwrite(Param(f"{prefix}_age5"), where=_BAND_5),
        Overwrite(Param(f"{prefix}_age15"), where=_BAND_15),
    )


def _infectiousness(*, match_golden: bool) -> list[tuple[Any, Any]]:
    """Per-compartment infectiousness weights for ``ForceOfInfection``.

    CLIN_STRATA is ``(clin, subclin)``. The default puts
    ``rel_infectiousness_subclin`` on subclinical disease. ``match_golden``
    copies summer3's wrap, which puts that weight on clinical disease.
    See ``reference/summer3-bugs.md``.
    """
    clinical = (
        (CLINICAL["clin"], Param("rel_infectiousness_subclin"))
        if match_golden
        else (CLINICAL["subclin"], Param("rel_infectiousness_subclin"))
    )
    return [
        (INFECTIOUS["low"], Param("rel_infectiousness_lowinf")),
        clinical,
        (_YOUNG, 0.0),
    ]


def _force_of_infection(*, match_golden: bool) -> ForceOfInfection:
    return ForceOfInfection(
        "foi",
        infectious=STATE["active"],
        group_by=AGE,
        kind=FOIKind.FREQUENCY,
        contact_rate=1.0,
        mixing=MixingMatrix(
            AGE,
            FieldRef(("mixing",)),
            normalize="none",
            check_reciprocal=False,
        ),
        infectiousness=_infectiousness(match_golden=match_golden),
    )


def _seed_peak_rate() -> Any:
    """Triangular pulse on the calendar axis."""
    calendar = _calendar()
    return clip(
        Param("seed_peak_rate")
        * (1.0 - abs(calendar - Param("seed_peak_time")) / Param("seed_duration")),
        0.0,
    )


def build_model(*, match_golden: bool = False) -> FlowModel:
    """Declare the ``tweak_cases_calib`` flows. Names match summer3.

    Default infectiousness matches the docstrings. ``match_golden`` selects the
    summer3 clinical wrap; destination and progression patches run after
    ``compile``.
    """
    model = FlowModel(PMAP)
    foi = _force_of_infection(match_golden=match_golden)
    for comp in INFECT_COMPS:
        suscept = "cleared" if comp in ("cleared", "recovered") else comp
        adjust: tuple[Any, ...] = (
            Multiply(Param("raw_transmission_rate") * Param(f"rel_sus_{suscept}")),
        )
        if comp == "mtb_naive":
            adjust = adjust + (Multiply(Param("rel_sus_children"), where=_YOUNG),)
        model.add_flow(
            TransitionFlow(
                f"infect_{comp}",
                STATE[comp],
                STATE["incipient"],
                foi,
                adjust=adjust,
            )
        )
    model.add_flow(
        TransitionFlow(
            "clearance",
            STATE["contained"],
            STATE["cleared"],
            Param("clearance_rate"),
        )
    )
    model.add_flow(
        TransitionFlow(
            "breakdown",
            STATE["contained"],
            STATE["incipient"],
            Param("breakdown_rate"),
        )
    )
    model.add_flow(
        TransitionFlow(
            "infectiousness_gain",
            INFECTIOUS["low"],
            INFECTIOUS["high"],
            Param("infectiousness_gain_rate"),
        )
    )
    model.add_flow(
        TransitionFlow(
            "infectiousness_loss",
            INFECTIOUS["high"],
            INFECTIOUS["low"],
            Param("infectiousness_loss_rate"),
        )
    )
    model.add_flow(
        TransitionFlow(
            "clinical_develop",
            CLINICAL["subclin"],
            CLINICAL["clin"],
            Param("clinical_progression_rate"),
        )
    )
    model.add_flow(
        TransitionFlow(
            "clinical_regress",
            CLINICAL["clin"],
            CLINICAL["subclin"],
            Param("clinical_regression_rate"),
        )
    )
    model.add_flow(
        TransitionFlow(
            "self_recovery",
            STATE["active"] & CLINICAL["subclin"],
            STATE["recovered"],
            Param("self_recovery_rate"),
        )
    )
    model.add_flow(
        TransitionFlow(
            "tb_mortality",
            STATE["active"] & CLINICAL["clin"],
            STATE["mtb_naive"] & AGE["0"],
            0.0,
            adjust=(
                Overwrite(Param("tb_mortality_rate_lowinf"), where=INFECTIOUS["low"]),
                Overwrite(Param("tb_mortality_rate_inf"), where=INFECTIOUS["high"]),
            ),
        )
    )
    model.add_flow(
        TransitionFlow(
            "containment",
            STATE["incipient"],
            STATE["contained"],
            0.0,
            adjust=_latency_adjust("containment_rate"),
        )
    )
    # Dest adjustments split progressions by infectiousness. Compile scatters
    # 1/2 onto the two strata; the default path restores weight 1 so
    # ``[1 - p, p]`` is the whole split. ``match_golden`` keeps the half.
    model.add_flow(
        TransitionFlow(
            "progression",
            STATE["incipient"],
            CLINICAL["subclin"],
            0.0,
            adjust=_latency_adjust("progression_rate")
            + (
                Multiply(
                    1.0 - Param("progression_prop_infectious"),
                    where=Dest(INFECTIOUS["low"]),
                ),
                Multiply(
                    Param("progression_prop_infectious"),
                    where=Dest(INFECTIOUS["high"]),
                ),
            ),
        )
    )
    model.add_flow(
        TransitionFlow(
            "ageing",
            Everything(),
            Everything(),
            1.0,
            pairing=TraitChain.from_breakpoints(AGE),
        )
    )
    model.add_flow(
        TransitionFlow(
            "seed_peak",
            STATE["mtb_naive"],
            STATE["incipient"],
            _seed_peak_rate(),
        )
    )
    model.add_flow(
        TransitionFlow(
            "detection",
            STATE["active"] & CLINICAL["clin"],
            STATE["treatment"],
            defer(_detection_rate)(
                _calendar(),
                Param("detect_rate_current"),
                Param("rel_detect_2007"),
                Param("rel_detect_1986"),
                Param("rel_detect_2021"),
            ),
        )
    )
    death_rate = _DEATH_RATE.interp(arg=_calendar())
    for age in AGES:
        model.add_flow(
            TransitionFlow(
                f"nontb_deaths_{age}",
                AGE[age],
                STATE["mtb_naive"] & AGE["0"],
                death_rate,
            )
        )
    # Absolute inflow. Rates from the UN increment series can be negative;
    # EntryFlow scatters the value as-is.
    model.add_flow(
        EntryFlow(
            "entry_births",
            STATE["mtb_naive"] & AGE["0"],
            FieldRef(("entry_rate",)),
        )
    )
    model.add_flow(
        TransitionFlow(
            "success",
            STATE["treatment"],
            STATE["recovered"],
            FieldRef(("success_rate",)),
        )
    )
    model.add_flow(
        TransitionFlow(
            "relapse",
            STATE["treatment"],
            CLINICAL["subclin"] & INFECTIOUS["low"],
            FieldRef(("relapse_rate",)),
        )
    )
    model.add_flow(
        TransitionFlow(
            "rx_death",
            STATE["treatment"],
            STATE["mtb_naive"] & AGE["0"],
            FieldRef(("rx_death_rate",)),
        )
    )
    model.add_flow(
        TransitionFlow(
            "acf",
            STATE["active"],
            STATE["treatment"],
            FieldRef(("acf_rate",)),
        )
    )
    model.set_initial_population({STATE["mtb_naive"] & AGE[age]: POP_PER_AGE for age in AGES})
    return model


def _derived_field(params: dict[str, Any], *, y: Any, t: Any) -> dict[str, Any]:
    """Default corrected path: spline mixing + smooth entry."""
    return force_of_infection(params, y=y, t=t, mixing=spline_mixing, entry=entry_rate_at)


def _golden_derived_field(params: dict[str, Any], *, y: Any, t: Any) -> dict[str, Any]:
    """Golden path: year-frozen matrix + step entry (summer3 vector field pieces)."""
    return force_of_infection(params, y=y, t=t, mixing=None, entry=birth_rate_at)


def _copy_summer3_destination_index(compiled: Any) -> None:
    """Point each many-to-one edge at compartment ``i``, not the declared dest.

    summer3 stores the right compartment object and the wrong index. The
    runner uses the index. Must run before the vector field is traced.
    """
    for name in _INDEX_BUG_FLOWS:
        flow = compiled.flows[name]
        count = int(np.asarray(flow.src_idx).size)
        indices = np.arange(count, dtype=np.int32)
        indices.flags.writeable = False
        object.__setattr__(flow, "dest_idx", indices)


def _use_full_progression_weight(compiled: Any) -> None:
    """Drop the extra ``1/2`` scatter so ``[1 - p, p]`` is the whole split."""
    flow = compiled.flows["progression"]
    ones = np.ones(np.asarray(flow.weight).shape, dtype=np.float64)
    ones.flags.writeable = False
    object.__setattr__(flow, "weight", ones)


def get_compiled(*, match_golden: bool = False) -> Any:
    """Compile once per mode. ``derived_fn`` is module-level so the jit cache is stable.

    Default is the docstring model with folded-spline contact and smooth entry.
    ``match_golden=True`` copies the three summer3 bugs and keeps the year-frozen
    matrix plus step entry so trajectories track ``tests/golden/``.
    """
    if match_golden not in _COMPILED:
        field = _golden_derived_field if match_golden else _derived_field
        compiled = build_model(match_golden=match_golden).compile(derived_fn=field)
        if match_golden:
            _copy_summer3_destination_index(compiled)
        else:
            _use_full_progression_weight(compiled)
        _COMPILED[match_golden] = compiled
    return _COMPILED[match_golden]


def save_plan(compiled: Any) -> SavePlan:
    """Compartments and flows only.

    An empty ``SavePlan`` also saves every parameter path. Those paths can
    share names with flows and would overwrite the flow outputs.
    """
    requests = {"compartments": SaveRequest(Compartments())}
    for name in compiled.order:
        requests[name] = SaveRequest(FlowMass(name))
    return SavePlan(requests=requests, ts=np_save_ts())


def np_save_ts() -> Any:
    return np.arange(N_TIMES)


def compile_model(
    *,
    match_golden: bool = False,
    mixing: Any = _DEFAULT,
    entry: Any = _DEFAULT,
) -> Any:
    """Compile a model that is not stored in the module cache.

    Default (``match_golden=False``): :func:`spline_mixing` and
    :data:`entry_rate_at` (smooth). ``match_golden=True``: ``get_norm_c_matrix``
    and the step entry lookup. Pass ``mixing`` / ``entry`` to override.
    Arrays on ``params`` are traced inputs: a new table of the same shape does
    not recompile. Callers that use the spline should fold via
    :func:`fold_spline_contact` (or :func:`prepare_run_params`) once per draw.
    """
    if mixing is _DEFAULT:
        mixing = None if match_golden else spline_mixing
    if entry is _DEFAULT:
        entry = birth_rate_at if match_golden else entry_rate_at

    def field(params: dict[str, Any], *, y: Any, t: Any) -> dict[str, Any]:
        return force_of_infection(params, y=y, t=t, mixing=mixing, entry=entry)

    compiled = build_model(match_golden=match_golden).compile(derived_fn=field)
    if match_golden:
        _copy_summer3_destination_index(compiled)
    else:
        _use_full_progression_weight(compiled)
    return compiled


def solve(compiled: Any, params: dict[str, Any], **solver: Any) -> Any:
    """Integrate with the same default solver settings as :func:`run`.

    Keyword arguments are forwarded to ``CompiledModel.run``. Pass a
    :class:`~summer4.solvers.Diffrax` backend (see :mod:`vacf.solvers`) to
    set ``dtmax`` or other ``PIDController`` knobs. Defaults stay on the tight
    uncapped tolerances used for forward-run goldens.
    """
    from vacf.solvers import FORWARD_SOLVER

    settings: dict[str, Any] = {
        "t0": 0.0,
        "t1": float(N_TIMES),
        "dt": 0.1,
        "save": save_plan(compiled),
        **FORWARD_SOLVER,
    }
    settings.update(solver)
    return compiled.run(params, **settings)


def run(params: dict[str, Any], *, match_golden: bool = False, **solver: Any) -> Any:
    """Integrate index time ``0 … 301`` with Dopri5.

    Default sends replacement deaths to age 0, lets the progression split
    sum to one, applies ``rel_infectiousness_subclin`` to subclinical
    disease, uses the folded spline contact matrix, and a smooth entry rate.
    ``match_golden=True`` copies the three summer3 bugs and the year-frozen
    matrix plus step entry so the run tracks ``tests/golden/``.

    Defaults are the golden settings: ``rtol=atol=1e-9``, uncapped step,
    ``max_steps=20_000``. Calibration uses :data:`vacf.solvers.CALIBRATION_SOLVER`
    (``rtol=atol=1e-5``, ``dtmax=1``, ``max_steps=4000``). Pass ``**solver`` to
    override.
    """
    prepared = prepare_run_params(params, match_golden=match_golden)
    return solve(get_compiled(match_golden=match_golden), prepared, **solver)


def calendar_years() -> Any:
    import numpy as np

    return np.arange(START_TIME, END_TIME, 1.0)
