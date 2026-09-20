"""Vietnam TB model from ``tweak_cases_calib``, on summer4.

Solver ``t`` is the summer3 index ``0 … 300``. Calendar year is ``t + 1800``.
``acf_coverage`` is 0 in ``BASE_PARAMS``, so the ACF flow is identically zero
on the default run.
"""

from __future__ import annotations

from typing import Any

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
    Transform,
    TransitionFlow,
)
from tb_macro.constants import (
    ACF_MIN_AGE,
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
from tb_macro.curves import get_cos_multicurve, get_scale_data, get_triang_vals
from tb_macro.demography import birth_rate_at
from tb_macro.mixing import get_norm_c_matrix
from tb_macro.static_data import StaticInputs, load_static

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
# lands in compartment i. See reference/summer3-bugs.md.
_INDEX_BUG_FLOWS: tuple[str, ...] = tuple(f"nontb_deaths_{age}" for age in AGES) + (
    "tb_mortality",
    "rx_death",
)
_AGE_BREAKS = jnp.array(AGE_STRATA, dtype=jnp.float64)
# Loaded at import, never inside ``derived_fn``. A lazy load under jit leaks tracers.
STATIC: StaticInputs = load_static()


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
    # GLUE(KI10): replaced by WP13 — Interp is scalar-only, so each age is
    # its own jnp.interp. Eight ages, unrolled. Folding the eight into one
    # searchsorted was measured and is not faster: XLA already shares the
    # search between them, and the strided row gather it needs costs more
    # than the eight contiguous column gathers. See plans/mixing-spline.md.
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
    release: bool = False,
    mixing: Any = None,
) -> dict[str, Any]:
    """Per-step mixing matrix, FOI, death rates, treatment hazards, ACF.

    # GLUE(TM5): replaced by WP14. Infectiousness is two properties, under-15s
    # do not transmit, and the contact matrix is rebuilt every step
    # (spectral radius included). ``ForceOfInfection`` cannot say that.
    # GLUE(KI6): the matrix is ``get_norm_c_matrix`` from the original file.
    ``mixing(calendar, params)``, when set, replaces that call. ``run`` does not set it.
    """
    static = STATIC
    calendar = t + START_TIME
    pop = PropertyData(PMAP, y)
    # CLIN_STRATA is (clin, subclin). summer3 wraps
    # [rel_infectiousness_subclin, 1] in that order, so the name does not
    # match the stratum. The release model puts the weight on subclin.
    # See reference/summer3-bugs.md.
    inf_w = jnp.array([params["rel_infectiousness_lowinf"], 1.0])
    rel_sub = params["rel_infectiousness_subclin"]
    clin_w = jnp.array([1.0, rel_sub]) if release else jnp.array([rel_sub, 1.0])
    weighted = (
        pop.keep(STATE["active"], 0.0)
        * pop.broadcast_over(INFECTIOUS, inf_w)
        * pop.broadcast_over(CLINICAL, clin_w)
    )
    ipops = weighted.sum_over(AGE).data
    total = pop.sum_over(AGE).data
    age_infect = jnp.where(_AGE_BREAKS < YOUNG_END_AGE, 0.0, 1.0)
    pressure = age_infect * ipops / total
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
    foi = mixing_matrix @ pressure
    deaths = _interp_deaths(calendar, static.death_times, static.death_rates)
    tsr = get_cos_multicurve(calendar, static.tsr_x, static.tsr_y)
    unsucc = get_cos_multicurve(calendar, static.unsucc_x, static.unsucc_y)
    success, relapse, rx_death = _outcome_rates(params["rx_duration"], unsucc, tsr, deaths)
    screen = _acf_screen(calendar, params)
    acf_inf = jnp.array(
        [
            screen * params["acf_sensitivity_low"] * params["prop_lowinf_bactpos"],
            screen * params["acf_sensitivity_high"],
        ]
    )
    age_mask = jnp.where(_AGE_BREAKS >= ACF_MIN_AGE, 1.0, 0.0)
    acf = (pop.broadcast_over(INFECTIOUS, acf_inf) * pop.broadcast_over(AGE, age_mask)).keep(
        STATE["active"], 0.0
    )

    derived = dict(params)
    derived["foi"] = pop.broadcast_over(AGE, foi)
    derived["success_rate"] = pop.broadcast_over(AGE, success)
    derived["relapse_rate"] = pop.broadcast_over(AGE, relapse)
    derived["rx_death_rate"] = pop.broadcast_over(AGE, rx_death)
    derived["acf_rate"] = acf
    derived["entry_rate"] = birth_rate_at(t, static.entry_times, static.entry_rates, START_TIME)
    for index, age in enumerate(AGES):
        derived[f"death_{age}"] = deaths[index]
    return derived


def _triang_rate(prev: Any, calendar: Any, peak_time: Any, peak_height: Any, width: Any) -> Any:
    del prev
    return get_triang_vals(calendar, peak_time, peak_height, width)


def _detection_rate(
    prev: Any,
    calendar: Any,
    current: Any,
    rel_2007: Any,
    rel_1986: Any,
    rel_2021: Any,
) -> Any:
    """Cosine detection scale-up. 1986 is current × rel_2007 × rel_1986."""
    del prev
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


def _ageing_pairing() -> TraitChain:
    # GLUE(TM3): replaced by WP13 — TraitChain.from_breakpoints.
    pairs = tuple((str(AGE_STRATA[i]), str(AGE_STRATA[i + 1])) for i in range(len(AGE_STRATA) - 1))
    widths = tuple(1.0 / (AGE_STRATA[i + 1] - AGE_STRATA[i]) for i in range(len(AGE_STRATA) - 1))
    return TraitChain(AGE, pairs, rates=widths)


def build_model() -> FlowModel:
    """Declare the ``tweak_cases_calib`` flows. Names match summer3."""
    model = FlowModel(PMAP)
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
                FieldRef(("foi",)),
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
    # summer3 scatters 1/2 onto the two infectious strata, then multiplies by
    # [1 - p, p]. The static edge weight stays 1/2; the parameter is a dest
    # adjustment on top, not instead.
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
            pairing=_ageing_pairing(),
        )
    )
    # GLUE(TM4): calendar seed. Time() is the index.
    model.add_flow(
        TransitionFlow(
            "seed_peak",
            STATE["mtb_naive"],
            STATE["incipient"],
            0.0,
            adjust=(
                Transform(
                    _triang_rate,
                    Time() + START_TIME,
                    Param("seed_peak_time"),
                    Param("seed_peak_rate"),
                    Param("seed_duration"),
                ),
            ),
        )
    )
    # GLUE(TM4): cosine detection. No second constant detection flow.
    model.add_flow(
        TransitionFlow(
            "detection",
            STATE["active"] & CLINICAL["clin"],
            STATE["treatment"],
            0.0,
            adjust=(
                Transform(
                    _detection_rate,
                    Time() + START_TIME,
                    Param("detect_rate_current"),
                    Param("rel_detect_2007"),
                    Param("rel_detect_1986"),
                    Param("rel_detect_2021"),
                ),
            ),
        )
    )
    for age in AGES:
        model.add_flow(
            TransitionFlow(
                f"nontb_deaths_{age}",
                AGE[age],
                STATE["mtb_naive"] & AGE["0"],
                FieldRef((f"death_{age}",)),
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


def _parity_field(params: dict[str, Any], *, y: Any, t: Any) -> dict[str, Any]:
    """Golden-matching field. Copies the summer3 clinical-weight wrap."""
    return force_of_infection(params, y=y, t=t, release=False)


def _release_field(params: dict[str, Any], *, y: Any, t: Any) -> dict[str, Any]:
    """Docstring field. ``rel_infectiousness_subclin`` multiplies ``subclin``."""
    return force_of_infection(params, y=y, t=t, release=True)


def _copy_summer3_destination_index(compiled: Any) -> None:
    """Point each many-to-one edge at compartment ``i``, not the declared dest.

    summer3 stores the right compartment object and the wrong index. The
    runner uses the index. Must run before the vector field is traced.
    """
    import numpy as np

    for name in _INDEX_BUG_FLOWS:
        flow = compiled.flows[name]
        count = int(np.asarray(flow.src_idx).size)
        indices = np.arange(count, dtype=np.int32)
        indices.flags.writeable = False
        object.__setattr__(flow, "dest_idx", indices)


def _use_full_progression_weight(compiled: Any) -> None:
    """Drop the extra ``1/2`` scatter so ``[1 - p, p]`` is the whole split."""
    import numpy as np

    flow = compiled.flows["progression"]
    ones = np.ones(np.asarray(flow.weight).shape, dtype=np.float64)
    ones.flags.writeable = False
    object.__setattr__(flow, "weight", ones)


def get_compiled(release: bool = False) -> Any:
    """Compile once per mode. ``derived_fn`` is module-level so the jit cache is stable.

    ``release=False`` matches the summer3 golden, bugs included.
    ``release=True`` is the model the docstrings describe.
    """
    if release not in _COMPILED:
        field = _release_field if release else _parity_field
        compiled = build_model().compile(derived_fn=field)
        if release:
            _use_full_progression_weight(compiled)
        else:
            _copy_summer3_destination_index(compiled)
        _COMPILED[release] = compiled
    return _COMPILED[release]


def save_plan(compiled: Any) -> SavePlan:
    """Compartments and flows only.

    An empty ``SavePlan`` also saves every parameter path. Those paths can
    share names with flows and would overwrite the flow traces.
    """
    requests = {"compartments": SaveRequest(Compartments())}
    for name in compiled.order:
        requests[name] = SaveRequest(FlowMass(name))
    return SavePlan(requests=requests, ts=np_save_ts())


def np_save_ts() -> Any:
    import numpy as np

    return np.arange(N_TIMES)


def compile_model(*, release: bool = False, mixing: Any = None) -> Any:
    """Compile a model that is not stored in the golden cache.

    ``mixing(calendar, params)`` replaces ``get_norm_c_matrix`` when it is not
    ``None``. ``run`` never passes one, so the golden path stays on the
    baseline matrix. Arrays on ``params`` are traced inputs: a new table of
    the same shape does not recompile.
    """

    def field(params: dict[str, Any], *, y: Any, t: Any) -> dict[str, Any]:
        return force_of_infection(params, y=y, t=t, release=release, mixing=mixing)

    compiled = build_model().compile(derived_fn=field)
    if release:
        _use_full_progression_weight(compiled)
    else:
        _copy_summer3_destination_index(compiled)
    return compiled


def solve(compiled: Any, params: dict[str, Any], **solver: Any) -> Any:
    """Integrate with the same default solver settings as :func:`run`.

    Keyword arguments are forwarded to ``CompiledModel.run``. That includes
    the diffrax controls installed by :mod:`tb_macro.step_controls` (``dtmax``,
    ``dtmin``, the PID coefficients, or a ``stepsize_controller``). Defaults
    stay on the golden tolerances.
    """
    settings: dict[str, Any] = {
        "t0": 0.0,
        "t1": float(N_TIMES),
        "dt": 0.1,
        "solver": "dopri5",
        "rtol": 1e-9,
        "atol": 1e-9,
        "max_steps": 20_000,
        "save": save_plan(compiled),
    }
    settings.update(solver)
    return compiled.run(params, **settings)


def run(params: dict[str, Any], *, release: bool = False, **solver: Any) -> Any:
    """Integrate index time ``0 … 301`` with Dopri5.

    ``release=False`` copies the three summer3 bugs and is what the goldens
    record. ``release=True`` sends replacement deaths to age 0, lets the
    progression split sum to one, and applies ``rel_infectiousness_subclin``
    to subclinical disease.

    summer4's ``run`` accepts ``dtmax`` and the other ``PIDController`` knobs
    via :mod:`tb_macro.step_controls`. This function does not pass them: the
    goldens were generated without a step cap, at ``rtol=atol=1e-9``. The
    tb_macroeconomics notebooks are looser. They pass only ``max_steps=4000``
    and inherit the summer3 runner: Dopri5, ``dt0=0.1``,
    ``PIDController(rtol=1e-5, atol=1e-5, dtmax=1)``. Pass those through
    ``**solver`` to match a notebook, not a golden.
    """
    return solve(get_compiled(release=release), params, **solver)


def calendar_years() -> Any:
    import numpy as np

    return np.arange(START_TIME, END_TIME, 1.0)
