"""Build and run the three-age TB macro model on summer4.

``passive_detection`` (constant ``detection``) and ``detection`` (time-varying
scale-up on clinical active disease) both move ``active`` into ``treatment``.
Both are kept so the trajectory matches the original library. They are not a
modelling recommendation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from jax import numpy as jnp

from summer4 import (
    Compartments,
    Everything,
    FieldRef,
    FlowMass,
    FlowModel,
    Multiply,
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
    AGES,
    BASE_PARAMS,
    CLINICAL_STRATA,
    DISEASE_STATES,
    END_YEAR,
    INFECT_COMPS,
    INFECTIOUS_STRATA,
    POP_PER_AGE,
    START_YEAR,
)
from tb_macro.functions import get_triang_vals, tanh_based_scaleup

STATE = Property("disease_state", DISEASE_STATES)
AGE = Property("age", AGES)
INFECTIOUS = Property("infectious", INFECTIOUS_STRATA)
CLINICAL = Property("clinical", CLINICAL_STRATA)

PMAP: PropertyMap = (
    PropertyMap.from_property(STATE)
    .stratify(AGE)
    .stratify(INFECTIOUS, STATE["active"])
    .stratify(CLINICAL, STATE["active"])
)

SAVE_TS: np.ndarray = np.arange(START_YEAR, END_YEAR + 1.0)
_MIXING = jnp.ones((len(AGES), len(AGES)))
_COMPILED: Any = None


def force_of_infection(params: dict[str, Any], *, y: Any, t: Any) -> dict[str, Any]:
    """Per-age FOI ``ones(3, 3) @ (I_age / N_age ** exp) * contact_rate``.

    ``I`` is unweighted ``active``. The struct passes every parameter through
    so ``Param`` still resolves, and adds ``foi`` broadcast over age.

    # GLUE(TM5): replaced by WP14 — ForceOfInfection(kind="generalised").
    """
    del t
    pop = PropertyData(PMAP, y)
    infectious = pop.keep(STATE["active"], 0.0).sum_over(AGE).data
    total = pop.sum_over(AGE).data
    pressure = infectious / total ** params["freq_dens_exponent"]
    foi = _MIXING @ pressure * params["contact_rate"]
    derived = dict(params)
    derived["foi"] = pop.broadcast_over(AGE, foi)
    return derived


def _triang_rate(prev: Any, t_index: Any, peak_time: Any, peak_height: Any, width: Any) -> Any:
    """``Transform`` passes the previous rate; the pulse replaces it."""
    del prev
    return get_triang_vals(t_index, peak_time, peak_height, width)


def _detection_rate(
    prev: Any,
    t_index: Any,
    recent: Any,
    shape: Any,
    inflection: Any,
    past_frac: Any,
) -> Any:
    """``recent_detection_rate * tanh_based_scaleup(..., 1)``."""
    del prev
    return recent * tanh_based_scaleup(t_index, shape, inflection, past_frac, 1.0)


def _seed_adjust() -> tuple[Transform, ...]:
    # GLUE(TM4): replaced by WP13. Time() is the calendar year; the original
    # functions take the save index, which is year - 1800.
    return (
        Transform(
            _triang_rate,
            Time() - START_YEAR,
            Param("seed_peak_time"),
            Param("seed_peak_rate"),
            Param("seed_duration"),
        ),
    )


def _detection_adjust() -> tuple[Transform, ...]:
    # GLUE(TM4): replaced by WP13.
    return (
        Transform(
            _detection_rate,
            Time() - START_YEAR,
            Param("recent_detection_rate"),
            Param("passive_detection_shape"),
            Param("passive_detection_inflection"),
            Param("passive_detection_past_frac"),
        ),
    )


def build_model() -> FlowModel:
    """Declare the twenty original flows. Names are unchanged."""
    model = FlowModel(PMAP)
    for comp in INFECT_COMPS:
        model.add_flow(
            TransitionFlow(
                f"infect_{comp}",
                STATE[comp],
                STATE["incipient"],
                FieldRef(("foi",)),
                adjust=(Multiply(Param(f"rel_sus_{comp}")),),
            )
        )
    model.add_flow(
        TransitionFlow("containment", STATE["incipient"], STATE["contained"], Param("contain"))
    )
    model.add_flow(
        TransitionFlow("clearance", STATE["contained"], STATE["cleared"], Param("clearance_rate"))
    )
    model.add_flow(
        TransitionFlow("breakdown", STATE["contained"], STATE["incipient"], Param("breakdown_rate"))
    )
    # Destination leaves infectious unspecified, so each edge weight is 1/2.
    model.add_flow(
        TransitionFlow(
            "progression",
            STATE["incipient"],
            STATE["active"] & CLINICAL["subclin"],
            Param("progression"),
        )
    )
    model.add_flow(
        TransitionFlow(
            "increase_infectiousness",
            INFECTIOUS["low"],
            INFECTIOUS["high"],
            Param("increase_infect"),
        )
    )
    model.add_flow(
        TransitionFlow(
            "decrease_infectiousness",
            INFECTIOUS["high"],
            INFECTIOUS["low"],
            Param("decrease_infect"),
        )
    )
    model.add_flow(
        TransitionFlow(
            "clinical_develop",
            CLINICAL["subclin"],
            CLINICAL["clin"],
            Param("clinical_development"),
        )
    )
    model.add_flow(
        TransitionFlow(
            "clinical_regress",
            CLINICAL["clin"],
            CLINICAL["subclin"],
            Param("clinical_regression"),
        )
    )
    model.add_flow(
        TransitionFlow(
            "self_recovery",
            STATE["active"] & CLINICAL["subclin"],
            STATE["recovered"],
            Param("self_recovery"),
        )
    )
    model.add_flow(
        TransitionFlow(
            "passive_detection",
            STATE["active"],
            STATE["treatment"],
            Param("detection"),
        )
    )
    model.add_flow(
        TransitionFlow(
            "treatment_recovery",
            STATE["treatment"],
            STATE["recovered"],
            Param("treatment_recovery"),
        )
    )
    model.add_flow(
        TransitionFlow(
            "treatment_relapse",
            STATE["treatment"],
            STATE["active"] & CLINICAL["subclin"] & INFECTIOUS["low"],
            Param("treatment_relapse"),
        )
    )
    # GLUE(TM3): replaced by WP13 — TraitChain.from_breakpoints(age).
    model.add_flow(
        TransitionFlow(
            "ageing_0_to_5",
            Everything(),
            Everything(),
            1.0,
            pairing=TraitChain(AGE, (("0", "5"),), rates=(1.0 / 5.0,)),
        )
    )
    model.add_flow(
        TransitionFlow(
            "ageing_5_to_15",
            Everything(),
            Everything(),
            1.0,
            pairing=TraitChain(AGE, (("5", "15"),), rates=(1.0 / 10.0,)),
        )
    )
    model.add_flow(
        TransitionFlow(
            "seed_peak",
            STATE["mtb_naive"],
            STATE["incipient"],
            0.0,
            adjust=_seed_adjust(),
        )
    )
    model.add_flow(
        TransitionFlow(
            "detection",
            STATE["active"] & CLINICAL["clin"],
            STATE["treatment"],
            0.0,
            adjust=_detection_adjust(),
        )
    )
    # TM6 is full: InitialPopulation. Each (mtb_naive, age) selector matches one
    # compartment, so the ragged even-split is not exercised by this seed.
    model.set_initial_population({STATE["mtb_naive"] & AGE[age]: POP_PER_AGE for age in AGES})
    return model


def get_compiled() -> Any:
    """Compile once. ``derived_fn`` is module-level so the jit cache is stable."""
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_model().compile(derived_fn=force_of_infection)
    return _COMPILED


def run(params: dict[str, Any]) -> Any:
    """Integrate 1800–1999 with Dopri5 at the golden tolerances."""
    compiled = get_compiled()
    # summer4 passes ``dt`` through as Dopri5's initial step. The summer3
    # runner uses dt0=0.1 (and dtmax=1, which summer4 cannot set). dt=1.0
    # leaves about 1e-6 relative error; 0.1 stays inside the parity tolerance.
    # The save grid is still one point per year.
    return compiled.run(
        params,
        t0=START_YEAR,
        t1=END_YEAR,
        dt=0.1,
        solver="dopri5",
        rtol=1e-9,
        atol=1e-9,
        max_steps=20_000,
        save=save_plan(compiled),
    )


def save_plan(compiled: Any) -> SavePlan:
    """Compartments and flows only.

    An empty ``SavePlan`` also saves every ``Param`` path. Those paths share
    names with flows (``detection``, ``progression``, ``self_recovery``,
    ``treatment_recovery``, ``treatment_relapse``) and would overwrite the
    flow traces.
    """
    requests = {"compartments": SaveRequest(Compartments())}
    for name in compiled.order:
        requests[name] = SaveRequest(FlowMass(name))
    return SavePlan(requests=requests, ts=SAVE_TS)
