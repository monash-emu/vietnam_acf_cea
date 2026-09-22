"""Summer4 port of the summer3 calibration likelihood.

The original notebook (``notebooks/04-Calibration.ipynb`` on
``tweak_cases_calib``) builds one log-density and hands it to NUTS. The
density is the sum of seven terms in ``calibration.make_log_likelihood``:
latent infection, notifications, TB deaths, adult pulmonary prevalence, the
prevalence decline between the two surveys, the high-infectious share of
prevalent disease, and the Canberra distance of the mixing matrix. This
module is that function on the summer4 run, with the same error models.

Counts and ratios are normal on the log scale with ``COUNT_LOG_SD``.
Proportions are normal on the logit scale with ``PROP_LOGIT_SD``. The
Canberra distance is normal around zero with standard deviation
``mixing_dist_sd``, which is itself a calibrated parameter. Multi-year
series are averaged so each series weighs about as much as a single point.
A solve that does not succeed contributes ``-1e10`` instead of the density.

``build_mixing_cases`` is the comparison set for that density. Every case
leaves ``bg_mixing``, ``a_spread`` and ``pc_strength`` free and folds them
once per draw. That is the calibration path: nothing is fixed at prepare.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import arviz as az
import numpy as np
from arviz import InferenceData
from jax import jit, lax
from jax import numpy as jnp
from numpyro import distributions as dist
from numpyro.infer import MCMC

from tb_macro.constants import (
    AGE_STRATA,
    INFECTED_STATES,
    ISO3,
    N_TIMES,
    START_TIME,
    YOUNG_END_AGE,
)
from tb_macro.contact import prepare_smooth, prepare_spline, prepare_stepwise
from tb_macro.contact.pchip import eval_jax
from tb_macro.contact.stepwise import year_index_jax
from tb_macro.inputs import get_norm_conmat, load_who_outcomes
from tb_macro.mixing import canberra_distance, get_norm_c_matrix, normalise_by_spectral_radius
from tb_macro.model import PMAP, STATIC, compile_model, save_plan
from tb_macro.parameters import BASE_PARAMS
from tb_macro.targets import (
    INF_PREV_TARGET,
    LATENT_TARGET,
    NOTIF_TARGET,
    PREV_DECLINE_TARGET,
    PULM_PREV_TARGET,
)

# Same widths as the summer3 calibration module.
COUNT_LOG_SD = 0.1
PROP_LOGIT_SD = 0.2
MIXING_TARGET_YEAR = 2025.0
_EPS = 1e-32
_FAILED = -1.0e10

# The calibration notebook passes only ``max_steps`` and inherits the summer3
# runner: Dopri5, ``dt0=0.1``, ``rtol=atol=1e-5``, ``dtmax=1``.
CALIBRATION_SOLVER: dict[str, Any] = {
    "t0": 0.0,
    "t1": float(N_TIMES),
    "dt": 0.1,
    "solver": "dopri5",
    "rtol": 1e-5,
    "atol": 1e-5,
    "max_steps": 4000,
    "dtmax": 1.0,
}

# Solver span. Before the weight table, weights clip and fertility still
# moves, which is what ``get_norm_c_matrix`` does. Stepwise and spline are
# built on this span so a draw sees the same years as the baseline.
_CONTACT_YEARS = np.arange(START_TIME, START_TIME + N_TIMES + 1.0)

TERM_NAMES: tuple[str, ...] = (
    "latent",
    "notifications",
    "deaths",
    "prevalence",
    "decline",
    "infectiousness",
    "mixing",
)

Fold = Callable[[dict[str, Any]], dict[str, Any]]
Mixing = Callable[[Any, dict[str, Any]], Any]


def _rows(predicate: Callable[[dict[str, str]], bool]) -> Any:
    chosen = [i for i, traits in enumerate(PMAP.to_dicts()) if predicate(traits)]
    if not chosen:
        raise ValueError("compartment selection is empty")
    return jnp.asarray(chosen, dtype=jnp.int32)


_ADULT = {str(age) for age in AGE_STRATA if age >= YOUNG_END_AGE}
ADULT_ROWS = _rows(lambda traits: traits["age"] in _ADULT)
LATENT_ROWS = _rows(
    lambda traits: traits["age"] in _ADULT and traits["disease_state"] in INFECTED_STATES
)
HIGH_ROWS = _rows(lambda traits: traits.get("infectious") == "high" and traits["age"] in _ADULT)
LOW_ROWS = _rows(lambda traits: traits.get("infectious") == "low" and traits["age"] in _ADULT)
TREATMENT_ROWS = _rows(
    lambda traits: traits["disease_state"] == "treatment" and traits["age"] in _ADULT
)


def _logit(x: Any) -> Any:
    """Log-odds, clipped off 0 and 1."""
    clipped = jnp.clip(x, _EPS, 1.0 - _EPS)
    return jnp.log(clipped) - jnp.log1p(-clipped)


def _log_normal_log_prob(modelled: Any, target: Any, sd: float = COUNT_LOG_SD) -> Any:
    """Normal log-density of ``log(modelled)`` around ``log(target)``."""
    modelled = jnp.maximum(modelled, _EPS)
    target = jnp.maximum(jnp.asarray(target), _EPS)
    return dist.Normal(jnp.log(target), sd).log_prob(jnp.log(modelled))


def _logit_normal_log_prob(modelled: Any, target: Any, sd: float = PROP_LOGIT_SD) -> Any:
    """Normal log-density of ``logit(modelled)`` around ``logit(target)``."""
    return dist.Normal(_logit(jnp.asarray(target)), sd).log_prob(_logit(modelled))


def _interp(values: Any, query_times: Any) -> Any:
    """Linear interpolation from annual solver output onto calendar times.

    Row ``k`` of a saved trace is calendar year ``START_TIME + k``.
    """
    series = jnp.squeeze(jnp.asarray(values))
    years = START_TIME + jnp.arange(series.shape[0], dtype=series.dtype)
    return jnp.interp(jnp.asarray(query_times, dtype=series.dtype), years, series)


def _output_data(result: Any, name: str) -> Any:
    values = result[name].values
    return values.data if hasattr(values, "data") else values


def _sum_rows(compartments: Any, rows: Any) -> Any:
    return jnp.sum(compartments[:, rows], axis=1)


def _flow_total(result: Any, name: str) -> Any:
    return jnp.sum(_output_data(result, name), axis=-1)


def _adult_pulmonary(compartments: Any, time: Any) -> tuple[Any, Any, Any]:
    """High-infectious count, pulmonary prevalence, and adult population.

    The low-infectious contribution uses ``BASE_PARAMS["prop_lowinf_bactpos"]``,
    not the draw. That parameter is not in the calibration prior, and the
    summer3 likelihood does the same.
    """
    frac = BASE_PARAMS["prop_lowinf_bactpos"]
    high = _interp(_sum_rows(compartments, HIGH_ROWS), time)
    low = _interp(_sum_rows(compartments, LOW_ROWS), time)
    on_treatment = _interp(_sum_rows(compartments, TREATMENT_ROWS), time)
    adult = _interp(_sum_rows(compartments, ADULT_ROWS), time)
    pulmonary = high + low * frac + on_treatment
    return high, pulmonary, adult


def _series(values: Any) -> tuple[Any, Any]:
    clean = values.dropna()
    times = jnp.asarray(clean.index.to_numpy(), dtype=jnp.float64)
    observed = jnp.asarray(clean.to_numpy(), dtype=jnp.float64)
    return times, observed


def _targets() -> dict[str, Any]:
    """Host arrays for the seven terms. Built once per likelihood."""
    decline = PREV_DECLINE_TARGET.sort_index()
    _, _, who_mort = load_who_outcomes(ISO3)
    death_times, death_values = _series(who_mort)
    return {
        "latent_time": jnp.asarray(LATENT_TARGET.index[0], dtype=jnp.float64),
        "latent_value": jnp.asarray(LATENT_TARGET.iloc[0] / 1e2, dtype=jnp.float64),
        "notif_times": jnp.asarray(NOTIF_TARGET.index.to_numpy(), dtype=jnp.float64),
        "notif_values": jnp.asarray(NOTIF_TARGET.to_numpy(), dtype=jnp.float64),
        "death_times": death_times,
        "death_values": death_values,
        "prev_time": jnp.asarray(PULM_PREV_TARGET.index[0], dtype=jnp.float64),
        "prev_value": jnp.asarray(PULM_PREV_TARGET.iloc[0] / 1e5, dtype=jnp.float64),
        "decline_base": jnp.asarray(decline.index[0], dtype=jnp.float64),
        "decline_end": jnp.asarray(decline.index[1], dtype=jnp.float64),
        "decline_value": jnp.asarray(decline.iloc[1] / decline.iloc[0], dtype=jnp.float64),
        "inf_time": jnp.asarray(INF_PREV_TARGET.index[0], dtype=jnp.float64),
        "inf_value": jnp.asarray(INF_PREV_TARGET.iloc[0], dtype=jnp.float64),
        "conmat": get_norm_conmat(ISO3),
    }


def _terms(
    compartments: Any,
    result: Any,
    matrix: Any,
    mixing_dist_sd: Any,
    targets: dict[str, Any],
) -> dict[str, Any]:
    """The seven log-density contributions. Each is a scalar."""
    latent = _interp(_sum_rows(compartments, LATENT_ROWS), targets["latent_time"])
    adult = _interp(_sum_rows(compartments, ADULT_ROWS), targets["latent_time"])
    notifications = _interp(_flow_total(result, "detection"), targets["notif_times"])
    deaths = _interp(_flow_total(result, "tb_mortality"), targets["death_times"])
    deaths = deaths + _interp(_flow_total(result, "rx_death"), targets["death_times"])
    _, pulmonary, pop = _adult_pulmonary(compartments, targets["prev_time"])
    _, end_prev, end_pop = _adult_pulmonary(compartments, targets["decline_end"])
    _, base_prev, base_pop = _adult_pulmonary(compartments, targets["decline_base"])
    high, inf_prev, _ = _adult_pulmonary(compartments, targets["inf_time"])
    end_prop = end_prev / (end_pop + _EPS)
    base_prop = base_prev / (base_pop + _EPS)
    distance = canberra_distance(matrix, targets["conmat"])
    return {
        "latent": _logit_normal_log_prob(latent / (adult + _EPS), targets["latent_value"]),
        "notifications": _log_normal_log_prob(notifications, targets["notif_values"]).mean(),
        "deaths": _log_normal_log_prob(deaths, targets["death_values"]).mean(),
        "prevalence": _logit_normal_log_prob(pulmonary / (pop + _EPS), targets["prev_value"]),
        "decline": _log_normal_log_prob(end_prop / (base_prop + _EPS), targets["decline_value"]),
        "infectiousness": _logit_normal_log_prob(high / (inf_prev + _EPS), targets["inf_value"]),
        "mixing": dist.Normal(0.0, mixing_dist_sd).log_prob(distance),
    }


def _baseline_matrix(params: dict[str, Any]) -> Any:
    static = STATIC
    return get_norm_c_matrix(
        static.age_weights,
        static.weight_ends,
        static.pops,
        static.pop_ends,
        static.fert,
        static.fert_ends,
        MIXING_TARGET_YEAR,
        params["bg_mixing"],
        params["a_spread"],
        params["pc_strength"],
    )


def make_log_likelihood(
    compiled: Any,
    *,
    mixing: Mixing | None = None,
    fold: Fold | None = None,
    solver: dict[str, Any] | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Log-likelihood of one parameter draw, and the seven terms that sum to it.

    ``mixing(calendar, params)`` replaces ``get_norm_c_matrix`` inside the
    solve and in the Canberra term, so both see the same matrix. ``fold``
    runs once per draw, before the solve, and its arrays are passed in
    through ``params``. Leave both as ``None`` for the baseline matrix.

    The returned function is jitted. ``total`` is the density NUTS should
    factor. It is ``-1e10`` when the solver does not report success. The
    predicate is stopped, so a failed solve contributes a zero cotangent
    rather than a NaN one.
    """
    settings = dict(CALIBRATION_SOLVER)
    if solver:
        settings.update(solver)
    settings["save"] = save_plan(compiled)
    targets = _targets()

    @jit
    def evaluate(params: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(params)
        if fold is not None:
            prepared.update(fold(prepared))
        result = compiled.run(prepared, **settings)
        compartments = _output_data(result, "compartments")
        if mixing is None:
            matrix = _baseline_matrix(prepared)
        else:
            matrix = mixing(MIXING_TARGET_YEAR, prepared)
        terms = _terms(compartments, result, matrix, prepared["mixing_dist_sd"], targets)
        total = jnp.zeros(())
        for name in TERM_NAMES:
            total = total + terms[name]
        ok = lax.stop_gradient(jnp.asarray(result.solver.result_code) == 0)
        evaluated = dict(terms)
        evaluated["total"] = jnp.where(ok, total, _FAILED)
        evaluated["ok"] = ok
        evaluated["num_steps"] = result.solver.num_steps
        evaluated["num_rejected"] = result.solver.num_rejected_steps
        return evaluated

    return evaluate


@dataclass(frozen=True, slots=True)
class MixingCase:
    """A contact matrix the calibration likelihood can be pointed at.

    ``fold`` is ``None`` for the baseline, which rebuilds the kernel in
    every stage. The other cases fold all three mixing scalars once per draw.
    """

    name: str
    mixing: Mixing | None
    fold: Fold | None


def _static_arrays() -> tuple[Any, ...]:
    static = STATIC
    return (
        static.age_weights,
        static.weight_ends,
        static.pops,
        static.pop_ends,
        static.fert,
        static.fert_ends,
    )


def _stepwise_case() -> MixingCase:
    contact = prepare_stepwise(*_static_arrays(), years=_CONTACT_YEARS)

    def mixing(calendar: Any, params: dict[str, Any]) -> Any:
        index = year_index_jax(calendar, contact.year0, contact.year1)
        return params["contact_table"][index]

    def fold(params: dict[str, Any]) -> dict[str, Any]:
        ready = contact.materialize(params["bg_mixing"], params["a_spread"], params["pc_strength"])
        return {"contact_table": ready.table}

    return MixingCase("stepwise", mixing, fold)


def _smooth_case() -> MixingCase:
    """PCHIP of the inputs, folded, with the eigenproblem still in the stage."""
    contact = prepare_smooth(*_static_arrays())
    knot_x = contact.knot_x
    pop_y = contact.pop_y
    pop_m = contact.pop_m

    def mixing(calendar: Any, params: dict[str, Any]) -> Any:
        mixed = eval_jax(calendar, knot_x, params["offset_y"], params["offset_m"])
        pop = eval_jax(calendar, knot_x, pop_y, pop_m)
        return normalise_by_spectral_radius(mixed * pop[None, :])

    def fold(params: dict[str, Any]) -> dict[str, Any]:
        ready = contact.materialize(params["bg_mixing"], params["a_spread"], params["pc_strength"])
        return {"offset_y": ready.offset_y, "offset_m": ready.offset_m}

    return MixingCase("smooth", mixing, fold)


def _spline_case() -> MixingCase:
    contact = prepare_spline(*_static_arrays(), years=_CONTACT_YEARS)

    def mixing(calendar: Any, params: dict[str, Any]) -> Any:
        return params["curve"].at(calendar)

    def fold(params: dict[str, Any]) -> dict[str, Any]:
        ready = contact.materialize(params["bg_mixing"], params["a_spread"], params["pc_strength"])
        if ready.curve is None:
            raise ValueError("prepare_spline did not fit a curve")
        return {"curve": ready.curve}

    return MixingCase("spline", mixing, fold)


def build_mixing_cases() -> tuple[MixingCase, ...]:
    """Baseline and the three folded builders, all three matrix scalars free.

    Free in the stage is the same function as folded, at a higher price, so
    it is not a separate case. Smooth is fit on the weight table and is
    constant before 1950. Stepwise and spline cover the solver span.
    """
    return (
        MixingCase("baseline", None, None),
        _stepwise_case(),
        _smooth_case(),
        _spline_case(),
    )


def compile_case(case: MixingCase, *, release: bool = False) -> Any:
    """Compile one case. ``release=False`` is the model the notebook calibrated."""
    return compile_model(release=release, mixing=case.mixing)


MCMC_EXTRA_FIELDS: tuple[str, ...] = (
    "potential_energy",
    "energy",
    "num_steps",
    "accept_prob",
)


def inference_data_from_mcmc(mcmc: MCMC) -> InferenceData:
    """Convert a finished NumPyro MCMC run to ArviZ.

    ArviZ 1.x returns an xarray ``DataTree``. ``sample_stats.lp`` is already the
    joint log posterior (``-potential_energy``, higher better). Do not assign
    ``lp`` with the old ``InferenceData`` tuple syntax — on a ``DataTree`` that
    stores an object scalar and ``to_netcdf`` fails.
    """
    return az.from_numpyro(mcmc, log_likelihood=False)
