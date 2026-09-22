"""Inspect-notebook indicators, computed the way ``outputs.py`` on the branch does.

Remote calibration also pickles scenario outputs under the summer3 indicator
names (``pulm_prev``, ``age_pop``, ``treatment``, …) so
``notebooks/07-InspectMassiveRun.ipynb`` can load them unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from arviz import InferenceData
from jax import numpy as jnp

from tb_macro.calibration import CALIBRATION_SOLVER, MixingCase, build_mixing_cases, compile_case
from tb_macro.constants import (
    AGES,
    AGE_STRATA,
    DISEASE_STATES,
    INFECTED_STATES,
    N_TIMES,
    PREV_STATES,
    START_TIME,
)
from tb_macro.model import PMAP, get_compiled, solve
from tb_macro.parameters import BASE_PARAMS
from tb_macro.utils import add_midyear_points


def _data(output: Any) -> Any:
    values = output.values
    return values.data if hasattr(values, "data") else values


def flow_total(result: Any, name: str) -> Any:
    """Sum a flow's per-edge instantaneous mass. Shape ``(time,)``."""
    return jnp.sum(_data(result[name]), axis=-1)


def _rows_for_age(src: Any, age: str) -> Any:
    rows = [
        i
        for i, comp in enumerate(src)
        if "age" in PMAP.to_dicts()[int(comp)] and PMAP.to_dicts()[int(comp)]["age"] == age
    ]
    return jnp.asarray(rows, dtype=jnp.int32)


def flow_by_age(result: Any, name: str) -> dict[str, Any]:
    """Sum flow mass by the source compartment's age."""
    compiled = get_compiled()
    src = compiled.flows[name].src_idx
    data = _data(result[name])
    out: dict[str, Any] = {}
    for age in AGES:
        rows = _rows_for_age(src, age)
        out[age] = jnp.sum(data[:, rows], axis=-1) if rows.size else jnp.zeros(data.shape[0])
    return out


def compartments_by_label(result: Any) -> dict[str, Any]:
    """summer3 labels: stratum values joined in stratification order."""
    data = _data(result["compartments"])
    order = ("disease_state", "age", "infectious", "clinical")
    labelled: dict[str, Any] = {}
    for row, traits in enumerate(PMAP.to_dicts()):
        name = "_".join(traits[key] for key in order if key in traits)
        labelled[name] = data[:, row]
    return labelled


def _sum_states_by_age(result: Any, states: tuple[str, ...]) -> dict[str, Any]:
    data = _data(result["compartments"])
    out = {age: jnp.zeros(data.shape[0]) for age in AGES}
    for row, traits in enumerate(PMAP.to_dicts()):
        if traits["disease_state"] not in states:
            continue
        out[traits["age"]] = out[traits["age"]] + data[:, row]
    return out


def age_population(result: Any) -> dict[str, Any]:
    """Total population by modelled age."""
    return _sum_states_by_age(result, DISEASE_STATES)


def age_prevalence(result: Any) -> dict[str, Any]:
    """``active`` plus ``treatment``, by age."""
    return _sum_states_by_age(result, PREV_STATES)


def age_latent(result: Any) -> dict[str, Any]:
    """Everyone not ``mtb_naive``, by age."""
    return _sum_states_by_age(result, INFECTED_STATES)


def age_treatment(result: Any) -> dict[str, Any]:
    """Population in the treatment compartment, by age."""
    return _sum_states_by_age(result, ("treatment",))


def age_pulmonary_prevalence(result: Any) -> dict[str, Any]:
    """High infectious, plus ``prop_lowinf_bactpos`` of low, plus on treatment.

    The original multiplies by ``BASE_PARAMS["prop_lowinf_bactpos"]``, not the
    run's parameter. None of the parity fixtures change that value.
    """
    data = _data(result["compartments"])
    high = {age: jnp.zeros(data.shape[0]) for age in AGES}
    low = {age: jnp.zeros(data.shape[0]) for age in AGES}
    on_rx = {age: jnp.zeros(data.shape[0]) for age in AGES}
    frac = BASE_PARAMS["prop_lowinf_bactpos"]
    for row, traits in enumerate(PMAP.to_dicts()):
        age = traits["age"]
        column = data[:, row]
        if traits.get("infectious") == "high":
            high[age] = high[age] + column
        elif traits.get("infectious") == "low":
            low[age] = low[age] + column
        if traits["disease_state"] == "treatment":
            on_rx[age] = on_rx[age] + column
    return {age: high[age] + low[age] * frac + on_rx[age] for age in AGES}


def age_incidence(result: Any) -> dict[str, Any]:
    """Progression into active disease, by source age."""
    return flow_by_age(result, "progression")


def age_notifications(result: Any) -> dict[str, Any]:
    """Routine detection plus ACF, by source age."""
    routine = flow_by_age(result, "detection")
    acf = flow_by_age(result, "acf")
    return {age: routine[age] + acf[age] for age in AGES}


def age_deaths(result: Any) -> dict[str, Any]:
    """Community TB deaths plus deaths on treatment, by source age."""
    community = flow_by_age(result, "tb_mortality")
    treated = flow_by_age(result, "rx_death")
    return {age: community[age] + treated[age] for age in AGES}


def disease_state_totals(result: Any) -> dict[str, Any]:
    """Compartment sums by disease state, for the notebook plot."""
    data = _data(result["compartments"])
    totals: dict[str, Any] = {}
    for row, traits in enumerate(PMAP.to_dicts()):
        state = traits["disease_state"]
        column = data[:, row]
        totals[state] = column if state not in totals else totals[state] + column
    return totals


INDICATORS: tuple[tuple[str, Any], ...] = (
    ("incidence", age_incidence),
    ("prevalence", age_prevalence),
    ("pulmonary", age_pulmonary_prevalence),
    ("notifications", age_notifications),
    ("deaths", age_deaths),
    ("population", age_population),
)

# summer3 remote pickle keys (``scripts/massive/calibrate.py`` on tweak_cases_calib).
REMOTE_INDICATORS: tuple[tuple[str, Callable[[Any], dict[str, Any]]], ...] = (
    ("incidence", age_incidence),
    ("prevalence", age_prevalence),
    ("pulm_prev", age_pulmonary_prevalence),
    ("latent", age_latent),
    ("notifications", age_notifications),
    ("deaths", age_deaths),
    ("age_pop", age_population),
    ("treatment", age_treatment),
)


def _annual_years() -> np.ndarray:
    return np.arange(START_TIME, START_TIME + N_TIMES, dtype=float)


def age_dict_to_frame(by_age: dict[str, Any]) -> pd.DataFrame:
    """Annual age series to a mid-year-augmented DataFrame with summer3 columns."""
    frame = pd.DataFrame(
        {age: np.asarray(values) for age, values in by_age.items()},
        index=_annual_years(),
    )
    frame.columns = frame.columns.map(str)
    out = add_midyear_points(frame)
    if is_age_stratified_output(out):
        out.columns.name = "age_group"
    return out


def is_age_stratified_output(output: pd.DataFrame) -> bool:
    """Whether a single output frame has one column per modelled age group."""
    age_labels = {str(age) for age in AGE_STRATA}
    return len(output.columns) == len(age_labels) and set(map(str, output.columns)) == age_labels


def get_posterior_samples(idata: InferenceData, n_samples: int) -> Any:
    """Draw ``n_samples`` posterior points without replacement (seed 0)."""
    posterior = idata.posterior.stack(sample=("chain", "draw"))
    n_available = int(posterior.sizes["sample"])
    n_take = min(n_samples, n_available)
    rng = np.random.default_rng(0)
    idxs = rng.choice(n_available, size=n_take, replace=False)
    return posterior.isel(sample=idxs)


def save_sampled_outputs(
    path: Path,
    outputs: list[dict[str, list[pd.DataFrame]]],
    sample_labels: list[str],
) -> None:
    """Pickle nested outputs and sample labels from ``rerun_model_for_outputs``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"outputs": outputs, "sample_labels": sample_labels}, path)


def load_sampled_outputs(
    path: Path,
) -> tuple[list[dict[str, list[pd.DataFrame]]], list[str]]:
    """Load nested outputs and sample labels saved by ``save_sampled_outputs``."""
    payload = pd.read_pickle(path)
    return payload["outputs"], payload["sample_labels"]


def _spline_case() -> MixingCase:
    return next(case for case in build_mixing_cases() if case.name == "spline")


def rerun_model_for_outputs(
    idata: InferenceData,
    scen_params: list[dict[str, float]],
    samples: Any,
    *,
    case: MixingCase | None = None,
) -> tuple[list[dict[str, list[pd.DataFrame]]], list[str]]:
    """Re-run the calibrated model on accepted draws for each scenario.

    Uses the folded spline contact matrix (same path as the remote likelihood).
    Solver output is annual on 1 January; mid-year points are inserted so each
    indicator sits on a six-month grid.
    """
    mixing_case = case if case is not None else _spline_case()
    compiled = compile_case(mixing_case)
    sample_labels: list[str] = []
    outputs: list[dict[str, list[pd.DataFrame]]] = [
        {name: [] for name, _ in REMOTE_INDICATORS} for _ in scen_params
    ]
    for i in range(int(samples.sizes["sample"])):
        label = f"chain_{int(samples['chain'][i])}/draw_{int(samples['draw'][i])}"
        sample_labels.append(label)
        drawn = {name: float(samples[name].isel(sample=i)) for name in idata.posterior.data_vars}
        for scenario_index, scenario in enumerate(scen_params):
            params = dict(BASE_PARAMS)
            params.update(drawn)
            params.update(scenario)
            if mixing_case.fold is not None:
                params.update(mixing_case.fold(params))
            result = solve(compiled, params, **CALIBRATION_SOLVER)
            for name, func in REMOTE_INDICATORS:
                outputs[scenario_index][name].append(age_dict_to_frame(func(result)))
    return outputs, sample_labels
