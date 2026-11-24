"""Regenerate calibration parity goldens with summer3wip.

Run via ``pixi run -e reference golden``. Evaluates the seven log-density
terms and their sum for ``base``, ``nuts_start``, and uniform samples inside
``PARAM_BOUNDS``. Do not edit the parquet/json to fit summer4.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import diffrax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "reference" / "snapshot"
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(SNAPSHOT / "src" / "vacf"))

from calibration_golden import (  # noqa: E402
    CALIBRATION_SEED,
    N_RANDOM_DRAWS,
    SUMMER3_CALIBRATION_SOLVER,
    TERM_NAMES,
    write_manifest,
    write_params,
    write_terms,
)
from vacf.calibration import (  # noqa: E402
    get_death_log_likelihood,
    get_infprop_log_likelihood,
    get_latent_log_likelihood,
    get_mixing_log_likelihood,
    get_notification_log_likelihood,
    get_prev_decline_log_likelihood,
    get_pulm_prev_log_likelihood,
)
from vacf.constants import AGE_STRATA, END_TIME, ISO3, START_TIME  # noqa: E402
from vacf.demography import prepare_pop_data_for_entries  # noqa: E402
from vacf.epi import add_flows_to_model, get_base_model, initialise_pops  # noqa: E402
from vacf.inputs import (
    get_norm_conmat,
    load_demography,
    load_fertility,
    load_who_outcomes,
)  # noqa: E402
from vacf.parameters import BASE_PARAMS, PARAM_BOUNDS, get_nuts_init_values  # noqa: E402


def calibration_draws(*, seed: int, n_random: int) -> list[tuple[str, dict[str, float]]]:
    rng = np.random.default_rng(seed)
    draws: list[tuple[str, dict[str, float]]] = [
        ("base", dict(BASE_PARAMS)),
        ("nuts_start", dict(BASE_PARAMS) | get_nuts_init_values()),
    ]
    for index in range(n_random):
        sampled = {
            name: float(rng.uniform(low, high)) for name, (low, high) in PARAM_BOUNDS.items()
        }
        draws.append((f"draw_{index:02d}", dict(BASE_PARAMS) | sampled))
    return draws


def _build_model() -> tuple[Any, ...]:
    group_popsize, death_rates, age_weights = load_demography(ISO3)
    fert_padded = load_fertility(ISO3)
    tsr, death_in_unsucc, who_mort = load_who_outcomes(ISO3)
    epi, disease, age, clin, infect = get_base_model(START_TIME, END_TIME)
    start_apops = [1000.0] * len(AGE_STRATA)
    entry_times, entry_rates = prepare_pop_data_for_entries(
        group_popsize, START_TIME, sum(start_apops)
    )
    add_flows_to_model(
        epi,
        disease,
        age,
        clin,
        infect,
        age_weights,
        group_popsize,
        fert_padded,
        death_rates,
        tsr,
        death_in_unsucc,
        entry_times,
        entry_rates,
    )
    initialise_pops(epi, disease, age, start_apops)
    weight_arr = jnp.array(age_weights)
    weight_ends = jnp.array(age_weights.index[[0, -1]])
    pop_arr = jnp.array(group_popsize)
    pop_ends = jnp.array(group_popsize.index[[0, -1]])
    fert_arr = jnp.array(fert_padded)
    fert_ends = jnp.array(fert_padded.index[[0, -1]])
    conmat = get_norm_conmat(ISO3)
    return (
        epi,
        disease,
        age,
        infect,
        who_mort,
        weight_arr,
        weight_ends,
        pop_arr,
        pop_ends,
        fert_arr,
        fert_ends,
        conmat,
    )


def _solver_kwargs() -> dict[str, Any]:
    return {
        "stepsize_controller": diffrax.PIDController(
            rtol=SUMMER3_CALIBRATION_SOLVER["rtol"],
            atol=SUMMER3_CALIBRATION_SOLVER["atol"],
            dtmax=SUMMER3_CALIBRATION_SOLVER["dtmax"],
        ),
        "max_steps": SUMMER3_CALIBRATION_SOLVER["max_steps"],
    }


def _terms(
    results: dict[str, Any],
    params: dict[str, float],
    disease: Any,
    age: Any,
    infect: Any,
    who_mort: Any,
    weight_arr: Any,
    weight_ends: Any,
    pop_arr: Any,
    pop_ends: Any,
    fert_arr: Any,
    fert_ends: Any,
    conmat: Any,
) -> dict[str, float]:
    values = {
        "latent": float(get_latent_log_likelihood(results, disease, age)),
        "notifications": float(get_notification_log_likelihood(results)),
        "deaths": float(get_death_log_likelihood(results, who_mort)),
        "prevalence": float(get_pulm_prev_log_likelihood(results, disease, age, infect)),
        "decline": float(get_prev_decline_log_likelihood(results, disease, age, infect)),
        "infectiousness": float(get_infprop_log_likelihood(results, disease, age, infect)),
        "mixing": float(
            get_mixing_log_likelihood(
                params,
                weight_arr,
                weight_ends,
                pop_arr,
                pop_ends,
                fert_arr,
                fert_ends,
                conmat,
            )
        ),
    }
    values["total"] = float(sum(values[name] for name in TERM_NAMES))
    return values


def main() -> None:
    built = _build_model()
    (
        epi,
        disease,
        age,
        infect,
        who_mort,
        weight_arr,
        weight_ends,
        pop_arr,
        pop_ends,
        fert_arr,
        fert_ends,
        conmat,
    ) = built
    solver_kwargs = _solver_kwargs()
    rows: list[dict[str, Any]] = []
    params_by_draw: dict[str, dict[str, float]] = {}
    draws = calibration_draws(seed=CALIBRATION_SEED, n_random=N_RANDOM_DRAWS)
    for name, params in draws:
        print(f"calibration golden {name}", flush=True)
        results = epi.run(params, solver_kwargs=solver_kwargs)
        rows.append(
            {
                "draw": name,
                **_terms(
                    results,
                    params,
                    disease,
                    age,
                    infect,
                    who_mort,
                    weight_arr,
                    weight_ends,
                    pop_arr,
                    pop_ends,
                    fert_arr,
                    fert_ends,
                    conmat,
                ),
            }
        )
        params_by_draw[name] = params
    write_manifest(
        seed=CALIBRATION_SEED,
        n_random=N_RANDOM_DRAWS,
        draws=[name for name, _ in draws],
    )
    write_terms(rows)
    write_params(params_by_draw)
    print(f"wrote calibration goldens for {len(rows)} draws", flush=True)


if __name__ == "__main__":
    main()
