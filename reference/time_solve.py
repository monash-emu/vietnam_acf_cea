"""Time one JIT and then several post-JIT summer3 solves.

Prints ``JIT <seconds>`` and ``POST <t1> <t2> ...``. Does not write parquets.
Run with the reference environment's Python.

Solver settings are the ones the tb_macroeconomics notebooks use: only
``max_steps=4000`` is passed, and the summer3 runner supplies Dopri5,
``dt0=0.1``, ``PIDController(rtol=1e-5, atol=1e-5, dtmax=1)``.
"""

from __future__ import annotations

import time

import numpy as np
import make_golden  # noqa: F401  — inserts the snapshot on sys.path before tb_macro

from tb_macro.constants import AGE_STRATA, END_TIME, ISO3, START_TIME  # noqa: E402
from tb_macro.demography import prepare_pop_data_for_entries  # noqa: E402
from tb_macro.epi import add_flows_to_model, get_base_model, initialise_pops  # noqa: E402
from tb_macro.inputs import load_demography, load_fertility, load_who_outcomes  # noqa: E402
from tb_macro.parameters import BASE_PARAMS, PARAM_BOUNDS  # noqa: E402

N_SETS = 6
SEED = 0


def draw_params(n_sets: int, seed: int) -> list[dict[str, float]]:
    """Uniform draws on ``PARAM_BOUNDS``. Other keys stay at ``BASE_PARAMS``."""
    rng = np.random.default_rng(seed)
    drawn: list[dict[str, float]] = []
    for _ in range(n_sets):
        params = dict(BASE_PARAMS)
        for name, (low, high) in PARAM_BOUNDS.items():
            params[name] = float(rng.uniform(low, high))
        drawn.append(params)
    return drawn


def main() -> None:
    """Construct once, discard one compiled solve, then time each draw."""
    sets = draw_params(N_SETS, SEED)
    group_popsize, death_rates, age_weights = load_demography(ISO3)
    fert_padded = load_fertility(ISO3)
    tsr, death_in_unsucc, _who_mort = load_who_outcomes(ISO3)
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
    # Notebooks pass only max_steps. The summer3 runner then keeps Dopri5,
    # dt0=0.1, PIDController(rtol=1e-5, atol=1e-5, dtmax=1), throw=False.
    # That is what notebooks/03-Optimisation.ipynb, 04-Calibration.ipynb,
    # 06-ManualRun.ipynb and 07-BatchOptimisation.ipynb run, via SOLVER_KWARGS.
    solver_kwargs = {"max_steps": 4000}

    def once(params: dict[str, float]) -> float:
        started = time.perf_counter()
        results = epi.run(params, solver_kwargs=solver_kwargs)
        _ = results["compartments"]
        return time.perf_counter() - started

    jit_s = once(sets[0])
    post = [once(params) for params in sets]
    print(f"JIT {jit_s:.6f}", flush=True)
    print("POST " + " ".join(f"{value:.6f}" for value in post), flush=True)


if __name__ == "__main__":
    main()
