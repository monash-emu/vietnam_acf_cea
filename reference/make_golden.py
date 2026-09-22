"""Regenerate parity goldens with summer3wip.

Run only via ``pixi run -e reference golden``. The reference environment pins
summer3wip at ``15ad51b`` (the rev in ``tweak_cases_calib``'s pixi.toml) and
must not share a solve with summer4.

The vendored tree is ``tb_macroeconomics`` ``tweak_cases_calib`` at ``9cc8cf8``.
Construction matches ``notebooks/03-InspectRun.ipynb`` in ``MODE == "base"``:
flat 1000 per age, Vietnam demography, ``add_flows_to_model``.

summer4 cannot set ``dtmax``. The original notebook uses the runner default
``rtol=atol=1e-5`` and ``dtmax=1``. These goldens tighten that to ``1e-9`` and
``max_steps=20_000`` so reference error sits under the parity tolerance, the
same reason the three-age port did. Do not edit the parquet to fit summer4.

Calibration log-density goldens are written by ``reference/make_calibration_golden.py``
(also run from ``pixi run -e reference golden``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import diffrax
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "reference" / "snapshot"
sys.path.insert(0, str(SNAPSHOT / "src" / "tb_macro"))

from tb_macro.constants import (  # noqa: E402
    AGE_STRATA,
    END_TIME,
    ISO3,
    START_TIME,
)
from tb_macro.demography import prepare_pop_data_for_entries  # noqa: E402
from tb_macro.epi import add_flows_to_model, get_base_model, initialise_pops  # noqa: E402
from tb_macro.inputs import load_demography, load_fertility, load_who_outcomes  # noqa: E402
from tb_macro.outputs import (  # noqa: E402
    get_age_deaths,
    get_age_inc,
    get_age_notifs,
    get_age_pop,
    get_age_prev,
    get_age_pulm_prev,
)
from tb_macro.parameters import BASE_PARAMS, get_nuts_init_values  # noqa: E402

OUT = ROOT / "tests" / "golden"
YEARS = np.arange(START_TIME, END_TIME, 1.0)
FLOW_TOTALS = (
    "progression",
    "detection",
    "acf",
    "tb_mortality",
    "rx_death",
    "entry_births",
    "success",
    "relapse",
    "seed_peak",
    "infect_mtb_naive",
    "infect_contained",
    "infect_cleared",
    "infect_recovered",
    *(f"nontb_deaths_{age}" for age in AGE_STRATA),
)
INDICATOR_FNS = (
    ("incidence", get_age_inc),
    ("prevalence", get_age_prev),
    ("pulmonary", get_age_pulm_prev),
    ("notifications", get_age_notifs),
    ("deaths", get_age_deaths),
    ("population", get_age_pop),
)


def fixtures() -> dict[str, dict[str, float]]:
    return {
        "base": dict(BASE_PARAMS),
        "nuts_start": dict(BASE_PARAMS) | get_nuts_init_values(),
        "acf": dict(BASE_PARAMS) | {"acf_coverage": 0.8},
    }


def build(params: dict[str, float]) -> Any:
    """Construct and run the model the way the inspect notebook does."""
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
    solver_kwargs = {
        "stepsize_controller": diffrax.PIDController(rtol=1e-9, atol=1e-9, dtmax=1.0),
        "max_steps": 20_000,
    }
    results = epi.run(params, solver_kwargs=solver_kwargs)
    return results, disease, age, clin, infect


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, path, compression="snappy")


def _indicator_frame(managed: Any) -> np.ndarray:
    data = np.asarray(managed.data)
    if data.ndim != 2 or data.shape[1] != len(AGE_STRATA):
        raise ValueError(f"expected (time, {len(AGE_STRATA)}) indicator, got {data.shape}")
    return data


def _save(name: str, params: dict[str, float], built: tuple[Any, ...]) -> None:
    results, disease, age, clin, infect = built
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)

    compartments = results["compartments"].to_pandas_df()
    compartments = compartments.reset_index(drop=True)
    compartments.insert(0, "time", YEARS)
    _write_parquet(compartments, folder / "compartments.parquet")

    missing = [flow for flow in FLOW_TOTALS if flow not in results["flows"]]
    if missing:
        raise KeyError(f"{name} missing flows {missing}; have {sorted(results['flows'])}")
    columns: dict[str, np.ndarray] = {"time": YEARS}
    for flow_name in FLOW_TOTALS:
        total = np.asarray(results["flows"][flow_name].sum(to_dims="time").data).reshape(-1)
        columns[flow_name] = total
    _write_parquet(pd.DataFrame(columns), folder / "flows.parquet")

    indicator_cols: dict[str, np.ndarray] = {"time": YEARS}
    for label, fn in INDICATOR_FNS:
        values = _indicator_frame(fn(results, age, disease, clin, infect))
        for index, stratum in enumerate(AGE_STRATA):
            indicator_cols[f"{label}_{stratum}"] = values[:, index]
    _write_parquet(pd.DataFrame(indicator_cols), folder / "indicators.parquet")
    yaml = "".join(f"{key}: {float(value):.16g}\n" for key, value in params.items())
    (folder / "params.yaml").write_text(yaml)


def main() -> None:
    for name, params in fixtures().items():
        print(f"running {name}", flush=True)
        _save(name, params, build(params))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
