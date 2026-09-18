"""Regenerate parity goldens with summer3wip.

Run only via ``pixi run -e reference golden``. The reference environment pins
summer3wip at 22d3aa6 and must not share a solve with summer4.

The vendored tree is tb_macroeconomics 1d4fbd6 (the last commit that still has
both detection flows). Upstream has since grown past the model this port
reimplements: eight age bands, a dynamic mixing matrix, demography. The golden
model is the three-age ``ones(3, 3)`` model specified in
``plans/tb-macro-summer4-port.plan.md``. The tanh scale-up is the vendored
function. The triangular seed is the same formula; JAX 0.11 removed
``jnp.clip(..., a_min=)``, so the golden uses ``jnp.maximum``.
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
sys.path.insert(0, str(ROOT / "reference"))

from tb_macro_original.utils import tanh_based_scaleup  # noqa: E402

from summer3.epi import (  # noqa: E402
    CategoryData,
    CompartmentMap,
    CompartmentalEpiModel,
    Stratification,
    TransitionFlow,
    strat_data_from_pandas,
)
from summer3.graph import CompartmentValues, Parameter, Time, defer  # noqa: E402

from jax import numpy as jnp  # noqa: E402


def get_triang_vals(time: Any, peak_time: Any, peak_height: Any, width: Any) -> Any:
    """Vendored formula. ``jnp.clip(..., a_min=)`` does not exist on JAX 0.11."""
    time_from_peak = jnp.absolute(time - peak_time)
    return jnp.maximum(peak_height * (1.0 - time_from_peak / width), 0.0)


OUT = ROOT / "tests" / "golden"
START_YEAR = 1800.0
END_EXCLUSIVE = 2000.0
AGES = (0, 5, 15)
DISEASE = (
    "mtb_naive",
    "incipient",
    "contained",
    "cleared",
    "active",
    "treatment",
    "recovered",
)
INFECT = ("mtb_naive", "contained", "cleared", "recovered")
AGE_NAMES = tuple(str(age) for age in AGES)

BASE_PARAMS: dict[str, float] = {
    "contact_rate": 0.8,
    "freq_dens_exponent": 0.5392614891179714,
    "rel_sus_mtb_naive": 1.0,
    "rel_sus_contained": 0.3324848208129065,
    "rel_sus_cleared": 0.7101012998861186,
    "rel_sus_recovered": 0.7101012998861186,
    "contain": 2.0,
    "clearance_rate": 0.05643858771640714,
    "breakdown_rate": 0.5678500778834155,
    "progression": 0.1,
    "increase_infect": 2.799432998282645,
    "decrease_infect": 1.0,
    "clinical_development": 2.280456422265371,
    "clinical_regression": 1.0,
    "self_recovery": 0.4,
    "detection": 3.1435769682295085,
    "treatment_recovery": 0.5,
    "treatment_relapse": 0.0,
    "seed_peak_time": 30.0,
    "seed_peak_rate": 0.01,
    "seed_duration": 10.0,
    "recent_detection_rate": 0.7,
    "passive_detection_shape": 0.12804084477668393,
    "passive_detection_inflection": 40.0,
    "passive_detection_past_frac": 0.7117933876877114,
}
REINFECTION: dict[str, float] = {
    "rel_sus_mtb_naive": 0.5,
    "rel_sus_contained": 0.5,
    "rel_sus_cleared": 0.5,
    "rel_sus_recovered": 0.5,
    "contain": 0.1,
    "clearance_rate": 0.1,
    "breakdown_rate": 0.1,
    "freq_dens_exponent": 0.7,
    "detection": 0.3,
    "treatment_relapse": 0.05,
}


def infect_process(
    compartment_values: Any,
    contact_rate: Any,
    freq_dens_exponent: Any,
    age_cats: Any,
    infectious_compartments: Any,
) -> CategoryData:
    """``ones(3, 3) @ (I_age / N_age ** exp) * contact_rate``. ``I`` is active."""
    infect_pop = age_cats.product(infectious_compartments)
    infectious = compartment_values.sumcats(infect_pop).data
    total = compartment_values.sumcats(age_cats).data
    pressure = infectious / total**freq_dens_exponent
    foi = jnp.ones((len(AGE_NAMES), len(AGE_NAMES))) @ pressure * contact_rate
    return CategoryData(age_cats, foi)


def build(params: dict[str, float]) -> Any:
    """Construct and run the model the way the original notebook does."""
    disease = Stratification("disease_state", list(DISEASE))
    humans = CompartmentMap.new(disease)
    age = humans.stratify(Stratification("age", list(AGE_NAMES)))
    infectious = Stratification("infectious", ["low", "high"])
    humans.stratify(infectious, (disease, ["active"]))
    clinical = Stratification("clinical", ["subclin", "clin"])
    humans.stratify(clinical, (disease, ["active"]))
    times = pd.Index(np.arange(START_YEAR, END_EXCLUSIVE, 1.0))
    epi = CompartmentalEpiModel(humans, times)

    for comp in INFECT:
        foi = defer(infect_process)(
            CompartmentValues,
            Parameter("contact_rate", 0.0) * Parameter(f"rel_sus_{comp}", 0.0),
            Parameter("freq_dens_exponent", 1.0),
            age.categories(),
            disease["active"],
        )
        epi.add_flow(TransitionFlow(f"infect_{comp}", disease[comp], disease["incipient"], foi))

    epi.add_flow(
        TransitionFlow(
            "containment", disease["incipient"], disease["contained"], Parameter("contain", 0.0)
        )
    )
    epi.add_flow(
        TransitionFlow(
            "clearance", disease["contained"], disease["cleared"], Parameter("clearance_rate", 0.0)
        )
    )
    epi.add_flow(
        TransitionFlow(
            "breakdown",
            disease["contained"],
            disease["incipient"],
            Parameter("breakdown_rate", 0.0),
        )
    )
    epi.add_flow(
        TransitionFlow(
            "progression", disease["incipient"], clinical["subclin"], Parameter("progression", 0.0)
        )
    )
    epi.add_flow(
        TransitionFlow(
            "increase_infectiousness",
            infectious["low"],
            infectious["high"],
            Parameter("increase_infect", 0.0),
        )
    )
    epi.add_flow(
        TransitionFlow(
            "decrease_infectiousness",
            infectious["high"],
            infectious["low"],
            Parameter("decrease_infect", 0.0),
        )
    )
    epi.add_flow(
        TransitionFlow(
            "clinical_develop",
            clinical["subclin"],
            clinical["clin"],
            Parameter("clinical_development", 0.0),
        )
    )
    epi.add_flow(
        TransitionFlow(
            "clinical_regress",
            clinical["clin"],
            clinical["subclin"],
            Parameter("clinical_regression", 0.0),
        )
    )
    epi.add_flow(
        TransitionFlow(
            "self_recovery",
            (disease["active"], clinical["subclin"]),
            disease["recovered"],
            Parameter("self_recovery", 0.0),
        )
    )
    epi.add_flow(
        TransitionFlow(
            "passive_detection",
            disease["active"],
            disease["treatment"],
            Parameter("detection", 0.0),
        )
    )
    epi.add_flow(
        TransitionFlow(
            "treatment_recovery",
            disease["treatment"],
            disease["recovered"],
            Parameter("treatment_recovery", 0.0),
        )
    )
    epi.add_flow(
        TransitionFlow(
            "treatment_relapse",
            disease["treatment"],
            (clinical["subclin"], infectious["low"]),
            Parameter("treatment_relapse", 0.0),
        )
    )
    epi.add_flow(TransitionFlow("ageing_0_to_5", age["0"], age["5"], 1.0 / 5.0))
    epi.add_flow(TransitionFlow("ageing_5_to_15", age["5"], age["15"], 1.0 / 10.0))
    epi.add_flow(
        TransitionFlow(
            "seed_peak",
            disease["mtb_naive"],
            disease["incipient"],
            defer(get_triang_vals)(
                Time,
                Parameter("seed_peak_time", 0.0),
                Parameter("seed_peak_rate", 0.0),
                Parameter("seed_duration", 0.0),
            ),
        )
    )
    rate = Parameter("recent_detection_rate", 0.0) * defer(tanh_based_scaleup)(
        Time,
        Parameter("passive_detection_shape", 0.0),
        Parameter("passive_detection_inflection", 0.0),
        Parameter("passive_detection_past_frac", 0.0),
        1.0,
    )
    epi.add_flow(
        TransitionFlow(
            "detection", (disease["active"], clinical["clin"]), disease["treatment"], rate
        )
    )

    pops = pd.Series(index=list(AGE_NAMES), data=np.array([1000.0, 1000.0, 1000.0]))
    splits = [0.0] * len(DISEASE)
    splits[DISEASE.index("mtb_naive")] = 1.0
    epi.set_initial_population(
        strat_data_from_pandas(pops, age),
        [CategoryData(disease.categories(), jnp.array(splits))],
    )
    solver_kwargs = {
        "stepsize_controller": diffrax.PIDController(rtol=1e-9, atol=1e-9, dtmax=1.0),
        "max_steps": 20_000,
    }
    return epi.run(params, solver_kwargs=solver_kwargs)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, path, compression="snappy")


def _source_ages(flow: Any) -> list[str]:
    container = flow.indices["source"].index
    ages: list[str] = []
    for comp in container.compartments:
        found = "?"
        for strat, stratum in comp.strata:
            if getattr(strat, "name", None) == "age":
                found = stratum if isinstance(stratum, str) else str(stratum[0])
        ages.append(found)
    return ages


def _save(name: str, params: dict[str, float], results: Any) -> None:
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    years = np.arange(START_YEAR, END_EXCLUSIVE, 1.0)

    compartments = results["compartments"].to_pandas_df()
    compartments = compartments.reset_index(drop=True)
    compartments.insert(0, "time", years)
    _write_parquet(compartments, folder / "compartments.parquet")

    columns: dict[str, np.ndarray] = {"time": years}
    age_flows = [f"infect_{comp}" for comp in INFECT] + ["detection"]
    for flow_name, flow in results["flows"].items():
        total = np.asarray(flow.sum(to_dims="time").data).reshape(-1)
        columns[flow_name] = total
        if flow_name in age_flows:
            data = np.asarray(flow.data)
            for age in _source_ages(flow):
                key = f"{flow_name}__age_{age}"
                if key not in columns:
                    columns[key] = np.zeros(len(years))
            for edge, age in enumerate(_source_ages(flow)):
                columns[f"{flow_name}__age_{age}"] = (
                    columns[f"{flow_name}__age_{age}"] + data[:, edge]
                )
    _write_parquet(pd.DataFrame(columns), folder / "flows.parquet")

    target = (
        results["flows"]["infect_mtb_naive"].sum(to_dims="time").to_pandas_df().rolling(7).sum()
    )
    values = np.asarray(target["data"]).reshape(-1)[7:60:7]
    indices = np.arange(7, 60, 7)
    _write_parquet(
        pd.DataFrame({"index": indices, "time": START_YEAR + indices, "value": values}),
        folder / "target.parquet",
    )
    yaml = "".join(f"{key}: {value:.16g}\n" for key, value in params.items())
    (folder / "params.yaml").write_text(yaml)


def main() -> None:
    fixtures = {"base": BASE_PARAMS, "reinfection": BASE_PARAMS | REINFECTION}
    for name, params in fixtures.items():
        print(f"running {name}", flush=True)
        _save(name, params, build(params))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
