#!/usr/bin/env python3
"""Write the summer3-style notebook ports under notebooks/summer3/."""

from __future__ import annotations

import nbformat as nbf
from pathlib import Path

OUT = Path(__file__).resolve().parent


def _nb(cells: list) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "tb-macro-summer4 (pixi)",
            "language": "python",
            "name": "tb-macro-summer4",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb["cells"] = cells
    return nb


def _md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def _code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def write(name: str, cells: list) -> None:
    path = OUT / name
    nbf.write(_nb(cells), path)
    print(f"wrote {path.relative_to(OUT.parent.parent)}")


def notebook_01() -> None:
    write(
        "01-Inputs.ipynb",
        [
            _md(
                """# UN and WHO inputs

Downloads demographic and TB programme CSVs into `data/`, the same way
`tweak_cases_calib` `notebooks/01-Inputs.ipynb` does.

Loaders in `tb_macro.inputs` still pin the **vendored** filenames already in
this repository. Running this notebook writes new timestamped files; it does
not switch the model until those paths in `inputs.py` are updated.

Use the **tb-macro-summer4 (pixi)** kernel. Needs network access."""
            ),
            _md("## UN demographic data provenance\n\n"
                "All UN demographic files downloaded from "
                "https://population.un.org/wpp/downloads?folder=Standard%20Projections&group=CSV%20format\n\n"
                "### Population size\n\n"
                'Second link "1950-2100, medium (gz)" in "Population row".'),
            _code(
                """
import pandas as pd
from datetime import UTC, datetime

from tb_macro.constants import DATA_PATH

url = (
    "https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/"
    "CSV_FILES/WPP2024_Population1JanuaryByAge5GroupSex_Medium.csv.gz"
)
data = pd.read_csv(url)

select_columns = ["ISO3_code", "Location", "Time", "AgeGrp", "AgeGrpStart", "PopTotal"]
select_iso3s = ["KIR", "IDN", "VNM", "PHL"]
select_data = data.loc[data["ISO3_code"].isin(select_iso3s), select_columns]
timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%MZ")
filename = f"un_population_{timestamp}.csv"
out = DATA_PATH / "population" / filename
out.parent.mkdir(parents=True, exist_ok=True)
select_data.to_csv(out)
out
"""
            ),
            _md("### Mortality"),
            _code(
                """
url = (
    "https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/"
    "CSV_FILES/WPP2024_DeathsBySingleAgeSex_Medium_1950-2023.csv.gz"
)
data_1 = pd.read_csv(url)
url = (
    "https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/"
    "CSV_FILES/WPP2024_DeathsBySingleAgeSex_Medium_2024-2100.csv.gz"
)
data_2 = pd.read_csv(url)
data = pd.concat([data_1, data_2])
select_columns = ["ISO3_code", "Location", "Time", "AgeGrp", "AgeGrpStart", "DeathTotal"]
select_data = data.loc[data["ISO3_code"].isin(select_iso3s), select_columns]
timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%MZ")
filename = f"un_mortality_{timestamp}.csv"
out = DATA_PATH / "population" / filename
select_data.to_csv(out, index=False)
out
"""
            ),
            _md("### Fertility"),
            _code(
                """
url = (
    "https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/"
    "CSV_FILES/WPP2024_Fertility_by_Age1.csv.gz"
)
data = pd.read_csv(url)
select_columns = ["ISO3_code", "Location", "Time", "AgeGrp", "AgeGrpStart", "ASFR", "Variant"]
select_data = data.loc[
    data["ISO3_code"].isin(select_iso3s) & (data["Variant"] == "Medium"), select_columns
]
timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%MZ")
filename = f"un_fertility_{timestamp}.csv"
out = DATA_PATH / "population" / filename
select_data.to_csv(out, index=False)
out
"""
            ),
            _md(
                """## WHO data

### Treatment outcomes

From https://www.who.int/teams/global-programme-on-tuberculosis-and-lung-health/data"""
            ),
            _code(
                """
data = pd.read_csv("https://extranet.who.int/tme/generateCSV.asp?ds=outcomes")
timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%MZ")
filename = f"who_outcomes_{timestamp}.csv"
out = DATA_PATH / "who" / filename
out.parent.mkdir(parents=True, exist_ok=True)
data.to_csv(out)
out
"""
            ),
            _md("### Indicators"),
            _code(
                """
url = "https://extranet.who.int/tme/generateCSV.asp?ds=estimates"
data = pd.read_csv(url)
select_iso3s = ["KIR", "IDN", "VNM", "PHL"]
select_data = data.loc[data["iso3"].isin(select_iso3s)]
timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%MZ")
filename = f"who_indicators_{timestamp}.csv"
out = DATA_PATH / "who" / filename
select_data.to_csv(out)
out
"""
            ),
        ],
    )


def notebook_02() -> None:
    write(
        "02-ConmatInputs.ipynb",
        [
            _md(
                """# Contact-matrix inputs

Writes a single-age population CSV and runs `conmat_matrix.R` so
`data/mixing/conmat_all_{ISO3}.csv` matches summer3 `02-ConmatInputs.ipynb`.

Requires R with the `conmat` package. The model already vendors
`conmat_all_VNM.csv`; only re-run when regenerating mixing inputs.

Use the **tb-macro-summer4 (pixi)** kernel."""
            ),
            _code(
                """
import subprocess

from tb_macro.constants import DATA_PATH, ISO3
from tb_macro.inputs import write_conmat_pop_csv

conmat_dir = DATA_PATH / "mixing"
countries = [ISO3]
year = 2025
for iso3 in countries:
    path = write_conmat_pop_csv(iso3, year)
    print(path)
    completed = subprocess.run(
        ["Rscript", str(conmat_dir / "conmat_matrix.R"), iso3, str(year), str(conmat_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    print(completed.stdout)
    if completed.returncode != 0:
        print(completed.stderr)
        raise RuntimeError(f"Rscript failed with code {completed.returncode}")
    print(conmat_dir / f"conmat_all_{iso3}.csv")
"""
            ),
        ],
    )


def notebook_03() -> None:
    write(
        "03-InspectRun.ipynb",
        [
            _md(
                """# Single-run outputs

Summer4 port of `tweak_cases_calib` `notebooks/03-InspectRun.ipynb`. Same
figure groups: population structure, infection source, and mixing.

Uses `run(params)` — corrected model, folded spline contact, smooth entry —
not the golden-matching path. Mixing heatmaps are recomputed with
`spline_mixing` (summer4 does not save `computed_values["dynamic_mm"]`).

`MODE`:
- `base` — `BASE_PARAMS`
- `optimise` — short Adam fit of the calibration likelihood (see also
  `notebooks/06-Optimisation.ipynb`)
- `random` / `map` — draw from a saved ArviZ `.nc` under `outputs/` or
  `outputs/local/`

Use the **tb-macro-summer4 (pixi)** kernel. If this folder is the notebook
cwd, the `plots` helper imports via the cell below."""
            ),
            _code(
                """
import sys
from pathlib import Path

import numpy as np
import optax
from jax import jit, nn, value_and_grad
from jax import numpy as jnp

from tb_macro.calibration import (
    TERM_NAMES,
    build_mixing_cases,
    compile_case,
    make_log_likelihood,
)
from tb_macro.constants import END_TIME, ISO3, LOCAL_OUTPUT_PATH, OUTPUT_PATH
from tb_macro.model import run
from tb_macro.parameters import BASE_PARAMS, PARAM_BOUNDS, get_nuts_init_values

# Allow `from plots import ...` when the kernel cwd is the repo root or this folder.
_here = Path.cwd()
for candidate in (_here, _here / "notebooks" / "summer3", _here / "summer3"):
    if (candidate / "plots.py").is_file():
        sys.path.insert(0, str(candidate))
        break

from plots import (
    load_group_popsize,
    plot_age_population_comparison,
    plot_comp_distributions,
    plot_dynamic_mixing_matrix,
    plot_infection_source_props,
    plot_mean_contacts_by_age,
    plot_mixing_target_comparison,
    plot_single_age_s_matrix_components,
)
"""
            ),
            _code(
                """
MODE = "base"  # "base", "optimise", "random", or "map"

CALIB_SOURCE = "local"  # "local" or "remote"
RUN_ID = None  # e.g. "20260917T1241Z" stamp, or None for newest .nc
FOLDER = None  # remote task folder under OUTPUT_PATH, or None for newest
SAMPLE_SEED = 0
"""
            ),
            _code(
                """
import arviz as az

calib_names = [p for p in PARAM_BOUNDS if p != "mixing_dist_sd"]


def _newest_nc(directory: Path) -> Path:
    files = sorted(directory.glob("*.nc"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No .nc under {directory}")
    return files[-1]


def load_idata():
    if CALIB_SOURCE == "local":
        directory = LOCAL_OUTPUT_PATH
        path = directory / f"{RUN_ID}.nc" if RUN_ID else _newest_nc(directory)
    elif CALIB_SOURCE == "remote":
        root = OUTPUT_PATH
        if FOLDER is not None:
            directory = root / FOLDER
        else:
            dirs = sorted(
                (p for p in root.iterdir() if p.is_dir() and any(p.glob("*.nc"))),
                key=lambda p: p.stat().st_mtime,
            )
            if not dirs:
                raise FileNotFoundError(f"No run dirs under {root}")
            directory = dirs[-1]
        path = directory / f"{RUN_ID}.nc" if RUN_ID else _newest_nc(directory)
    else:
        raise ValueError(f"Unknown CALIB_SOURCE: {CALIB_SOURCE}")
    if not path.exists():
        raise FileNotFoundError(path)
    return az.from_netcdf(path), path


group_popsize = load_group_popsize(ISO3)
"""
            ),
            _code(
                """
run_params = dict(BASE_PARAMS)
opt_history = None
sample_info = None

if MODE == "base":
    pass

elif MODE == "optimise":
    case = next(item for item in build_mixing_cases() if item.name == "spline")
    compiled = compile_case(case)
    evaluate = make_log_likelihood(compiled, mixing=case.mixing, fold=case.fold)
    names = tuple(PARAM_BOUNDS)
    low = jnp.asarray([PARAM_BOUNDS[n][0] for n in names])
    high = jnp.asarray([PARAM_BOUNDS[n][1] for n in names])
    start = get_nuts_init_values()

    def constrain(raw):
        return low + (high - low) * nn.sigmoid(raw)

    def unconstrain(value):
        unit = (value - low) / (high - low)
        return jnp.log(unit) - jnp.log1p(-unit)

    def unpack(raw):
        params = dict(BASE_PARAMS)
        for name, value in zip(names, constrain(raw), strict=True):
            params[name] = value
        return params

    raw = unconstrain(jnp.asarray([start[n] for n in names]))

    def loss(point):
        return -evaluate(unpack(point))["total"]

    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(raw)

    @jit
    def step(point, state):
        value, grad = value_and_grad(loss)(point)
        updates, state = optimizer.update(grad, state, point)
        return optax.apply_updates(point, updates), state, value

    opt_history = []
    for i in range(40):
        raw, opt_state, value = step(raw, opt_state)
        opt_history.append(float(value))
        if i % 10 == 0:
            print(f"step {i:3d}  nll {opt_history[-1]:.4f}")
    run_params = unpack(raw)
    run_params = {k: float(v) if hasattr(v, "item") else v for k, v in run_params.items()}
    print("optimise done", {n: run_params[n] for n in names})

elif MODE in {"random", "map"}:
    idata, nc_path = load_idata()
    stacked = idata.posterior.stack(sample=("chain", "draw"))
    if hasattr(stacked, "dataset"):
        stacked = stacked.dataset
    n_samples = int(stacked.sizes["sample"])
    if MODE == "random":
        rng = np.random.default_rng(SAMPLE_SEED)
        idx = int(rng.integers(n_samples))
    else:
        lp = idata.sample_stats["lp"].stack(sample=("chain", "draw"))
        idx = int(np.asarray(lp).argmax())
    sample = stacked.isel(sample=idx)
    drawn = {k: float(sample[k].values) for k in sample.data_vars}
    run_params = dict(BASE_PARAMS)
    run_params.update(drawn)
    sample_info = {"path": str(nc_path), "index": idx}
    print(sample_info)
    print({k: run_params[k] for k in PARAM_BOUNDS})

else:
    raise ValueError(MODE)

result = run(run_params)
print("ran", MODE)
"""
            ),
            _md("# Population structure"),
            _code(
                """
plot_comp_distributions(result, group_popsize, 1950.0, END_TIME)
"""
            ),
            _code(
                """
plot_age_population_comparison(result, group_popsize, range(1970, 2021, 10))
"""
            ),
            _md("# Infection source"),
            _code(
                """
plot_infection_source_props(result, 1950.0, END_TIME)
"""
            ),
            _md("# Mixing"),
            _code(
                """
plot_dynamic_mixing_matrix(run_params, 1970.0, 10.0, 3)
"""
            ),
            _code(
                """
plot_mixing_target_comparison(run_params, ISO3)
"""
            ),
            _code(
                """
plot_single_age_s_matrix_components(run_params)
"""
            ),
            _code(
                """
plot_mean_contacts_by_age(run_params, ISO3)
"""
            ),
        ],
    )


def notebook_04() -> None:
    write(
        "04-Calibration.ipynb",
        [
            _md(
                """# Calibration

Summer4 port of `tweak_cases_calib` `notebooks/04-Calibration.ipynb`.

Same density as `tb_macro.calibration.make_log_likelihood` (seven terms +
Canberra mixing distance), uniform priors on every name in `PARAM_BOUNDS`,
NUTS tree depth 5, four parallel chains, `PRNGKey(2)`, started at
`get_nuts_init_values()`. Warmup and samples use `N_RUNS_LOCAL` (40), matching
the summer3 local notebook. The longer 100/100 run stays in
`notebooks/05-Calibration.ipynb`.

Corrected default model + folded spline. Not `BayesianModel` — summer4 stock
likelihoods do not cover the log/logit/Canberra terms yet.

Restart the kernel and run from the top so the device-count cell runs before
JAX imports. Use **tb-macro-summer4 (pixi)**."""
            ),
            _md("## Four CPU devices"),
            _code(
                """
import os

flags = os.environ.get("XLA_FLAGS", "").split()
kept = [
    flag
    for flag in flags
    if not flag.startswith("--xla_force_host_platform_device_count=")
]
os.environ["XLA_FLAGS"] = " ".join(["--xla_force_host_platform_device_count=4", *kept])

import jax

if jax.local_device_count() < 4:
    raise RuntimeError(
        f"JAX has {jax.local_device_count()} device(s). Restart the kernel so this "
        "cell can set the device count before JAX is imported."
    )

from datetime import UTC, datetime
from pathlib import Path

import arviz as az
import numpyro
from jax.random import PRNGKey
from numpyro import distributions as dist
from numpyro import infer

from tb_macro.calibration import (
    MCMC_EXTRA_FIELDS,
    build_mixing_cases,
    compile_case,
    inference_data_from_mcmc,
    make_log_likelihood,
)
from tb_macro.constants import LOCAL_OUTPUT_PATH, N_RUNS_LOCAL
from tb_macro.parameters import BASE_PARAMS, PARAM_BOUNDS, get_nuts_init_values
from tb_macro.utils import write_run_log

print(f"devices {jax.local_device_count()}  N_RUNS_LOCAL {N_RUNS_LOCAL}")
"""
            ),
            _md("## Likelihood"),
            _code(
                """
case = next(item for item in build_mixing_cases() if item.name == "spline")
compiled = compile_case(case)
evaluate = make_log_likelihood(compiled, mixing=case.mixing, fold=case.fold)
print(case.name)
"""
            ),
            _md("## Sample"),
            _code(
                """
priors = {name: dist.Uniform(low, high) for name, (low, high) in PARAM_BOUNDS.items()}
for name, (low, high) in PARAM_BOUNDS.items():
    print(f"{name:28s}  Uniform({low}, {high})")


def model() -> None:
    sampled = {name: numpyro.sample(name, prior) for name, prior in priors.items()}
    params = dict(BASE_PARAMS)
    params.update(sampled)
    numpyro.factor("ll", evaluate(params)["total"])


kernel = infer.NUTS(
    model,
    max_tree_depth=5,
    init_strategy=infer.init_to_value(values=get_nuts_init_values()),
)
mcmc = infer.MCMC(
    kernel,
    num_warmup=N_RUNS_LOCAL,
    num_samples=N_RUNS_LOCAL,
    num_chains=4,
    chain_method="parallel",
    progress_bar=True,
)
if mcmc.chain_method != "parallel":
    raise RuntimeError(
        f"Refusing to run with chain_method={mcmc.chain_method!r} on "
        f"{jax.local_device_count()} devices."
    )
print(f"chain_method {mcmc.chain_method}")
mcmc.run(PRNGKey(2), extra_fields=MCMC_EXTRA_FIELDS)
"""
            ),
            _md("## Posterior"),
            _code(
                """
idata = inference_data_from_mcmc(mcmc)
LOCAL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%MZ")
nc_path = LOCAL_OUTPUT_PATH / f"{timestamp}.nc"
idata.to_netcdf(nc_path, engine="h5netcdf")
write_run_log(
    LOCAL_OUTPUT_PATH / f"{timestamp}.log",
    extra={
        "idata": nc_path.name,
        "mixing": case.name,
        "match_golden": False,
        "entry": "smooth",
        "num_warmup": N_RUNS_LOCAL,
        "num_samples": N_RUNS_LOCAL,
        "num_chains": 4,
    },
)
print(nc_path)
az.summary(idata)
"""
            ),
        ],
    )


def notebook_05() -> None:
    write(
        "05-Outputs.ipynb",
        [
            _md(
                """# Calibration outputs

Summer4 port of `tweak_cases_calib` `notebooks/05-Outputs.ipynb`.

Loads an ArviZ `.nc` and optional `_outputs.pkl` from a local or massive run,
summarises the posterior, overlays priors, and plots count/rate panels against
calibration targets. Age bands are the **model** ages (summed for totals) —
single-year regroup from summer3 is not ported.

Use the **tb-macro-summer4 (pixi)** kernel."""
            ),
            _code(
                """
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from tb_macro.constants import (
    LOCAL_OUTPUT_PATH,
    N_OUTPUT_SAMPLES,
    OUTPUT_PATH,
    SCENARIO_PARAMS,
    YOUNG_END_AGE,
)
from tb_macro.inputs import load_who_outcomes
from tb_macro.outputs import (
    get_posterior_samples,
    load_sampled_outputs,
    rerun_model_for_outputs,
)
from tb_macro.parameters import PARAM_NAMES
from tb_macro.targets import (
    LATENT_TARGET,
    NOTIF_TARGET,
    PREV_DECLINE_TARGET,
    PULM_PREV_TARGET,
)
from tb_macro.constants import ISO3

_here = Path.cwd()
for candidate in (_here, _here / "notebooks" / "summer3", _here / "summer3"):
    if (candidate / "plots.py").is_file():
        sys.path.insert(0, str(candidate))
        break

from plots import plot_outputs_panels, plot_prior_post

SOURCE = "local"  # "local" or "remote"
RUN_ID = None  # stamp like "20260917T1241Z", or None for newest
FOLDER = None  # remote task id under OUTPUT_PATH, or None for newest
"""
            ),
            _code(
                """
def _newest_nc(directory: Path) -> Path:
    files = sorted(directory.glob("*.nc"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No .nc under {directory}")
    return files[-1]


if SOURCE == "local":
    run_dir = LOCAL_OUTPUT_PATH
    nc_path = run_dir / f"{RUN_ID}.nc" if RUN_ID else _newest_nc(run_dir)
elif SOURCE == "remote":
    root = OUTPUT_PATH
    if FOLDER is not None:
        run_dir = root / FOLDER
    else:
        dirs = sorted(
            (p for p in root.iterdir() if p.is_dir() and any(p.glob("*.nc"))),
            key=lambda p: p.stat().st_mtime,
        )
        if not dirs:
            raise FileNotFoundError(f"No run dirs under {root}")
        run_dir = dirs[-1]
    nc_path = run_dir / f"{RUN_ID}.nc" if RUN_ID else _newest_nc(run_dir)
else:
    raise ValueError(SOURCE)

stamp = nc_path.stem
outputs_path = nc_path.with_name(f"{stamp}_outputs.pkl")
print(f"run_dir {run_dir}")
print(f"idata   {nc_path}")
print(f"pickle  {outputs_path}  exists={outputs_path.exists()}")

idata = az.from_netcdf(nc_path)
_, _, who_mort = load_who_outcomes(ISO3)
"""
            ),
            _md("## Posterior summary"),
            _code(
                """
summary = az.summary(idata)
summary = summary.rename(index=lambda k: PARAM_NAMES.get(k, k))
summary[["mean", "sd", "r_hat"]]
"""
            ),
            _code(
                """
req_vars = [v for v in idata.posterior.data_vars if v != "mixing_dist_sd"]
plot_prior_post(idata, req_vars)
"""
            ),
            _md("## Scenario outputs"),
            _code(
                """
if outputs_path.exists():
    outputs, sample_labels = load_sampled_outputs(outputs_path)
    print(f"loaded {outputs_path.name}")
else:
    print("pickle missing; re-running posterior samples")
    samples = get_posterior_samples(idata, N_OUTPUT_SAMPLES)
    outputs, sample_labels = rerun_model_for_outputs(idata, SCENARIO_PARAMS, samples)


def _frames_to_multi(frames: list[pd.DataFrame], labels: list[str]) -> pd.DataFrame:
    parts = []
    for label, frame in zip(labels, frames, strict=True):
        wide = frame.copy()
        wide.columns = pd.MultiIndex.from_product([[label], wide.columns])
        parts.append(wide)
    return pd.concat(parts, axis=1)


scenario = 0
age_pop = _frames_to_multi(outputs[scenario]["age_pop"], sample_labels)
incidence = _frames_to_multi(outputs[scenario]["incidence"], sample_labels)
notifications = _frames_to_multi(outputs[scenario]["notifications"], sample_labels)
prevalence = _frames_to_multi(outputs[scenario]["prevalence"], sample_labels)
deaths = _frames_to_multi(outputs[scenario]["deaths"], sample_labels)
latent = _frames_to_multi(outputs[scenario]["latent"], sample_labels)
pulm_prev = _frames_to_multi(outputs[scenario]["pulm_prev"], sample_labels)

bad = (age_pop < -1e-3).any().groupby(level=0).any()
good_labels = [lab for lab in sample_labels if not bool(bad.get(lab, False))]
print(f"samples {len(sample_labels)}  kept {len(good_labels)}")


def sum_ages(df: pd.DataFrame) -> pd.DataFrame:
    kept = df.loc[:, pd.IndexSlice[good_labels, :]]
    return kept.T.groupby(level=0).sum().T


total_pop = sum_ages(age_pop)
incs = sum_ages(incidence)
notifs = sum_ages(notifications)
prevs = sum_ages(prevalence)
tb_deaths = sum_ages(deaths)
adult_ages = [
    age
    for age in age_pop.columns.get_level_values(1).unique()
    if int(age) >= YOUNG_END_AGE
]
adult_pop = sum_ages(age_pop.loc[:, pd.IndexSlice[:, adult_ages]])
pulm = sum_ages(pulm_prev.loc[:, pd.IndexSlice[:, adult_ages]])
lat = sum_ages(latent.loc[:, pd.IndexSlice[:, adult_ages]])
"""
            ),
            _code(
                """
plot_outputs_panels(
    prevs,
    incs,
    notifs,
    NOTIF_TARGET,
    tb_deaths,
    who_mort,
    lat,
    None,
    total_pop,
    1990.0,
    2040.0,
    "count",
    pulm,
    None,
    adult_pop,
)
"""
            ),
            _code(
                """
plot_outputs_panels(
    prevs,
    incs,
    notifs,
    NOTIF_TARGET,
    tb_deaths,
    None,
    lat,
    LATENT_TARGET,
    total_pop,
    1990.0,
    2040.0,
    "rate",
    pulm,
    PULM_PREV_TARGET,
    adult_pop,
    PREV_DECLINE_TARGET,
)
"""
            ),
            _md("## Trace"),
            _code(
                """
import plotly.graph_objects as go
from plotly.subplots import make_subplots

posterior = idata.posterior
dataset = posterior.dataset if hasattr(posterior, "dataset") else posterior
names = [v for v in dataset.data_vars if v != "mixing_dist_sd"]
n_chains = int(dataset.sizes["chain"])
fig = make_subplots(rows=len(names), cols=2, subplot_titles=[f"{n} posterior" for n in names for _ in (0, 1)][: 2 * len(names)])
for i, name in enumerate(names):
    values = np.asarray(dataset[name])
    for chain in range(n_chains):
        fig.add_trace(
            go.Scatter(y=values[chain], mode="lines", line=dict(width=1), opacity=0.7, showlegend=False),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(x=values[chain], showlegend=False, opacity=0.5),
            row=i + 1,
            col=2,
        )
fig.update_layout(height=180 * len(names), width=900, template="plotly_white", title_text="Trace")
fig.show()
"""
            ),
        ],
    )


def notebook_06() -> None:
    write(
        "06-Documentation.ipynb",
        [
            _md(
                """# Methods

Summer4 stand-in for `tweak_cases_calib` `notebooks/06-Documentation.ipynb`.

The original notebook rendered Google-style `Notes:` sections through
`document.get_func_notes`. Those Notes were not ported onto every summer4
builder. This notebook states the same topics against the modules that exist
here, and prints the fixed-parameter and prior tables from `parameters.py`.

Default model: corrected replacement deaths, progression weights that sum to
one, `rel_infectiousness_subclin` on subclinical disease, folded spline
contact, smooth entry. `match_golden=True` restores the three summer3 bugs
plus year-frozen mixing and step entry for golden parity. Solver backends live
in `tb_macro.solvers` (`Diffrax` + `PIDController`); calibration uses
`CALIBRATION_SOLVER`.

Use the **tb-macro-summer4 (pixi)** kernel."""
            ),
            _md(
                """## Demographics

Age breaks are `AGE_STRATA` in `tb_macro.constants`. UN population, mortality
and fertility loaders are in `tb_macro.inputs`; entry rates and smooth vs step
birth lookups are in `tb_macro.demography`. Replacement (non-TB) deaths feed
age 0 on the default path.

## Natural history and detection

Flows are declared in `tb_macro.model.build_model`: infection via
`ForceOfInfection`, latency and clinical transitions, cosine detection
scale-up (`tb_macro.curves.get_cos_multicurve`), treatment and ACF hazards
filled through `derived_fn`, triangular seeding, and ageing.

## Mixing

Static year-frozen construction is `tb_macro.mixing.get_norm_c_matrix`. The
default run folds a cubic spline (`tb_macro.contact` + `fold_spline_contact` /
`spline_mixing` in `model.py`) once per parameter draw.

## Calibration

`tb_macro.calibration.make_log_likelihood` is the seven-term summer3 density
(latent, notifications, TB deaths, adult pulmonary prevalence, survey decline,
high-infectious share, Canberra mixing distance). It is hand-rolled: summer4
`BayesianModel` stock likelihoods do not yet express those transforms. Targets
are in `tb_macro.targets`."""
            ),
            _md("## Fixed parameters"),
            _code(
                """
import pandas as pd

from tb_macro.parameters import BASE_PARAMS, PARAM_BOUNDS, PARAM_NAMES

fixed = {k: v for k, v in BASE_PARAMS.items() if k not in PARAM_BOUNDS}
pd.DataFrame(
    {
        "parameter": [PARAM_NAMES.get(k, k) for k in fixed],
        "name": list(fixed),
        "value": list(fixed.values()),
    }
)
"""
            ),
            _md("## Prior ranges"),
            _code(
                """
pd.DataFrame(
    {
        "parameter": [PARAM_NAMES.get(k, k) for k in PARAM_BOUNDS],
        "name": list(PARAM_BOUNDS),
        "low": [PARAM_BOUNDS[k][0] for k in PARAM_BOUNDS],
        "high": [PARAM_BOUNDS[k][1] for k in PARAM_BOUNDS],
        "base": [BASE_PARAMS[k] for k in PARAM_BOUNDS],
    }
)
"""
            ),
        ],
    )


def main() -> None:
    notebook_01()
    notebook_02()
    notebook_03()
    notebook_04()
    notebook_05()
    notebook_06()
    (OUT / "__init__.py").write_text('"""Summer3-style notebook ports (summer4 APIs)."""\n')


if __name__ == "__main__":
    main()
