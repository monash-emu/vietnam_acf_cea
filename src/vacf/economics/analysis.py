"""Running the study: calibration, scenario runs, and the comparisons between them.

Notebooks orchestrate; the logic lives here. Everything is parameterised by the arms in
:mod:`vacf.economics.study`, so changing the specification changes the analysis without
editing either this module or the notebook.
"""

from __future__ import annotations

import logging
from functools import cache
from pathlib import Path
from typing import Any

import arviz as az

import numpy as np
import pandas as pd

from vacf.calibration import (
    MCMC_EXTRA_FIELDS,
    build_mixing_cases,
    compile_case,
    inference_data_from_mcmc,
    make_log_likelihood,
    target_trajectories,
)
from vacf.constants import AGES
from vacf.model import prepare_run_params, solve
from vacf.outputs import INDICATORS, get_posterior_samples
from vacf.parameters import BASE_PARAMS, HOMOGENEOUS_PARAM_BOUNDS, get_nuts_init_values
from vacf.solvers import CALIBRATION_SOLVER
from vacf.utils import write_run_log

from vacf.economics import contract, costing, indicators, study
from vacf.economics.cea import efficiency_frontier
from vacf.economics.costs import cost_items_from_spreadsheet, total_costs
from vacf.economics.dalys import age_weighted_daly_values, total_dalys
from vacf.economics.inputs import load_age_daly_table


def homogeneous_case():
    """The mixing assumption used throughout this study."""
    return next(case for case in build_mixing_cases() if case.name == "homogeneous")


@cache
def _compiled_homogeneous() -> Any:
    """Compiled once per process, so repeated solves across posterior draws reuse it."""
    return compile_case(homogeneous_case())


# ---------------------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------------------


def calibrate(
    num_warmup: int = 100,
    num_samples: int = 100,
    num_chains: int = 4,
    *,
    seed: int = 0,
    max_tree_depth: int = 5,
    progress_bar: bool = True,
) -> Any:
    """Sample the posterior under homogeneous mixing.

    Twelve parameters are estimated: the sixteen of the upstream model less the four that
    only exist to shape a contact-data-derived mixing matrix. The Canberra mixing term is
    omitted from the likelihood for the same reason.

    The defaults are a short local chain, enough to exercise the pipeline. Production runs
    belong on the cluster — see ``scripts/massive/``.

    Args:
        num_warmup: Warmup iterations per chain.
        num_samples: Retained samples per chain.
        num_chains: Number of chains.
        seed: PRNG seed.
        max_tree_depth: NUTS tree depth.
        progress_bar: Whether to show sampling progress.

    Returns:
        ArviZ inference data.

    Notes:
    -----
    The model is calibrated by Hamiltonian Monte Carlo (the No-U-Turn
    Sampler) to Vietnamese data on tuberculosis infection prevalence,
    case notifications, deaths, disease prevalence, the decline in
    prevalence between surveys, and the proportion of prevalent
    disease that is bacteriologically positive. Uniform priors are
    applied over the ranges given in the parameter table.
    """
    import numpyro
    from jax.random import PRNGKey
    from numpyro import distributions as dist
    from numpyro import infer

    case = homogeneous_case()
    evaluate = make_log_likelihood(
        compile_case(case), mixing=case.mixing, fold=case.fold, include_mixing_term=False
    )
    priors = {
        name: dist.Uniform(low, high) for name, (low, high) in HOMOGENEOUS_PARAM_BOUNDS.items()
    }

    def model() -> None:
        sampled = {name: numpyro.sample(name, prior) for name, prior in priors.items()}
        params = dict(BASE_PARAMS)
        params.update(sampled)
        numpyro.factor("ll", evaluate(params)["total"])

    init = {k: v for k, v in get_nuts_init_values().items() if k in HOMOGENEOUS_PARAM_BOUNDS}
    kernel = infer.NUTS(
        model, max_tree_depth=max_tree_depth, init_strategy=infer.init_to_value(values=init)
    )
    mcmc = infer.MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="parallel" if num_chains > 1 else "sequential",
        progress_bar=progress_bar,
    )
    mcmc.run(PRNGKey(seed), extra_fields=MCMC_EXTRA_FIELDS)
    return inference_data_from_mcmc(mcmc)


def posterior_medians(idata: Any) -> dict[str, float]:
    """Posterior median of each calibrated parameter.

    A central point estimate for the deterministic comparison. Full uncertainty propagation
    runs the scenarios across posterior draws instead.
    """
    posterior = idata.posterior
    dataset = posterior.dataset if hasattr(posterior, "dataset") else posterior
    return {
        name: float(np.median(np.asarray(dataset[name].values)))
        for name in HOMOGENEOUS_PARAM_BOUNDS
        if name in dataset
    }


def calibration_fit(calibrated: dict[str, float]) -> pd.DataFrame:
    """Model trajectories of the calibration targets at one calibrated point.

    Solved as the likelihood solves it: ``BASE_PARAMS`` plus ``calibrated``, no scenario
    overrides, homogeneous mixing, calibration solver. Pair with
    :func:`vacf.calibration.observed_targets`.
    """
    params = dict(BASE_PARAMS)
    params.update(calibrated)
    result = solve(_compiled_homogeneous(), params, **CALIBRATION_SOLVER)
    return target_trajectories(result)


# ---------------------------------------------------------------------------------------
# Scenario runs
# ---------------------------------------------------------------------------------------


def run_scenario(arm, calibrated: dict[str, float] | None = None) -> tuple[Any, dict[str, Any]]:
    """Solve one arm under homogeneous mixing.

    Returns:
        The solved result and the full parameter set it was solved with.
    """
    params = dict(BASE_PARAMS)
    if calibrated:
        params.update(calibrated)
    params.update(arm.params)
    prepared = prepare_run_params(params)
    return solve(_compiled_homogeneous(), prepared, **CALIBRATION_SOLVER), params


def run_all(arms=None, calibrated: dict[str, float] | None = None) -> dict[str, dict[str, Any]]:
    """Solve every arm and derive both epidemiological and economic outputs.

    Returns:
        Arm name to ``{"arm", "params", "epi", "economic"}``.
    """
    arms = arms if arms is not None else study.SCENARIOS
    out: dict[str, dict[str, Any]] = {}
    for arm in arms:
        result, params = run_scenario(arm, calibrated)
        economic = indicators.economic_indicators(result, params)
        contract.validate(economic, list(AGES))
        out[arm.name] = {
            "arm": arm,
            "params": params,
            "epi": epi_frame(result),
            "economic": economic,
        }
    return out


def epi_frame(result: Any) -> pd.DataFrame:
    """Epidemiological indicators, summed across age, indexed by year."""
    years = indicators.model_years()
    data = {name: sum(np.asarray(v) for v in func(result).values()) for name, func in INDICATORS}
    return pd.DataFrame(data, index=pd.Index(years, name="year"))


# ---------------------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------------------


def _window(frame: pd.DataFrame, horizon: tuple[int, int]) -> pd.DataFrame:
    return frame.loc[horizon[0] : horizon[1]]


def epi_comparison(runs: dict, horizon: tuple[int, int] | None = None) -> pd.DataFrame:
    """Cumulative epidemiological burden per arm, and the difference from baseline."""
    horizon = horizon or study.HORIZON
    rows = {}
    for name, run in runs.items():
        window = _window(run["epi"], horizon)
        rows[run["arm"].label] = {
            "cum_incidence": window["incidence"].sum(),
            "cum_notifications": window["notifications"].sum(),
            "cum_deaths": window["deaths"].sum(),
            "prevalence_end": run["epi"].loc[horizon[1], "prevalence"],
        }
    table = pd.DataFrame(rows).T

    baseline = runs["baseline"]["arm"].label
    for column in ("cum_incidence", "cum_deaths"):
        table[f"{column}_averted"] = table.loc[baseline, column] - table[column]
        table[f"{column}_averted_pct"] = (
            100.0 * table[f"{column}_averted"] / table.loc[baseline, column]
        )
    return table


def economic_comparison(
    runs: dict, horizon: tuple[int, int] | None = None, items: dict | None = None
) -> pd.DataFrame:
    """Discounted costs and DALYs per arm, split by perspective and DALY component."""
    horizon = horizon or study.HORIZON
    items = items if items is not None else cost_items_from_spreadsheet()
    daly_values = age_weighted_daly_values(
        load_age_daly_table(), indicators.single_age_population(horizon[0]), list(AGES)
    )

    rows = {}
    for name, run in runs.items():
        frame = run["economic"]
        costs = total_costs(
            frame,
            costing.cost_structure_for(name, items),
            discount_rate=costing.DISCOUNT_RATE,
            reference_year=costing.REFERENCE_YEAR,
            start_year=horizon[0],
            end_year=horizon[1],
        )
        dalys = total_dalys(
            frame,
            list(AGES),
            daly_values,
            costing.YLD_WEIGHT.mean,
            discount_rate=costing.DISCOUNT_RATE,
            reference_year=costing.REFERENCE_YEAR,
            start_year=horizon[0],
            end_year=horizon[1],
        )
        rows[run["arm"].label] = {
            "n_screened": _window(frame, horizon)[contract.N_SCREENED].sum(),
            "cost_provider": costs["total_provider"],
            "cost_patient": costs["total_patient"],
            "cost": costs["total"],
            "ylls": dalys["ylls"],
            "post_tb_dalys": dalys["post_tb_dalys"],
            "ylds": dalys["ylds"],
            "dalys": dalys["total"],
        }
    return pd.DataFrame(rows).T


def incremental(economic: pd.DataFrame) -> pd.DataFrame:
    """Efficiency frontier, dominance and ICERs across arms."""
    return efficiency_frontier(economic[["cost", "dalys"]].copy())


# ---------------------------------------------------------------------------------------
# Posterior uncertainty
# ---------------------------------------------------------------------------------------


#: Long tables written by :func:`run_posterior`, one parquet file each.
POSTERIOR_TABLES: tuple[str, ...] = (
    "epi",
    "economic",
    "calibration_fit",
    "epi_summary",
    "economic_summary",
    "incremental",
)


def posterior_draws(idata: Any, n_draws: int) -> pd.DataFrame:
    """``n_draws`` posterior points sampled without replacement (seed 0), one row each.

    Indexed by ``sample``; ``chain`` and ``draw`` locate each point in ``idata``.
    """
    samples = get_posterior_samples(idata, n_draws)
    frame = pd.DataFrame(
        {
            "chain": np.asarray(samples["chain"].values, dtype=int),
            "draw": np.asarray(samples["draw"].values, dtype=int),
            **{
                name: np.asarray(samples[name].values, dtype=float)
                for name in HOMOGENEOUS_PARAM_BOUNDS
            },
        }
    )
    frame.index.name = "sample"
    return frame


def _by_arm(table: pd.DataFrame, runs: dict, sample: int) -> pd.DataFrame:
    """Comparison tables are indexed by arm label; add the arm name and the sample."""
    names = {run["arm"].label: name for name, run in runs.items()}
    out = table.rename_axis("label").reset_index()
    out.insert(0, "arm", out["label"].map(names))
    out.insert(0, "sample", sample)
    return out


def run_posterior(draws: pd.DataFrame, arms=None, logger: Any = None) -> dict[str, pd.DataFrame]:
    """Every arm at every posterior draw, as long tables with ``sample`` and ``arm`` columns.

    Parameters
    ----------
    draws
        Output of :func:`posterior_draws`.
    arms
        Arms to run; defaults to :data:`vacf.economics.study.SCENARIOS`.
    logger
        Optional logger for progress.

    Returns
    -------
    dict
        ``epi`` and ``economic`` are annual per draw and arm. ``economic`` holds the contract
        columns that costing consumes, so costs and DALYs can be recomputed from it without
        re-solving. ``calibration_fit`` is annual per draw. ``epi_summary``,
        ``economic_summary`` and ``incremental`` are the study-horizon comparisons per draw.
    """
    items = cost_items_from_spreadsheet()
    tables: dict[str, list[pd.DataFrame]] = {name: [] for name in POSTERIOR_TABLES}
    for count, (sample, row) in enumerate(draws.iterrows(), start=1):
        calibrated = {name: float(row[name]) for name in HOMOGENEOUS_PARAM_BOUNDS}
        runs = run_all(arms, calibrated)
        for name, run in runs.items():
            for key in ("epi", "economic"):
                frame = run[key].reset_index()
                frame.insert(0, "arm", name)
                frame.insert(0, "sample", sample)
                tables[key].append(frame)
        fit = calibration_fit(calibrated).reset_index()
        fit.insert(0, "sample", sample)
        tables["calibration_fit"].append(fit)
        economic = economic_comparison(runs, items=items)
        tables["epi_summary"].append(_by_arm(epi_comparison(runs), runs, sample))
        tables["economic_summary"].append(_by_arm(economic, runs, sample))
        tables["incremental"].append(_by_arm(incremental(economic), runs, sample))
        if logger is not None and (count % 10 == 0 or count == len(draws)):
            logger.info("posterior draws run: %d / %d", count, len(draws))
    return {name: pd.concat(frames, ignore_index=True) for name, frames in tables.items()}


# ---------------------------------------------------------------------------------------
# Summarising a stored posterior run
# ---------------------------------------------------------------------------------------


def load_posterior_outputs(directory: Path | str) -> dict[str, Any]:
    """Everything ``scripts/massive/calibrate.py`` writes to one run directory.

    Returns the :data:`POSTERIOR_TABLES`, ``draws``, ``posterior_summary`` and ``idata``.
    """
    directory = Path(directory)
    out: dict[str, Any] = {
        name: pd.read_parquet(directory / f"{name}.parquet")
        for name in ("draws", *POSTERIOR_TABLES)
    }
    out["posterior_summary"] = pd.read_csv(directory / "posterior_summary.csv", index_col=0)
    out["idata"] = az.from_netcdf(directory / "idata.nc")
    return out


def quantile_bands(
    table: pd.DataFrame, column: str, by: list[str], level: float = 0.95
) -> pd.DataFrame:
    """Median and central ``level`` interval of ``column`` across draws, per ``by`` group."""
    tail = (1.0 - level) / 2.0
    grouped = table.groupby(by, sort=False)[column]
    return pd.DataFrame(
        {
            "median": grouped.median(),
            "lower": grouped.quantile(tail),
            "upper": grouped.quantile(1.0 - tail),
        }
    )


def interval_summary(
    table: pd.DataFrame, columns: list[str], by: str = "label", level: float = 0.95
) -> pd.DataFrame:
    """:func:`quantile_bands` for several columns, side by side as ``(column, statistic)``."""
    return pd.concat(
        {column: quantile_bands(table, column, [by], level) for column in columns}, axis=1
    )


def with_prevalence_decline(fit: pd.DataFrame, base_time: float) -> pd.DataFrame:
    """Add ``decline``: each draw's adult pulmonary prevalence relative to ``base_time``.

    The decline target is a ratio of prevalence at two survey rounds, so this is the
    trajectory it constrains; it is 1 at ``base_time`` by construction.
    """
    out = fit.copy()
    base = {
        sample: np.interp(base_time, group["year"], group["prevalence"])
        for sample, group in out.groupby("sample")
    }
    out["decline"] = out["prevalence"] / out["sample"].map(base)
    return out


def incremental_vs_baseline(economic_summary: pd.DataFrame) -> pd.DataFrame:
    """Each campaign arm's extra cost and DALYs averted relative to baseline, per draw.

    Pairing within a draw means uncertainty shared by both arms (the calibrated epidemic)
    does not widen the interval of the difference.
    """
    is_base = economic_summary["arm"] == study.BASELINE.name
    base = economic_summary[is_base].set_index("sample")
    arms = economic_summary[~is_base]
    out = arms[["sample", "arm", "label"]].reset_index(drop=True)
    out["inc_cost"] = arms["cost"].to_numpy() - base.loc[arms["sample"], "cost"].to_numpy()
    out["dalys_averted"] = base.loc[arms["sample"], "dalys"].to_numpy() - arms["dalys"].to_numpy()
    return out


def expected_incremental(economic_summary: pd.DataFrame) -> pd.DataFrame:
    """Efficiency frontier and ICERs on costs and DALYs averaged over draws.

    The usual summary of a probabilistic analysis. Per-draw ICERs are unstable: an arm that
    is dominated in some draws has no ICER there, and ratios of noisy differences are
    heavy-tailed.
    """
    means = economic_summary.groupby("label", sort=False)[["cost", "dalys"]].mean()
    return incremental(means)


# ---------------------------------------------------------------------------------------
# Running and storing a full analysis
# ---------------------------------------------------------------------------------------


def run_and_store(
    directory: Path | str,
    *,
    n_runs: int,
    n_chains: int,
    n_draws: int,
    seed: int = 0,
    logger: logging.Logger | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Calibrate, run every arm at posterior draws, and write the run directory.

    The single implementation behind ``scripts/massive/calibrate.py`` and the run notebook,
    so local and cluster runs have the same layout. Writes ``idata.nc``,
    ``posterior_summary.csv``, ``draws.parquet``, the :data:`POSTERIOR_TABLES` as parquet,
    the shareable summaries under ``csv/``, and ``run.log``.

    Parameters
    ----------
    directory
        Run directory. Refuses to replace an existing posterior unless ``overwrite``.
    n_runs
        Warmup iterations, and retained samples, per chain.
    n_chains
        Chains, run in parallel; JAX must already see at least this many devices.
    n_draws
        Posterior draws pushed through every arm.

    Returns
    -------
    dict
        The outputs, keyed as :func:`load_posterior_outputs` returns them.
    """
    import jax

    logger = logger or logging.getLogger(__name__)
    directory = Path(directory)
    if (directory / "idata.nc").exists() and not overwrite:
        raise FileExistsError(f"{directory} already holds a posterior; pass overwrite=True.")
    if n_chains > 1 and jax.local_device_count() < n_chains:
        raise RuntimeError(
            f"JAX sees {jax.local_device_count()} devices for {n_chains} chains, so they would "
            "run one after another. Set the device count before JAX is imported."
        )
    directory.mkdir(parents=True, exist_ok=True)
    logger.info(study.describe())
    logger.info("chains %d, warmup = samples = %d, posterior draws %d", n_chains, n_runs, n_draws)

    idata = calibrate(num_warmup=n_runs, num_samples=n_runs, num_chains=n_chains, seed=seed)
    idata.to_netcdf(directory / "idata.nc", engine="h5netcdf")
    summary = az.summary(idata)
    summary.to_csv(directory / "posterior_summary.csv")
    worst_rhat = float(summary["r_hat"].max())
    divergences = int(idata.sample_stats["diverging"].values.sum())
    logger.info("calibration done: worst r_hat %.3f, divergences %d", worst_rhat, divergences)

    draws = posterior_draws(idata, n_draws)
    draws.to_parquet(directory / "draws.parquet")
    tables = run_posterior(draws, logger=logger)
    for name, table in tables.items():
        table.to_parquet(directory / f"{name}.parquet", index=False)

    outputs = {"draws": draws, **tables, "posterior_summary": summary, "idata": idata}
    write_summary_csvs(outputs, directory / "csv")
    write_run_log(
        directory / "run.log",
        extra={
            "task": directory.name,
            "mixing": "homogeneous",
            "n_chains": n_chains,
            "n_runs": n_runs,
            "n_draws": len(draws),
            "seed": seed,
            "worst_r_hat": round(worst_rhat, 4),
            "divergences": divergences,
            "arms": ", ".join(arm.name for arm in study.SCENARIOS),
            "study": study.describe(),
        },
    )
    logger.info("wrote outputs to %s", directory)
    return outputs


# ---------------------------------------------------------------------------------------
# Shareable CSV summaries
# ---------------------------------------------------------------------------------------

#: Statistics across posterior draws in the shared CSV files.
SUMMARY_QUANTILES: dict[str, float] = {
    "median": 0.5,
    "iqr_low": 0.25,
    "iqr_high": 0.75,
    "ci95_low": 0.025,
    "ci95_high": 0.975,
}

EPI_OUTCOMES: tuple[str, ...] = (
    "cum_incidence",
    "cum_incidence_averted",
    "cum_incidence_averted_pct",
    "cum_deaths",
    "cum_deaths_averted",
    "cum_deaths_averted_pct",
    "cum_notifications",
    "prevalence_end",
)
ECONOMIC_OUTCOMES: tuple[str, ...] = (
    "n_screened",
    "cost_provider",
    "cost_patient",
    "cost",
    "ylls",
    "ylds",
    "post_tb_dalys",
    "dalys",
)
INCREMENTAL_OUTCOMES: tuple[str, ...] = ("inc_cost", "dalys_averted")
TRAJECTORY_INDICATORS: tuple[str, ...] = (
    "incidence",
    "notifications",
    "deaths",
    "prevalence",
    "pulmonary",
    "population",
)

#: Unit written next to each quantity in the CSV files.
UNITS: dict[str, str] = {
    "cum_incidence": "new active TB, cumulative over horizon",
    "cum_incidence_averted": "new active TB averted vs no ACF, cumulative",
    "cum_incidence_averted_pct": "% of no-ACF incidence",
    "cum_deaths": "TB deaths, cumulative over horizon",
    "cum_deaths_averted": "TB deaths averted vs no ACF, cumulative",
    "cum_deaths_averted_pct": "% of no-ACF deaths",
    "cum_notifications": "notifications (routine and ACF), cumulative over horizon",
    "prevalence_end": "people with active TB or on treatment, end of horizon",
    "n_screened": "people screened, cumulative over horizon, undiscounted",
    "cost_provider": "2025 US$, discounted",
    "cost_patient": "2025 US$, discounted",
    "cost": "2025 US$, discounted",
    "ylls": "DALYs (years of life lost), discounted",
    "ylds": "DALYs (years lived with disability), discounted",
    "post_tb_dalys": "DALYs (post-TB sequelae), discounted",
    "dalys": "DALYs, discounted",
    "inc_cost": "2025 US$ vs no ACF, discounted",
    "dalys_averted": "DALYs vs no ACF, discounted",
    "incidence": "new active TB per year",
    "notifications": "notifications per year (routine and ACF)",
    "deaths": "TB deaths per year",
    "prevalence": "people with active TB or on treatment",
    "pulmonary": "people with bacteriologically positive pulmonary TB or on treatment",
    "population": "people",
}


def distribution_table(
    table: pd.DataFrame, columns: tuple[str, ...], by: list[str]
) -> pd.DataFrame:
    """One row per ``by`` group and quantity, with :data:`SUMMARY_QUANTILES` across draws."""
    grouped = table.groupby(by, sort=False)
    frames = []
    for column in columns:
        stats = pd.DataFrame(
            {name: grouped[column].quantile(q) for name, q in SUMMARY_QUANTILES.items()}
        ).reset_index()
        stats.insert(len(by), "quantity", column)
        stats.insert(len(by) + 1, "unit", UNITS[column])
        frames.append(stats)
    return pd.concat(frames, ignore_index=True)


def write_summary_csvs(
    outputs: dict[str, Any], directory: Path | str, first_year: int = 2000
) -> list[Path]:
    """Median, IQR and 95% interval of the epidemiological and economic outputs, as CSV.

    ``outputs`` is a stored run as :func:`load_posterior_outputs` returns it. Annual
    trajectories run from ``first_year`` to the end of the study horizon.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    keys = ["arm", "label"]
    labels = {arm.name: arm.label for arm in study.SCENARIOS}
    epi = outputs["epi"]
    epi = epi[epi["year"].between(first_year, study.HORIZON[1])]
    epi = epi.assign(label=epi["arm"].map(labels), year=epi["year"].astype(int))

    frontier = (
        outputs["incremental"]
        .assign(on_frontier=lambda t: t["status"] == "frontier")
        .groupby("label", sort=False)["on_frontier"]
        .mean()
        .rename("share_of_draws_on_frontier")
    )
    cost_effectiveness = expected_incremental(outputs["economic_summary"]).join(frontier)
    cost_effectiveness = cost_effectiveness.rename_axis("label").reset_index()
    names = {label: name for name, label in labels.items()}
    cost_effectiveness.insert(0, "arm", cost_effectiveness["label"].map(names))

    tables = {
        "epi_outcomes": distribution_table(outputs["epi_summary"], EPI_OUTCOMES, keys),
        "economic_outcomes": distribution_table(
            outputs["economic_summary"], ECONOMIC_OUTCOMES, keys
        ),
        "incremental_vs_baseline": distribution_table(
            incremental_vs_baseline(outputs["economic_summary"]), INCREMENTAL_OUTCOMES, keys
        ),
        "cost_effectiveness": cost_effectiveness,
        "epi_trajectories": distribution_table(epi, TRAJECTORY_INDICATORS, [*keys, "year"]),
    }
    paths = []
    for name, table in tables.items():
        path = directory / f"{name}.csv"
        table.to_csv(path, index=False)
        paths.append(path)
    readme = directory / "README.txt"
    readme.write_text(_csv_readme(outputs, tables, first_year))
    return [*paths, readme]


def _csv_readme(outputs: dict[str, Any], tables: dict[str, pd.DataFrame], first_year: int) -> str:
    worst_rhat = float(outputs["posterior_summary"]["r_hat"].max())
    converged = "converged" if worst_rhat <= 1.01 else "NOT CONVERGED: do not report"
    start, end = study.HORIZON
    return f"""Vietnam active case finding: summary outputs
{study.describe()}

Posterior draws: {len(outputs["draws"])}. Worst r_hat {worst_rhat:.3f} ({converged}).

Statistics are across posterior draws: median; interquartile range (iqr_low, iqr_high:
25th and 75th percentiles); 95% interval (ci95_low, ci95_high: 2.5th and 97.5th
percentiles). Intervals reflect calibration (parameter) uncertainty only: screening
sensitivities, costs and disability weights are fixed at their point values.

Costs are 2025 US$. Costs and DALYs are discounted at {costing.DISCOUNT_RATE:.0%} a year to
{costing.REFERENCE_YEAR} and accumulated over {start}-{end}. Differences from no active case
finding are taken within each draw before summarising.

Files
  epi_outcomes.csv             cumulative burden and burden averted, {start}-{end}
  economic_outcomes.csv        people screened, costs and DALYs by component
  incremental_vs_baseline.csv  extra cost and DALYs averted vs no active case finding
  cost_effectiveness.csv       efficiency frontier on costs and DALYs averaged over draws;
                               each ICER (US$ per DALY averted) is against the next cheaper
                               option on the frontier; share_of_draws_on_frontier is the
                               fraction of draws in which the option is not dominated
  epi_trajectories.csv         annual indicators by arm, {first_year}-{end}

Each file has one row per arm and quantity (and year for trajectories), with a unit column.
"""
