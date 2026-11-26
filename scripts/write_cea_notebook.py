"""Generate the Vietnam ACF notebooks.

  notebooks/10-ACF-CEA-Run.ipynb      calibrate, run every arm at posterior draws, store
  notebooks/11-ACF-CEA-Results.ipynb  load a stored run, plot it, write shareable CSVs
  notebooks/12-ACF-Slides.ipynb       Beamer slides on the model, fits and epi projections

Kept as a script so the notebooks are reproducible and reviewable as source rather than as
JSON blobs. Run with ``pixi run python scripts/write_cea_notebook.py``. This overwrites both
notebooks, including any outputs saved in them.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


SCENARIO_TABLE = """
| Scenario | Screening |
| --- | --- |
| `baseline` | None; routine services only |
| `acf_xpert` | Xpert as the front-line test |
| `acf_cxr_xpert` | Chest X-ray front-line, confirmatory Xpert for X-ray positives |

The second algorithm tests in series, so a case must be X-ray positive **and** Xpert
positive to be found. It is therefore less sensitive than front-line Xpert and cheaper per
person screened — the trade-off this analysis quantifies.

Everything the analysis simulates is defined in `vacf.economics.study`. Change it there,
not here.
"""

RUN_CELLS = [
    md("""
# Active case finding in Vietnam: run the analysis

Calibrates the homogeneous-mixing model, runs every arm in `vacf.economics.study` at each of
`N_DRAWS` posterior draws, and stores the outputs under `outputs/<RUN_NAME>/`. It calls
`analysis.run_and_store`, the same function as `scripts/massive/calibrate.py`, so a local
run and a MASSIVE run have the same layout. Load and plot a run with
[11-ACF-CEA-Results](11-ACF-CEA-Results.ipynb).

Twelve parameters are estimated by NUTS: the upstream model's sixteen, less the four that
only shape a contact-data-derived mixing matrix, which are unidentifiable under homogeneous
age mixing. The Canberra mixing term is dropped from the likelihood for the same reason.

Production runs belong on MASSIVE. Locally, 100 warmup and 100 samples on 8 chains took
about 40 minutes and did not converge (worst r_hat 1.28): enough to develop the results
notebook, not to report.

Restart the kernel and run from the top: the number of devices is fixed when JAX is first
imported.
"""),
    code("""
# Must run before JAX is imported, or NumPyro runs the chains one after another.
import os

N_CHAINS = 8

flags = [f for f in os.environ.get("XLA_FLAGS", "").split()
         if not f.startswith("--xla_force_host_platform_device_count=")]
os.environ["XLA_FLAGS"] = " ".join([f"--xla_force_host_platform_device_count={N_CHAINS}", *flags])

import jax

if jax.local_device_count() < N_CHAINS:
    raise RuntimeError(
        f"JAX sees {jax.local_device_count()} device(s). Restart the kernel and run this "
        "cell first, otherwise the chains run sequentially."
    )
print(f"devices: {jax.local_device_count()}")
"""),
    code("""
import logging
import warnings
from datetime import UTC, datetime

from vacf.constants import OUTPUT_PATH
from vacf.economics import analysis, study

RUN_NAME = f"local-{datetime.now(UTC):%Y%m%dT%H%MZ}"
N_RUNS = 100  # warmup iterations, and retained samples, per chain
N_DRAWS = 100  # posterior draws run through every arm
SEED = 0

logger = logging.getLogger("acf")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)

print(study.describe())
print(f"run {RUN_NAME}: {N_CHAINS} chains, {N_RUNS} warmup + {N_RUNS} samples, {N_DRAWS} draws")
banner = study.provisional_warning()
if banner:
    warnings.warn(banner, stacklevel=1)
"""),
    code("""
RUN = OUTPUT_PATH / RUN_NAME
outputs = analysis.run_and_store(
    RUN, n_runs=N_RUNS, n_chains=N_CHAINS, n_draws=N_DRAWS, seed=SEED, logger=logger
)
"""),
    code("""
summary = outputs["posterior_summary"]
worst_rhat = summary["r_hat"].max()
print(f"wrote {RUN}")
print(f"worst r_hat {worst_rhat:.3f}")
if worst_rhat > 1.01:
    print("NOT CONVERGED. Use this run for development only.")
summary[["mean", "sd", "ess_bulk", "r_hat"]]
"""),
    md("""
To inspect this run, open [11-ACF-CEA-Results](11-ACF-CEA-Results.ipynb) and set
`RUN = OUTPUT_PATH / "<RUN_NAME>"`, using the name printed above.
"""),
]

RESULTS_CELLS = [
    md(
        """
# Active case finding in Vietnam: results

Loads a stored run of the analysis and summarises it: fit to the calibration targets,
epidemiological impact, costs and DALYs, and cost-effectiveness, each as a median with an
interval across posterior draws. Also writes the CSV files to share with collaborators.
"""
        + SCENARIO_TABLE
        + """
Runs come from [10-ACF-CEA-Run](10-ACF-CEA-Run.ipynb) locally or from
`scripts/massive/calibrate.py` on MASSIVE; both write the same layout under
`outputs/<run>/`. Nothing here re-runs the model.
"""
    ),
    code("""
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from vacf.calibration import observed_targets
from vacf.constants import OUTPUT_PATH
from vacf.economics import analysis, study

pd.set_option("display.width", 200)

print(study.describe())

banner = study.provisional_warning()
if banner:
    warnings.warn(banner, stacklevel=1)
    print("\\n*** " + banner + " ***")
"""),
    md("""
## Load a run

Set `RUN` to a run directory. `outputs/local-sample-n100` is a short local run that has not
converged, for development only; for reporting, point at a downloaded MASSIVE job
(`outputs/<job_id>`).

Intervals are the central 95% of draws. Differences between arms are taken within each draw,
so uncertainty they share (the calibrated epidemic) does not widen the interval of the
difference.
"""),
    code("""
RUN = OUTPUT_PATH / "local-sample-n100"
posterior = analysis.load_posterior_outputs(RUN)

worst_rhat = posterior["posterior_summary"]["r_hat"].max()
print(f"{RUN.name}: {len(posterior['draws'])} posterior draws, worst r_hat {worst_rhat:.3f}")
if worst_rhat > 1.01:
    print("NOT CONVERGED. Treat these results as a pipeline demonstration only.")
posterior["posterior_summary"][["mean", "sd", "ess_bulk", "r_hat"]]
"""),
    md("""
### Screening sensitivity

Values follow the Kiribati methods appendix. The full algorithm (symptom screening, chest
X-ray and Xpert) *defines* the model's active TB, so it has sensitivity one in every state
by construction; the X-ray approach is scored by asking whether it would also have found
each person the full algorithm found. Symptomatic disease is detected with certainty
because symptom screening is the first step and defines symptom status in the model.

Because the confirmatory Xpert step has sensitivity one, the serial product for scenario 2
reduces exactly to the sensitivity of chest X-ray alone.
"""),
    code("""
pd.DataFrame(study.sensitivity_table()).T[list(study.DISEASE_STATES)]
"""),
    md("""
## Fit to calibration targets

Median and 95% interval of each target quantity across draws, against the data in the units
the likelihood compares.
"""),
    code("""
observed = observed_targets()
fit = analysis.with_prevalence_decline(
    posterior["calibration_fit"], base_time=observed["decline"].index[0]
)
fit = fit[fit["year"].between(1990, 2025)]

# (column, title, y-axis label, display scale)
FIT_PANELS = [
    ("notifications", "Notifications", "Per year", 1.0),
    ("deaths", "TB deaths", "Per year", 1.0),
    ("latent", "Ever infected, adults", "%", 100.0),
    ("prevalence", "Pulmonary TB prevalence, adults", "Per 100,000", 1e5),
    ("decline", "Prevalence relative to first survey", "Ratio", 1.0),
    ("infectiousness", "High-infectious share of prevalent TB", "%", 100.0),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
for ax, (name, title, ylabel, scale) in zip(axes.flat, FIT_PANELS):
    bands = analysis.quantile_bands(fit, name, ["year"]) * scale
    ax.fill_between(bands.index, bands["lower"], bands["upper"], color="tab:red", alpha=0.25,
                    label="95% interval")
    ax.plot(bands.index, bands["median"], color="tab:red", label="Median")
    target = observed[name].loc[1990:2025]
    ax.scatter(target.index, target * scale, color="black", zorder=3, label="Target")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
for ax in axes[-1]:
    ax.set_xlabel("Year")
axes[0, 0].legend(frameon=False)
fig.suptitle("Fit to calibration targets across posterior draws")
fig.tight_layout()
plt.show()
"""),
    md("""
## Epidemiological impact

Cumulative incidence and deaths over the study horizon, and the share averted relative to no
active case finding.
"""),
    code("""
analysis.interval_summary(
    posterior["epi_summary"],
    ["cum_incidence", "cum_deaths", "cum_incidence_averted_pct", "cum_deaths_averted_pct"],
).round(1)
"""),
    code("""
start, end = study.ACF_START - 5, study.HORIZON[1]
recent = posterior["epi"][posterior["epi"]["year"].between(start, end)]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, (column, title) in zip(axes, [("incidence", "TB incidence"), ("deaths", "TB deaths")]):
    bands = analysis.quantile_bands(recent, column, ["arm", "year"])
    for arm in study.SCENARIOS:
        band = bands.loc[arm.name]
        (line,) = ax.plot(band.index, band["median"], label=arm.label)
        ax.fill_between(band.index, band["lower"], band["upper"], color=line.get_color(), alpha=0.2)
    ax.axvspan(study.ACF_START, study.ACF_START + study.ACF_DURATION, color="grey", alpha=0.12)
    ax.set_title(title)
    ax.set_ylabel("Per year")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
axes[0].legend(frameon=False)
fig.suptitle("Median and 95% interval across draws; grey marks the campaign")
fig.tight_layout()
plt.show()
"""),
    md("""
## Costs and DALYs

Discounted totals over the study horizon: people screened in millions, costs in US$
millions, DALYs in thousands.
"""),
    code("""
econ_draws = posterior["economic_summary"].assign(
    screened_millions=lambda t: t["n_screened"] / 1e6,
    cost_musd=lambda t: t["cost"] / 1e6,
    dalys_thousands=lambda t: t["dalys"] / 1e3,
)
analysis.interval_summary(
    econ_draws, ["screened_millions", "cost_musd", "dalys_thousands"]
).round(1)
"""),
    code("""
medians = posterior["economic_summary"].groupby("label", sort=False).median(numeric_only=True)
costs = medians[["cost_provider", "cost_patient"]].rename(
    columns={"cost_provider": "Provider", "cost_patient": "Patient"}
)
dalys = medians[["ylls", "ylds", "post_tb_dalys"]].rename(
    columns={"ylls": "YLL", "ylds": "YLD", "post_tb_dalys": "Post-TB"}
)

fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
(costs / 1e6).plot.barh(stacked=True, ax=axes[0])
axes[0].set_xlabel("US$ millions, discounted")
axes[0].set_title("Cost by perspective")
(dalys / 1e3).plot.barh(stacked=True, ax=axes[1])
axes[1].set_xlabel("Thousands, discounted")
axes[1].set_title("DALYs by component")
axes[0].invert_yaxis()
axes[0].set_ylabel("")
fig.suptitle("Median of each component across draws")
fig.tight_layout()
plt.show()
"""),
    md("""
## Cost-effectiveness

The first table pairs each campaign with no active case finding within every draw. The
efficiency frontier and ICERs are then computed on costs and DALYs averaged over draws, the
usual summary of a probabilistic analysis: per-draw ICERs are undefined in draws where an
option is dominated, and ratios of noisy differences are heavy-tailed. An option is
**dominated** when another is both cheaper and averts more DALYs, and **extended-dominated**
when a combination of other options on the frontier does better; each ICER is against the
next cheaper option on the frontier. The count table shows how often each option is on the
frontier across draws.
"""),
    code("""
paired = analysis.incremental_vs_baseline(posterior["economic_summary"]).assign(
    inc_cost_musd=lambda t: t["inc_cost"] / 1e6,
    dalys_averted_thousands=lambda t: t["dalys_averted"] / 1e3,
)
analysis.interval_summary(paired, ["inc_cost_musd", "dalys_averted_thousands"]).round(1)
"""),
    code("""
expected = analysis.expected_incremental(posterior["economic_summary"])
display(expected[["cost", "dalys", "status", "inc_cost", "inc_dalys_averted", "icer"]])

labels = [arm.label for arm in study.SCENARIOS]
pd.crosstab(posterior["incremental"]["label"], posterior["incremental"]["status"]).reindex(labels)
"""),
    code("""
fig, ax = plt.subplots(figsize=(7, 5.5))
for label, group in paired.groupby("label", sort=False):
    ax.scatter(group["dalys_averted_thousands"], group["inc_cost_musd"], s=14, alpha=0.6,
               label=label)
ax.axhline(0, color="grey", linewidth=0.8)
ax.axvline(0, color="grey", linewidth=0.8)
ax.set_xlabel("DALYs averted vs baseline (thousands, discounted)")
ax.set_ylabel("Incremental cost vs baseline (US$ millions, discounted)")
ax.set_title("Cost-effectiveness plane: one point per posterior draw")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()
"""),
    md("""
## CSV files to share

Median, interquartile range and 95% interval of the epidemiological and economic outputs,
with a unit for every quantity, in `<run>/csv/` alongside a README that defines them. Runs
written by the current scripts already include these; this cell (re)writes them from the
stored tables, for runs made before they were added.
"""),
    code("""
paths = analysis.write_summary_csvs(posterior, RUN / "csv")
for path in paths:
    print(path.relative_to(OUTPUT_PATH))
pd.read_csv(RUN / "csv" / "epi_outcomes.csv").head(6)
"""),
    md("""
## Caveats

1. **Sensitivities are carried over from Kiribati.** They assume Vietnamese active TB is
   delimited the same way — by what the full algorithm detects — and that X-ray performance
   transfers. The two asymptomatic X-ray values, 0.2 and 0.9, lie within the uniform
   priors estimated there, 𝒰(0.07, 0.52) and 𝒰(0.72, 0.93). They cannot be identified from
   Vietnamese data, since screening is off during calibration, so vary them over those
   ranges in one-way sensitivity analysis (`study.CXR_SENSITIVITY_RANGE`).
2. **Check convergence.** The worst r_hat is printed when a run is loaded. Do not report
   results from a run above 1.01.
3. **Intervals reflect calibration uncertainty only.** Screening sensitivities, costs and
   disability weights are fixed at their point values; vary them in sensitivity analysis.
4. **Homogeneous mixing** is assumed, so no contact survey data informs the age pattern of
   transmission, and the upstream posterior does not transfer.
5. **Coverage is annual.** 80% per year over ten years means roughly ten screening rounds,
   so total screening volume is about ten times the eligible population and screening cost
   scales with campaign duration.
"""),
]


SLIDES_CELLS = [
    md("""
# Slides: transmission model and epidemiological projections

Builds a Beamer deck, as a `.tex` file and its figures, from a stored run: model structure
and assumptions, calibration and fits, scenarios, and projected epidemiological impact.
Every number on the slides is read from the run and from `vacf.economics.study`, so
rebuilding against the MASSIVE run updates the deck.

The deck reuses the theme (`Unipd`) and diagrams of the Kiribati talk
(`images/TB Model.pdf`, `images/ICE_TB.png`, `images/Classif.pdf`), so it compiles in that
folder. Copy `vietnam_model_slides.tex` and the `images_2/` folder from `OUT` into the
Kiribati slides folder, and build there as usual. Check that the reused model diagram still
matches this model, which has no preventive treatment and no hard-to-reach stratum.
"""),
    code(r"""
from vacf.constants import OUTPUT_PATH
from vacf.economics import analysis, slides, study

RUN = OUTPUT_PATH / "local-sample-n100"
OUT = RUN / "slides"

DETAILS = dict(
    title="Active TB case finding in Viet Nam: transmission model and epidemiological projections",
    short_title="Viet Nam ACF TB modelling",
    author="Romain Ragonnet, PhD, MSc",
    short_author="Romain Ragonnet",
    institute=(
        r"Epidemiological Modelling Unit (EMU)\\"
        r"School of Public Health and Preventive Medicine\\"
        r"Monash University, Melbourne, Australia"
    ),
    date=r"\today",
)
"""),
    code("""
posterior = analysis.load_posterior_outputs(RUN)
tex = slides.write_slides(posterior, OUT, run_name=RUN.name, **DETAILS)
print(f"wrote {tex}")
for path in sorted((OUT / slides.FIGURE_DIR).glob("*.pdf")):
    print(f"      {path}")
"""),
    md("## Preview of the figures"),
    code("""
display(slides.fit_figure(posterior))
for arm in study.SCENARIOS[1:]:
    display(slides.impact_figure(posterior, arm.name))
display(slides.cumulative_figure(posterior))
"""),
]


def _notebook(cells: list[dict]) -> dict:
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:02d}"  # nbformat requires a stable id per cell
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "vacf (pixi)", "language": "python", "name": "vacf"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


OUTPUTS = {
    NOTEBOOKS / "10-ACF-CEA-Run.ipynb": RUN_CELLS,
    NOTEBOOKS / "11-ACF-CEA-Results.ipynb": RESULTS_CELLS,
    NOTEBOOKS / "12-ACF-Slides.ipynb": SLIDES_CELLS,
}

if __name__ == "__main__":
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    for path, cells in OUTPUTS.items():
        path.write_text(json.dumps(_notebook(cells), indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {path} ({len(cells)} cells)")
