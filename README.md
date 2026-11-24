# Vietnam active case finding: epidemiological impact and cost-effectiveness

## Running the analysis

The environment is managed with [pixi](https://pixi.sh/). You need pixi **0.84 or newer**;
older versions cannot read this repository's lock file.

```bash
pixi install          # one-off; creates .pixi/envs/default
pixi run test         # full test suite
pixi run cea          # scenario comparison, printed to the terminal (~1 min)
```

`pixi run cea --calibrate` fits the model first. That is slow and the default chain is too
short to converge; production fits belong on the cluster (`scripts/massive/`).

### Notebook

```bash
pixi run install-kernel   # one-off; registers "Python (vietnam_acf_cea)" at user level
pixi run notebook         # opens JupyterLab in notebooks/
```

Select the **Python (vietnam_acf_cea)** kernel, then:

- `10-ACF-CEA-Run.ipynb` calibrates the model, runs every scenario at 100 posterior draws
  and stores the outputs under `outputs/<run>/`. It calls the same function as
  `scripts/massive/calibrate.py`, so local and cluster runs have the same layout.
- `11-ACF-CEA-Results.ipynb` loads a stored run and plots it. It never re-runs the model.

Each run directory includes `csv/`: the median, interquartile range and 95% interval of
the epidemiological and economic outputs, with units and a README, to share with
collaborators.

### In VS Code

The kernel is registered at *user* level deliberately. VS Code only finds an
environment-local kernelspec once it has discovered the pixi interpreter, which it will not
do when the workspace is opened on a parent folder rather than on this repository. A
user-level kernel is visible from any workspace.

If it still does not appear: **Select Kernel → Jupyter Kernel… → Python (vietnam_acf_cea)**.
Failing that, run *Developer: Reload Window*, since VS Code caches the kernel list.

Note that `.vscode/settings.json` uses `${workspaceFolder}`, so its interpreter path is only
correct when VS Code is opened directly on this folder. Opening the parent folder works for
the notebook but leaves linting and the integrated terminal pointing elsewhere.

The kernelspec sets `XLA_FLAGS` for four devices. The run notebook's first cell raises
that to its `N_CHAINS` (8, as on MASSIVE) before JAX is imported, and raises an error if
JAX was already imported, since the chains would otherwise run one after another.

Both notebooks are generated from `scripts/write_cea_notebook.py`, so edit that and run
`pixi run write-notebook` rather than hand-editing the JSON.

### What the analysis simulates

Everything is defined in `src/vacf/economics/study.py` — scenarios, campaign duration,
coverage, eligibility age and screening sensitivities. Change it there; the notebook, the
command-line runner and the tests all read from it.

---

## Model provenance

The Vietnam tuberculosis model from
[`monash-emu/tb_macroeconomics`](https://github.com/monash-emu/tb_macroeconomics)
branch `tweak_cases_calib` (`9cc8cf8`), reimplemented on
[summer4](https://github.com/monash-emu/summer4) at `c9548d5` (after
`v0.2.0a5`).

The default model is the corrected docstring version: replacement deaths go to
age 0, progression weights sum to one, and `rel_infectiousness_subclin` applies
to subclinical disease. Contact uses the folded spline matrix; entry births use
the smooth (C0) lookup. Parity against the summer3 goldens (bugs included,
year-frozen mixing, step entry) is opt-in via `match_golden=True`. This is not
the three-age model in `plans/vacf-port.plan.md`. Forward-run
goldens are `base`, `nuts_start`, and `acf`. `vacf.calibration` is the
likelihood from the summer3 calibration notebook (seven custom log/logit-normal
terms plus a Canberra mixing distance). That density stays hand-rolled for now:
summer4's `BayesianModel` stock likelihoods (Normal / Poisson / NB on raw
outputs) do not cover those transforms. `notebooks/05-Calibration.ipynb`
samples it with that notebook's priors: 100 warmup and 100 samples on 4 chains,
folded spline contact matrix. `notebooks/06-Optimisation.ipynb` maximises the
same likelihood with Adam, on the same bounds. `reference/bench_mixing_nuts.py`
is the sampler as a script. `scripts/massive/` is the SLURM entry point for an
8-chain remote run on massive/M3 of the Vietnam ACF study: it calibrates the
homogeneous-mixing model, runs every arm at 100 posterior draws, and writes the
posterior (netCDF) and per-draw epidemiological, economic and calibration-fit
tables (parquet) under `outputs/<job_id>/`. The script's docstring lists the files.

```bash
pixi install
pixi run test
```

Regenerate goldens only in the reference environment, which pins summer3wip
`15ad51b` and does not install summer4:

```bash
pixi run -e reference golden
```

That writes forward-run parity under `tests/golden/{base,nuts_start,acf}/` and
calibration log-density parity under `tests/golden/calibration/` (eight draws:
`base`, `nuts_start`, and six uniform samples inside `PARAM_BOUNDS`, seed
`20240922`). Those goldens include the three summer3 bugs; see
`reference/summer3-bugs.md`.

`notebooks/01-InspectRun.ipynb` is the user gate: the default corrected run,
with the disease-state plot. `notebooks/02-ReleaseModel.ipynb` compares that
to `run(..., match_golden=True)` against the summer3 parquet.

In VS Code, use the default pixi environment as the notebook kernel. From the
repository root:

```bash
pixi install
pixi run install-kernel
```

Then open a notebook and choose **vacf (pixi)**. The workspace
setting `python.defaultInterpreterPath` points at `.pixi/envs/default/bin/python`,
which is the same environment. Do not use the `reference` environment for these
notebooks: it does not install summer4.

## Workarounds

`grep -n GLUE src` finds the stand-ins a later summer4 package can delete.

- **Treatment / ACF / entry** — competing treatment hazards, the ACF screen
  rate, and absolute entry births remain `FieldRef`s filled by `derived_fn`.
  Entry uses `smooth_birth_rate_at` by default; `match_golden=True` keeps the
  step lookup.
- **Detection** — piecewise half-cosine scale-up is `defer(get_cos_multicurve)`
  (no half-cosine rate node). The triangular seed is a native `clip` / `abs`
  rate tree; ageing is `TraitChain.from_breakpoints`; infection is
  `ForceOfInfection` with selector-keyed infectiousness; nontb deaths use
  `Data.table(...).interp`. Contact is the folded spline on the default path
  (`match_golden` rebuilds `get_norm_c_matrix`).
- **Entry births** can be negative. `EntryFlow` applies the value as an
  absolute inflow, which is what summer3 does.
- **Progression** restores weight `1` after compile so
  `1 - progression_prop_infectious` and `progression_prop_infectious` sum to
  one. `match_golden=True` keeps summer3's extra 1/2 scatter.
- **Solver.** Goldens use Dopri5 via `vacf.solvers.FORWARD_SOLVER`
  (`rtol=atol=1e-9`, uncapped step, `max_steps=20_000`). Calibration uses
  `vacf.solvers.CALIBRATION_SOLVER` (`rtol=atol=1e-5`, `dtmax=1`,
  `max_steps=4000`) as a `Diffrax` backend — summer4 `run` has no `dtmax=`
  keyword; the controller travels inside `solver=`.
- **Calibration API.** summer4 `v0.2.0a5` closed ports row TM8 with
  `BayesianModel`; this port keeps the summer3 seven-term density until those
  terms are expressible as `OutputSet` / `Target` likelihoods.
- **Float64.** Enabled in `vacf/__init__.py`. summer3 turns it on at import.
