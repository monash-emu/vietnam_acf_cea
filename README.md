# TB macro model on summer4

The Vietnam tuberculosis model from
[`monash-emu/tb_macroeconomics`](https://github.com/monash-emu/tb_macroeconomics)
branch `tweak_cases_calib` (`9cc8cf8`), reimplemented on
[summer4](https://github.com/monash-emu/summer4) `v0.2.0a4`.

Numerical parity against that branch is the definition of correct. This is not
the three-age model in `plans/tb-macro-summer4-port.plan.md`. Forward runs are
`base`, `nuts_start`, and `acf`. `tb_macro.calibration` is the likelihood from
the summer3 calibration notebook. `notebooks/05-Calibration.ipynb` samples it
with that notebook's priors: 100 warmup and 100 samples on 4 chains, folded
spline contact matrix. `notebooks/06-Optimisation.ipynb` maximises the same
likelihood with Adam, on the same bounds. `reference/bench_mixing_nuts.py` is
the sampler as a script.

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
`20240922`).

`notebooks/01-InspectRun.ipynb` is the user gate: the default run against the
`base` golden, with the disease-state plot. `notebooks/02-ReleaseModel.ipynb`
runs the same parameters with the three summer3 bugs turned off
(`run(..., release=True)`) and plots that against the golden-matching run.

In VS Code, use the default pixi environment as the notebook kernel. From the
repository root:

```bash
pixi install
pixi run install-kernel
```

Then open a notebook and choose **tb-macro-summer4 (pixi)**. The workspace
setting `python.defaultInterpreterPath` points at `.pixi/envs/default/bin/python`,
which is the same environment. Do not use the `reference` environment for these
notebooks: it does not install summer4.

## Workarounds

`grep -n GLUE src` finds the stand-ins a later summer4 package can delete.

- **KI6** — the spectrally normalised contact matrix is still rebuilt inside
  `derived_fn` (or supplied by a calibration `mixing` callback). Folded
  stepwise / spline cases already pass a table; the baseline path does not.
- **Treatment / ACF / entry** — competing treatment hazards, the ACF screen
  rate, and absolute entry births remain `FieldRef`s filled by `derived_fn`.
- **Detection** — piecewise half-cosine scale-up is `defer(get_cos_multicurve)`
  (no half-cosine rate node). The triangular seed is a native `clip` / `abs`
  rate tree; ageing is `TraitChain.from_breakpoints`; infection is
  `ForceOfInfection` with selector-keyed infectiousness; nontb deaths use
  `Data.table(...).interp`.
- **Entry births** can be negative. `EntryFlow` applies the value as an
  absolute inflow, which is what summer3 does.
- **Progression** keeps summer3's 1/2 scatter, then multiplies by
  `1 - progression_prop_infectious` and `progression_prop_infectious`.
- **Solver.** Goldens use Dopri5, initial step `0.1`, `rtol=atol=1e-9`,
  `max_steps=20_000`, and no `dtmax`. The tb_macroeconomics notebooks pass only
  `max_steps=4000` and inherit the summer3 runner: `rtol=atol=1e-5`, `dtmax=1`.
  `tb_macro.step_controls` patches `CompiledModel.run` so those controls work.
  `run()` does not pass them.
- **Float64.** Enabled in `tb_macro/__init__.py`. summer3 turns it on at import.
