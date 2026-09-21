# TB macro model on summer4

The Vietnam tuberculosis model from
[`monash-emu/tb_macroeconomics`](https://github.com/monash-emu/tb_macroeconomics)
branch `tweak_cases_calib` (`9cc8cf8`), reimplemented on
[summer4](https://github.com/monash-emu/summer4) `v0.2.0a3`.

Numerical parity against that branch is the definition of correct. This is not
the three-age model in `plans/tb-macro-summer4-port.plan.md`. Forward runs are
`base`, `nuts_start`, and `acf`. `tb_macro.calibration` is the likelihood from
the summer3 calibration notebook, and `reference/bench_mixing_nuts.py` compares
mixing-matrix variants under a short NUTS run of that likelihood.

```bash
pixi install
pixi run test
```

Regenerate goldens only in the reference environment, which pins summer3wip
`15ad51b` and does not install summer4:

```bash
pixi run -e reference golden
```

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

- **TM3** — ageing is one hand-written `TraitChain`, rate `1/width` on each of
  the seven steps `0→3→5→10→15→18→40→65`.
- **TM4** — the triangular seed and the cosine detection scale-up are
  `Transform`s of the copied functions. Solver `t` is the summer3 index;
  calendar year is `t + 1800`.
- **TM5 / KI6** — force of infection, the spectrally normalised mixing matrix,
  per-age death rates, treatment hazards and the ACF rate are one `derived_fn`.
  `ForceOfInfection` cannot express infectiousness on two properties, the
  under-15 transmission mask, or a matrix rebuilt every step.
- **KI10** — death rates are eight `jnp.interp` calls. `Interp` is scalar-only.
- **Entry births** can be negative. `EntryFlow` applies the value as an
  absolute inflow, which is what summer3 does.
- **Progression** keeps summer3's 1/2 scatter, then multiplies by
  `1 - progression_prop_infectious` and `progression_prop_infectious`.
- **Solver.** Goldens use Dopri5, initial step `0.1`, `rtol=atol=1e-9`,
  `max_steps=20_000`, and no `dtmax`. The tb_macroeconomics notebooks pass only
  `max_steps=4000` and inherit the summer3 runner: `rtol=atol=1e-5`, `dtmax=1`.
  `CompiledModel.run` accepts those controls. `run()` does not pass them.
- **Float64.** Enabled in `tb_macro/__init__.py`. summer3 turns it on at import.
