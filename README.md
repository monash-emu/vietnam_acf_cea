# tb-macro-summer4

TB natural-history model from [`monash-emu/tb_macroeconomics`](https://github.com/monash-emu/tb_macroeconomics),
reimplemented on [summer4](https://github.com/monash-emu/summer4) `v0.2.0a1`.

Numerical parity against summer3wip goldens is the definition of correct. The
model is the three-age (`0`, `5`, `15`) specification in
`plans/tb-macro-summer4-port.plan.md`: homogeneous `ones(3, 3)` mixing, both
detection flows kept. Upstream `tb_macroeconomics` has since added demography
and a dynamic mixing matrix; those are not in this port.

```bash
pixi install
pixi run test
```

Regenerate goldens only in the reference environment, which pins summer3wip and
does not install summer4:

```bash
pixi run -e reference golden
```

`notebooks/01-TBModel.ipynb` is the user gate: run it and check that the
posterior for `contact_rate` covers 0.25.

## Workarounds for the summer4 handoff

These are the gaps the port papers over at tag `v0.2.0a1`. `grep -n GLUE src`
finds the ones a later package deletes.

- **TM3** — ageing is a hand-written `TraitChain` with rates `1/5` and `1/10`.
- **TM4** — triangular seed and tanh detection scale-up are `Transform`s.
  Calendar time is shifted by 1800 before those functions, because the original
  solver's `t` is the save index. The vendored `jnp.clip(..., a_min=)` does not
  exist on JAX 0.11; both sides use `jnp.maximum(..., 0)`.
- **TM5** — force of infection is a `derived_fn`: `ones(3, 3) @ (I / N**exp) *
  contact_rate`, with `rel_sus_*` as `Multiply` adjustments. `I` is unweighted
  `active`.
- **TM6** — no glue. `InitialPopulation` already does this. The seed is one
  compartment per age, so the ragged even-split is not exercised.
- **TM8** — hand-written numpyro. `CompiledModel.run` is not vmap-safe, so
  chains are sequential rather than `vectorized`. Current numpyro rejects the
  original `Poisson(target).log_prob(continuous_sum)`; the port uses the same
  formula via `gammaln`. The two leading target entries are 0, and `log(0)`
  makes the gradient NaN, so those entries are left out of the factor.
- **Solver.** summer4 uses `dt` as Dopri5's initial step and cannot set
  `dtmax`. The reference uses `dt0=0.1` and `dtmax=1`. The port passes
  `dt=0.1`. With that, compartments, flows and the target match the goldens at
  `rtol=1e-6`, `atol=1e-9`.
- **Save plan.** An empty `SavePlan` also stores every parameter path, and
  those paths overwrite flows that share a name (`detection`, `progression`,
  `self_recovery`, `treatment_recovery`, `treatment_relapse`). The run saves
  compartments and flows explicitly.
- **Float64.** The port turns on `jax_enable_x64`. summer4 does not. The
  reference does, inside summer3.
- **Both detection flows are kept**, constant `passive_detection` and the
  time-varying `detection` on clinical active disease.

