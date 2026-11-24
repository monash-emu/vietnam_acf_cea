# Parallel NUTS chains crash in the diffrax adjoint

Record of a failure, not a fix. The 4×100 calibration comparison was not run.
`reference/bench_mixing_nuts.py` on this branch asks for parallel chains and
then dies. A single chain of the same likelihood works.

## What we wanted

Four chains, 100 warmup and 100 samples, of the summer4 port of the summer3
calibration likelihood, for each mixing-matrix variant. The chains have to
run at the same time. A sequential fallback is not acceptable: one chain of
the baseline already takes about 16 seconds per NUTS iteration, so four
sequential chains of 200 iterations is most of a day.

## Two separate problems

### NumPyro will silently run the chains one after another

JAX on this machine has one CPU device. NumPyro's default `chain_method` is
`"parallel"`, and `MCMC.__init__` rewrites that to `"sequential"` when
`jax.local_device_count()` is less than `num_chains`, with a warning. The
rewrite is in `numpyro/infer/mcmc.py`, just after the `chain_method` check.

`numpyro.set_host_device_count(4)` only works if it runs before JAX is
imported. It sets

```
XLA_FLAGS=--xla_force_host_platform_device_count=4
```

Importing `vacf` imports JAX, so the flag has to be set from `sys.argv`
at the top of `reference/bench_mixing_nuts.py`, before any JAX import. The
script does that, and it raises if `mcmc.chain_method` is not `"parallel"`.

Checked in a fresh process: without the flag, `jax.local_device_count()` is
1. With the flag set first, it is 4, and a trivial normal model samples with
`chain_method == "parallel"`.

### The real likelihood then crashes inside `pmap`

The device count is not enough. A one-step smoke test of the spline
likelihood,

```
pixi run python reference/bench_mixing_nuts.py --variants spline --warmup 1 --samples 1
```

prints `devices 4  chains 4` and `nuts chain_method parallel  chains 4
devices 4`, then dies. The stack is `MCMC.run` → `jax.pmap` →
`initialize_model` → `value_and_grad` → diffrax's recursive-checkpoint
adjoint → equinox `checkpointed_while_loop` backward:

```
ValueError: Closure-converted function called with different dynamic
arguments to the example arguments provided
```

The mismatch is `result=diffrax._solution.RESULTS`: `<traced>` on the call,
`<unknown>` on the closure the checkpoint was built with. A single chain,
`jit` but not `pmap`, does not do that closure conversion, and the same
likelihood's gradient matches a one-sided difference. The bug is `pmap` over
the adjoint, not the density.

The 100-step run was not started.

## What to try

Four processes, one chain each, is the path that already works. Split
`PRNGKey(2)` so each process gets its own key, run
`reference/bench_mixing_nuts.py --chains 1` four times, and concatenate the
draws. That is real parallelism and it does not go through `pmap`.

`chain_method="vectorized"` is one `vmap` on the single device. Worth one
smoke test, because `vmap` does not closure-convert the way `pmap` does. If
it raises the same `RESULTS` error, drop it.

Do not let NumPyro fall back to sequential. The guard in `_run_nuts` is
there so a later change cannot do that quietly.

Changing the diffrax adjoint (`DirectAdjoint`, `BacksolveAdjoint`) might
make `pmap` work, and it also changes the memory and the gradient. Only do
that if process parallelism is rejected. The golden `run()` path is not this
code; `make_log_likelihood` is.
