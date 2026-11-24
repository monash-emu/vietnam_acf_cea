# Spline mixing matrix

Successor to `plans/mixing-matrix.md`, which stands as written. That note built
`prepare_stepwise` and `prepare_smooth` and left two things on the table: a
stepwise field that still jumps on every 1 January, and a smooth field that
still solves an 8×8 eigenproblem inside every solver stage. `prepare_spline`
removes both. The golden model is still unchanged; `force_of_infection` still
calls `get_norm_c_matrix` unless a `mixing` callable is passed.

## Where the time actually was

`reference/time_mixing.py` measures a whole solve, and the number that matters
is not the wall time of `compiled.run`. Roughly 1.2 s of every `compiled.run`
call is equinox re-tracing `diffeqsolve`: `summer4.solvers.diffrax_backend`
rebuilds the `SubSaveAt` closures on each call, and `filter_jit` treats a
function as static and keyed by identity, so the cache misses every time. That
constant is the same for every mixing variant and it hides the differences
between them. A calibration does not pay it, because it jits the whole
log-density once. The script therefore reports both the raw call and the same
solve inside one outer `jax.jit`.

Under the outer jit, on this branch and this machine, six draws at
`rtol = atol = 1e-9`, from `reference/time_mixing.py`:

| mixing                                | median solve | steps | rejected |
| ------------------------------------- | -----------: | ----: | -------: |
| `get_norm_c_matrix` (baseline)        |      2.125 s | 8 986 |    4 195 |
| `prepare_smooth`, free in the stage   |      0.999 s | 7 144 |    3 242 |
| `prepare_stepwise`, folded per draw   |      0.501 s | 8 976 |    4 192 |
| `prepare_spline`, folded per draw     |      0.411 s | 7 306 |    3 373 |
| `prepare_spline` + `--entry smooth`   |      0.192 s | 3 402 |      527 |

`notebooks/04-MixingRuntime.ipynb` reproduces this against the summer3 snapshot
and agrees: 2.14 s baseline, 0.50 s stepwise, 0.77 s smooth, 0.40 s spline.

5.2× on the mixing matrix alone, 11× once the other 1 January discontinuity
goes too. The per-parameter-set fold is 0.59 ms and its reverse pass 1.01 ms,
against 1.62 ms and 5.92 ms for the stepwise table: the `eig` VJP is most of
that difference, and the spline has no `eig`.

Splitting that into a marginal cost per solver step and a step count separates
the two effects, and they are genuinely separate:

| mixing            | µs / step | steps at `1e-11` |
| ----------------- | --------: | ---------------: |
| baseline          |     221.5 |           15 370 |
| stepwise, folded  |      61.0 |           15 068 |
| spline, folded    |      61.0 |           13 331 |
| spline + C1 entry |      66.2 |            7 556 |

The first column is what `prepare_stepwise` already won: not rebuilding a
121×121 kernel and an eigenproblem per stage is worth 3.6× on the step itself.
The second column is what is left, and it is a smoothness problem, not an
arithmetic one.

## Rejected steps are the remaining cost

At `rtol = 1e-9` the baseline takes 8 658 steps and **rejects 4 162 of them**.
Dopri5 under a PID controller assumes a smooth field; every discontinuity is a
step proposed, evaluated over six stages, and thrown away.

Two discontinuities cause almost all of them:

- the mixing matrix, which `get_year_index` freezes on each calendar year and
  which therefore jumps every 1 January;
- `demography.birth_rate_at`, a `searchsorted` step lookup, which jumps on the
  same dates.

Removing them one at a time, all else equal:

| field                                | steps | rejected |
| ------------------------------------ | ----: | -------: |
| baseline                             | 8 658 |    4 162 |
| constant matrix, step entry rate     | 6 790 |    3 239 |
| constant matrix, C1 entry rate       | 3 032 |      564 |
| stepwise table, C1 entry rate        | 8 522 |    4 092 |
| **spline, C1 entry rate**            | 3 327 |      480 |

Neither fix helps much alone. A C1 mixing matrix under a step entry rate still
rejects 3 250 steps; a C1 entry rate under a stepwise matrix still rejects
4 092. Together the rejections fall by a factor of eight, and a spline matrix
lands within 4% of the step count of a *constant* matrix, which is the floor.

The entry rate is not part of this work and is not changed here.
`demography.smooth_birth_rate_at` is provided, unused, so the measurement can
be reproduced and so the choice stays with the modeller: it is a different
vector field and the goldens do not cover it.

## What `prepare_spline` is

One knot per calendar year, the normalised matrix evaluated there, and a
Catmull-Rom cubic between.

```
C_k      = (bg B_k + Sa_k(a) + pc S_pc_k) * p_k         once per parameter set
u_k, v_k, rho_k                                          once per parameter set
Chat_k   = C_k / rho_k                                   once per parameter set
C(t)     = cubic(Chat, t) / rayleigh(cubic(u, t), ., cubic(v, t))    per stage
```

Only the last line is in the solver, and it is a gather and a Horner cubic:
0.04 µs against the baseline stage's 50 µs, measured inside a `fori_loop` so
that dispatch is not what is being timed. The knot components `B_k`, `S_pc_k`,
`L_k` and `p_k` are the demography and are built once per process (70 ms);
folding the three scalars and fitting the curve is 0.3 ms per parameter set.

Because the knots are the measurement years, the curve passes through
`get_norm_c_matrix` at every year of the weight table to `1e-15`.

### Interpolating the normalised matrix

`plans/mixing-matrix.md` argued against exactly this, on the grounds that a
blend of two matrices of spectral radius 1 does not have spectral radius 1, and
that the resulting within-year factor multiplies the force of infection and so
the gradient of every infection parameter. The argument is correct. What it
lacked was a number: on annual knots the drift is `3e-5`, against an
interpolation error in the entries themselves of `4e-4` on entries of size
`1e-1`. The effect it warns about is an order of magnitude smaller than the
error it was protecting.

It is also removable for two matrix–vector products. A positive contact matrix
has a simple real dominant root, so store its left and right dominant vectors
as curves of their own and divide by the Rayleigh quotient `u C v / u v` of the
interpolated matrix. The quotient is stationary at the true pair, so an
interpolated pair that is wrong by `e` gives a radius wrong by `e²`: the
measured drift falls from `3e-5` to `1e-9`. That is `pin_radius`, and it is on
by default; turning it off saves nothing measurable in a solve.

### No eigensolver anywhere

`normalise_by_spectral_radius` calls `jnp.linalg.eigvals`: a general complex
eigensolver, on a matrix whose dominant root is known to be real and simple,
and one that JAX only lowers on CPU. That last point is not a detail — it is
why the present vector field cannot run on a GPU at all.

`vacf.contact.perron` replaces it with normalise-and-square. Squaring a
matrix squares `|λ₂/λ₁|`, so `n` squarings are worth `2**n` power iterations;
ten of them are ten batched 8×8 products. The result is numerically rank one,
so its row and column sums are the dominant vectors, and the radius is their
Rayleigh quotient.

The worst corner of the Vietnam parameter box (`bg_mixing` 0.001, `a_spread` 2,
`pc_strength` 0.05 — a nearly diagonal matrix) has `|λ₂/λ₁| = 0.90`. Plain power
iteration reaches only `1e-5` there after 32 steps; squaring reaches `3e-15` at
every corner tested. It is also 3–5× faster than `eigvals` — 0.34 ms against 1.8 ms on 302 knots,
0.19 against 0.63 on the 152 of the weight table — and runs on any backend.

The vectors carry `stop_gradient`, so the cotangent is `u (dC) v / u v`. That
is not an approximation of the dominant eigenvalue's derivative, it is the
derivative, which is why it matches finite differences to nine digits.

### Catmull-Rom, not PCHIP

Slopes are central differences on unit knots, one-sided at the two ends. The
rule is linear in the values, so a cotangent on `bg_mixing`, `a_spread` or
`pc_strength` passes through the fit unchanged and the fit needs no derivative
rule of its own — `contact.pchip.eval_jax` had to declare a `custom_jvp` that
refuses to differentiate its knot data, and this does not. PCHIP's monotone
limiter exists to stop overshoot from driving a non-negative series negative; a
normalised contact matrix is bounded away from zero while `bg_mixing` is
positive, so the limiter would only flatten real curvature at local extrema.

The rule is local: one segment reads four knots. `SplineContact.matrix` with
free scalars fits exactly those four, and agrees with the materialised curve to
`3e-16`. "Free in the stage" and "folded per draw" are therefore the same
function at two prices, not two approximations.

### Annual knots only

`prepare_smooth` interpolates the inputs and then aggregates; `prepare_spline`
aggregates the measured years and then interpolates. Between years the two
disagree by about `4e-4`. Neither is the reference for the other: the
demographic series are annual and nothing in the data says what happens in
July. The cheaper convention is the one that does not need 121-length gathers
at four knots per year.

## Precomputation layers

The three scalars can be fixed at three different times, which is what a
calibration needs.

- **Fixed at prepare** (`prepare_spline(..., a_spread=9.87)`): folded into
  `offset` on the host, no longer a tracer, and the lag tensor is dropped.
  Once per calibration.
- **Folded per parameter set** (`contact.materialize(bg, a, pc)`): 0.3 ms,
  fully differentiable in all three. This is the intended calibration path.
- **Free in the stage** (`contact.matrix(t, bg, a, pc)`): the four-knot local
  fit. Correct and still cheaper than the baseline, but there is no reason to
  use it when `materialize` exists.

`SplineCurve` is registered as a pytree with `year0` static, so a curve crosses
a `jit` boundary as one argument and a new draw does not recompile.

## How much of a solve is the mixing matrix

A fixed-step solve (`ConstantStepSize`, `dt = 0.02`, 15 100 steps) makes every
case evaluate the right-hand side exactly the same number of times, so the
remaining difference is the arithmetic that was swapped out. Compiling all the
cases first and then timing them round-robin brings the repeatability to 0.3%;
timing them in build order does not, and is worth about 7%.

Replacing the matrix with a single constant 8×8 — no lookup, no time
dependence — is the floor:

| mixing in the stage       | solve   | share that is the matrix |
| ------------------------- | ------: | -----------------------: |
| `get_norm_c_matrix`       | 0.929 s |                    ~75%  |
| stepwise table            | 0.213 s |                     ~0%  |
| spline curve              | 0.232 s |                     ~0%  |
| constant 8×8              | 0.233 s |                       —  |

**Three quarters of the baseline's per-step work is building the contact
matrix.** Once it is precomputed, whether by a table or a curve, its cost is
indistinguishable from having no time-varying matrix at all: the three lower
rows sit inside each other's noise, and the in-jit stage timings in
`notebooks/03-MixingComparison.ipynb` put them at 0.03–0.11 µs against a step
of roughly 50 µs.

That is the reason the spline's advantage is entirely the step count. Its
stage is in fact 3.4× *more* expensive than the stepwise gather. Both are noise.

## What else is time-varying

With the spline in place, replacing each remaining time-varying input by a
constant. The figures are that input *and everything downstream of it*: a
constant death rate also makes `_outcome_rates` loop-invariant, so XLA hoists
the whole chain, and the number is an upper bound on what the lookup itself
costs.

| replaced by a constant        | saved   | share of the solve |
| ----------------------------- | ------: | -----------------: |
| death rates (`_interp_deaths`)| 0.046 s |               ~20% |
| entry rate (`birth_rate_at`)  | 0.012 s |                ~5% |
| TSR and unsuccessful-treatment curves | 0.004 s |         ~2% |
| ACF screen                    |     0   |                 0% |
| all four                      | 0.061 s |               ~26% |

The ACF campaign is free because `acf_coverage` is 0 during calibration and
XLA folds the branch away.

### Two micro-optimisations that were tried and are not worth it

Both were implemented, checked against the goldens, A/B-tested with old and new
compiled in one process and timed round-robin, and reverted. The comments in
`model.py` and `curves.py` record the result so nobody repeats them.

- **One `searchsorted` for all eight ages in `_interp_deaths`** instead of eight
  unrolled `jnp.interp` calls. Measured 3% *slower*. The premise was wrong: the
  eight calls share one scalar time and one knot array, so XLA already
  eliminates the duplicate searches, and the strided row gather the fused
  version needs costs more than eight contiguous column gathers.
- **`jnp.searchsorted` in place of `curves.binary_search_sum_ge`**, which is a
  hand-rolled `lax.while_loop` evaluated twice per stage. Provably the same
  integer, and 1.7% faster, which is inside the run-to-run spread.

The honest conclusion is that after the mixing matrix there is no arithmetic
left in this right-hand side worth chasing. The floor — constant matrix, no
time-varying inputs at all — is 0.169 s of the 0.232 s spline solve. The rest
is summer4's own per-step machinery and the 302 save points.

## A silent zero gradient in the smooth path

Found while wiring `prepare_smooth` into the outer-jit comparison, and fixed
here because it is a correctness bug, not a performance one.

`SmoothContact.materialize` called `float(bg_mixing)` on its arguments, so the
folded smooth path — the fast one, the one a calibration would use — could not
be traced at all. Making it traceable then exposed the real problem underneath:
`contact.pchip.eval_jax` is a `custom_jvp` whose rule propagated only the
tangent of `t` and dropped the tangents of the knot values and slopes. Folding
a mixing scalar puts it *into* the knot values, so `jax.grad` of a folded
smooth matrix returned exactly zero for `bg_mixing`, `a_spread` and
`pc_strength`, with no error. A NUTS run would have sampled those three from
the prior and never said so.

The cubic is linear in the knot values and slopes, so the fix is the same
cubic evaluated on the tangents. The folded gradient now matches the free path
and central differences. `test_pchip_gradient_reaches_the_knot_values` and
`test_smooth_folded_gradients_match_the_free_path` are the regressions.

`prepare_spline` never had this: its fold is JAX throughout, and its
interpolation has no `custom_jvp` because Catmull-Rom slopes are linear in the
values and need no derivative rule of their own.

## Not done

- `force_of_infection` still calls `get_norm_c_matrix`. Swapping it is a
  separate change and must not happen under the parity tests: the spline is a
  different vector field between 1 January dates.
- The entry rate stays a step lookup. Without that change most of the step-count
  win above is unavailable.
- The ~1.2 s re-trace in `compiled.run` is a summer4 harness issue, not a model
  one, and is worth reporting upstream.
