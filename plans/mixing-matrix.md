# Mixing matrix variants

Record of the design for a faster contact matrix, and for a smooth-gradient
alternative. The golden model is unchanged: `force_of_infection` still calls
`get_norm_c_matrix`. The new code lives in `tb_macro.contact` and is compared
in `notebooks/03-MixingComparison.ipynb`.

## What is expensive

`get_norm_c_matrix` runs on every right-hand-side evaluation. For Vietnam that
builds a 121×121 kernel (single ages 0 through 120), projects it to the 8 age
bands, column-scales by population, and divides by the spectral radius of the
8×8.

The kernel is three pieces:

- background, a constant `bg_mixing` on every age pair;
- assortative mixing, `(1/a) exp(-|i-j|/a)`, which depends only on `a_spread`;
- parent–child mixing, `pc_strength` times fertility at the younger person's
  birth year, indexed by the age gap.

`get_year_index` clips time to the table and truncates toward zero. Age gaps
are integers, so `floor(t - k) = floor(t) - k`. Weights, populations, fertility
and therefore the whole matrix depend on the three scalars and on `floor(t)`
only. The matrix is constant on each calendar year. The cast's derivative in
time is zero, so a Dopri5 step that crosses 1 January evaluates two different
matrices while the stepper assumes one smooth field. Parameter gradients of
`bg_mixing`, `a_spread` and `pc_strength` still exist: the index does not
depend on them. Gradients in time do not.

XLA will not notice that successive solver stages share a year. The 121×121
`exp`, the fertility gather and the eigenproblem are repeated for every stage.

## What factors

Projection `S = W K Wᵀ` is linear in `K`, and `K` is linear in the three
scalars, so

```
S(t) = bg * B(t) + Sa(t, a) + pc * S_pc(t)
C(t) = S(t) * p(t)
C_norm = C / ρ(C)
```

`B[g,h]` is the product of the two band totals of `W`. It is not a filled
matrix of ones: a band whose weights sum to 0 contributes a 0 row. After the
usual within-band normalisation those totals are 1, and `B` is the all-ones
matrix. Parent–child does not depend on any parameter. Assortative mixing is a
dot product of a lag tensor with `(1/a) exp(-k/a)`:

```
L[g,h,k] = sum_{i in g, j in h, |i-j|=k} W_i W_j
Sa[g,h] = (1/a) sum_k L[g,h,k] exp(-k/a)
```

`L`, `S_pc`, `B` and `p` are functions of the demographic tables only. The only
nonlinear step is the spectral radius, so it is applied after the scalars are
combined. It is homogeneous of degree 1: an overall scale disappears, relative
weights do not.

A fixed scalar is folded into the stored curves at prepare time and is not a
tracer. `a_spread` is the only one that rebuilds an assortative curve. `bg`
and `pc` are scales on curves that already exist.

## Do not interpolate the normalised matrix

Linearly blending two normalised matrices does not keep the dominant
eigenvalue at 1. That within-year factor multiplies the force of infection and
therefore the gradient of every infection parameter, not only the mixing
scalars. Blending the inputs and then normalising is a different function from
blending the normalised output, because projection is quadratic in `W` and the
radius does not commute with addition.

The smooth variant treats the annual tables as samples, interpolates those
series, aggregates, and only then divides by the radius.

## Two variants

Both agree with `get_norm_c_matrix` at integer years inside the weight table
(1950 through 2101 for the current Vietnam extract). Between those years they
differ on purpose.

**Stepwise** (`prepare_stepwise`) is the same vector field as the baseline.
One record per integer year: `B`, `S_pc`, `L`, `p`, built in NumPy with the
same clip-then-truncate fertility index, including birth year `t - min(i,j)`.
`materialize(bg, a, pc)` dots the lag tensor, adds the scaled pieces,
column-scales and normalises. That runs once per draw, outside a solver. The
step indexes the resulting `(n_years, 8, 8)` table with the same truncation as
`get_year_index`. Exact on the closed interval from the first weight year to
the last. Past that interval the baseline's fertility lookup can still move
while weights have already clipped; callers should stay inside the table.

**Smooth** (`prepare_smooth`) replaces the floor with a Fritsch–Carlson PCHIP
through weights, fertility and population. No new dependency: slopes are fit
in NumPy, the cubic is evaluated with Horner on JAX arrays so `jax.grad` sees
a C1 function. This is not `jnp.interp` and not the half-cosine used for
detection. PCHIP is local and does not overshoot, so non-negative demographic
series stay non-negative and the Perron root stays simple. After the
interpolant, each age band and each fertility row is renormalised so they stay
distributions. That division is the identity at measurement years.

Parent–child uses `fert(t - min(i,j))`, not `fert(floor(t) - min(i,j))`. The
exact aggregation of those curves is sampled at four knots per year. The step
evaluates PCHIP of the 8×8 curves (and of `L` if `a_spread` is free), dots the
length-121 kernel, column-scales, then calls `eigvals`. The radius is 1 at
fractional times. Four knots is an implementation detail of the stored curves,
not a second definition: `SmoothContact.exact_matrix` aggregates the
interpolated inputs directly, and the stored curves must stay within `1e-3`
relative of that. If they do not, add knots. Do not loosen the bound.

Hermite cubics are linear, so a fixed `a_spread` contracts both the lag-tensor
values and their slopes. That is the same function as interpolating `L` and
then dotting, and the step no longer touches `L`.

## Gradients

Stepwise: `dC/dt = 0` between integers. Cotangents in the three scalars land
on the active year only. Smooth: `dC/dt` is the derivative of the cubic, and
the left and right derivatives agree at a knot. Cotangents in `bg` and `pc`
are 8×8 scalings plus the radius. Cotangents in `a` hit `exp(-k/a)` either in
the step (free `a`, small dot) or once when the curve is folded. Neither path
casts time to an integer.

The 8×8 eigenproblem stays in the smooth step because that is what keeps the
radius equal to 1 along the path. Its vector-Jacobian product is stable while
the dominant eigenvalue is simple, which a positive contact matrix gives.

## Separation roadmap

The contact code does not import `model` or summer4. Age breaks and `MAX_AGE`
have defaults from `tb_macro.constants` but are arguments. Demographic arrays
are arguments, not closures over Vietnam.

1. This branch: `ContactMatrix` plus the two builders. The baseline function
   stays the model, so the goldens stay valid.
2. Once stepwise has stayed exact, `force_of_infection` can index a
   materialised table instead of calling `get_norm_c_matrix`. That is still
   the summer3 vector field. Smooth stays opt-in; it is a different field
   between 1 January dates and must not be swapped in under the parity tests.
3. Drop the default import of the Vietnam age breaks. Every caller passes
   breaks, fertility, weights and populations.
4. Move `tb_macro.contact` to its own package when a second model needs it.
   `tests/test_contact.py` and `notebooks/03-MixingComparison.ipynb` are the
   checks that move with it.

Entry rates are still a step lookup. Making the mixing matrix C1 does not make
the full right-hand side C1.
