# Summer3 bugs the goldens include

`tweak_cases_calib` at `9cc8cf8`, run on summer3wip `15ad51b`, does not do what its
docstrings say. The parquet goldens are that run, bugs included. Parity tests
copy the bugs with `match_golden=True`. The default model removes them; the same
parameter values will not mean the same epidemic.

Audited by comparing every actualized edge of the reference model with the
summer4 model, on the same state. Flow **sums** match. Three behaviours do not
match the written model. `run(params, match_golden=True)` copies all three,
which is why it tracks the golden. `run(params)` (the default) removes them.

## 1. Many-to-one destinations use `arange`, not the destination

summer3 `reconcile_broadcast`, many-to-one branch
(`summer3/proto.py`, the `len(dest.compartments) == 1` case):

```python
out_dest_comps = np.repeat(dest.compartments[0], len(src))
out_dest_indices = np.arange(len(src), dtype=np.int32)
```

The compartment **objects** are the real destination, repeated. The runner
ignores those and scatters with `parent_indices`, which is `0 .. n-1`: the
first `n` compartments of the map, in map order. Edge `i` of the flow is
deposited in compartment `i`, whatever that is.

The declared destination of all three flows below is `(mtb_naive, age 0)`.
`run(params)` does that. `run(params, match_golden=True)` copies the index bug
so the golden matches.

Map order, first 16 compartments (the only ones these indices can hit):

| index | compartment |
|---|---|
| 0–7 | `mtb_naive` at ages `0, 3, 5, 10, 15, 18, 40, 65` |
| 8–15 | `incipient` at those same ages |

### `nontb_deaths_{age}` (eight flows)

Source, in map order, 10 compartments at that age:

`mtb_naive`, `incipient`, `contained`, `cleared`, `active/low/clin`,
`active/low/subclin`, `active/high/clin`, `active/high/subclin`, `treatment`,
`recovered`.

Every age uses the **same** destinations, indices `0 .. 9`. A death in
`recovered` at age 65 is added to `incipient_3`, same as `recovered` at age 0.
Only edge 0 (`mtb_naive` of that age → `mtb_naive_0`) is the birth the
docstring describes.

### `tb_mortality`

Source is clinical active, map order: for each age in
`0, 3, 5, 10, 15, 18, 40, 65`, low then high. 16 edges, destinations
indices `0 .. 15`. `active_0_low_clin` happens to land on `mtb_naive_0`.
`active_65_high_clin` lands on `incipient_65`.

### `rx_death`

Source is `treatment` in age order, 8 edges, destinations indices `0 .. 7`.
Those eight compartments are exactly `mtb_naive` at each age, so a treatment
death at age 40 is added to `mtb_naive_40`. That is an accident of layout, not
a same-age replacement. It is still not `(mtb_naive, age 0)`.

### Why the early years still matched

At `t = 0` only `mtb_naive` is occupied, so the only nonzero death edge is
edge 0, which is correct. `rx_death` is zero until detection starts in 1957.
The bug shows up once the seed creates other disease states, which is why
`mtb_naive_0` and the younger ages diverge from the 1830s while `entry_births`
and the mixing matrix stay exact. Comparing summed flow mass cannot see it.

### Default fix

Repeat the destination's real index:

```python
dest_index = int(np.asarray(dest.parent_indices)[0])
out_dest_indices = np.full(len(src), dest_index, dtype=np.int32)
```

In this repo that means one `TransitionFlow` into `(mtb_naive, age 0)` per
flow, which is already written. The default path leaves those indices alone;
only `match_golden=True` remaps them.

## 2. Progression is split twice

`add_latency_flows` asks for a dest adjustment
`[1 - p, p]` on `(low, high)`, with `p = progression_prop_infectious` (`0.2`
in `BASE_PARAMS`). The docstring says that fraction of progressions enter high
infectiousness and the rest enter low, so the weights should sum to 1.

summer3 has already scattered `1 / n` onto the two infectious strata before
the adjustment runs (`reconcile_broadcast`, scatter branch, adjustment
`1 / len(scatter_strat.strata)`). The adjustment multiplies that. Effective
weights are `0.5 * (1 - p)` and `0.5 * p`: `0.4` and `0.1` at the default,
total outflow half the named progression rate.

`match_golden=True` keeps the half scatter. The default restores weight `1` so
`[1 - p, p]` is the whole split.

## 3. `rel_infectiousness_subclin` multiplies `clin`

`infect_process` documents the weight as applying to subclinical disease.
`CLIN_STRATA` is `("clin", "subclin")`, and the wrap is

```python
clinical_cats.wrap(jnp.array([rel_infect_subclin, 1.0]))
```

so `clin` gets `rel_infectiousness_subclin` and `subclin` gets `1`. The name
does not match the stratum. The low/high wrap is in stratum order and is not
this bug.

`match_golden=True` copies the wrap. The default puts the weight on `subclin`.

## Not bugs

Checked, and either already what the code says or already matching on both
sides. Do not change them to chase the golden.

- Other edges (infection, latency, ageing, seed, detection, ACF, success,
  relapse, self-recovery, within-active flows) pair the same compartments in
  summer3 and summer4. Self-recovery collapsing infectiousness is the
  intended strata collapse, not a bad index.
- `recovered` uses `rel_sus_cleared`. The infection-flow notes say so.
- Ages whose lower bound is under 15 do not transmit. `rel_sus_children`
  applies only to `mtb_naive`.
- No ageing out of `65`. Oldest-band death rates are doubled before use.
- The seed is a triangle on calendar time, zero outside `peak ± width`.
- Treatment-success knots are sorted; death-among-unsuccessful is not. Copy
  that order. It is how the series is stored, not a scatter bug.
- `entry_births` is an absolute increment and can be negative. It matched
  the golden with relative error 0.
- summer4 cannot set `dtmax`. That is not why the trajectories differ: the
  vector fields disagree on a frozen golden state, and tightening `rtol`
  does not close the gap.
