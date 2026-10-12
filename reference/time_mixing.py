"""Time a whole solve under each mixing matrix. Runs in the default environment.

``reference/time_solve.py`` times the summer3 snapshot. This times summer4 and
compares ``get_norm_c_matrix`` with the three builders in ``tb_macro.contact``.
The golden path is untouched: every case here passes a ``mixing`` callable to
``compile_model``.

Roughly 1.2 s of every ``compiled.run`` call is equinox re-tracing
``diffeqsolve`` — ``summer4.solvers.diffrax_backend`` rebuilds the ``SubSaveAt``
closures on each call and ``filter_jit`` keys a function by identity, so the
cache misses every time. That constant is identical across the cases and swamps
what is being compared, so the table also reports the same solve wrapped in one
outer ``jax.jit``, which is what a calibration does to its log-density.

``--entry step`` swaps ``entry_rate_at`` back to the discontinuous
``birth_rate_at``. The default model uses the smooth (C0) lookup; the step
field is what ``match_golden`` and the summer3 goldens record. The flag is here
because mixing comparisons need both entry conventions. See
``plans/mixing-spline.md``.

    pixi run python reference/time_mixing.py
    pixi run python reference/time_mixing.py --entry step
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Callable

import jax
import numpy as np
from jax import numpy as jnp

import tb_macro.demography as demography
import tb_macro.model as model
from tb_macro.contact import prepare_smooth, prepare_spline, prepare_stepwise
from tb_macro.contact.spline import SplineCurve
from tb_macro.model import STATIC, compile_model, save_plan
from tb_macro.parameters import BASE_PARAMS, PARAM_BOUNDS
from tb_macro.solvers import FORWARD_SOLVER

N_SETS = 6
SEED = 0
# The solver queries well before the weight table starts; the year tables cover
# the whole span and clip each series the way ``get_year_index`` clips it.
YEARS = np.arange(1800.0, 2102.0)
SOLVER = dict(t0=0.0, t1=302.0, dt=0.1, **FORWARD_SOLVER)
ARRAYS = (
    STATIC.age_weights,
    STATIC.weight_ends,
    STATIC.pops,
    STATIC.pop_ends,
    STATIC.fert,
    STATIC.fert_ends,
)

Fold = Callable[[Any], dict[str, Any]] | None
Mixing = Callable[[Any, dict[str, Any]], Any] | None


def draw_params(n_sets: int, seed: int) -> list[dict[str, float]]:
    """Uniform draws on ``PARAM_BOUNDS``. Other keys stay at ``BASE_PARAMS``."""
    rng = np.random.default_rng(seed)
    drawn: list[dict[str, float]] = []
    for _ in range(n_sets):
        params = dict(BASE_PARAMS)
        for name, (low, high) in PARAM_BOUNDS.items():
            params[name] = float(rng.uniform(low, high))
        drawn.append(params)
    return drawn


def _theta(params: dict[str, float]) -> Any:
    return jnp.asarray(
        [params["bg_mixing"], params["a_spread"], params["pc_strength"]], dtype=jnp.float64
    )


def stepwise_case() -> tuple[Mixing, Fold]:
    """A normalised year table, indexed with the baseline's truncation."""
    contact = prepare_stepwise(*ARRAYS, years=YEARS)
    start, stop = float(YEARS[0]), float(YEARS[-1])

    def mixing(calendar: Any, params: dict[str, Any]) -> Any:
        index = jnp.clip(jnp.clip(calendar, start, stop) - start, 0, YEARS.size - 1)
        return params["cmat"][index.astype(jnp.int32)]

    def fold(theta: Any) -> dict[str, Any]:
        return {"cmat": contact.materialize(theta[0], theta[1], theta[2]).table}

    return mixing, fold


def spline_case() -> tuple[Mixing, Fold]:
    """A cubic through the normalised annual matrices."""
    contact = prepare_spline(*ARRAYS, years=YEARS)
    year0 = float(YEARS[0])

    def mixing(calendar: Any, params: dict[str, Any]) -> Any:
        return params["curve"].at(calendar)

    def fold(theta: Any) -> dict[str, Any]:
        curve = contact.materialize(theta[0], theta[1], theta[2]).curve
        assert curve is not None and curve.year0 == year0
        return {"curve": curve}

    return mixing, fold


def spline_free_case() -> tuple[Mixing, Fold]:
    """The same function, refitted from four knots inside every stage."""
    contact = prepare_spline(*ARRAYS, years=YEARS)

    def mixing(calendar: Any, params: dict[str, Any]) -> Any:
        return contact.matrix(
            calendar, params["bg_mixing"], params["a_spread"], params["pc_strength"]
        )

    return mixing, None


def smooth_case() -> tuple[Mixing, Fold]:
    """PCHIP of the inputs, with the eigenproblem still in the stage."""
    contact = prepare_smooth(*ARRAYS)

    def mixing(calendar: Any, params: dict[str, Any]) -> Any:
        return contact.matrix(
            calendar, params["bg_mixing"], params["a_spread"], params["pc_strength"]
        )

    return mixing, None


def time_case(label: str, mixing: Mixing, fold: Fold, sets: list[dict[str, float]]) -> None:
    """One discarded solve for the compile, then ``N_SETS`` timed ones, twice over.

    ``raw`` calls ``compiled.run`` and pays the re-trace. ``jit`` wraps the same
    call so it is traced once, which is the cost a calibration sees.
    """
    compiled = compile_model(mixing=mixing)
    kwargs = dict(SOLVER, save=save_plan(compiled))

    def raw(params: dict[str, float]) -> Any:
        prepared = dict(params)
        if fold is not None:
            prepared.update(fold(_theta(params)))
        return compiled.run(prepared, **kwargs)

    @jax.jit
    def wrapped(theta: Any) -> Any:
        prepared = dict(BASE_PARAMS)
        prepared["bg_mixing"], prepared["a_spread"], prepared["pc_strength"] = theta
        if fold is not None:
            prepared.update(fold(theta))
        return compiled.run(prepared, **kwargs)["compartments"].values.data

    def measure(call: Callable[[Any], Any], inputs: list[Any]) -> tuple[float, float, Any]:
        started = time.perf_counter()
        result = call(inputs[0])
        jax.block_until_ready(result)
        first = time.perf_counter() - started
        taken = []
        for item in inputs:
            started = time.perf_counter()
            result = call(item)
            jax.block_until_ready(result)
            taken.append(time.perf_counter() - started)
        return first, float(np.median(taken)), result

    _, raw_median, result = measure(raw, sets)
    compile_s, jit_median, _ = measure(wrapped, [_theta(item) for item in sets])
    info = result.solver
    print(
        f"{label:26} steps {int(info.num_steps):6d} "
        f"rejected {int(info.num_rejected_steps):5d}  "
        f"raw {raw_median:6.3f} s  jit {jit_median:6.3f} s  "
        f"(compile {compile_s:5.2f} s)",
        flush=True,
    )


def main() -> None:
    """Time every mixing variant on the same draws."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", choices=("step", "smooth"), default="smooth")
    args = parser.parse_args()
    if args.entry == "step":
        model.entry_rate_at = demography.birth_rate_at
    else:
        model.entry_rate_at = demography.smooth_birth_rate_at

    sets = draw_params(N_SETS, SEED)
    print(f"entry rate: {args.entry}; {N_SETS} draws, rtol=atol={SOLVER['rtol']:g}")
    started = time.perf_counter()
    prepare_stepwise(*ARRAYS, years=YEARS)
    stepwise_prepare = time.perf_counter() - started
    started = time.perf_counter()
    prepare_spline(*ARRAYS, years=YEARS)
    spline_prepare = time.perf_counter() - started
    print(
        f"prepare, once per process: stepwise {stepwise_prepare:.3f} s, spline {spline_prepare:.3f} s"
    )

    time_case("baseline", model.baseline_mixing, None, sets)
    time_case("stepwise, folded", *stepwise_case(), sets)
    time_case("spline, folded", *spline_case(), sets)
    time_case("spline, free in stage", *spline_free_case(), sets)
    time_case("smooth, free in stage", *smooth_case(), sets)


if __name__ == "__main__":
    main()
