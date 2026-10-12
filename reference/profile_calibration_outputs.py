"""Profile the full-calibration outputs model run under an outer ``jax.jit``.

Remote calibration finishes with ``rerun_model_for_outputs``: folded spline,
corrected model, ``CALIBRATION_SOLVER``. That path calls ``compiled.run``
without an outer jit and pays equinox re-trace. Calibration's likelihood wraps
the same solve in ``@jit``. This script profiles the **compute** of that solve
the way a stable outer jit sees it, via ``jax.profiler``.

Writes TensorBoard / Perfetto artefacts under ``outputs/profile/`` (gitignored)
and prints XLA-op / family hotspot tables from the Perfetto JSON.

    pixi run python reference/profile_calibration_outputs.py
    pixi run python reference/profile_calibration_outputs.py --mode likelihood
    pixi run python reference/profile_calibration_outputs.py --mode grad --repeats 3
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import jax
from jax import numpy as jnp

from tb_macro.calibration import (
    CALIBRATION_SOLVER,
    build_mixing_cases,
    compile_case,
    make_log_likelihood,
)
from tb_macro.constants import OUTPUT_PATH, SCENARIO_PARAMS
from tb_macro.model import save_plan
from tb_macro.parameters import BASE_PARAMS, PARAM_BOUNDS, get_nuts_init_values

# Bookkeeping / wrapper names excluded from the named-op table.
_SKIP_PREFIXES: tuple[str, ...] = (
    "ThunkExecutor::",
    "PjRt",
    "CommonPjRt",
    "ParseArguments",
    "Wait for",
    "ThreadpoolListener",
    "$",
    "end: ",
    "PjitFunction",
    "calib_",
)

# First-match buckets for family rollup (order matters).
_FAMILIES: tuple[str, ...] = (
    "wrapped_scatter",
    "gather",
    "while",
    "cond",
    "dot",
    "call",
    "copy",
    "select",
    "multiply",
    "negate",
    "dynamic-slice",
    "broadcast",
    "reduce",
    "concatenate",
    "transpose",
    "reshape",
    "bitcast",
    "add",
    "div",
)


def _spline_case() -> Any:
    return next(item for item in build_mixing_cases() if item.name == "spline")


def _base_params(scenario: dict[str, float] | None = None) -> dict[str, Any]:
    params = dict(BASE_PARAMS)
    params.update(get_nuts_init_values())
    if scenario:
        params.update(scenario)
    return params


def _build_solve() -> tuple[Callable[..., Any], Any]:
    """Outer-jit solve matching ``rerun_model_for_outputs`` settings."""
    case = _spline_case()
    compiled = compile_case(case)
    settings = dict(CALIBRATION_SOLVER, save=save_plan(compiled))
    params = _base_params(SCENARIO_PARAMS[0] if SCENARIO_PARAMS else None)
    if case.fold is not None:
        params = dict(params)
        params.update(case.fold(params))

    @jax.jit
    def run(prepared: dict[str, Any]) -> Any:
        result = compiled.run(prepared, **settings)
        data = result["compartments"].values.data
        return jnp.sum(data), result.solver.num_steps

    return run, params


def _theta_bundle() -> tuple[Any, tuple[str, ...], dict[str, Any], Any]:
    """Float vector over ``PARAM_BOUNDS`` plus a jitted density on that vector."""
    case = _spline_case()
    compiled = compile_case(case)
    evaluate = make_log_likelihood(compiled, mixing=case.mixing, fold=case.fold)
    base = _base_params()
    names = tuple(PARAM_BOUNDS)
    theta0 = jnp.asarray([base[name] for name in names], dtype=jnp.float64)

    def unpack(theta: Any) -> dict[str, Any]:
        params = dict(base)
        params.update({name: theta[i] for i, name in enumerate(names)})
        return params

    @jax.jit
    def density(theta: Any) -> Any:
        return evaluate(unpack(theta))["total"]

    return evaluate, names, base, (density, theta0)


def _build_likelihood() -> tuple[Callable[..., Any], Any]:
    evaluate, _names, base, _ = _theta_bundle()
    return evaluate, base


def _build_grad() -> tuple[Callable[..., Any], Any]:
    """``value_and_grad`` on the float calibration vector (NUTS-like cost)."""
    _evaluate, _names, _base, (density, theta0) = _theta_bundle()

    @jax.jit
    def value_and_grad(theta: Any) -> Any:
        return jax.value_and_grad(density)(theta)

    return value_and_grad, theta0


def _warmup(call: Callable[..., Any], args: Any) -> Any:
    started = time.perf_counter()
    out = call(args)
    jax.block_until_ready(out)
    print(f"warmup (compile+first) {time.perf_counter() - started:.3f} s", flush=True)
    return out


def _timed_runs(call: Callable[..., Any], args: Any, repeats: int) -> list[float]:
    times: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        out = call(args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - started)
    return times


def _family(name: str) -> str:
    for key in _FAMILIES:
        if name == key or name.startswith(f"{key}.") or name.startswith(f"{key}_"):
            return key
    if "fusion" in name:
        return "other_fusion"
    return "other"


def _summarize_perfetto(path: Path, top_n: int) -> None:
    """Aggregate Perfetto chrome-trace events into family and op tables."""
    with gzip.open(path, "rt") as handle:
        payload = json.load(handle)
    events = payload.get("traceEvents", payload) if isinstance(payload, dict) else payload
    by_name: dict[str, dict[str, float]] = defaultdict(lambda: {"sum": 0.0, "n": 0.0})
    thunk_exec_us = 0.0
    thunk_wait_us = 0.0
    for event in events:
        if not isinstance(event, dict):
            continue
        duration = float(event.get("dur", 0) or 0)
        if duration <= 0:
            continue
        name = str(event.get("name", ""))
        if name == "ThunkExecutor::Execute":
            thunk_exec_us += duration
        elif name.startswith("ThunkExecutor::Execute (wait"):
            thunk_wait_us += duration
        if not name or any(name.startswith(prefix) for prefix in _SKIP_PREFIXES):
            continue
        by_name[name]["sum"] += duration
        by_name[name]["n"] += 1.0

    print(
        f"\nThunkExecutor::Execute sum={thunk_exec_us / 1000:.1f} ms  "
        f"wait={thunk_wait_us / 1000:.1f} ms",
        flush=True,
    )
    families: dict[str, float] = defaultdict(float)
    for name, stats in by_name.items():
        families[_family(name)] += stats["sum"]
    family_total = sum(families.values()) or 1.0
    print("Named-op families (share of named thunk children):", flush=True)
    print(f"{'ms':>10}  {'%':>6}  family", flush=True)
    for name, duration in sorted(families.items(), key=lambda item: -item[1])[:15]:
        print(
            f"{duration / 1000:10.2f}  {100.0 * duration / family_total:5.1f}%  {name}",
            flush=True,
        )

    print(f"\nTop named ops by sum dur (top {top_n}):", flush=True)
    print(f"{'ms':>10}  {'n':>6}  {'avg_us':>8}  name", flush=True)
    ranked = sorted(by_name.items(), key=lambda item: -item[1]["sum"])
    for name, stats in ranked[:top_n]:
        avg_us = stats["sum"] / stats["n"]
        print(
            f"{stats['sum'] / 1000:10.3f}  {int(stats['n']):6d}  {avg_us:8.2f}  {name}",
            flush=True,
        )


def main() -> None:
    """Warm up, time, and ``jax.profiler.trace`` the chosen workload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("solve", "likelihood", "grad"),
        default="solve",
        help="solve: outputs ODE; likelihood: jitted density; grad: value_and_grad",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Profiled runs after warmup")
    parser.add_argument("--top", type=int, default=25, help="Named-op rows to print")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="TensorBoard / Perfetto directory (default: outputs/profile/...)",
    )
    args = parser.parse_args()

    builders = {
        "solve": _build_solve,
        "likelihood": _build_likelihood,
        "grad": _build_grad,
    }
    call, call_args = builders[args.mode]()
    log_dir = args.log_dir or (OUTPUT_PATH / "profile" / f"calibration_outputs_{args.mode}")
    print(
        f"mode={args.mode}  mixing=spline  match_golden=False  "
        f"solver={CALIBRATION_SOLVER['solver']}  "
        f"rtol={CALIBRATION_SOLVER['rtol']}  max_steps={CALIBRATION_SOLVER['max_steps']}",
        flush=True,
    )
    out = _warmup(call, call_args)
    if args.mode == "solve":
        print(f"  num_steps={int(out[1])}", flush=True)
    elif args.mode == "likelihood":
        print(
            f"  total={float(out['total']):.4g}  ok={bool(out['ok'])}  "
            f"steps={int(out['num_steps'])}",
            flush=True,
        )

    times = _timed_runs(call, call_args, max(1, args.repeats))
    print(
        f"timed {len(times)} runs: median={sorted(times)[len(times) // 2]:.4f} s  "
        f"min={min(times):.4f} s  max={max(times):.4f} s",
        flush=True,
    )

    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"tracing {args.repeats} runs -> {log_dir}", flush=True)
    with jax.profiler.trace(str(log_dir), create_perfetto_trace=True):
        for _ in range(args.repeats):
            with jax.profiler.TraceAnnotation(f"calib_outputs_{args.mode}"):
                jax.block_until_ready(call(call_args))

    perfetto = sorted(log_dir.rglob("perfetto_trace.json.gz"))
    if not perfetto:
        raise SystemExit(f"No perfetto_trace.json.gz under {log_dir}")
    latest = perfetto[-1]
    print(f"perfetto: {latest}", flush=True)
    xplane = sorted(log_dir.rglob("*.xplane.pb"))
    if xplane:
        print(f"xplane:   {xplane[-1]}", flush=True)
    _summarize_perfetto(latest, args.top)
    print(
        "\nOpen the Perfetto file at https://ui.perfetto.dev "
        "(or TensorBoard with the profile plugin on the log dir).",
        flush=True,
    )


if __name__ == "__main__":
    main()
