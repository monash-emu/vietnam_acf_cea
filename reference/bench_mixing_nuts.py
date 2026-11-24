"""NUTS run of the calibration likelihood on a mixing-matrix variant.

The density is the summer4 port of the likelihood in the summer3 calibration
notebook. Every case samples the notebook's prior, including all three mixing
scalars. Nothing is fixed when the contact matrix is prepared.

The default is the folded spline: one curve per draw, evaluated inside the
solve. The run is 4 chains, 100 warmup and 100 samples, tree depth 4. Chains
use NumPyro's ``parallel`` method. That method silently becomes ``sequential``
when JAX sees fewer CPU devices than chains, so this script sets
``--xla_force_host_platform_device_count`` before JAX is imported and refuses
to sample if the fallback still happens. Parallel chains need JAX 0.7.2: from
0.8, ``pmap`` is ``jit(shard_map)`` and the diffrax adjoint crashes. See
``plans/parallel-nuts.md``.

    pixi run python reference/bench_mixing_nuts.py --probe-only
    pixi run python reference/bench_mixing_nuts.py
"""

from __future__ import annotations

import argparse
import os
import sys


def _requested_chains(argv: list[str]) -> int:
    """Read ``--chains`` before JAX is imported. The default is 4."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--chains", type=int, default=4)
    known, _ = parser.parse_known_args(argv)
    return known.chains


def _enable_parallel_chains(n_chains: int) -> None:
    """One CPU device per chain, before JAX reads ``XLA_FLAGS``."""
    if n_chains <= 1:
        return
    current = os.environ.get("XLA_FLAGS", "").split()
    kept = [
        flag for flag in current if not flag.startswith("--xla_force_host_platform_device_count=")
    ]
    os.environ["XLA_FLAGS"] = " ".join(
        [f"--xla_force_host_platform_device_count={n_chains}", *kept]
    )


_enable_parallel_chains(_requested_chains(sys.argv[1:]))

import gc
import time
from typing import Any

import jax  # noqa: E402
import numpy as np
from jax import numpy as jnp
from jax.random import PRNGKey
import numpyro
from numpyro import distributions as dist
from numpyro import infer

from vacf.calibration import (
    TERM_NAMES,
    MixingCase,
    build_mixing_cases,
    compile_case,
    make_log_likelihood,
)
from vacf.parameters import BASE_PARAMS, PARAM_BOUNDS, get_nuts_init_values

# Printed gradients, and the ones checked against a one-sided difference.
_FOCUS = ("bg_mixing", "a_spread", "pc_strength", "raw_transmission_rate")
_DIFFERENCED = ("bg_mixing", "a_spread", "pc_strength")


def _theta(values: dict[str, float], names: tuple[str, ...]) -> Any:
    return jnp.asarray([values[name] for name in names], dtype=jnp.float64)


def _unpack(theta: Any, names: tuple[str, ...]) -> dict[str, Any]:
    params = dict(BASE_PARAMS)
    for name, value in zip(names, theta, strict=True):
        params[name] = value
    return params


def _finite_difference(
    evaluate: Any,
    theta: Any,
    names: tuple[str, ...],
    autodiff: np.ndarray,
) -> dict[str, tuple[float, float]]:
    """One-sided difference over 1% of the prior width, next to the autodiff slope.

    The step is large enough that solver noise at ``rtol=1e-5`` does not decide
    the sign, and small enough to stay inside the prior.
    """
    slopes: dict[str, tuple[float, float]] = {}
    base = float(evaluate(_unpack(theta, names))["total"])
    for name in _DIFFERENCED:
        index = names.index(name)
        low, high = PARAM_BOUNDS[name]
        step = 1.0e-2 * (high - low)
        shifted = theta.at[index].add(step)
        bumped = float(evaluate(_unpack(shifted, names))["total"])
        slopes[name] = ((bumped - base) / step, float(autodiff[index]))
    return slopes


def _probe(
    case: MixingCase,
    evaluate: Any,
    names: tuple[str, ...],
    theta: Any,
) -> dict[str, Any]:
    """Steady forward and gradient cost at the NUTS start, plus the terms."""
    params = _unpack(theta, names)
    print("  compiling forward", flush=True)

    def total(point: Any) -> Any:
        return evaluate(_unpack(point, names))["total"]

    started = time.perf_counter()
    first = evaluate(params)
    jax.block_until_ready(first["total"])
    compile_forward = time.perf_counter() - started

    started = time.perf_counter()
    forward = evaluate(params)
    jax.block_until_ready(forward["total"])
    forward_s = time.perf_counter() - started
    print("  compiling gradient", flush=True)

    grad_fn = jax.jit(jax.value_and_grad(total))
    started = time.perf_counter()
    _, first_grad = grad_fn(theta)
    jax.block_until_ready(first_grad)
    compile_grad = time.perf_counter() - started

    started = time.perf_counter()
    value, grad = grad_fn(theta)
    jax.block_until_ready(grad)
    grad_s = time.perf_counter() - started

    gradient = np.asarray(grad)
    print("  finite differences", flush=True)
    ratios = _finite_difference(evaluate, theta, names, gradient)
    terms = {name: float(forward[name]) for name in TERM_NAMES}
    return {
        "case": case.name,
        "total": float(value),
        "ok": bool(forward["ok"]),
        "num_steps": int(forward["num_steps"]),
        "num_rejected": int(forward["num_rejected"]),
        "terms": terms,
        "gradient": gradient,
        "ratios": ratios,
        "compile_forward": compile_forward,
        "forward_s": forward_s,
        "compile_grad": compile_grad,
        "grad_s": grad_s,
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(left, right) / denom)


def _run_nuts(
    evaluate: Any,
    *,
    warmup: int,
    samples: int,
    depth: int,
    chains: int,
) -> dict[str, Any]:
    """Chains from the notebook's initial point. ``PRNGKey(2)`` matches it."""
    priors = {name: dist.Uniform(low, high) for name, (low, high) in PARAM_BOUNDS.items()}

    def model() -> None:
        sampled = {name: numpyro.sample(name, prior) for name, prior in priors.items()}
        params = dict(BASE_PARAMS)
        params.update(sampled)
        numpyro.factor("ll", evaluate(params)["total"])

    kernel = infer.NUTS(
        model,
        max_tree_depth=depth,
        init_strategy=infer.init_to_value(values=get_nuts_init_values()),
    )
    mcmc = infer.MCMC(
        kernel,
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        chain_method="parallel",
        progress_bar=True,
    )
    if chains > 1 and mcmc.chain_method != "parallel":
        raise RuntimeError(
            f"Refusing to run {chains} chains with chain_method={mcmc.chain_method!r} "
            f"on {jax.local_device_count()} devices. NumPyro draws chains sequentially "
            "when there are fewer devices than chains. Set "
            "XLA_FLAGS=--xla_force_host_platform_device_count=N before importing JAX."
        )
    print(
        f"  nuts chain_method {mcmc.chain_method}  chains {chains}  "
        f"devices {jax.local_device_count()}",
        flush=True,
    )
    # ``MCMC.run`` returns once the parallel map is dispatched. Reading the
    # draws is what waits for the chains. Time through that read.
    started = time.perf_counter()
    mcmc.run(
        PRNGKey(2),
        extra_fields=("accept_prob", "diverging", "num_steps", "adapt_state.step_size"),
    )
    extra = mcmc.get_extra_fields()
    draws = mcmc.get_samples(group_by_chain=False)
    jax.block_until_ready(draws)
    elapsed = time.perf_counter() - started
    mixing = {name: np.asarray(draws[name]) for name in _DIFFERENCED}
    return {
        "elapsed": elapsed,
        "divergences": int(np.sum(np.asarray(extra["diverging"]))),
        "accept": float(np.mean(np.asarray(extra["accept_prob"]))),
        "num_steps": float(np.mean(np.asarray(extra["num_steps"]))),
        "step_size": float(np.mean(np.asarray(extra["adapt_state.step_size"]))),
        "mixing": mixing,
        "n": int(np.asarray(extra["accept_prob"]).size),
    }


def _print_probe(row: dict[str, Any], baseline: np.ndarray | None, names: tuple[str, ...]) -> None:
    print(
        f"{row['case']}: ll {row['total']:.4f}  ok {row['ok']}  "
        f"steps {row['num_steps']}  rejected {row['num_rejected']}",
        flush=True,
    )
    print(
        f"  time  forward {row['forward_s']:.2f}s (compile {row['compile_forward']:.1f}s)  "
        f"grad {row['grad_s']:.2f}s (compile {row['compile_grad']:.1f}s)",
        flush=True,
    )
    terms = "  ".join(f"{name} {row['terms'][name]:.3f}" for name in TERM_NAMES)
    print(f"  terms {terms}", flush=True)
    pieces = []
    for name in _FOCUS:
        index = names.index(name)
        pieces.append(f"{name} {row['gradient'][index]:.4g}")
    print(f"  grad  {'  '.join(pieces)}", flush=True)
    compared = []
    for name in _DIFFERENCED:
        finite, auto = row["ratios"][name]
        ratio = finite / auto if abs(auto) > 1.0e-8 else float("nan")
        compared.append(f"{name} fd {finite:.4g} ad {auto:.4g} ({ratio:.2f})")
    print(f"  fd/ad {'  '.join(compared)}", flush=True)
    if baseline is not None:
        cosine = _cosine(row["gradient"], baseline)
        gaps = []
        for name in _DIFFERENCED:
            index = names.index(name)
            scale = max(abs(float(baseline[index])), 1.0e-8)
            gaps.append(f"{name} {abs(row['gradient'][index] - baseline[index]) / scale:.3f}")
        print(f"  vs baseline  cosine {cosine:.4f}  rel {', '.join(gaps)}", flush=True)


def _print_nuts(case: str, nuts: dict[str, Any]) -> None:
    print(
        f"  nuts {case}  {nuts['elapsed']:.1f}s  draws {nuts['n']}  "
        f"divergences {nuts['divergences']}  "
        f"accept {nuts['accept']:.3f}  "
        f"leapfrog {nuts['num_steps']:.1f}  step {nuts['step_size']:.3g}",
        flush=True,
    )
    for name, values in nuts["mixing"].items():
        print(
            f"    {name} mean {float(values.mean()):.5g}  std {float(values.std()):.5g}",
            flush=True,
        )


def main() -> None:
    """Probe the requested variants, then sample unless ``--probe-only``.

    The default variant is the folded spline.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--variants", nargs="*", default=["spline"])
    args = parser.parse_args()

    if args.chains < 1:
        raise ValueError("--chains must be at least 1")
    print(
        f"devices {jax.local_device_count()}  chains {args.chains}",
        flush=True,
    )
    names = tuple(PARAM_BOUNDS)
    theta = _theta(get_nuts_init_values(), names)
    started = time.perf_counter()
    cases = build_mixing_cases()
    print(f"prepare contacts: {time.perf_counter() - started:.2f}s", flush=True)
    if args.variants:
        wanted = set(args.variants)
        cases = tuple(case for case in cases if case.name in wanted)

    baseline_grad: np.ndarray | None = None
    for case in cases:
        print(f"\n== {case.name}", flush=True)
        compiled = compile_case(case)
        evaluate = make_log_likelihood(compiled, mixing=case.mixing, fold=case.fold)
        row = _probe(case, evaluate, names, theta)
        if case.name == "baseline":
            baseline_grad = row["gradient"]
        _print_probe(row, None if case.name == "baseline" else baseline_grad, names)
        finite = np.isfinite(row["total"]) and bool(np.all(np.isfinite(row["gradient"])))
        if args.probe_only or not row["ok"] or not finite:
            if not row["ok"] or not finite:
                print("  skipping NUTS: likelihood or gradient is not finite", flush=True)
        else:
            nuts = _run_nuts(
                evaluate,
                warmup=args.warmup,
                samples=args.samples,
                depth=args.depth,
                chains=args.chains,
            )
            _print_nuts(case.name, nuts)
        del compiled, evaluate, row
        jax.clear_caches()
        gc.collect()
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
