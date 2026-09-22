"""Remote NUTS calibration for massive / M3.

Writes the same artefacts as summer3 ``scripts/massive/calibrate.py`` on
``tweak_cases_calib``:

  ``outputs/<task>/<stamp>.nc``
  ``outputs/<task>/<stamp>_outputs.pkl``
  ``outputs/<task>/<stamp>.log``
  ``outputs/<task>/run.log``

The density is the summer4 folded-spline likelihood (``notebooks/05-Calibration.ipynb``):
every name in ``PARAM_BOUNDS`` is sampled, ``release=False``, tree depth 5,
started at ``get_nuts_init_values()``, ``PRNGKey(2)``. Scenario reruns after
sampling use ``SCENARIO_PARAMS`` (baseline and ACF coverage 0.8).

Device count is set before ``tb_macro`` (and JAX) is imported so NumPyro keeps
``chain_method="parallel"``. JAX is pinned to 0.7.2 for that path. The count
must match ``tb_macro.constants.N_CHAINS_REMOTE``.

Usage (from the repo root, via the SLURM wrappers)::

    ./scripts/massive/launch.sh 500 24:00:00
"""

from __future__ import annotations

import os
import sys

# Must match ``tb_macro.constants.N_CHAINS_REMOTE``. Set before any JAX import.
_N_CHAINS = 8


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


_enable_parallel_chains(_N_CHAINS)

import gc  # noqa: E402
from datetime import UTC, datetime  # noqa: E402

import jax  # noqa: E402
import numpyro  # noqa: E402
from jax.random import PRNGKey  # noqa: E402
from numpyro import distributions as dist  # noqa: E402
from numpyro import infer  # noqa: E402

from tb_macro.calibration import (  # noqa: E402
    MCMC_EXTRA_FIELDS,
    build_mixing_cases,
    compile_case,
    inference_data_from_mcmc,
    make_log_likelihood,
)
from tb_macro.constants import (  # noqa: E402
    N_CHAINS_REMOTE,
    N_OUTPUT_SAMPLES,
    OUTPUT_PATH,
    SCENARIO_PARAMS,
)
from tb_macro.outputs import (  # noqa: E402
    get_posterior_samples,
    rerun_model_for_outputs,
    save_sampled_outputs,
)
from tb_macro.parameters import BASE_PARAMS, PARAM_BOUNDS, get_nuts_init_values  # noqa: E402
from tb_macro.utils import get_git_provenance, get_logger, write_run_log  # noqa: E402

if N_CHAINS_REMOTE != _N_CHAINS:
    raise RuntimeError(
        f"N_CHAINS_REMOTE={N_CHAINS_REMOTE} but device count was set to {_N_CHAINS} "
        "before import; update both together."
    )


def main(task: str, n_runs: int) -> None:
    """Run remote NUTS and write idata, sampled outputs, and provenance logs."""
    path = OUTPUT_PATH / task
    path.mkdir(parents=True, exist_ok=True)
    logger = get_logger(path / "run.log")
    provenance = get_git_provenance()
    logger.info(
        "git commit %s (dirty=%s) branch %s",
        provenance["commit"],
        provenance["dirty"],
        provenance["branch"],
    )
    logger.info(
        "devices %s  chains %s  n_runs %s",
        jax.local_device_count(),
        N_CHAINS_REMOTE,
        n_runs,
    )

    case = next(item for item in build_mixing_cases() if item.name == "spline")
    compiled = compile_case(case)
    evaluate = make_log_likelihood(compiled, mixing=case.mixing, fold=case.fold)
    logger.info("mixing case %s", case.name)

    priors = {name: dist.Uniform(low, high) for name, (low, high) in PARAM_BOUNDS.items()}

    def model() -> None:
        sampled = {name: numpyro.sample(name, prior) for name, prior in priors.items()}
        params = dict(BASE_PARAMS)
        params.update(sampled)
        numpyro.factor("ll", evaluate(params)["total"])

    kernel = infer.NUTS(
        model,
        max_tree_depth=5,
        init_strategy=infer.init_to_value(values=get_nuts_init_values()),
    )
    mcmc = infer.MCMC(
        kernel,
        num_warmup=n_runs,
        num_samples=n_runs,
        num_chains=N_CHAINS_REMOTE,
        chain_method="parallel",
        progress_bar=True,
    )
    if mcmc.chain_method != "parallel":
        raise RuntimeError(
            f"Refusing to run with chain_method={mcmc.chain_method!r} on "
            f"{jax.local_device_count()} devices. Set "
            "XLA_FLAGS=--xla_force_host_platform_device_count="
            f"{N_CHAINS_REMOTE} before importing JAX."
        )
    mcmc.run(PRNGKey(2), extra_fields=MCMC_EXTRA_FIELDS)
    idata = inference_data_from_mcmc(mcmc)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%MZ")
    idata.to_netcdf(path / f"{timestamp}.nc", engine="h5netcdf")

    # Free NUTS XLAs before the scenario reruns.
    del mcmc, kernel, evaluate, model
    jax.clear_caches()
    gc.collect()

    samples = get_posterior_samples(idata, N_OUTPUT_SAMPLES)
    outputs, sample_labels = rerun_model_for_outputs(
        idata,
        SCENARIO_PARAMS,
        samples,
        case=case,
    )
    save_sampled_outputs(path / f"{timestamp}_outputs.pkl", outputs, sample_labels)
    write_run_log(
        path / f"{timestamp}.log",
        extra={
            "task": task,
            "n_runs": n_runs,
            "idata": f"{timestamp}.nc",
            "outputs": f"{timestamp}_outputs.pkl",
            "scenario_params": SCENARIO_PARAMS,
            "mixing": case.name,
        },
    )
    logger.info(
        "Wrote %s, %s and %s",
        f"{timestamp}.nc",
        f"{timestamp}_outputs.pkl",
        f"{timestamp}.log",
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <task_id> <n_runs>")
    main(sys.argv[1], int(sys.argv[2]))
