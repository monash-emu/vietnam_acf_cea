"""Remote calibration and posterior scenario runs for the Vietnam ACF study on massive/M3.

Calls :func:`vacf.economics.analysis.run_and_store`, the same function as
``notebooks/10-ACF-CEA-Run.ipynb``: calibrate the homogeneous-mixing model, run every arm in
:mod:`vacf.economics.study` at ``n_draws`` posterior draws, and write under
``outputs/<task>/``:

  idata.nc                    posterior (ArviZ netCDF), with NUTS diagnostics
  posterior_summary.csv       ``az.summary`` of the posterior
  draws.parquet               parameter values of each sampled draw
  epi.parquet                 annual epidemiological indicators, per draw and arm
  economic.parquet            annual economic indicators (costing inputs), per draw and arm
  calibration_fit.parquet     annual model counterparts of the targets, per draw
  epi_summary.parquet         cumulative burden over the study horizon, per draw and arm
  economic_summary.parquet    discounted costs and DALYs, per draw and arm
  incremental.parquet         efficiency frontier and ICERs, per draw
  csv/                        median, IQR and 95% interval of the outputs, for sharing
  run.log                     provenance and convergence summary
  progress.log                progress messages

Costs and DALYs use the cost inputs at run time. ``economic.parquet`` holds everything
costing consumes, so they can be recomputed locally without re-running this.

Usage, from the repo root via the SLURM wrapper::

    ./scripts/massive/launch.sh <n_runs> <walltime>

``n_runs`` is warmup and retained samples per chain. Locally, for a quick check::

    pixi run python scripts/massive/calibrate.py smoke <n_runs> <n_draws>
"""

from __future__ import annotations

import os
import sys

# One CPU device per chain, set before JAX is imported. Must match N_CHAINS_REMOTE.
_N_CHAINS = 8
_kept = [
    flag
    for flag in os.environ.get("XLA_FLAGS", "").split()
    if not flag.startswith("--xla_force_host_platform_device_count=")
]
os.environ["XLA_FLAGS"] = " ".join([f"--xla_force_host_platform_device_count={_N_CHAINS}", *_kept])

from vacf.constants import N_CHAINS_REMOTE, OUTPUT_PATH  # noqa: E402
from vacf.economics import analysis  # noqa: E402
from vacf.utils import get_logger  # noqa: E402

N_POSTERIOR_DRAWS = 100
SEED = 0


def main(task: str, n_runs: int, n_draws: int = N_POSTERIOR_DRAWS) -> None:
    if N_CHAINS_REMOTE != _N_CHAINS:
        raise RuntimeError(f"N_CHAINS_REMOTE={N_CHAINS_REMOTE} but {_N_CHAINS} devices were set.")
    path = OUTPUT_PATH / task
    path.mkdir(parents=True, exist_ok=True)
    analysis.run_and_store(
        path,
        n_runs=n_runs,
        n_chains=_N_CHAINS,
        n_draws=n_draws,
        seed=SEED,
        logger=get_logger(path / "progress.log"),
    )


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit(f"usage: {sys.argv[0]} <task_id> <n_runs> [n_draws]")
    main(sys.argv[1], int(sys.argv[2]), *(int(arg) for arg in sys.argv[3:]))
