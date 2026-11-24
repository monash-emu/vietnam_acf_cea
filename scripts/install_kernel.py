"""Register this project's Jupyter kernel at user level.

The upstream task installed into the environment's own prefix
(``.pixi/envs/default/share/jupyter/kernels``). JupyterLab launched from the project finds
that, but VS Code only does if it has already discovered the pixi interpreter — which it
will not when the workspace is opened on a parent folder rather than on this repository.
Installing at user level makes the kernel visible from any workspace, matching how the
other projects on this machine are set up.

The kernelspec also carries ``XLA_FLAGS``, so the four NUTS chains run in parallel whether
or not the notebook's first cell was executed first.

    pixi run install-kernel
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

KERNEL_NAME = "vacf"
DISPLAY_NAME = "Python (vietnam_acf_cea)"

#: Four chains sample in parallel only if JAX sees four devices, and the device count must
#: be set before JAX is imported. Putting it in the kernelspec guarantees that.
KERNEL_ENV = {"XLA_FLAGS": "--xla_force_host_platform_device_count=4"}


def user_kernel_dir() -> Path:
    """Where ``--user`` installs.

    Resolved explicitly rather than by kernel-name lookup: a stale environment-local
    kernelspec of the same name takes precedence in lookup order, so looking the name up
    can return — and then patch — the wrong file.
    """
    from jupyter_core.paths import jupyter_data_dir

    return Path(jupyter_data_dir()) / "kernels" / KERNEL_NAME


def main() -> int:
    subprocess.run(
        [
            sys.executable, "-m", "ipykernel", "install",
            "--user", "--name", KERNEL_NAME, "--display-name", DISPLAY_NAME,
        ],
        check=True,
    )

    spec_path = user_kernel_dir() / "kernel.json"
    spec = json.loads(spec_path.read_text())
    spec["env"] = {**spec.get("env", {}), **KERNEL_ENV}
    spec_path.write_text(json.dumps(spec, indent=1) + "\n")

    print(f"\nRegistered '{DISPLAY_NAME}' at {spec_path}")
    print(f"Interpreter: {spec['argv'][0]}")
    print("In VS Code: Select Kernel -> Jupyter Kernel... -> " + DISPLAY_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
