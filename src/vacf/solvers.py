"""Diffrax backends for ``CompiledModel.run``.

summer4 ``run(solver=)`` takes a :class:`~summer4.solvers.Diffrax` holding the
caller's own solver and ``PIDController`` (notebook 23). That is how ``dtmax``
reaches diffrax — there is no ``dtmax=`` keyword on ``run`` itself.
"""

from __future__ import annotations

from typing import Any


def dopri5(
    *,
    rtol: float,
    atol: float,
    dtmax: float | None = None,
    **pid: Any,
) -> Any:
    """Dopri5 with a ``PIDController``. Omit ``dtmax`` for an uncapped step."""
    import diffrax
    from summer4.solvers import Diffrax

    controller = diffrax.PIDController(rtol=rtol, atol=atol, dtmax=dtmax, **pid)
    return Diffrax(diffrax.Dopri5(), stepsize_controller=controller)


# Forward-run / golden tolerances: uncapped step, matching ``tests/golden/``.
FORWARD_SOLVER: dict[str, Any] = {
    "solver": dopri5(rtol=1e-9, atol=1e-9),
    "max_steps": 20_000,
}

# summer3 calibration notebook: Dopri5, rtol=atol=1e-5, dtmax=1, max_steps=4000.
CALIBRATION_SOLVER: dict[str, Any] = {
    "solver": dopri5(rtol=1e-5, atol=1e-5, dtmax=1.0),
    "max_steps": 4000,
}
