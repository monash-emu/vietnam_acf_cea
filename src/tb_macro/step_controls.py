"""Pass diffrax step controls through summer4's solver.

summer4 v0.2.0a2 builds ``PIDController`` from ``rtol`` and ``atol`` only.
``install`` replaces that call so ``CompiledModel.run`` accepts the rest of the
controller, and a ready-made ``stepsize_controller``. Importing ``tb_macro``
installs it. The golden ``run()`` does not pass the new arguments.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Any

_CONTROLS: contextvars.ContextVar[StepControls | None] = contextvars.ContextVar(
    "tb_macro_step_controls", default=None
)


@dataclass(frozen=True, slots=True)
class StepControls:
    """Scalar ``PIDController`` knobs, or a controller object."""

    dtmin: float | None = None
    dtmax: float | None = None
    pcoeff: float = 0.0
    icoeff: float = 1.0
    dcoeff: float = 0.0
    force_dtmin: bool = True
    factormin: float = 0.2
    factormax: float = 10.0
    safety: float = 0.9
    error_order: float | None = None
    stepsize_controller: Any = None
    throw: bool = False

    def pid_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``diffrax.PIDController``, besides rtol/atol."""
        return {
            "dtmin": self.dtmin,
            "dtmax": self.dtmax,
            "pcoeff": self.pcoeff,
            "icoeff": self.icoeff,
            "dcoeff": self.dcoeff,
            "force_dtmin": self.force_dtmin,
            "factormin": self.factormin,
            "factormax": self.factormax,
            "safety": self.safety,
            "error_order": self.error_order,
        }


def _patched_diffrax_solve(
    model: Any,
    *,
    y0: object,
    params: object,
    spec: Any,
    groups: tuple[Any, ...],
    plan: Any,
    solver: str | Any,
    solver_stats: bool,
) -> Any:
    """``diffrax_solve`` with the active :class:`StepControls` applied."""
    import jax.numpy as jnp

    from summer4.flows.stages import Prepared
    from summer4.jax.state import unpack_state
    from summer4.results.result import SolverInfo
    from summer4.solvers.base import SolveOutput
    from summer4.solvers.diffrax_backend import (
        _ensure_eqx_modules,
        _filtered_plan,
        _import_diffrax,
        _result_code,
        resolve_diffrax_solver,
    )

    diffrax = _import_diffrax()
    solver_inst, solver_name = resolve_diffrax_solver(solver)
    vf_cls, save_cls = _ensure_eqx_modules()

    prepared = params if isinstance(params, Prepared) else model.prepare(params)
    y_arr, _rebox = unpack_state(y0, model.pmap)
    controls = _CONTROLS.get()

    term = diffrax.ODETerm(vf_cls(model))
    subs: dict[str, Any] = {}
    for group in groups:
        group_plan = _filtered_plan(plan, group.keys)
        requests = tuple((key, group_plan.requests[key].what) for key in group.keys)
        keep = group_plan.flow_reads()
        subs[group.name] = diffrax.SubSaveAt(
            ts=jnp.asarray(group.ts),
            fn=save_cls(model, requests, keep),
        )
    saveat = diffrax.SaveAt(subs=subs, dense=bool(spec.dense))

    if controls is not None and controls.stepsize_controller is not None:
        controller: Any = controls.stepsize_controller
        dt0: float | None = float(spec.dt)
    elif spec.rtol is not None or spec.atol is not None or _pid_requested(controls):
        pid = {} if controls is None else controls.pid_kwargs()
        controller = diffrax.PIDController(
            rtol=1e-3 if spec.rtol is None else float(spec.rtol),
            atol=1e-6 if spec.atol is None else float(spec.atol),
            **pid,
        )
        dt0 = float(spec.dt)
    else:
        controller = diffrax.ConstantStepSize()
        dt0 = float(spec.dt)

    t0 = float(spec.t0)
    if spec.t1 is not None:
        t1 = float(spec.t1)
    elif spec.steps is not None:
        t1 = t0 + float(spec.dt) * int(spec.steps)
    else:
        raise ValueError("diffrax backend requires t1 or steps.")
    for group in groups:
        if group.ts.size:
            t1 = max(t1, float(group.ts[-1]))

    max_steps = 4096 if spec.max_steps is None else int(spec.max_steps)
    if spec.dense and spec.max_steps is None:
        max_steps = max(max_steps, 4096)
    throw = False if controls is None else controls.throw

    sol = diffrax.diffeqsolve(
        term,
        solver_inst,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y_arr,
        args=prepared,
        saveat=saveat,
        stepsize_controller=controller,
        max_steps=max_steps,
        throw=throw,
    )

    saved: dict[str, Any] = {}
    for group in groups:
        group_ys = sol.ys[group.name]
        for key in group.keys:
            saved[key] = group_ys[key]

    stats: SolverInfo | None = None
    if solver_stats:
        state = sol.stats
        stats = SolverInfo(
            solver=solver_name,
            num_steps=state.get("num_steps"),
            num_accepted_steps=state.get("num_accepted_steps"),
            num_rejected_steps=state.get("num_rejected_steps"),
            result_code=_result_code(sol.result),
            max_steps=state.get("max_steps", max_steps),
            dense=bool(spec.dense),
        )
    dense = sol.interpolation if spec.dense else None
    return SolveOutput(saved=saved, stats=stats, dense=dense)


def _pid_requested(controls: StepControls | None) -> bool:
    if controls is None:
        return False
    return (
        controls.dtmin is not None
        or controls.dtmax is not None
        or controls.error_order is not None
        or controls.pcoeff != 0.0
        or controls.icoeff != 1.0
        or controls.dcoeff != 0.0
        or controls.force_dtmin is not True
        or controls.factormin != 0.2
        or controls.factormax != 10.0
        or controls.safety != 0.9
    )


def install() -> None:
    """Replace summer4's diffrax entry points. Safe to call more than once."""
    from summer4.flows import compiled as compiled_mod
    from summer4.solvers import diffrax_backend

    if getattr(compiled_mod.CompiledModel.run, "_tb_step_controls", False):
        return
    diffrax_backend.diffrax_solve = _patched_diffrax_solve
    original = compiled_mod.CompiledModel.run

    def run(
        self: Any,
        params: object,
        y0: object | None = None,
        *,
        t0: float,
        t1: float | None = None,
        dt: float,
        steps: int | None = None,
        save: Any = None,
        solver: str | Any = "euler",
        epoch: Any = None,
        rtol: float | None = None,
        atol: float | None = None,
        max_steps: int | None = None,
        dtmin: float | None = None,
        dtmax: float | None = None,
        pcoeff: float = 0.0,
        icoeff: float = 1.0,
        dcoeff: float = 0.0,
        force_dtmin: bool = True,
        factormin: float = 0.2,
        factormax: float = 10.0,
        safety: float = 0.9,
        error_order: float | None = None,
        stepsize_controller: Any = None,
        throw: bool = False,
    ) -> Any:
        use_euler = solver == "euler" or (isinstance(solver, str) and solver.lower() == "euler")
        if use_euler and (
            dtmin is not None or dtmax is not None or stepsize_controller is not None
        ):
            raise ValueError("step-size controls apply to adaptive diffrax solvers, not euler.")
        token = _CONTROLS.set(
            StepControls(
                dtmin=dtmin,
                dtmax=dtmax,
                pcoeff=pcoeff,
                icoeff=icoeff,
                dcoeff=dcoeff,
                force_dtmin=force_dtmin,
                factormin=factormin,
                factormax=factormax,
                safety=safety,
                error_order=error_order,
                stepsize_controller=stepsize_controller,
                throw=throw,
            )
        )
        try:
            return original(
                self,
                params,
                y0,
                t0=t0,
                t1=t1,
                dt=dt,
                steps=steps,
                save=save,
                solver=solver,
                epoch=epoch,
                rtol=rtol,
                atol=atol,
                max_steps=max_steps,
            )
        finally:
            _CONTROLS.reset(token)

    run._tb_step_controls = True  # type: ignore[attr-defined]
    compiled_mod.CompiledModel.run = run  # type: ignore[method-assign]
