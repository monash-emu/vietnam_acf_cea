"""Health-economic analysis layered on the Vietnam TB model.

Entirely post-processing: every function here reads solved ``vacf`` output and none
modifies the model. ``src/vacf`` therefore stays byte-identical to upstream, its golden
parity tests keep covering it unchanged, and ``git pull upstream main`` continues to work.

That separation also means a change to a unit cost, a discount rate or the analytic
perspective is re-run in seconds and never requires re-running the calibration.
"""

from vacf.economics import (
    analysis,
    cea,
    contract,
    costing,
    costs,
    dalys,
    discounting,
    indicators,
    inputs,
    scenarios,
    study,
)

__all__ = [
    "analysis",
    "cea",
    "contract",
    "costing",
    "costs",
    "dalys",
    "discounting",
    "indicators",
    "inputs",
    "scenarios",
    "study",
]
