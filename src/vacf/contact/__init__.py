"""Contact matrices that do not rebuild a 121×121 kernel inside a solver step.

``prepare_stepwise`` matches ``get_norm_c_matrix``. ``prepare_smooth`` treats
the annual demographic tables as samples of smooth curves. ``prepare_spline``
is the default contact path for ``run`` / calibration: it normalises at each
calendar year and joins those matrices with a cubic, so a solver stage never
sees an eigenproblem. The golden / ``match_golden`` path still calls
``get_norm_c_matrix``. See ``plans/mixing-matrix.md``.
"""

from __future__ import annotations

from typing import Any, Protocol

from vacf.contact.smooth import SmoothContact, prepare_smooth
from vacf.contact.spline import SplineContact, prepare_spline
from vacf.contact.stepwise import StepwiseContact, prepare_stepwise

__all__ = [
    "ContactMatrix",
    "SmoothContact",
    "SplineContact",
    "StepwiseContact",
    "prepare_smooth",
    "prepare_spline",
    "prepare_stepwise",
]


class ContactMatrix(Protocol):
    """What a later package can depend on, without the Vietnam model."""

    def matrix(
        self,
        time: Any,
        bg_mixing: Any = None,
        a_spread: Any = None,
        pc_strength: Any = None,
    ) -> Any:
        """Normalised age-band contact matrix at ``time``."""

    def materialize(
        self,
        bg_mixing: Any = None,
        a_spread: Any = None,
        pc_strength: Any = None,
    ) -> Any:
        """Fold free mixing scalars into stored curves or a year table."""
