"""TB macroeconomics natural-history model on summer4.

JAX float64 is on because the summer3wip reference solves in float64. summer4
does not enable it, and parity at 1e-6 needs the same width.
"""

import jax

jax.config.update("jax_enable_x64", True)

from tb_macro.step_controls import install as _install_step_controls

_install_step_controls()

__version__ = "0.1.0"
