"""SLSQP solver — Rust-backed Python bindings."""

from importlib.metadata import version

__version__ = version("rslsqp")

from rslsqp.slsqp_module import (
    GradientMode,
    LinesearchMode,
    NnlsMode,
    SlsqpStatus,
    SlsqpResult,
    SlsqpSolver,
)

# SciPy-compatible interface
from rslsqp.scipy_compat import (
    minimize,
    OptimizeResult,
)

__all__ = [
    "GradientMode",
    "LinesearchMode",
    "NnlsMode",
    "SlsqpStatus",
    "SlsqpResult",
    "SlsqpSolver",
    "minimize",
    "OptimizeResult",
]
