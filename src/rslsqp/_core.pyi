"""Type stubs for the ``rslsqp._core`` native extension module.

This module is implemented in Rust via PyO3/maturin and exposes the SLSQP
solver core, workspace types, enum types, and helper functions to Python.
"""

from typing import Callable

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Enum types
# ---------------------------------------------------------------------------

class GradientMode:
    """How gradients are supplied or approximated."""

    USER: GradientMode
    """User-supplied gradient callback."""
    BACKWARD: GradientMode
    """Backward finite differences."""
    FORWARD: GradientMode
    """Forward finite differences."""
    CENTRAL: GradientMode
    """Central finite differences."""

    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __int__(self) -> int: ...

class LinesearchMode:
    """Line-search strategy."""

    INEXACT: LinesearchMode
    """Inexact (Armijo-type) line-search."""
    EXACT: LinesearchMode
    """Exact (golden-section / parabolic) line-search."""

    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __int__(self) -> int: ...

class NnlsMode:
    """Which non-negative least-squares method to use."""

    NNLS: NnlsMode
    """Original NNLS algorithm."""
    BVLS: NnlsMode
    """Newer BVLS algorithm."""

    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __int__(self) -> int: ...

class SlsqpStatus:
    """Solver status returned by the SLSQP reverse-communication protocol."""

    CONVERGED: SlsqpStatus
    """Mode 0 — solver converged within requested accuracy."""
    FUNC_EVAL_REQUIRED: SlsqpStatus
    """Mode 1 — solver requests a function evaluation."""
    GRAD_EVAL_REQUIRED: SlsqpStatus
    """Mode -1 — solver requests a gradient evaluation."""
    USER_STOP: SlsqpStatus
    """Mode -2 — user called ``abort()``."""
    TOO_MANY_EQUALITY_CONSTRAINTS: SlsqpStatus
    """Mode 2 — more equality constraints than variables."""
    LSQ_ITERATIONS_EXCEEDED: SlsqpStatus
    """Mode 3 — LSQ sub-problem exceeded 3·n iterations."""
    INCOMPATIBLE_INEQUALITY: SlsqpStatus
    """Mode 4 — inequality constraints are incompatible."""
    SINGULAR_MATRIX_E: SlsqpStatus
    """Mode 5 — singular matrix E in LSQ sub-problem."""
    SINGULAR_MATRIX_C: SlsqpStatus
    """Mode 6 — singular matrix C in LSQ sub-problem."""
    RANK_DEFICIENT_HFTI: SlsqpStatus
    """Mode 7 — rank-deficient equality constraint sub-problem (HFTI)."""
    POSITIVE_DIRECTIONAL_DERIVATIVE: SlsqpStatus
    """Mode 8 — positive directional derivative for line-search."""
    MAX_ITERATIONS_REACHED: SlsqpStatus
    """Mode 9 — exceeded ``max_iter`` iterations."""
    INVALID_X_SIZE: SlsqpStatus
    """Mode -100 — ``x`` has wrong length."""
    INVALID_LINESEARCH_MODE: SlsqpStatus
    """Mode -101 — invalid line-search mode value."""
    FUNCTION_NOT_ASSOCIATED: SlsqpStatus
    """Mode -102 — objective function not provided."""
    GRADIENT_NOT_ASSOCIATED: SlsqpStatus
    """Mode -103 — gradient function not provided."""
    INVALID_GRADIENT_MODE: SlsqpStatus
    """Mode -104 — invalid gradient mode value."""
    INVALID_PERTURBATION_STEP: SlsqpStatus
    """Mode -105 — invalid FD perturbation step."""
    UNKNOWN: SlsqpStatus
    """Any unrecognised mode value."""

    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __int__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

class SlsqpWorkspace:
    """Pre-allocated workspace for the SLSQP solver.

    Owns all sub-arrays and internal state so that :func:`slsqp` can reuse
    them across reverse-communication iterations without allocation.

    Parameters
    ----------
    n : int
        Number of optimisation variables.
    m : int
        Total number of constraints.
    meq : int
        Number of equality constraints (``0 <= meq <= m``).
    """

    def __init__(self, n: int, m: int, meq: int) -> None: ...
    def reset(self) -> None:
        """Reset all workspace arrays to zero for reuse across optimisations."""
        ...

# ---------------------------------------------------------------------------
# Result type for the high-level ``optimize`` function
# ---------------------------------------------------------------------------

class RustOptimizeResult:
    """Result returned by :func:`optimize`.

    All attributes are read-only properties.
    """

    @property
    def x(self) -> npt.NDArray[np.float64]:
        """Optimal variable vector."""
        ...
    @property
    def fun(self) -> float:
        """Objective function value at the solution."""
        ...
    @property
    def constraints(self) -> npt.NDArray[np.float64]:
        """Constraint values at the solution."""
        ...
    @property
    def status(self) -> SlsqpStatus:
        """Solver exit status."""
        ...
    @property
    def message(self) -> str:
        """Human-readable status message."""
        ...
    @property
    def iterations(self) -> int:
        """Number of iterations performed."""
        ...
    @property
    def success(self) -> bool:
        """``True`` if the solver converged."""
        ...
    @property
    def nfev(self) -> int:
        """Number of function evaluations."""
        ...
    @property
    def njev(self) -> int:
        """Number of gradient evaluations."""
        ...

# ---------------------------------------------------------------------------
# Core solver — one reverse-communication step
# ---------------------------------------------------------------------------

def slsqp(
    m: int,
    meq: int,
    la: int,
    n: int,
    x: npt.NDArray[np.float64],
    xl: npt.NDArray[np.float64],
    xu: npt.NDArray[np.float64],
    f: float,
    c: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    a: npt.NDArray[np.float64],
    acc: float,
    iter: int,
    mode: int,
    ws: SlsqpWorkspace,
    alphamin: float,
    alphamax: float,
    tolf: float,
    toldf: float,
    toldx: float,
    max_iter_ls: int,
    nnls_mode: NnlsMode,
    infinite_bound: float,
) -> tuple[float, int, int]:
    """Perform one step of the SLSQP reverse-communication solver.

    Parameters
    ----------
    m : int
        Total number of constraints.
    meq : int
        Number of equality constraints.
    la : int
        Leading dimension of the constraint array (``max(1, m)``).
    n : int
        Number of optimisation variables.
    x : ndarray, shape (n,)
        Current variable vector (modified in-place).
    xl : ndarray, shape (n,)
        Lower bounds.
    xu : ndarray, shape (n,)
        Upper bounds.
    f : float
        Current objective value.
    c : ndarray, shape (la,)
        Constraint values (modified in-place).
    g : ndarray, shape (n+1,)
        Objective gradient (modified in-place).
    a : ndarray, shape (la, n+1)
        Constraint Jacobian (modified in-place).
    acc : float
        Convergence accuracy.
    iter : int
        Remaining iteration count.
    mode : int
        Current solver mode (0 = initial, 1 = func eval, -1 = grad eval).
    ws : SlsqpWorkspace
        Pre-allocated workspace.
    alphamin : float
        Minimum line-search step length.
    alphamax : float
        Maximum line-search step length.
    tolf : float
        Absolute function value tolerance (negative = disabled).
    toldf : float
        Function change tolerance (negative = disabled).
    toldx : float
        Variable change tolerance (negative = disabled).
    max_iter_ls : int
        Maximum NNLS/BVLS iterations (0 = ``3*n``).
    nnls_mode : NnlsMode
        Which NNLS implementation to use.
    infinite_bound : float
        Threshold for infinite bounds.

    Returns
    -------
    tuple[float, int, int]
        ``(acc, iter, mode)`` — updated accuracy, iteration count, and mode.
    """
    ...

# ---------------------------------------------------------------------------
# High-level optimise — entire loop in Rust
# ---------------------------------------------------------------------------

def optimize(
    func: Callable[[npt.NDArray[np.float64]], tuple[float, npt.NDArray[np.float64]]],
    grad: Callable[
        [npt.NDArray[np.float64]],
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    ]
    | None,
    x0: npt.NDArray[np.float64],
    xl: npt.NDArray[np.float64],
    xu: npt.NDArray[np.float64],
    m: int,
    meq: int,
    max_iter: int,
    acc: float,
    gradient_mode: GradientMode,
    gradient_delta: float,
    linesearch_mode: LinesearchMode,
    alphamin: float,
    alphamax: float,
    tolf: float,
    toldf: float,
    toldx: float,
    max_iter_ls: int,
    nnls_mode: NnlsMode,
    infinite_bound: float,
    workspace: SlsqpWorkspace | None = None,
) -> RustOptimizeResult:
    """Run the entire SLSQP optimization loop in Rust.

    Only crosses the Python boundary for function/gradient evaluations.

    Parameters
    ----------
    func : callable
        Python callable ``(x_array) -> (f, c)`` where *f* is a float and
        *c* is a 1-D constraint array.
    grad : callable or None
        Python callable ``(x_array) -> (g, a)`` providing analytic gradients,
        or ``None`` for finite-difference mode.
    x0 : ndarray, shape (n,)
        Initial guess.
    xl : ndarray, shape (n,)
        Lower bounds (NaN for unbounded).
    xu : ndarray, shape (n,)
        Upper bounds (NaN for unbounded).
    m : int
        Total number of constraints.
    meq : int
        Number of equality constraints.
    max_iter : int
        Maximum number of iterations.
    acc : float
        Convergence tolerance.
    gradient_mode : GradientMode
        How gradients are computed.
    gradient_delta : float
        Step size for finite differences.
    linesearch_mode : LinesearchMode
        Line-search strategy.
    alphamin : float
        Minimum step length.
    alphamax : float
        Maximum step length.
    tolf : float
        Absolute function tolerance (negative = disabled).
    toldf : float
        Function change tolerance (negative = disabled).
    toldx : float
        Variable change tolerance (negative = disabled).
    max_iter_ls : int
        Maximum NNLS iterations.
    nnls_mode : NnlsMode
        Which NNLS implementation to use.
    infinite_bound : float
        Threshold for infinite bounds.
    workspace : SlsqpWorkspace or None
        Optional pre-allocated workspace for reuse.

    Returns
    -------
    RustOptimizeResult
        Optimisation result with fields ``x``, ``fun``, ``constraints``,
        ``status``, ``message``, ``iterations``, and ``success``.
    """
    ...

# ---------------------------------------------------------------------------
# Finite-difference gradient helper
# ---------------------------------------------------------------------------

def compute_fd_gradients(
    func: Callable[[npt.NDArray[np.float64]], tuple[float, npt.NDArray[np.float64]]],
    x: npt.NDArray[np.float64],
    gradient_mode: GradientMode,
    gradient_delta: float,
    m: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute finite-difference gradients entirely in Rust.

    Eliminates the Python-level ``for ig in range(n)`` loop, keeping all
    perturbation/differencing arithmetic in Rust and only crossing the
    Python boundary for function evaluations.

    Parameters
    ----------
    func : callable
        Python callable ``(x_array) -> (f, c)`` where *f* is a float and
        *c* is a 1-D constraint array.
    x : ndarray, shape (n,)
        Current point.
    gradient_mode : GradientMode
        Finite-difference mode (BACKWARD, FORWARD, or CENTRAL).
        Must not be ``GradientMode.USER``.
    gradient_delta : float
        FD step size.
    m : int
        Number of constraints.

    Returns
    -------
    tuple[ndarray, ndarray]
        ``(g, a)`` — objective gradient of shape ``(n,)`` and constraint
        Jacobian of shape ``(m, n)``.
    """
    ...
