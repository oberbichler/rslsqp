"""Object-oriented interface to the SLSQP solver.

This module provides :class:`SlsqpSolver`, a Pythonic convenience wrapper
around the low-level :func:`rslsqp._core.slsqp` reverse-communication
routine.  It handles workspace allocation, the iteration loop, optional
finite-difference gradient approximation, and callback dispatch.

Ported from Jacob Williams' Fortran ``slsqp_module`` (BSD license).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

from rslsqp._core import SlsqpWorkspace, slsqp
from rslsqp._core import optimize as _rust_optimize
from rslsqp._core import compute_fd_gradients as _rust_fd_gradients

# Enums and status helper — single source of truth lives in Rust.
from rslsqp._core import (
    GradientMode,
    LinesearchMode,
    NnlsMode,
    SlsqpStatus,
)

# Machine epsilon
_EPMACH: float = float(np.finfo(np.float64).eps)

# Lookup table for converting raw mode int → SlsqpStatus enum.
# PyO3 enums do not support construction via ``SlsqpStatus(int_val)``
# and are not iterable, so we list all variants explicitly.
_STATUS_FROM_MODE: dict[int, SlsqpStatus] = {
    0: SlsqpStatus.CONVERGED,
    1: SlsqpStatus.FUNC_EVAL_REQUIRED,
    -1: SlsqpStatus.GRAD_EVAL_REQUIRED,
    -2: SlsqpStatus.USER_STOP,
    2: SlsqpStatus.TOO_MANY_EQUALITY_CONSTRAINTS,
    3: SlsqpStatus.LSQ_ITERATIONS_EXCEEDED,
    4: SlsqpStatus.INCOMPATIBLE_INEQUALITY,
    5: SlsqpStatus.SINGULAR_MATRIX_E,
    6: SlsqpStatus.SINGULAR_MATRIX_C,
    7: SlsqpStatus.RANK_DEFICIENT_HFTI,
    8: SlsqpStatus.POSITIVE_DIRECTIONAL_DERIVATIVE,
    9: SlsqpStatus.MAX_ITERATIONS_REACHED,
    -100: SlsqpStatus.INVALID_X_SIZE,
    -101: SlsqpStatus.INVALID_LINESEARCH_MODE,
    -102: SlsqpStatus.FUNCTION_NOT_ASSOCIATED,
    -103: SlsqpStatus.GRADIENT_NOT_ASSOCIATED,
    -104: SlsqpStatus.INVALID_GRADIENT_MODE,
    -105: SlsqpStatus.INVALID_PERTURBATION_STEP,
}

__all__ = [
    "GradientMode",
    "LinesearchMode",
    "NnlsMode",
    "SlsqpStatus",
    "SlsqpResult",
    "SlsqpSolver",
]


# ---------------------------------------------------------------------------
# Callback type aliases
# ---------------------------------------------------------------------------

# objective + constraints
ObjectiveFunc = Callable[[np.ndarray], tuple[float, np.ndarray]]
"""``(x) -> (f, c)`` where *f* is the scalar objective and *c* is
the constraint vector of length *m* (equality constraints first)."""

# gradient callback
GradientFunc = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
"""``(x) -> (g, a)`` where *g* has shape ``(n,)`` (objective partials)
and *a* has shape ``(m, n)`` (constraint Jacobian)."""

# iteration callback
IterCallback = Callable[[int, np.ndarray, float, np.ndarray], None]
"""``(iteration, x, f, c) -> None``."""

# message callback
MsgCallback = Callable[[str], None]
"""``(message) -> None``."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SlsqpResult:
    """Result returned by :meth:`SlsqpSolver.optimize`."""

    x: np.ndarray
    """Optimal variable vector."""
    fun: float
    """Objective function value at the solution."""
    constraints: np.ndarray
    """Constraint values at the solution."""
    status: SlsqpStatus
    """Solver exit status (see :class:`SlsqpStatus`)."""
    message: str
    """Human-readable status message."""
    iterations: int
    """Number of iterations performed."""
    success: bool
    """``True`` if *status* is :attr:`SlsqpStatus.CONVERGED`."""


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------


class SlsqpSolver:
    """High-level, Pythonic interface to the SLSQP optimiser.

    Parameters
    ----------
    func : ObjectiveFunc
        Callable ``(x) -> (f, c)`` returning the scalar objective *f* and a
        1-D constraint array *c* of length *m* (equality constraints first).
        When there are no constraints, *c* should be an empty array.
    xl, xu : array_like, shape (n,)
        Lower / upper bounds on the variables.  Use ``np.nan`` (or
        ``±infinite_bound``) for unbounded variables.
    m : int
        Total number of constraints (``>= 0``).
    meq : int
        Number of *equality* constraints (``0 <= meq <= m``).
    max_iter : int
        Maximum number of major iterations.
    acc : float
        Convergence accuracy tolerance.
    grad : GradientFunc or None
        Callable ``(x) -> (g, a)`` providing analytic gradients.  Required
        when *gradient_mode* is :attr:`GradientMode.USER`; ignored
        otherwise.
    gradient_mode : GradientMode
        How gradients are computed (default: :attr:`GradientMode.USER`).
    gradient_delta : float
        Step size for finite-difference gradient approximation
        (BACKWARD / FORWARD / CENTRAL modes).
    linesearch_mode : LinesearchMode
        Line-search strategy (default: :attr:`LinesearchMode.INEXACT`).
    alphamin, alphamax : float
        Lower / upper bound for the line-search step (0 < alphamin < alphamax <= 1).
    tolf : float
        Stop if ``|f| < tolf``.  Negative means disabled.
    toldf : float
        Stop if ``|f_{k+1} - f_k| < toldf``.  Negative means disabled.
    toldx : float
        Stop if ``||x_{k+1} - x_k|| < toldx``.  Negative means disabled.
    max_iter_ls : int
        Maximum iterations in the NNLS/BVLS sub-problem (0 → ``3*n``).
    nnls_mode : NnlsMode
        Which NNLS implementation to use.
    iprint : int or None
        File descriptor for status messages (``0`` = silent, ``None`` = ``sys.stdout``).
    callback : IterCallback or None
        Called once per iteration with ``(iteration, x, f, c)``.
    msg_callback : MsgCallback or None
        Receives warning / error message strings.
    infinite_bound : float
        Value treated as "unbounded" for bounds (default: ``np.inf``).
    """

    def __init__(
        self,
        func: ObjectiveFunc,
        xl: np.ndarray,
        xu: np.ndarray,
        *,
        m: int = 0,
        meq: int = 0,
        max_iter: int = 100,
        acc: float = 1.0e-8,
        grad: GradientFunc | None = None,
        gradient_mode: GradientMode = GradientMode.USER,
        gradient_delta: float = 1.0e-8,
        linesearch_mode: LinesearchMode = LinesearchMode.INEXACT,
        alphamin: float = 0.1,
        alphamax: float = 1.0,
        tolf: float = -1.0,
        toldf: float = -1.0,
        toldx: float = -1.0,
        max_iter_ls: int = 0,
        nnls_mode: NnlsMode = NnlsMode.NNLS,
        iprint: int | None = None,
        callback: IterCallback | None = None,
        msg_callback: MsgCallback | None = None,
        infinite_bound: float = np.inf,
    ) -> None:
        xl = np.asarray(xl, dtype=float)
        xu = np.asarray(xu, dtype=float)
        n = xl.size

        # --- validation ------------------------------------------------
        if xu.size != n:
            raise ValueError(
                f"xl and xu must have the same length (got {n} vs {xu.size})"
            )
        if n < 1:
            raise ValueError(f"n must be >= 1 (got {n})")
        if m < 0:
            raise ValueError(f"m must be >= 0 (got {m})")
        if meq < 0 or meq > m:
            raise ValueError(f"meq must be in [0, m] (got meq={meq}, m={m})")
        # bounds consistency (ignore NaN entries)
        mask = ~np.isnan(xl) & ~np.isnan(xu)
        if np.any(xl[mask] > xu[mask]):
            bad = np.nonzero(xl[mask] > xu[mask])[0]
            raise ValueError(
                f"Lower bounds must be <= upper bounds. Violated at indices: {bad.tolist()}"
            )
        if not isinstance(linesearch_mode, LinesearchMode):
            raise ValueError(
                f"linesearch_mode must be a LinesearchMode enum value (got {linesearch_mode!r})"
            )
        if not isinstance(gradient_mode, GradientMode):
            raise ValueError(
                f"gradient_mode must be a GradientMode enum value (got {gradient_mode!r})"
            )
        if not isinstance(nnls_mode, NnlsMode):
            raise ValueError(
                f"nnls_mode must be a NnlsMode enum value (got {nnls_mode!r})"
            )
        if not (0.0 < alphamin < alphamax <= 1.0):
            raise ValueError(
                f"Need 0 < alphamin < alphamax <= 1 (got {alphamin}, {alphamax})"
            )
        if gradient_mode == GradientMode.USER and grad is None:
            raise ValueError(
                "grad callback must be provided when gradient_mode is USER"
            )
        if gradient_mode != GradientMode.USER and gradient_delta <= _EPMACH:
            raise ValueError(
                f"gradient_delta must be > machine epsilon for FD modes (got {gradient_delta})"
            )

        # --- store parameters ------------------------------------------
        self._func = func
        self._grad = grad
        self._callback = callback
        self._msg_callback = msg_callback
        self._iprint = iprint  # None → stdout

        self._n = n
        self._m = m
        self._meq = meq
        self._max_iter = max_iter
        self._acc = float(acc)
        self._xl = xl.copy()
        self._xu = xu.copy()
        self._gradient_mode = gradient_mode
        self._gradient_delta = float(gradient_delta)
        self._linesearch_mode = linesearch_mode
        self._alphamin = float(alphamin)
        self._alphamax = float(alphamax)
        self._tolf = float(tolf)
        self._toldf = float(toldf)
        self._toldx = float(toldx)
        self._max_iter_ls = int(max_iter_ls)
        self._nnls_mode = nnls_mode
        self._infinite_bound = abs(float(infinite_bound))
        self._use_rust_loop = False  # opt-in fast path

        # --- workspace allocation (Rust handles all sizing) ------------
        self._ws = SlsqpWorkspace(n, m, meq)

        # --- internal state (reset each optimize call) -----------------
        self._user_triggered_stop = False

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def abort(self) -> None:
        """Request the solver to stop after the current iteration."""
        self._user_triggered_stop = True

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def _report_message(self, msg: str) -> None:
        """Route a status / error message to the configured output."""
        if self._msg_callback is not None:
            self._msg_callback(msg)
        elif self._iprint is None:
            print(msg, file=sys.stdout)
        elif self._iprint != 0:
            print(msg, file=sys.stderr)

    # ------------------------------------------------------------------
    # Gradient computation
    # ------------------------------------------------------------------

    def _compute_gradients(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(g, a)`` – objective gradient and constraint Jacobian.

        Dispatches to the user callback or to finite differences depending
        on *gradient_mode*.  For FD modes the perturbation loop runs in Rust
        via :func:`rslsqp._core.compute_fd_gradients`, eliminating the
        Python-level ``for ig in range(n)`` overhead.
        """
        if self._gradient_mode == GradientMode.USER:
            assert self._grad is not None
            return self._grad(x)

        # Delegate the FD loop to Rust — only Python→Rust boundary
        # crossings are the func evaluations themselves.
        return _rust_fd_gradients(
            self._func,
            x,
            self._gradient_mode,
            self._gradient_delta,
            self._m,
        )

    # ------------------------------------------------------------------
    # Main optimisation loop
    # ------------------------------------------------------------------

    def optimize(self, x0: np.ndarray) -> SlsqpResult:
        """Run the SLSQP optimiser starting from *x0*.

        Parameters
        ----------
        x0 : array_like, shape (n,)
            Initial guess.

        Returns
        -------
        SlsqpResult
            Named result with fields *x*, *fun*, *constraints*, *status*,
            *message*, *iterations*, and *success*.
        """
        x = np.array(x0, dtype=float, copy=True)
        n = self._n
        m = self._m

        if x.size != n:
            status = SlsqpStatus.INVALID_X_SIZE
            self._report_message(str(status))
            return SlsqpResult(
                x=x,
                fun=np.nan,
                constraints=np.empty(0),
                status=status,
                message=str(status),
                iterations=0,
                success=False,
            )

        # ── Fast path: entire loop in Rust ────────────────────────────
        # Use when explicitly opted in (no abort/callback support in Rust loop).
        if self._use_rust_loop:
            return self._optimize_rust(x)

        # ── Slow path: Python iteration loop ──────────────────────────
        return self._optimize_python(x)

    def _optimize_rust(self, x: np.ndarray) -> SlsqpResult:
        """Fast path: delegate the entire iteration loop to Rust."""
        n = self._n
        m = self._m
        grad_fn = self._grad if self._gradient_mode == GradientMode.USER else None

        result = _rust_optimize(
            self._func,
            grad_fn,
            x,
            self._xl,
            self._xu,
            m,
            self._meq,
            self._max_iter,
            self._acc,
            self._gradient_mode,
            self._gradient_delta,
            self._linesearch_mode,
            self._alphamin,
            self._alphamax,
            self._tolf,
            self._toldf,
            self._toldx,
            self._max_iter_ls,
            self._nnls_mode,
            self._infinite_bound,
            workspace=self._ws,
        )

        return SlsqpResult(
            x=np.asarray(result.x),
            fun=result.fun,
            constraints=np.asarray(result.constraints),
            status=result.status,
            message=result.message,
            iterations=result.iterations,
            success=result.success,
        )

    def _optimize_python(self, x: np.ndarray) -> SlsqpResult:
        """Slow path: Python iteration loop with callback support."""
        n = self._n
        m = self._m

        # Reset workspace & internal state
        self._user_triggered_stop = False
        self._ws.reset()

        # Allocate working arrays
        la = max(1, m)
        c = np.zeros(la, dtype=float)
        a = np.zeros((la, n + 1), dtype=float)
        g = np.zeros(n + 1, dtype=float)

        i_iter = 0
        iter_ = self._max_iter
        mode = 0

        # Line-search sign convention
        if self._linesearch_mode == LinesearchMode.EXACT:
            acc = -abs(self._acc)
        else:
            acc = abs(self._acc)

        f_val = 0.0
        cvec = np.zeros(m, dtype=float)

        # --- iteration loop -------------------------------------------
        while True:
            # --- function evaluation ---
            if mode == 0 or mode == 1:
                f_val, cvec_raw = self._func(x)
                cvec = np.asarray(cvec_raw, dtype=float)
                if m > 0:
                    c[:m] = cvec

            # --- gradient evaluation ---
            if mode == 0 or mode == -1:
                dfdx, dcdx = self._compute_gradients(x)
                g[:n] = dfdx
                if m > 0:
                    a[:m, :n] = dcdx

                # report iteration (initial guess is iteration 0)
                if self._callback is not None:
                    self._callback(i_iter, x.copy(), f_val, cvec.copy())
                i_iter += 1

            # --- call core solver ---
            acc, iter_, mode = slsqp(
                m,
                self._meq,
                la,
                n,
                x,
                self._xl,
                self._xu,
                f_val,
                c,
                g,
                a,
                acc,
                iter_,
                mode,
                self._ws,
                self._alphamin,
                self._alphamax,
                self._tolf,
                self._toldf,
                self._toldx,
                self._max_iter_ls,
                self._nnls_mode,
                self._infinite_bound,
            )

            if mode == 1 or mode == -1:
                pass  # continue to next evaluation
            else:
                # report final solution
                if mode == 0 and self._callback is not None:
                    self._callback(i_iter, x.copy(), f_val, cvec.copy())
                self._report_message(str(_STATUS_FROM_MODE.get(mode, SlsqpStatus.UNKNOWN)))
                break

            if self._user_triggered_stop:
                mode = -2
                self._report_message(str(_STATUS_FROM_MODE.get(mode, SlsqpStatus.UNKNOWN)))
                self._user_triggered_stop = False
                break

        status = _STATUS_FROM_MODE.get(mode, SlsqpStatus.UNKNOWN)
        return SlsqpResult(
            x=x,
            fun=f_val,
            constraints=cvec,
            status=status,
            message=str(status),
            iterations=iter_,
            success=(status == SlsqpStatus.CONVERGED),
        )
