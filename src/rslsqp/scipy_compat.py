"""SciPy-compatible interface for the SLSQP solver.

This module provides :func:`minimize`, a drop-in replacement for
``scipy.optimize.minimize(method='SLSQP')``.  It accepts the same
arguments and returns an :class:`OptimizeResult`-compatible object.

Usage
-----
Replace::

    from scipy.optimize import minimize
    result = minimize(fun, x0, method='SLSQP', bounds=bounds,
                      constraints=constraints, jac=jac, options=options)

with::

    from rslsqp.scipy_compat import minimize
    result = minimize(fun, x0, bounds=bounds,
                      constraints=constraints, jac=jac, options=options)

The returned ``result`` object has the same attributes as
:class:`scipy.optimize.OptimizeResult`: ``x``, ``fun``, ``jac``,
``nit``, ``nfev``, ``njev``, ``status``, ``success``, ``message``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, cast

import numpy as np

from rslsqp.slsqp_module import (
    GradientMode,
    LinesearchMode,
    NnlsMode,
    SlsqpSolver,
)

__all__ = ["minimize", "OptimizeResult"]


# ---------------------------------------------------------------------------
# OptimizeResult — mirrors scipy.optimize.OptimizeResult
# ---------------------------------------------------------------------------


@dataclass
class OptimizeResult:
    """Optimisation result compatible with :class:`scipy.optimize.OptimizeResult`.

    Attributes
    ----------
    x : ndarray
        Solution vector.
    fun : float
        Objective value at ``x``.
    jac : ndarray or None
        Jacobian (gradient) of the objective at ``x``, if available.
    nit : int
        Number of iterations performed.
    nfev : int
        Number of objective function evaluations.
    njev : int
        Number of Jacobian evaluations.
    status : int
        Integer status code (0 = converged).
    success : bool
        ``True`` if the optimiser converged.
    message : str
        Human-readable description of the termination cause.
    """

    x: np.ndarray = field(default_factory=lambda: np.empty(0))
    fun: float = 0.0
    jac: np.ndarray | None = None
    nit: int = 0
    nfev: int = 0
    njev: int = 0
    status: int = 0
    success: bool = False
    message: str = ""

    # -- dict-like access for SciPy compatibility --------------------------
    # scipy.optimize.OptimizeResult inherits from dict; these methods let
    # user code that does ``result["x"]`` or ``"x" in result`` work
    # transparently with our dataclass-based implementation.

    def __getitem__(self, key: str) -> Any:
        """Return attribute *key* via dict-style ``result[key]`` access."""
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        """Return ``True`` if *key* is a valid result attribute."""
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Return attribute *key*, or *default* if it does not exist."""
        return getattr(self, key, default)

    def keys(self) -> list[str]:
        """Return the list of result attribute names."""
        return [f.name for f in self.__dataclass_fields__.values()]


# ---------------------------------------------------------------------------
# Constraint helpers
# ---------------------------------------------------------------------------

# Type alias for a single scipy-style constraint dict.
ConstraintDict = dict[str, Any]


def _parse_constraints(
    constraints: ConstraintDict | Sequence[ConstraintDict],
) -> tuple[
    list[Callable],  # eq_funs
    list[Callable | None],  # eq_jacs
    list[Callable],  # ineq_funs
    list[Callable | None],  # ineq_jacs
]:
    """Parse scipy-style constraint dicts into ordered lists.

    Returns ``(eq_funs, eq_jacs, ineq_funs, ineq_jacs)`` where each list
    has one entry per constraint block.  ``*_jacs`` entries may be ``None``
    if the corresponding block has no Jacobian.
    """
    if isinstance(constraints, dict):
        cons_list: Sequence[ConstraintDict] = [cast(ConstraintDict, constraints)]
    else:
        cons_list = constraints

    eq_funs: list[Callable] = []
    eq_jacs: list[Callable | None] = []
    ineq_funs: list[Callable] = []
    ineq_jacs: list[Callable | None] = []

    for con in cons_list:
        ctype = con.get("type", "").lower()
        cfun = con["fun"]
        cjac = con.get("jac", None)
        cargs = con.get("args", ())

        # Wrap so that the constraint callable has the same signature as fun(x).
        if cargs:
            _cfun = cfun
            _cjac = cjac

            def _wrapped_fun(x: np.ndarray, _f=_cfun, _a=cargs) -> np.ndarray:
                return np.atleast_1d(_f(x, *_a))

            cfun = _wrapped_fun
            if _cjac is not None and callable(_cjac):

                def _wrapped_jac(x: np.ndarray, _j=_cjac, _a=cargs) -> np.ndarray:
                    return np.atleast_2d(_j(x, *_a))

                cjac = _wrapped_jac
        else:
            _cfun_orig = cfun
            _cjac_orig = cjac

            def _wrap_fun(x: np.ndarray, _f=_cfun_orig) -> np.ndarray:
                return np.atleast_1d(_f(x))

            cfun = _wrap_fun
            if _cjac_orig is not None and callable(_cjac_orig):

                def _wrap_jac(x: np.ndarray, _j=_cjac_orig) -> np.ndarray:
                    return np.atleast_2d(_j(x))

                cjac = _wrap_jac

        if ctype == "eq":
            eq_funs.append(cfun)
            eq_jacs.append(cjac if callable(cjac) else None)
        elif ctype == "ineq":
            ineq_funs.append(cfun)
            ineq_jacs.append(cjac if callable(cjac) else None)
        else:
            raise ValueError(
                f"Unknown constraint type '{ctype}'. Must be 'eq' or 'ineq'."
            )

    return eq_funs, eq_jacs, ineq_funs, ineq_jacs


def _parse_bounds(bounds: Any, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert scipy-style bounds to ``(xl, xu)`` arrays.

    Accepts:
    - ``None`` → unbounded.
    - A sequence of ``(lo, hi)`` tuples (length *n*); ``None`` in a
      position means unbounded in that direction.
    - An object with ``.lb`` and ``.ub`` attributes
      (``scipy.optimize.Bounds``-compatible).
    """
    if bounds is None:
        return np.full(n, np.nan), np.full(n, np.nan)

    # scipy.optimize.Bounds-style object
    if hasattr(bounds, "lb") and hasattr(bounds, "ub"):
        lb = np.asarray(bounds.lb, dtype=float)
        ub = np.asarray(bounds.ub, dtype=float)
        # scipy uses -inf/inf for unbounded; map to NaN for our solver
        xl = np.where(np.isfinite(lb), lb, np.nan)
        xu = np.where(np.isfinite(ub), ub, np.nan)
        return xl, xu

    # Sequence of (lo, hi) tuples
    xl = np.full(n, np.nan)
    xu = np.full(n, np.nan)
    for i, b in enumerate(bounds):
        if b is None:
            continue
        lo, hi = b
        if lo is not None:
            lo_f = float(lo)
            if np.isfinite(lo_f):
                xl[i] = lo_f
        if hi is not None:
            hi_f = float(hi)
            if np.isfinite(hi_f):
                xu[i] = hi_f
    return xl, xu


# ---------------------------------------------------------------------------
# minimize — the main public function
# ---------------------------------------------------------------------------


def minimize(
    fun: Callable,
    x0: np.ndarray | Sequence[float],
    args: tuple = (),
    jac: Callable | bool | str | None = None,
    bounds: Any = None,
    constraints: ConstraintDict | Sequence[ConstraintDict] = (),
    tol: float | None = None,
    callback: Callable | None = None,
    options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> OptimizeResult:
    """Minimise a scalar function of one or more variables using SLSQP.

    This function is a **drop-in replacement** for
    ``scipy.optimize.minimize(method='SLSQP')``.

    Parameters
    ----------
    fun : callable
        Objective function ``fun(x, *args) -> float``.
        If *jac* is ``True``, must return ``(f, g)`` where *g* is the
        gradient array.
    x0 : array_like, shape (n,)
        Initial guess.
    args : tuple, optional
        Extra arguments passed to *fun* (and *jac* if callable).
    jac : callable, bool, or str, optional
        Method for computing the gradient of the objective:

        - ``None`` — use forward finite differences (default).
        - callable ``jac(x, *args) -> ndarray`` — user-supplied gradient.
        - ``True`` — *fun* returns ``(f, g)`` in a single call.
        - ``'2-point'`` / ``'3-point'`` — forward / central finite differences.
        - ``'cs'`` — complex-step differentiation (falls back to central).
    bounds : sequence of (min, max) or Bounds, optional
        Variable bounds.  Use ``None`` or ``±np.inf`` for unbounded.
    constraints : dict or list of dict, optional
        Constraint definitions.  Each dict has keys:

        - ``'type'``: ``'eq'`` or ``'ineq'``
        - ``'fun'``: ``con(x, *args) -> float or array``
        - ``'jac'``: ``jac(x, *args) -> array`` (optional)
        - ``'args'``: extra arguments (optional)

        Equality constraints enforce ``con(x) == 0``;
        inequality constraints enforce ``con(x) >= 0``.
    tol : float, optional
        Convergence tolerance.  Overrides ``options['ftol']`` if set.
    callback : callable, optional
        Called after each iteration as ``callback(x)`` (scipy convention).
    options : dict, optional
        Solver options:

        - ``maxiter`` (int, default 100): maximum iterations.
        - ``ftol`` (float, default 1e-8): convergence accuracy.
        - ``disp`` (bool, default False): print convergence messages.
        - ``eps`` (float, default 1e-8): step for finite differences.
        - ``finite_diff_rel_step`` (float): alias for ``eps``.
        - ``nnls_mode`` (int, default 1): 1 = NNLS, 2 = BVLS.
    **kwargs
        Ignored (for forward-compatibility).

    Returns
    -------
    OptimizeResult
        Result object with attributes: ``x``, ``fun``, ``jac``, ``nit``,
        ``nfev``, ``njev``, ``status``, ``success``, ``message``.
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    n = x0.size

    # --- Parse options ----------------------------------------------------
    opts = dict(options or {})
    maxiter = int(opts.get("maxiter", 100))
    ftol = float(opts.get("ftol", 1e-8))
    disp = bool(opts.get("disp", False))
    eps = float(opts.get("eps", opts.get("finite_diff_rel_step", 1e-8)))
    nnls_mode_val = int(opts.get("nnls_mode", 1))
    _NNLS_MAP = {1: NnlsMode.NNLS, 2: NnlsMode.BVLS}
    if nnls_mode_val not in _NNLS_MAP:
        raise ValueError(f"nnls_mode must be 1 (NNLS) or 2 (BVLS), got {nnls_mode_val}")

    if tol is not None:
        ftol = float(tol)

    # --- Parse bounds -----------------------------------------------------
    xl, xu = _parse_bounds(bounds, n)

    # --- Parse constraints ------------------------------------------------
    eq_funs, eq_jacs, ineq_funs, ineq_jacs = _parse_constraints(constraints)
    meq = sum(np.atleast_1d(f(x0)).size for f in eq_funs) if eq_funs else 0
    mineq = sum(np.atleast_1d(f(x0)).size for f in ineq_funs) if ineq_funs else 0
    m = meq + mineq

    # --- Determine gradient mode ------------------------------------------
    has_analytic_obj_jac = False
    jac_returns_with_fun = False
    gradient_mode = GradientMode.FORWARD  # default: forward FD

    if jac is True:
        # fun returns (f, g)
        jac_returns_with_fun = True
        has_analytic_obj_jac = True
    elif callable(jac):
        has_analytic_obj_jac = True
    elif jac == "3-point" or jac == "cs":
        gradient_mode = GradientMode.CENTRAL
    elif jac == "2-point" or jac is None or jac is False:
        pass  # already FORWARD
    else:
        raise ValueError(f"Unrecognised jac={jac!r}")

    # Check if all constraint Jacobians are provided
    all_con_jacs_available = all(j is not None for j in eq_jacs + ineq_jacs)

    # Use analytic gradients only when the objective gradient AND all
    # constraint Jacobians are available; otherwise fall back to FD.
    use_analytic_grad = has_analytic_obj_jac and (m == 0 or all_con_jacs_available)
    if use_analytic_grad:
        gradient_mode = GradientMode.USER

    # --- Counters ---------------------------------------------------------
    nfev = 0
    njev = 0
    last_jac: np.ndarray | None = None

    # --- Build the combined objective + constraint function ----------------
    def combined_func(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal nfev, last_jac
        nfev += 1

        if jac_returns_with_fun:
            result = fun(x, *args)
            f_val = float(result[0])
            last_jac = np.asarray(result[1], dtype=float)
        else:
            f_val = float(fun(x, *args))

        # Assemble constraint vector: equalities first, then inequalities
        c_parts: list[np.ndarray] = []
        for cfun in eq_funs:
            c_parts.append(np.atleast_1d(cfun(x)))
        for cfun in ineq_funs:
            c_parts.append(np.atleast_1d(cfun(x)))

        if c_parts:
            c_vec = np.concatenate(c_parts)
        else:
            c_vec = np.empty(0)

        return f_val, c_vec

    # --- Build the combined gradient function (analytic mode only) ---------
    combined_grad = None
    if use_analytic_grad:

        def _combined_grad(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            nonlocal njev, last_jac
            njev += 1

            # Objective gradient
            if jac_returns_with_fun:
                # The gradient was already computed in combined_func
                if last_jac is not None:
                    g = last_jac.copy()
                else:
                    # Fallback: call fun again
                    result = fun(x, *args)
                    g = np.asarray(result[1], dtype=float)
            else:
                # callable(jac) is guaranteed here since use_analytic_grad
                # requires has_analytic_obj_jac which means jac is True or
                # callable.
                g = np.asarray(cast(Callable, jac)(x, *args), dtype=float)

            # Constraint Jacobian: rows are equalities first, then inequalities
            a_parts: list[np.ndarray] = []
            for cjac_fn in eq_jacs:
                if cjac_fn is not None:
                    a_parts.append(np.atleast_2d(cjac_fn(x)))
            for cjac_fn in ineq_jacs:
                if cjac_fn is not None:
                    a_parts.append(np.atleast_2d(cjac_fn(x)))

            if a_parts:
                a = np.vstack(a_parts)
            else:
                a = np.empty((0, n), dtype=float)

            return g, a

        combined_grad = _combined_grad

    # --- Callback adapter -------------------------------------------------
    iter_callback = None
    if callback is not None:

        def iter_callback_adapter(
            it: int, x: np.ndarray, f: float, c: np.ndarray
        ) -> None:
            # scipy's SLSQP callback receives just x
            callback(x)  # type: ignore[misc]

        iter_callback = iter_callback_adapter

    # --- Build and run the solver -----------------------------------------
    solver_kwargs: dict[str, Any] = dict(
        func=combined_func,
        xl=xl,
        xu=xu,
        m=m,
        meq=meq,
        max_iter=maxiter,
        acc=ftol,
        linesearch_mode=LinesearchMode.INEXACT,
        nnls_mode=_NNLS_MAP[nnls_mode_val],
        iprint=0,
        callback=iter_callback,
    )

    if use_analytic_grad:
        solver_kwargs["grad"] = combined_grad
    solver_kwargs["gradient_mode"] = gradient_mode
    if not use_analytic_grad:
        solver_kwargs["gradient_delta"] = eps

    if disp:
        messages: list[str] = []
        solver_kwargs["msg_callback"] = messages.append

    solver = SlsqpSolver(**solver_kwargs)

    # Use the fast Rust iteration loop when no callback or disp needs
    # Python-side dispatch.  The Rust loop keeps the entire solve in
    # compiled code and only crosses the Python boundary for func/grad
    # evaluations.
    if callback is None and not disp:
        solver._use_rust_loop = True

    result = solver.optimize(x0)

    if disp:
        for msg in messages:
            print(msg)

    # --- Attach last known jac (no extra evaluation) ----------------------
    final_jac = last_jac

    return OptimizeResult(
        x=result.x,
        fun=result.fun,
        jac=final_jac,
        nit=result.iterations,
        nfev=nfev,
        njev=njev,
        status=int(result.status),
        success=result.success,
        message=result.message,
    )
