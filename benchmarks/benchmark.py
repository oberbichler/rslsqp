"""Benchmark: rslsqp vs scipy SLSQP — drop-in replacement comparison.

Compares wall-clock time, CPU time, and iteration counts on realistic
optimisation problems.  Both solvers receive **identical arguments** —
the only difference is the import line:

    # scipy
    from scipy.optimize import minimize

    # rslsqp (drop-in replacement)
    from rslsqp import minimize

Both solvers receive **analytic gradients** so the benchmark isolates
solver-core overhead rather than finite-difference cost.
"""

from __future__ import annotations

import gc
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.optimize import minimize as scipy_minimize

from rslsqp import minimize as rslsqp_minimize

# ── reproducibility ──────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

# ── benchmark configuration ──────────────────────────────────────────────
N_WARMUP = 2  # warm-up runs (discarded)
N_REPEAT = 10  # timed runs


# ═══════════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class BenchResult:
    name: str
    solver: str
    n_vars: int
    n_constraints: int
    times_ms: list[float]
    cpu_times_ms: list[float]
    nit: int
    nfev: int
    njev: int
    fun: float
    success: bool

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms)

    @property
    def cpu_median_ms(self) -> float:
        return statistics.median(self.cpu_times_ms)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms)

    @property
    def stdev_ms(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Problem definitions
# ═══════════════════════════════════════════════════════════════════════════


def make_rosenbrock_unconstrained(n: int) -> dict[str, Any]:
    """Extended Rosenbrock (n variables, no constraints, with bounds)."""

    def fun(x: np.ndarray) -> float:
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))

    def jac(x: np.ndarray) -> np.ndarray:
        g = np.zeros_like(x)
        g[:-1] += -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
        g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
        return g

    x0 = np.full(n, -1.0)
    bounds = [(-5.0, 5.0)] * n

    return dict(
        name=f"Rosenbrock unconstrained (n={n})",
        fun=fun,
        jac=jac,
        x0=x0,
        bounds=bounds,
        constraints=[],
        n_vars=n,
        n_constraints=0,
    )


def make_rosenbrock_constrained(n: int) -> dict[str, Any]:
    """Extended Rosenbrock with inequality + equality constraints.

    Inequality: sum(x) >= n/2
    Equality:   x[0]^2 + x[1]^2 = 2
    """

    def fun(x: np.ndarray) -> float:
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))

    def jac(x: np.ndarray) -> np.ndarray:
        g = np.zeros_like(x)
        g[:-1] += -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
        g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
        return g

    x0 = np.full(n, 0.5)
    bounds = [(-5.0, 5.0)] * n

    constraints = [
        {
            "type": "ineq",
            "fun": lambda x, _n=n: np.array([np.sum(x) - _n / 2]),
            "jac": lambda x: np.ones((1, len(x))),
        },
        {
            "type": "eq",
            "fun": lambda x: np.array([x[0] ** 2 + x[1] ** 2 - 2.0]),
            "jac": lambda x: np.concatenate(
                [np.array([[2.0 * x[0], 2.0 * x[1]]]), np.zeros((1, len(x) - 2))],
                axis=1,
            ),
        },
    ]

    return dict(
        name=f"Rosenbrock constrained (n={n})",
        fun=fun,
        jac=jac,
        x0=x0,
        bounds=bounds,
        constraints=constraints,
        n_vars=n,
        n_constraints=2,
    )


def make_portfolio_optimisation(n: int) -> dict[str, Any]:
    """Markowitz mean-variance portfolio with n assets.

    Minimise  x^T Σ x  (variance)
    s.t.      μ^T x >= target_return       (inequality)
              sum(x) = 1                    (equality)
              0 <= x_i <= 1                 (bounds)
    """
    # Generate a realistic positive-definite covariance matrix
    A = RNG.standard_normal((n, n)) * 0.01
    cov = A.T @ A + np.eye(n) * 0.001  # ensure SPD
    mu = RNG.uniform(0.02, 0.15, n)  # expected returns
    target_return = float(np.median(mu))

    def fun(x: np.ndarray) -> float:
        return float(x @ cov @ x)

    def jac(x: np.ndarray) -> np.ndarray:
        return 2.0 * cov @ x

    x0 = np.full(n, 1.0 / n)
    bounds = [(0.0, 1.0)] * n

    constraints = [
        {
            "type": "eq",
            "fun": lambda x: np.array([np.sum(x) - 1.0]),
            "jac": lambda x: np.ones((1, len(x))),
        },
        {
            "type": "ineq",
            "fun": lambda x, _mu=mu, _tr=target_return: np.array([_mu @ x - _tr]),
            "jac": lambda x, _mu=mu: _mu.reshape(1, -1),
        },
    ]

    return dict(
        name=f"Portfolio optimisation (n={n})",
        fun=fun,
        jac=jac,
        x0=x0,
        bounds=bounds,
        constraints=constraints,
        n_vars=n,
        n_constraints=2,
    )


def make_quadratic_with_many_constraints(n: int, m_ineq: int) -> dict[str, Any]:
    """Convex quadratic with many linear inequality constraints.

    min  0.5 x^T H x + c^T x
    s.t. A_ineq x >= b_ineq    (m_ineq constraints)
         0 <= x_i <= 10        (bounds)
    """
    # Symmetric positive-definite Hessian
    L = RNG.standard_normal((n, n))
    H = L.T @ L + np.eye(n) * 0.5
    c_vec = RNG.standard_normal(n)

    # Generate feasible constraints: A x >= b where b is chosen so x0 is feasible
    x0 = np.full(n, 1.0)
    A_ineq = RNG.standard_normal((m_ineq, n))
    b_ineq = A_ineq @ x0 - RNG.uniform(0.5, 2.0, m_ineq)  # slack ensures feasibility

    def fun(x: np.ndarray) -> float:
        return float(0.5 * x @ H @ x + c_vec @ x)

    def jac(x: np.ndarray) -> np.ndarray:
        return H @ x + c_vec

    bounds = [(0.0, 10.0)] * n

    constraints = [
        {
            "type": "ineq",
            "fun": lambda x, _A=A_ineq, _b=b_ineq: _A @ x - _b,
            "jac": lambda x, _A=A_ineq: _A,
        },
    ]

    return dict(
        name=f"Quadratic + {m_ineq} ineq constraints (n={n})",
        fun=fun,
        jac=jac,
        x0=x0,
        bounds=bounds,
        constraints=constraints,
        n_vars=n,
        n_constraints=m_ineq,
    )


def make_least_squares_fitting(n: int) -> dict[str, Any]:
    """Non-linear least-squares curve fitting (n parameters).

    Fit y = sum_i  a_i * exp(-b_i * t)  to noisy data.
    n must be even (pairs of (a, b)).
    Bounded:  a_i in [0, 10], b_i in [0.01, 5].
    Equality: sum(a_i) = 5.
    """
    assert n % 2 == 0, "n must be even"
    n_terms = n // 2

    # Generate synthetic data
    t_data = np.linspace(0, 5, 200)
    true_a = RNG.uniform(0.5, 2.0, n_terms)
    true_b = RNG.uniform(0.1, 1.5, n_terms)
    y_clean = sum(true_a[i] * np.exp(-true_b[i] * t_data) for i in range(n_terms))
    y_data = y_clean + RNG.normal(0, 0.05, len(t_data))

    def _model(x: np.ndarray) -> np.ndarray:
        a = x[:n_terms]
        b = x[n_terms:]
        return sum(a[i] * np.exp(-b[i] * t_data) for i in range(n_terms))

    def fun(x: np.ndarray) -> float:
        residual = _model(x) - y_data
        return float(0.5 * np.sum(residual**2))

    def jac(x: np.ndarray) -> np.ndarray:
        a = x[:n_terms]
        b = x[n_terms:]
        residual = _model(x) - y_data
        g = np.zeros(n)
        for i in range(n_terms):
            exp_term = np.exp(-b[i] * t_data)
            g[i] = np.sum(residual * exp_term)  # d/da_i
            g[n_terms + i] = np.sum(residual * (-a[i] * t_data * exp_term))  # d/db_i
        return g

    x0 = np.ones(n)
    x0[:n_terms] = 5.0 / n_terms  # start feasible for the equality
    x0[n_terms:] = 0.5

    bounds = [(0.0, 10.0)] * n_terms + [(0.01, 5.0)] * n_terms

    constraints = [
        {
            "type": "eq",
            "fun": lambda x, _nt=n_terms: np.array([np.sum(x[:_nt]) - 5.0]),
            "jac": lambda x, _nt=n_terms: np.concatenate(
                [np.ones((1, _nt)), np.zeros((1, _nt))], axis=1
            ),
        },
    ]

    return dict(
        name=f"Nonlinear least-squares fitting (n={n})",
        fun=fun,
        jac=jac,
        x0=x0,
        bounds=bounds,
        constraints=constraints,
        n_vars=n,
        n_constraints=1,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Runners
# ═══════════════════════════════════════════════════════════════════════════


def _run_minimize(
    minimize_fn,
    problem: dict[str, Any],
    maxiter: int,
    ftol: float,
    solver_label: str,
    **extra_kwargs,
) -> BenchResult:
    """Time a scipy-compatible minimize() function."""
    times: list[float] = []
    cpu_times: list[float] = []
    result = None

    gc.disable()
    for i in range(N_WARMUP + N_REPEAT):
        t0 = time.perf_counter()
        c0 = time.process_time()
        result = minimize_fn(
            problem["fun"],
            problem["x0"],
            jac=problem["jac"],
            bounds=problem["bounds"] or None,
            constraints=problem["constraints"] or (),
            options={"maxiter": maxiter, "ftol": ftol, "disp": False},
            **extra_kwargs,
        )
        elapsed = (time.perf_counter() - t0) * 1000.0
        cpu_elapsed = (time.process_time() - c0) * 1000.0
        if i >= N_WARMUP:
            times.append(elapsed)
            cpu_times.append(cpu_elapsed)
    gc.collect()
    gc.enable()

    assert result is not None
    return BenchResult(
        name=problem["name"],
        solver=solver_label,
        n_vars=problem["n_vars"],
        n_constraints=problem["n_constraints"],
        times_ms=times,
        cpu_times_ms=cpu_times,
        nit=result.nit,
        nfev=result.nfev,
        njev=getattr(result, "njev", 0),
        fun=result.fun,
        success=result.success,
    )


def run_scipy(problem: dict[str, Any], maxiter: int, ftol: float) -> BenchResult:
    """Time scipy.optimize.minimize(method='SLSQP')."""
    return _run_minimize(
        lambda fun, x0, **kw: scipy_minimize(fun, x0, method="SLSQP", **kw),
        problem,
        maxiter,
        ftol,
        solver_label="scipy",
    )


def run_rslsqp(problem: dict[str, Any], maxiter: int, ftol: float) -> BenchResult:
    """Time rslsqp.minimize() — drop-in replacement."""
    return _run_minimize(
        rslsqp_minimize,
        problem,
        maxiter,
        ftol,
        solver_label="rslsqp",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Solution equivalence validation
# ═══════════════════════════════════════════════════════════════════════════


def _validate_solutions(r_scipy: BenchResult, r_rslsqp: BenchResult) -> None:
    """Check that both solvers agree on success and objective value."""
    for r in (r_scipy, r_rslsqp):
        if not r.success:
            print(f"  ⚠ {r.solver} did not converge (success=False, f={r.fun:.6e})")

    denom = max(abs(r_scipy.fun), 1.0)
    if abs(r_scipy.fun - r_rslsqp.fun) / denom > 1e-3:
        print(
            f"  ⚠ Solution mismatch: scipy f={r_scipy.fun:.6e}, "
            f"rslsqp f={r_rslsqp.fun:.6e}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Pretty-print
# ═══════════════════════════════════════════════════════════════════════════

HEADER = (
    f"{'Problem':<50} {'Solver':<8} {'Wall ms':>9} {'CPU ms':>9} "
    f"{'Mean ms':>9} {'± Std':>9} {'nit':>6} {'nfev':>7} {'njev':>7} "
    f"{'f(x*)':>14} {'OK':>4}"
)
SEP = "─" * len(HEADER)


def fmt_row(r: BenchResult) -> str:
    return (
        f"{r.name:<50} {r.solver:<8} {r.median_ms:>9.2f} {r.cpu_median_ms:>9.2f} "
        f"{r.mean_ms:>9.2f} {r.stdev_ms:>9.2f} {r.nit:>6} {r.nfev:>7} {r.njev:>7} "
        f"{r.fun:>14.6e} {'✓' if r.success else '✗':>4}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    maxiter = 500
    ftol = 1e-10

    problems: list[dict[str, Any]] = [
        # Unconstrained / bounded Rosenbrock at various sizes
        make_rosenbrock_unconstrained(50),
        make_rosenbrock_unconstrained(100),
        make_rosenbrock_unconstrained(200),
        # Constrained Rosenbrock
        make_rosenbrock_constrained(50),
        make_rosenbrock_constrained(100),
        # Portfolio optimisation
        make_portfolio_optimisation(50),
        make_portfolio_optimisation(100),
        make_portfolio_optimisation(200),
        # Quadratic with many constraints
        make_quadratic_with_many_constraints(50, 100),
        make_quadratic_with_many_constraints(100, 200),
        # Nonlinear least-squares
        make_least_squares_fitting(20),
        make_least_squares_fitting(40),
    ]

    results: list[tuple[BenchResult, BenchResult]] = []

    # Detect build mode from binary size heuristic
    import rslsqp._core as _rs

    so_path = Path(_rs.__file__)
    so_size = so_path.stat().st_size
    release_lib = Path("target/release/lib_core.dylib")
    debug_lib = Path("target/debug/lib_core.dylib")
    if release_lib.exists() and release_lib.stat().st_size == so_size:
        build_mode = "release ✓"
    elif debug_lib.exists() and debug_lib.stat().st_size == so_size:
        build_mode = "DEBUG ⚠️  (run: maturin develop --release)"
    else:
        build_mode = f"unknown ({so_size / 1024:.0f} KB)"

    print()
    print("=" * len(HEADER))
    print("  SLSQP Benchmark — drop-in replacement comparison")
    print()
    print("    from scipy.optimize import minimize    # ← scipy")
    print("    from rslsqp import minimize            # ← rslsqp (drop-in)")
    print()
    print(
        f"  Rust extension: {so_path.name}  ({so_size / 1024:.0f} KB, build: {build_mode})"
    )
    print(f"  {N_REPEAT} timed runs per solver per problem  (+ {N_WARMUP} warm-up)")
    print(f"  maxiter={maxiter}  ftol={ftol}")
    print()
    print(f"  Python: {platform.python_version()}")
    print(f"  NumPy:  {np.__version__}")
    print(f"  SciPy:  {scipy.__version__}")
    print(f"  CPU:    {platform.processor() or platform.machine()}")
    print(f"  OS:     {platform.platform()}")
    print("=" * len(HEADER))
    print()
    print(HEADER)
    print(SEP)

    for prob in problems:
        r_scipy = run_scipy(prob, maxiter, ftol)
        r_rslsqp = run_rslsqp(prob, maxiter, ftol)

        print(fmt_row(r_scipy))
        print(fmt_row(r_rslsqp))

        speedup = (
            r_scipy.median_ms / r_rslsqp.median_ms
            if r_rslsqp.median_ms > 0
            else float("inf")
        )
        tag = "faster" if speedup > 1.0 else "slower"
        print(f"{'':>60} rslsqp is {speedup:.2f}x {tag} than scipy")

        _validate_solutions(r_scipy, r_rslsqp)
        print(SEP)

        results.append((r_scipy, r_rslsqp))

    # ── summary ───────────────────────────────────────────────────────────
    print()
    print("Summary")
    print("───────")
    speedups: list[float] = []
    for r_sp, r_rs in results:
        s = r_sp.median_ms / r_rs.median_ms if r_rs.median_ms > 0 else float("inf")
        speedups.append(s)
        tag = "faster" if s > 1.0 else "slower"
        print(f"  {r_sp.name:<48} rslsqp {s:>6.2f}x {tag}")

    geo = float(np.exp(np.mean(np.log(speedups))))
    print()
    print(f"  Geometric mean speedup (rslsqp vs scipy): {geo:.2f}x")
    print()


if __name__ == "__main__":
    main()
