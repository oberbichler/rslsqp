"""Tests for slsqp_module – the Pythonic OO wrapper around the SLSQP solver."""

from __future__ import annotations

import numpy as np
import pytest

from rslsqp.slsqp_module import (
    GradientMode,
    LinesearchMode,
    NnlsMode,
    SlsqpStatus,
    SlsqpResult,
    SlsqpSolver,
)

# ---------------------------------------------------------------------------
# Helpers: Rosenbrock problem (same as Fortran test suite)
# ---------------------------------------------------------------------------


def rosenbrock_func(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Rosenbrock objective + one inequality constraint  1 - x0² - x1² >= 0."""
    f = 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2
    c = np.array([1.0 - x[0] ** 2 - x[1] ** 2])
    return f, c


def rosenbrock_grad(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Analytic gradient for the Rosenbrock + circle constraint."""
    g = np.array(
        [
            -400.0 * (x[1] - x[0] ** 2) * x[0] - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0] ** 2),
        ]
    )
    a = np.array([[-2.0 * x[0], -2.0 * x[1]]])
    return g, a


# ---------------------------------------------------------------------------
# Test: SlsqpStatus display
# ---------------------------------------------------------------------------


class TestSlsqpStatusDisplay:
    def test_converged(self) -> None:
        assert str(SlsqpStatus.CONVERGED) == "Required accuracy for solution obtained"

    def test_user_stop(self) -> None:
        assert "User-triggered" in str(SlsqpStatus.USER_STOP)

    def test_unknown(self) -> None:
        assert str(SlsqpStatus.UNKNOWN) == "Unknown slsqp error"

    def test_in_progress_modes(self) -> None:
        assert str(SlsqpStatus.FUNC_EVAL_REQUIRED) == "In progress"
        assert str(SlsqpStatus.GRAD_EVAL_REQUIRED) == "In progress"


# ---------------------------------------------------------------------------
# Test: input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_xl_xu_size_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            SlsqpSolver(
                func=rosenbrock_func,
                xl=np.array([-1.0]),
                xu=np.array([-1.0, 1.0]),
                grad=rosenbrock_grad,
            )

    def test_negative_m(self) -> None:
        with pytest.raises(ValueError, match="m must be"):
            SlsqpSolver(
                func=rosenbrock_func,
                xl=np.array([-1.0, -1.0]),
                xu=np.array([1.0, 1.0]),
                m=-1,
                grad=rosenbrock_grad,
            )

    def test_meq_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="meq must be"):
            SlsqpSolver(
                func=rosenbrock_func,
                xl=np.array([-1.0, -1.0]),
                xu=np.array([1.0, 1.0]),
                m=1,
                meq=2,
                grad=rosenbrock_grad,
            )

    def test_bounds_violation(self) -> None:
        with pytest.raises(ValueError, match="Lower bounds"):
            SlsqpSolver(
                func=rosenbrock_func,
                xl=np.array([2.0, -1.0]),
                xu=np.array([1.0, 1.0]),
                grad=rosenbrock_grad,
            )

    def test_invalid_linesearch_mode(self) -> None:
        with pytest.raises(ValueError, match="linesearch_mode"):
            SlsqpSolver(
                func=rosenbrock_func,
                xl=np.array([-1.0, -1.0]),
                xu=np.array([1.0, 1.0]),
                linesearch_mode=3,  # type: ignore[arg-type]
                grad=rosenbrock_grad,
            )

    def test_invalid_alphamin_alphamax(self) -> None:
        with pytest.raises(ValueError, match="alphamin"):
            SlsqpSolver(
                func=rosenbrock_func,
                xl=np.array([-1.0, -1.0]),
                xu=np.array([1.0, 1.0]),
                alphamin=0.9,
                alphamax=0.1,
                grad=rosenbrock_grad,
            )

    def test_user_gradient_without_callback(self) -> None:
        with pytest.raises(ValueError, match="grad callback"):
            SlsqpSolver(
                func=rosenbrock_func,
                xl=np.array([-1.0, -1.0]),
                xu=np.array([1.0, 1.0]),
                gradient_mode=GradientMode.USER,
                # no grad= provided
            )

    def test_fd_with_tiny_delta(self) -> None:
        with pytest.raises(ValueError, match="gradient_delta"):
            SlsqpSolver(
                func=rosenbrock_func,
                xl=np.array([-1.0, -1.0]),
                xu=np.array([1.0, 1.0]),
                gradient_mode=GradientMode.FORWARD,
                gradient_delta=0.0,
            )

    def test_invalid_x_size(self) -> None:
        solver = SlsqpSolver(
            func=rosenbrock_func,
            xl=np.array([-1.0, -1.0]),
            xu=np.array([1.0, 1.0]),
            m=1,
            grad=rosenbrock_grad,
        )
        result = solver.optimize(np.array([0.1]))  # wrong size
        assert result.status == SlsqpStatus.INVALID_X_SIZE
        assert not result.success


# ---------------------------------------------------------------------------
# Test: Rosenbrock with analytic gradients (matches Fortran test)
# ---------------------------------------------------------------------------


class TestRosenbrock:
    def test_rosenbrock_user_gradient(self) -> None:
        """Minimise Rosenbrock subject to x0² + x1² ≤ 1, analytic gradients."""
        solver = SlsqpSolver(
            func=rosenbrock_func,
            xl=np.array([-1.0, -1.0]),
            xu=np.array([1.0, 1.0]),
            m=1,
            meq=0,
            max_iter=100,
            acc=1e-8,
            grad=rosenbrock_grad,
            iprint=0,
        )
        result = solver.optimize(np.array([0.1, 0.1]))

        assert result.success, (
            f"Expected convergence, got status={result.status}: {result.message}"
        )
        assert result.x[0] == pytest.approx(0.7864, abs=0.01)
        assert result.x[1] == pytest.approx(0.6177, abs=0.01)
        assert result.status == SlsqpStatus.CONVERGED
        assert result.iterations > 0

    def test_rosenbrock_forward_fd(self) -> None:
        """Same problem, forward finite-difference gradients."""
        solver = SlsqpSolver(
            func=rosenbrock_func,
            xl=np.array([-1.0, -1.0]),
            xu=np.array([1.0, 1.0]),
            m=1,
            meq=0,
            max_iter=200,
            acc=1e-6,
            gradient_mode=GradientMode.FORWARD,
            gradient_delta=1e-7,
            iprint=0,
        )
        result = solver.optimize(np.array([0.1, 0.1]))

        assert result.success, f"FD-forward failed: {result.message}"
        assert result.x[0] == pytest.approx(0.7864, abs=0.02)
        assert result.x[1] == pytest.approx(0.6177, abs=0.02)

    def test_rosenbrock_backward_fd(self) -> None:
        """Same problem, backward finite-difference gradients."""
        solver = SlsqpSolver(
            func=rosenbrock_func,
            xl=np.array([-1.0, -1.0]),
            xu=np.array([1.0, 1.0]),
            m=1,
            meq=0,
            max_iter=200,
            acc=1e-6,
            gradient_mode=GradientMode.BACKWARD,
            gradient_delta=1e-7,
            iprint=0,
        )
        result = solver.optimize(np.array([0.1, 0.1]))

        assert result.success, f"FD-backward failed: {result.message}"
        assert result.x[0] == pytest.approx(0.7864, abs=0.02)
        assert result.x[1] == pytest.approx(0.6177, abs=0.02)

    def test_rosenbrock_central_fd(self) -> None:
        """Same problem, central finite-difference gradients."""
        solver = SlsqpSolver(
            func=rosenbrock_func,
            xl=np.array([-1.0, -1.0]),
            xu=np.array([1.0, 1.0]),
            m=1,
            meq=0,
            max_iter=200,
            acc=1e-6,
            gradient_mode=GradientMode.CENTRAL,
            gradient_delta=1e-7,
            iprint=0,
        )
        result = solver.optimize(np.array([0.1, 0.1]))

        assert result.success, f"FD-central failed: {result.message}"
        assert result.x[0] == pytest.approx(0.7864, abs=0.02)
        assert result.x[1] == pytest.approx(0.6177, abs=0.02)

    def test_rosenbrock_exact_linesearch(self) -> None:
        """Rosenbrock with exact line-search mode."""
        solver = SlsqpSolver(
            func=rosenbrock_func,
            xl=np.array([-1.0, -1.0]),
            xu=np.array([1.0, 1.0]),
            m=1,
            meq=0,
            max_iter=200,
            acc=1e-8,
            grad=rosenbrock_grad,
            linesearch_mode=LinesearchMode.EXACT,
            iprint=0,
        )
        result = solver.optimize(np.array([0.1, 0.1]))

        assert result.success, f"Exact linesearch failed: {result.message}"
        assert result.x[0] == pytest.approx(0.7864, abs=0.01)
        assert result.x[1] == pytest.approx(0.6177, abs=0.01)


# ---------------------------------------------------------------------------
# Test: unconstrained optimisation
# ---------------------------------------------------------------------------


class TestUnconstrained:
    def test_quadratic(self) -> None:
        """Minimise f(x) = x0² + x1² (unconstrained)."""

        def func(x: np.ndarray) -> tuple[float, np.ndarray]:
            return float(np.dot(x, x)), np.empty(0)

        def grad(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return 2.0 * x, np.empty((0, len(x)))

        solver = SlsqpSolver(
            func=func,
            xl=np.array([-10.0, -10.0]),
            xu=np.array([10.0, 10.0]),
            m=0,
            meq=0,
            max_iter=100,
            acc=1e-10,
            grad=grad,
            iprint=0,
        )
        result = solver.optimize(np.array([5.0, 3.0]))

        assert result.success
        np.testing.assert_allclose(result.x, [0.0, 0.0], atol=1e-6)
        assert result.fun == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Test: equality constraint
# ---------------------------------------------------------------------------


class TestEqualityConstraint:
    def test_equality_on_circle(self) -> None:
        """Minimise x0 + x1  subject to  x0² + x1² = 1."""

        def func(x: np.ndarray) -> tuple[float, np.ndarray]:
            f = x[0] + x[1]
            c = np.array([x[0] ** 2 + x[1] ** 2 - 1.0])
            return f, c

        def grad(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            g = np.array([1.0, 1.0])
            a = np.array([[2.0 * x[0], 2.0 * x[1]]])
            return g, a

        solver = SlsqpSolver(
            func=func,
            xl=np.array([-2.0, -2.0]),
            xu=np.array([2.0, 2.0]),
            m=1,
            meq=1,
            max_iter=100,
            acc=1e-10,
            grad=grad,
            iprint=0,
        )
        result = solver.optimize(np.array([1.0, 0.0]))

        assert result.success, f"eq-constraint failed: {result.message}"
        # Optimal on the unit circle: x = (-1/√2, -1/√2)
        expected = -1.0 / np.sqrt(2.0)
        assert result.x[0] == pytest.approx(expected, abs=1e-5)
        assert result.x[1] == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# Test: callback
# ---------------------------------------------------------------------------


class TestCallback:
    def test_iteration_callback(self) -> None:
        """Verify that the iteration callback is invoked."""
        log: list[int] = []

        def on_iter(it: int, x: np.ndarray, f: float, c: np.ndarray) -> None:
            log.append(it)

        solver = SlsqpSolver(
            func=rosenbrock_func,
            xl=np.array([-1.0, -1.0]),
            xu=np.array([1.0, 1.0]),
            m=1,
            meq=0,
            max_iter=100,
            acc=1e-8,
            grad=rosenbrock_grad,
            callback=on_iter,
            iprint=0,
        )
        result = solver.optimize(np.array([0.1, 0.1]))

        assert result.success
        # Iteration 0 is the initial guess
        assert 0 in log
        assert len(log) >= 2  # at least initial + one iteration

    def test_msg_callback(self) -> None:
        """Verify that the message callback receives the convergence message."""
        messages: list[str] = []

        solver = SlsqpSolver(
            func=rosenbrock_func,
            xl=np.array([-1.0, -1.0]),
            xu=np.array([1.0, 1.0]),
            m=1,
            meq=0,
            max_iter=100,
            acc=1e-8,
            grad=rosenbrock_grad,
            msg_callback=messages.append,
            iprint=0,
        )
        solver.optimize(np.array([0.1, 0.1]))

        assert any("accuracy" in m.lower() for m in messages)


# ---------------------------------------------------------------------------
# Test: abort
# ---------------------------------------------------------------------------


class TestAbort:
    def test_user_triggered_stop(self) -> None:
        """Abort after 3 iterations."""
        call_count = 0

        def func_with_abort(x: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal call_count, solver
            call_count += 1
            if call_count >= 6:  # stop after a few evaluations
                solver.abort()
            return rosenbrock_func(x)

        solver = SlsqpSolver(
            func=func_with_abort,
            xl=np.array([-1.0, -1.0]),
            xu=np.array([1.0, 1.0]),
            m=1,
            meq=0,
            max_iter=100,
            acc=1e-12,
            grad=rosenbrock_grad,
            iprint=0,
        )
        result = solver.optimize(np.array([0.1, 0.1]))

        assert result.status == SlsqpStatus.USER_STOP
        assert not result.success
        assert "User-triggered" in result.message


# ---------------------------------------------------------------------------
# Test: BVLS mode
# ---------------------------------------------------------------------------


class TestNnlsMode:
    def test_bvls_mode(self) -> None:
        """Rosenbrock with BVLS NNLS solver."""
        solver = SlsqpSolver(
            func=rosenbrock_func,
            xl=np.array([-1.0, -1.0]),
            xu=np.array([1.0, 1.0]),
            m=1,
            meq=0,
            max_iter=100,
            acc=1e-8,
            grad=rosenbrock_grad,
            nnls_mode=NnlsMode.BVLS,
            iprint=0,
        )
        result = solver.optimize(np.array([0.1, 0.1]))

        assert result.success, f"BVLS mode failed: {result.message}"
        assert result.x[0] == pytest.approx(0.7864, abs=0.01)


# ---------------------------------------------------------------------------
# Test: result dataclass fields
# ---------------------------------------------------------------------------


class TestResultFields:
    def test_result_has_all_fields(self) -> None:
        solver = SlsqpSolver(
            func=rosenbrock_func,
            xl=np.array([-1.0, -1.0]),
            xu=np.array([1.0, 1.0]),
            m=1,
            meq=0,
            max_iter=100,
            acc=1e-8,
            grad=rosenbrock_grad,
            iprint=0,
        )
        result = solver.optimize(np.array([0.1, 0.1]))

        assert isinstance(result, SlsqpResult)
        assert isinstance(result.x, np.ndarray)
        assert isinstance(result.fun, float)
        assert isinstance(result.constraints, np.ndarray)
        assert isinstance(result.status, SlsqpStatus)
        assert isinstance(result.message, str)
        assert isinstance(result.iterations, int)
        assert isinstance(result.success, bool)


# ---------------------------------------------------------------------------
# Test: NaN bounds = unbounded
# ---------------------------------------------------------------------------


class TestNanBounds:
    def test_nan_lower_bound(self) -> None:
        """NaN in xl should mean no lower bound."""

        def func(x: np.ndarray) -> tuple[float, np.ndarray]:
            return float((x[0] - 3.0) ** 2), np.empty(0)

        def grad(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return np.array([2.0 * (x[0] - 3.0)]), np.empty((0, 1))

        solver = SlsqpSolver(
            func=func,
            xl=np.array([np.nan]),
            xu=np.array([10.0]),
            m=0,
            max_iter=100,
            acc=1e-10,
            grad=grad,
            iprint=0,
        )
        result = solver.optimize(np.array([0.0]))

        assert result.success
        assert result.x[0] == pytest.approx(3.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Ported Fortran tests: slsqp_test_2.f90
# ---------------------------------------------------------------------------


class TestFortranTest2:
    """Port of slsqp_test_2.f90.

    Minimise  f = x0² + x1² + x2
    subject to:
        c0 = x0·x1 − x2 = 0       (equality)
        c1 = x2 − 1     ≥ 0       (inequality)
    bounds: −10 ≤ xi ≤ 10
    x0 = [1, 2, 3]
    Expected solution: x = [1, 1, 1], f = 3
    """

    @staticmethod
    def _func(x: np.ndarray) -> tuple[float, np.ndarray]:
        f = x[0] ** 2 + x[1] ** 2 + x[2]
        c = np.array([
            x[0] * x[1] - x[2],     # equality
            x[2] - 1.0,             # inequality
        ])
        return f, c

    @staticmethod
    def _grad(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        g = np.array([2.0 * x[0], 2.0 * x[1], 1.0])
        a = np.array([
            [x[1], x[0], -1.0],     # ∂c0/∂x
            [0.0, 0.0, 1.0],        # ∂c1/∂x
        ])
        return g, a

    def test_constrained_3var(self) -> None:
        solver = SlsqpSolver(
            func=self._func,
            xl=np.full(3, -10.0),
            xu=np.full(3, 10.0),
            m=2,
            meq=1,
            max_iter=100,
            acc=1e-7,
            grad=self._grad,
            linesearch_mode=LinesearchMode.INEXACT,
            alphamin=0.1,
            alphamax=0.5,
            iprint=0,
        )
        result = solver.optimize(np.array([1.0, 2.0, 3.0]))

        assert result.success, (
            f"slsqp_test_2 failed: status={result.status}, msg={result.message}"
        )
        np.testing.assert_allclose(result.x, [1.0, 1.0, 1.0], atol=1e-4)
        assert result.fun == pytest.approx(3.0, abs=1e-4)

    def test_constraints_satisfied(self) -> None:
        """Verify both constraints hold at the solution."""
        solver = SlsqpSolver(
            func=self._func,
            xl=np.full(3, -10.0),
            xu=np.full(3, 10.0),
            m=2,
            meq=1,
            max_iter=100,
            acc=1e-7,
            grad=self._grad,
            alphamin=0.1,
            alphamax=0.5,
            iprint=0,
        )
        result = solver.optimize(np.array([1.0, 2.0, 3.0]))
        assert result.success
        # c0 = x0*x1 - x2 == 0 (equality)
        assert result.constraints[0] == pytest.approx(0.0, abs=1e-5)
        # c1 = x2 - 1 >= 0 (inequality)
        assert result.constraints[1] >= -1e-5


# ---------------------------------------------------------------------------
# Ported Fortran tests: slsqp_test_3.f90
# ---------------------------------------------------------------------------


class TestFortranTest3:
    """Port of slsqp_test_3.f90.

    Rosenbrock with circle constraint, FD gradients (modes 1-3),
    gradient_delta=1e-5, message callback, iprint=0.
    """

    @pytest.mark.parametrize("gmode", [
        GradientMode.BACKWARD,
        GradientMode.FORWARD,
        GradientMode.CENTRAL,
    ])
    def test_rosenbrock_fd_modes(self, gmode: GradientMode) -> None:
        messages: list[str] = []

        solver = SlsqpSolver(
            func=rosenbrock_func,
            xl=np.array([-1.0, -1.0]),
            xu=np.array([1.0, 1.0]),
            m=1,
            meq=0,
            max_iter=100,
            acc=1e-8,
            gradient_mode=gmode,
            gradient_delta=1e-5,
            linesearch_mode=LinesearchMode.INEXACT,
            iprint=0,
            msg_callback=messages.append,
        )
        result = solver.optimize(np.array([0.1, 0.1]))

        assert result.success, (
            f"FD mode {gmode.name} failed: status={result.status}, msg={result.message}"
        )
        assert result.x[0] == pytest.approx(0.7864, abs=0.01)
        assert result.x[1] == pytest.approx(0.6177, abs=0.01)
        assert result.fun == pytest.approx(0.04567, abs=0.01)
        # message callback should have been invoked
        assert len(messages) > 0


# ---------------------------------------------------------------------------
# Ported Fortran tests: slsqp_test_71.f90 (Hock-Schittkowsky TP71)
# ---------------------------------------------------------------------------


class TestFortranTestTP71:
    """Port of slsqp_test_71.f90 — Hock-Schittkowsky problem 71.

    min   x0·x3·(x0 + x1 + x2) + x2
    s.t.  x0·x1·x2·x3 − x4 − 25 = 0      (equality)
          x0² + x1² + x2² + x3² − 40 = 0  (equality)
          1 ≤ x0,x1,x2,x3 ≤ 5
          0 ≤ x4
    x0 = [1, 5, 5, 1, −24]
    Optimal ≈ [1.0, 4.743, 3.821, 1.379, 0.0]
    """

    @staticmethod
    def _func(x: np.ndarray) -> tuple[float, np.ndarray]:
        f = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
        c = np.array([
            x[0] * x[1] * x[2] * x[3] - x[4] - 25.0,
            x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 - 40.0,
        ])
        return f, c

    @staticmethod
    def _grad(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        g = np.array([
            x[3] * (2.0 * x[0] + x[1] + x[2]),
            x[0] * x[3],
            x[0] * x[3] + 1.0,
            x[0] * (x[0] + x[1] + x[2]),
            0.0,
        ])
        a = np.zeros((2, 5))
        a[0, 0] = x[1] * x[2] * x[3]
        a[0, 1] = x[0] * x[2] * x[3]
        a[0, 2] = x[0] * x[1] * x[3]
        a[0, 3] = x[0] * x[1] * x[2]
        a[0, 4] = -1.0
        a[1, 0] = 2.0 * x[0]
        a[1, 1] = 2.0 * x[1]
        a[1, 2] = 2.0 * x[2]
        a[1, 3] = 2.0 * x[3]
        return g, a

    @pytest.mark.parametrize("nnls", [NnlsMode.NNLS, NnlsMode.BVLS])
    def test_tp71(self, nnls: NnlsMode) -> None:
        solver = SlsqpSolver(
            func=self._func,
            xl=np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
            xu=np.array([5.0, 5.0, 5.0, 5.0, 1e10]),
            m=2,
            meq=2,
            max_iter=100,
            acc=1e-8,
            grad=self._grad,
            linesearch_mode=LinesearchMode.INEXACT,
            nnls_mode=nnls,
            iprint=0,
        )
        result = solver.optimize(np.array([1.0, 5.0, 5.0, 1.0, -24.0]))

        assert result.success, (
            f"TP71 (nnls={nnls.name}) failed: status={result.status}, "
            f"msg={result.message}"
        )
        # Check against known optimal solution
        assert result.x[0] == pytest.approx(1.0, abs=0.01)
        assert result.x[1] == pytest.approx(4.743, abs=0.01)
        assert result.x[2] == pytest.approx(3.821, abs=0.01)
        assert result.x[3] == pytest.approx(1.379, abs=0.01)
        # x4 should be ~ 0 (slack variable absorbed by first constraint)
        assert result.x[4] == pytest.approx(0.0, abs=0.1)

    def test_tp71_constraints_satisfied(self) -> None:
        """Verify equality constraints hold at the solution."""
        solver = SlsqpSolver(
            func=self._func,
            xl=np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
            xu=np.array([5.0, 5.0, 5.0, 5.0, 1e10]),
            m=2,
            meq=2,
            max_iter=100,
            acc=1e-8,
            grad=self._grad,
            iprint=0,
        )
        result = solver.optimize(np.array([1.0, 5.0, 5.0, 1.0, -24.0]))
        assert result.success
        # Both constraints should be ≈ 0 (equality)
        np.testing.assert_allclose(result.constraints, [0.0, 0.0], atol=1e-4)


# ---------------------------------------------------------------------------
# Ported Fortran tests: slsqp_test_stopping_criterion.f90
# ---------------------------------------------------------------------------


class TestFortranStoppingCriterion:
    """Port of slsqp_test_stopping_criterion.f90.

    Rosenbrock with circle *equality* constraint (meq=1),
    NaN bounds (xl=[-1, NaN], xu=[NaN, 1]),
    tolf=0, toldf=0, toldx=0.
    Solution: x ≈ [0.7864, 0.6177]
    """

    @staticmethod
    def _func(x: np.ndarray) -> tuple[float, np.ndarray]:
        f = 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2
        c = np.array([1.0 - x[0] ** 2 - x[1] ** 2])
        return f, c

    @staticmethod
    def _grad(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        g = np.array([
            -400.0 * (x[1] - x[0] ** 2) * x[0] - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0] ** 2),
        ])
        a = np.array([[-2.0 * x[0], -2.0 * x[1]]])
        return g, a

    def test_stopping_with_nan_bounds(self) -> None:
        """Rosenbrock on unit circle with NaN bounds and extra stopping criteria."""
        solver = SlsqpSolver(
            func=self._func,
            xl=np.array([-1.0, np.nan]),
            xu=np.array([np.nan, 1.0]),
            m=1,
            meq=1,
            max_iter=100,
            acc=1e-8,
            grad=self._grad,
            linesearch_mode=LinesearchMode.INEXACT,
            tolf=0.0,
            toldf=0.0,
            toldx=0.0,
            iprint=0,
        )
        result = solver.optimize(np.array([0.1, 0.1]))

        assert result.success, (
            f"Stopping criterion test failed: status={result.status}, "
            f"msg={result.message}"
        )
        # Fortran expected values (within 1e-4 as in the Fortran test)
        assert result.x[0] == pytest.approx(0.78641515097183889, abs=1e-4)
        assert result.x[1] == pytest.approx(0.61769831659541152, abs=1e-4)

    def test_equality_constraint_satisfied(self) -> None:
        """The equality constraint x0² + x1² = 1 must hold."""
        solver = SlsqpSolver(
            func=self._func,
            xl=np.array([-1.0, np.nan]),
            xu=np.array([np.nan, 1.0]),
            m=1,
            meq=1,
            max_iter=100,
            acc=1e-8,
            grad=self._grad,
            tolf=0.0,
            toldf=0.0,
            toldx=0.0,
            iprint=0,
        )
        result = solver.optimize(np.array([0.1, 0.1]))
        assert result.success
        # c = 1 - x0² - x1² should be ≈ 0
        assert result.constraints[0] == pytest.approx(0.0, abs=1e-6)
        # Equivalently x0² + x1² ≈ 1
        assert result.x[0] ** 2 + result.x[1] ** 2 == pytest.approx(1.0, abs=1e-6)

    def test_objective_value(self) -> None:
        """Verify the objective function value at the solution."""
        solver = SlsqpSolver(
            func=self._func,
            xl=np.array([-1.0, np.nan]),
            xu=np.array([np.nan, 1.0]),
            m=1,
            meq=1,
            max_iter=100,
            acc=1e-8,
            grad=self._grad,
            tolf=0.0,
            toldf=0.0,
            toldx=0.0,
            iprint=0,
        )
        result = solver.optimize(np.array([0.1, 0.1]))
        assert result.success
        # Fortran expected: f ≈ 4.5674808719160388E-002
        assert result.fun == pytest.approx(0.04567, abs=1e-3)


# ---------------------------------------------------------------------------
# Workspace reuse across optimize() calls
# ---------------------------------------------------------------------------


class TestWorkspaceReuse:
    """Verify that calling optimize() multiple times on the same solver
    produces correct results — the cached workspace is properly reset."""

    def test_repeated_optimize_same_x0(self):
        """Two optimize() calls from the same starting point give identical results."""
        solver = SlsqpSolver(
            func=rosenbrock_func,
            xl=np.array([-10.0, -10.0]),
            xu=np.array([10.0, 10.0]),
            m=1,
            meq=0,
            max_iter=200,
            acc=1e-10,
            grad=rosenbrock_grad,
            iprint=0,
        )
        r1 = solver.optimize(np.array([-1.0, 1.0]))
        r2 = solver.optimize(np.array([-1.0, 1.0]))

        assert r1.success
        assert r2.success
        np.testing.assert_allclose(r1.x, r2.x, atol=1e-10)
        assert r1.fun == pytest.approx(r2.fun, abs=1e-12)

    def test_repeated_optimize_different_x0(self):
        """Workspace reuse does not contaminate across different starting points."""

        def quadratic(x):
            f = (x[0] - 3.0) ** 2 + (x[1] + 1.0) ** 2
            return f, np.empty(0)

        def quadratic_grad(x):
            g = np.array([2.0 * (x[0] - 3.0), 2.0 * (x[1] + 1.0)])
            a = np.empty((0, 2))
            return g, a

        solver = SlsqpSolver(
            func=quadratic,
            xl=np.array([-10.0, -10.0]),
            xu=np.array([10.0, 10.0]),
            max_iter=100,
            acc=1e-12,
            grad=quadratic_grad,
            iprint=0,
        )

        # First call from [0, 0]
        r1 = solver.optimize(np.array([0.0, 0.0]))
        assert r1.success
        np.testing.assert_allclose(r1.x, [3.0, -1.0], atol=1e-6)

        # Second call from [10, 10] — different starting point
        r2 = solver.optimize(np.array([10.0, 10.0]))
        assert r2.success
        np.testing.assert_allclose(r2.x, [3.0, -1.0], atol=1e-6)

        # Third call — yet another starting point
        r3 = solver.optimize(np.array([-5.0, 5.0]))
        assert r3.success
        np.testing.assert_allclose(r3.x, [3.0, -1.0], atol=1e-6)

    def test_workspace_reuse_with_constraints(self):
        """Workspace reuse works correctly for constrained problems."""
        solver = SlsqpSolver(
            func=rosenbrock_func,
            xl=np.array([-10.0, -10.0]),
            xu=np.array([10.0, 10.0]),
            m=1,
            meq=0,
            max_iter=200,
            acc=1e-8,
            grad=rosenbrock_grad,
            iprint=0,
        )

        results = [solver.optimize(np.array([-1.0, 1.0])) for _ in range(5)]
        for r in results:
            assert r.success
            np.testing.assert_allclose(r.x, results[0].x, atol=1e-10)

    def test_workspace_reuse_rust_loop(self):
        """Workspace reuse works with the Rust-loop fast path."""
        solver = SlsqpSolver(
            func=rosenbrock_func,
            xl=np.array([-10.0, -10.0]),
            xu=np.array([10.0, 10.0]),
            m=1,
            meq=0,
            max_iter=200,
            acc=1e-10,
            grad=rosenbrock_grad,
            iprint=0,
        )
        solver._use_rust_loop = True

        r1 = solver.optimize(np.array([-1.0, 1.0]))
        r2 = solver.optimize(np.array([-1.0, 1.0]))

        assert r1.success
        assert r2.success
        np.testing.assert_allclose(r1.x, r2.x, atol=1e-10)
        assert r1.fun == pytest.approx(r2.fun, abs=1e-12)


# ---------------------------------------------------------------------------
# Tests for Rust-accelerated compute_fd_gradients
# ---------------------------------------------------------------------------

from rslsqp._core import compute_fd_gradients


class TestRustFdGradients:
    """Tests for the Rust-accelerated finite-difference gradient computation.

    Validates that ``compute_fd_gradients`` produces correct gradients
    for all three FD modes and matches what the old Python loop used to
    compute.
    """

    @staticmethod
    def _quadratic_func(x: np.ndarray) -> tuple[float, np.ndarray]:
        """f(x) = x0^2 + 2*x1^2, no constraints."""
        f = x[0] ** 2 + 2.0 * x[1] ** 2
        return f, np.array([], dtype=float)

    @staticmethod
    def _constrained_func(x: np.ndarray) -> tuple[float, np.ndarray]:
        """f(x) = x0^2 + x1^2, c0 = x0 + x1 - 1, c1 = x0 - x1."""
        f = x[0] ** 2 + x[1] ** 2
        c = np.array([x[0] + x[1] - 1.0, x[0] - x[1]])
        return f, c

    @pytest.mark.parametrize("gmode", [
        GradientMode.FORWARD,
        GradientMode.BACKWARD,
        GradientMode.CENTRAL,
    ])
    def test_unconstrained_gradient(self, gmode: GradientMode) -> None:
        """FD gradient of a simple quadratic matches analytic gradient."""
        x = np.array([3.0, -2.0])
        g, a = compute_fd_gradients(self._quadratic_func, x, gmode, 1e-7, 0)

        # Analytic: df/dx0 = 2*x0 = 6, df/dx1 = 4*x1 = -8
        np.testing.assert_allclose(g, [6.0, -8.0], atol=1e-5)
        assert a.shape == (0, 2)

    @pytest.mark.parametrize("gmode", [
        GradientMode.FORWARD,
        GradientMode.BACKWARD,
        GradientMode.CENTRAL,
    ])
    def test_constrained_jacobian(self, gmode: GradientMode) -> None:
        """FD Jacobian of linear constraints is exact to FD precision."""
        x = np.array([1.5, 0.5])
        g, a = compute_fd_gradients(self._constrained_func, x, gmode, 1e-7, 2)

        # Analytic: df/dx0 = 2*x0 = 3, df/dx1 = 2*x1 = 1
        np.testing.assert_allclose(g, [3.0, 1.0], atol=1e-5)
        # Jacobian of [x0+x1-1, x0-x1] is [[1,1],[1,-1]]
        assert a.shape == (2, 2)
        np.testing.assert_allclose(a, [[1.0, 1.0], [1.0, -1.0]], atol=1e-5)

    def test_central_more_accurate_than_forward(self) -> None:
        """Central differences should be more accurate than forward."""
        x = np.array([1.0, 1.0])
        delta = 1e-5

        g_fwd, _ = compute_fd_gradients(
            self._quadratic_func, x, GradientMode.FORWARD, delta, 0,
        )
        g_cen, _ = compute_fd_gradients(
            self._quadratic_func, x, GradientMode.CENTRAL, delta, 0,
        )

        exact = np.array([2.0, 4.0])
        err_fwd = np.max(np.abs(g_fwd - exact))
        err_cen = np.max(np.abs(g_cen - exact))
        assert err_cen < err_fwd

    def test_x_not_mutated(self) -> None:
        """The input x array must not be mutated by the FD computation."""
        x = np.array([3.0, -2.0])
        x_orig = x.copy()
        compute_fd_gradients(self._quadratic_func, x, GradientMode.FORWARD, 1e-7, 0)
        np.testing.assert_array_equal(x, x_orig)

    def test_user_mode_raises(self) -> None:
        """GradientMode.USER should raise ValueError."""
        x = np.array([1.0, 1.0])
        with pytest.raises(ValueError, match="USER"):
            compute_fd_gradients(self._quadratic_func, x, GradientMode.USER, 1e-7, 0)

    def test_solver_uses_rust_fd(self) -> None:
        """SlsqpSolver with FD mode should converge using the Rust FD path.

        This is an integration test verifying the full pipeline through
        _compute_gradients → compute_fd_gradients → solver convergence.
        """
        solver = SlsqpSolver(
            func=rosenbrock_func,
            xl=np.array([-5.0, -5.0]),
            xu=np.array([5.0, 5.0]),
            m=1,
            meq=0,
            max_iter=200,
            acc=1e-8,
            gradient_mode=GradientMode.CENTRAL,
            gradient_delta=1e-6,
            iprint=0,
        )
        result = solver.optimize(np.array([0.1, 0.1]))
        assert result.success
        assert result.x[0] == pytest.approx(0.7864, abs=0.01)
        assert result.x[1] == pytest.approx(0.6177, abs=0.01)
