"""Tests for the scipy-compatible minimize() interface."""

from __future__ import annotations

import numpy as np
import pytest

from rslsqp.scipy_compat import minimize, OptimizeResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function: f(x) = 100*(x1 - x0²)² + (1 - x0)²."""
    return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of the Rosenbrock function."""
    return np.array([
        -400.0 * (x[1] - x[0] ** 2) * x[0] - 2.0 * (1.0 - x[0]),
        200.0 * (x[1] - x[0] ** 2),
    ])


def rosenbrock_with_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Rosenbrock returning (f, grad) in a single call."""
    return rosenbrock(x), rosenbrock_grad(x)


# ---------------------------------------------------------------------------
# Test: OptimizeResult
# ---------------------------------------------------------------------------


class TestOptimizeResult:
    def test_dict_access(self) -> None:
        r = OptimizeResult(x=np.array([1.0, 2.0]), fun=3.0, success=True)
        assert r["x"] is r.x
        assert r["fun"] == 3.0
        assert "success" in r
        assert "nonexistent" not in r

    def test_get_method(self) -> None:
        r = OptimizeResult()
        assert r.get("fun", 42.0) == 0.0  # default fun is 0.0
        assert r.get("nonexistent", 42) == 42

    def test_keys(self) -> None:
        r = OptimizeResult()
        k = r.keys()
        assert "x" in k
        assert "fun" in k
        assert "success" in k


# ---------------------------------------------------------------------------
# Test: unconstrained minimisation
# ---------------------------------------------------------------------------


class TestUnconstrained:
    def test_rosenbrock_no_jac(self) -> None:
        """Unconstrained Rosenbrock with finite-difference gradient."""
        result = minimize(rosenbrock, [0.5, 0.5], options={"maxiter": 200})
        assert result.success, f"Failed: {result.message}"
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-3)
        assert result.fun < 1e-5
        assert result.nfev > 0
        assert result.nit > 0

    def test_rosenbrock_with_jac_callable(self) -> None:
        """Unconstrained Rosenbrock with analytic gradient."""
        result = minimize(rosenbrock, [0.5, 0.5], jac=rosenbrock_grad,
                          options={"maxiter": 200})
        assert result.success, f"Failed: {result.message}"
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-3)
        assert result.njev > 0

    def test_rosenbrock_jac_true(self) -> None:
        """Unconstrained Rosenbrock with jac=True (fun returns (f, g))."""
        result = minimize(rosenbrock_with_grad, [0.5, 0.5], jac=True,
                          options={"maxiter": 200})
        assert result.success, f"Failed: {result.message}"
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-3)

    def test_simple_quadratic(self) -> None:
        """min x0² + x1² → solution at origin."""
        def fun(x):
            return x[0] ** 2 + x[1] ** 2

        result = minimize(fun, [5.0, 3.0])
        assert result.success
        np.testing.assert_allclose(result.x, [0.0, 0.0], atol=1e-4)
        assert result.fun < 1e-8


# ---------------------------------------------------------------------------
# Test: with bounds
# ---------------------------------------------------------------------------


class TestBounds:
    def test_bounds_as_tuples(self) -> None:
        """Bounds as list of (lo, hi) tuples."""
        def fun(x):
            return (x[0] - 3.0) ** 2 + (x[1] - 4.0) ** 2

        result = minimize(fun, [0.0, 0.0], bounds=[(0, 2), (0, 3)])
        assert result.success
        # Solution is clipped to the bound corner (2, 3)
        assert result.x[0] == pytest.approx(2.0, abs=1e-4)
        assert result.x[1] == pytest.approx(3.0, abs=1e-4)

    def test_bounds_none_entry(self) -> None:
        """None in bounds means unbounded."""
        def fun(x):
            return (x[0] - 5.0) ** 2

        result = minimize(fun, [0.0], bounds=[(None, None)])
        assert result.success
        assert result.x[0] == pytest.approx(5.0, abs=1e-3)

    def test_bounds_none_means_unbounded(self) -> None:
        """bounds=None → fully unbounded."""
        def fun(x):
            return (x[0] - 2.0) ** 2

        result = minimize(fun, [0.0], bounds=None)
        assert result.success
        assert result.x[0] == pytest.approx(2.0, abs=1e-3)

    def test_inf_bounds(self) -> None:
        """±inf in bounds means unbounded (scipy convention)."""
        def fun(x):
            return (x[0] - 3.0) ** 2

        result = minimize(fun, [0.0], bounds=[(-np.inf, np.inf)])
        assert result.success
        assert result.x[0] == pytest.approx(3.0, abs=1e-3)

    def test_bounds_object_with_lb_ub(self) -> None:
        """Accept objects with .lb/.ub attributes (Bounds-like)."""

        class FakeBounds:
            def __init__(self, lb, ub):
                self.lb = np.asarray(lb, dtype=float)
                self.ub = np.asarray(ub, dtype=float)

        def fun(x):
            return (x[0] - 5.0) ** 2 + (x[1] + 1.0) ** 2

        result = minimize(fun, [0.0, 0.0], bounds=FakeBounds([0, -10], [3, 10]))
        assert result.success
        assert result.x[0] == pytest.approx(3.0, abs=1e-3)
        assert result.x[1] == pytest.approx(-1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Test: inequality constraints
# ---------------------------------------------------------------------------


class TestInequalityConstraints:
    def test_single_inequality(self) -> None:
        """min x0 + x1 s.t. x0 + x1 >= 1 → solution on the boundary."""
        def fun(x):
            return x[0] + x[1]

        constraints = {"type": "ineq", "fun": lambda x: x[0] + x[1] - 1.0}
        result = minimize(fun, [0.5, 0.5], constraints=constraints,
                          bounds=[(-10, 10), (-10, 10)])
        assert result.success
        # The sum x0 + x1 should be ≈ 1 at the optimum
        assert (result.x[0] + result.x[1]) == pytest.approx(1.0, abs=1e-4)

    def test_rosenbrock_circle_constraint(self) -> None:
        """Rosenbrock s.t. x0² + x1² ≤ 1 (i.e. 1 - x0² - x1² ≥ 0)."""
        constraints = {
            "type": "ineq",
            "fun": lambda x: 1.0 - x[0] ** 2 - x[1] ** 2,
            "jac": lambda x: np.array([-2.0 * x[0], -2.0 * x[1]]),
        }
        result = minimize(rosenbrock, [0.1, 0.1],
                          jac=rosenbrock_grad,
                          bounds=[(-1, 1), (-1, 1)],
                          constraints=constraints)
        assert result.success, f"Failed: {result.message}"
        assert result.x[0] == pytest.approx(0.7864, abs=0.01)
        assert result.x[1] == pytest.approx(0.6177, abs=0.01)


# ---------------------------------------------------------------------------
# Test: equality constraints
# ---------------------------------------------------------------------------


class TestEqualityConstraints:
    def test_minimize_on_circle(self) -> None:
        """min x0 + x1 s.t. x0² + x1² = 1 → optimal at (-1/√2, -1/√2)."""
        def fun(x):
            return x[0] + x[1]

        def fun_jac(x):
            return np.array([1.0, 1.0])

        constraints = {
            "type": "eq",
            "fun": lambda x: x[0] ** 2 + x[1] ** 2 - 1.0,
            "jac": lambda x: np.array([2.0 * x[0], 2.0 * x[1]]),
        }
        result = minimize(fun, [1.0, 0.0], jac=fun_jac,
                          bounds=[(-2, 2), (-2, 2)],
                          constraints=constraints)
        assert result.success, f"Failed: {result.message}"
        expected = -1.0 / np.sqrt(2.0)
        assert result.x[0] == pytest.approx(expected, abs=1e-4)
        assert result.x[1] == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# Test: mixed equality + inequality
# ---------------------------------------------------------------------------


class TestMixedConstraints:
    def test_eq_and_ineq(self) -> None:
        """min x0² + x1² + x2 s.t. x0·x1 - x2 = 0, x2 ≥ 1."""
        def fun(x):
            return x[0] ** 2 + x[1] ** 2 + x[2]

        def fun_jac(x):
            return np.array([2.0 * x[0], 2.0 * x[1], 1.0])

        constraints = [
            {
                "type": "eq",
                "fun": lambda x: x[0] * x[1] - x[2],
                "jac": lambda x: np.array([x[1], x[0], -1.0]),
            },
            {
                "type": "ineq",
                "fun": lambda x: x[2] - 1.0,
                "jac": lambda x: np.array([0.0, 0.0, 1.0]),
            },
        ]
        result = minimize(fun, [1.0, 2.0, 3.0], jac=fun_jac,
                          bounds=[(-10, 10)] * 3,
                          constraints=constraints)
        assert result.success, f"Failed: {result.message}"
        np.testing.assert_allclose(result.x, [1.0, 1.0, 1.0], atol=1e-3)
        assert result.fun == pytest.approx(3.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Test: constraint with extra args
# ---------------------------------------------------------------------------


class TestConstraintArgs:
    def test_constraint_with_args(self) -> None:
        """Constraint function receives extra args."""
        def fun(x):
            return x[0] ** 2 + x[1] ** 2

        def con_with_arg(x, radius):
            return radius ** 2 - x[0] ** 2 - x[1] ** 2

        constraints = {
            "type": "ineq",
            "fun": con_with_arg,
            "args": (1.0,),
        }
        result = minimize(fun, [0.5, 0.5], constraints=constraints,
                          bounds=[(-2, 2), (-2, 2)])
        assert result.success
        # min is at origin, which is inside the circle
        np.testing.assert_allclose(result.x, [0.0, 0.0], atol=1e-3)

    def test_fun_with_args(self) -> None:
        """Objective function receives extra args."""
        def fun(x, a, b):
            return a * (x[0] - b) ** 2

        result = minimize(fun, [0.0], args=(2.0, 3.0))
        assert result.success
        assert result.x[0] == pytest.approx(3.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Test: callback
# ---------------------------------------------------------------------------


class TestCallback:
    def test_callback_is_called(self) -> None:
        """The callback should be called at least once."""
        calls: list[np.ndarray] = []

        def cb(x: np.ndarray) -> None:
            calls.append(x.copy())

        result = minimize(rosenbrock, [0.5, 0.5], callback=cb)
        assert result.success
        assert len(calls) > 0


# ---------------------------------------------------------------------------
# Test: tolerance / options
# ---------------------------------------------------------------------------


class TestOptions:
    def test_tol_overrides_ftol(self) -> None:
        """The tol= argument should override options['ftol']."""
        result = minimize(rosenbrock, [0.5, 0.5], tol=1e-4,
                          options={"maxiter": 200})
        assert result.success

    def test_disp_option(self, capsys: pytest.CaptureFixture[str]) -> None:
        """disp=True should print a message."""
        minimize(rosenbrock, [0.5, 0.5], options={"disp": True, "maxiter": 200})
        captured = capsys.readouterr()
        # Should have printed the convergence message
        assert len(captured.out) > 0

    def test_maxiter_option(self) -> None:
        """maxiter=1 should lead to non-convergence."""
        result = minimize(rosenbrock, [0.5, 0.5], options={"maxiter": 1})
        # With only 1 iteration, Rosenbrock likely won't converge
        # (but the solver should not crash)
        assert isinstance(result, OptimizeResult)


# ---------------------------------------------------------------------------
# Test: finite-difference mode strings
# ---------------------------------------------------------------------------


class TestJacStrings:
    def test_2point(self) -> None:
        result = minimize(rosenbrock, [0.5, 0.5], jac="2-point",
                          options={"maxiter": 200})
        assert result.success, f"Failed: {result.message}"
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-2)

    def test_3point(self) -> None:
        result = minimize(rosenbrock, [0.5, 0.5], jac="3-point",
                          options={"maxiter": 200})
        assert result.success, f"Failed: {result.message}"
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-2)

    def test_cs(self) -> None:
        """'cs' falls back to central differences."""
        result = minimize(rosenbrock, [0.5, 0.5], jac="cs",
                          options={"maxiter": 200})
        assert result.success, f"Failed: {result.message}"
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-2)


# ---------------------------------------------------------------------------
# Test: Hock-Schittkowsky TP71 (matches existing test suite)
# ---------------------------------------------------------------------------


class TestTP71:
    """HS problem 71 via the scipy-compatible interface."""

    @staticmethod
    def _fun(x):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    @staticmethod
    def _jac(x):
        return np.array([
            x[3] * (2.0 * x[0] + x[1] + x[2]),
            x[0] * x[3],
            x[0] * x[3] + 1.0,
            x[0] * (x[0] + x[1] + x[2]),
            0.0,
        ])

    def test_tp71(self) -> None:
        constraints = [
            {
                "type": "eq",
                "fun": lambda x: x[0] * x[1] * x[2] * x[3] - x[4] - 25.0,
                "jac": lambda x: np.array([
                    x[1] * x[2] * x[3],
                    x[0] * x[2] * x[3],
                    x[0] * x[1] * x[3],
                    x[0] * x[1] * x[2],
                    -1.0,
                ]),
            },
            {
                "type": "eq",
                "fun": lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 - 40.0,
                "jac": lambda x: np.array([
                    2.0 * x[0], 2.0 * x[1], 2.0 * x[2], 2.0 * x[3], 0.0,
                ]),
            },
        ]
        bounds = [(1, 5), (1, 5), (1, 5), (1, 5), (0, 1e10)]
        x0 = [1.0, 5.0, 5.0, 1.0, -24.0]

        result = minimize(self._fun, x0, jac=self._jac,
                          bounds=bounds, constraints=constraints)

        assert result.success, f"TP71 failed: {result.message}"
        assert result.x[0] == pytest.approx(1.0, abs=0.01)
        assert result.x[1] == pytest.approx(4.743, abs=0.01)
        assert result.x[2] == pytest.approx(3.821, abs=0.01)
        assert result.x[3] == pytest.approx(1.379, abs=0.01)


# ---------------------------------------------------------------------------
# Test: invalid constraint type
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unknown_constraint_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown constraint type"):
            minimize(rosenbrock, [0.5, 0.5],
                     constraints={"type": "bad", "fun": lambda x: 0})

    def test_invalid_jac(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised jac"):
            minimize(rosenbrock, [0.5, 0.5], jac="invalid_method")


# ---------------------------------------------------------------------------
# Test: result attributes match scipy convention
# ---------------------------------------------------------------------------


class TestResultFormat:
    def test_result_has_scipy_fields(self) -> None:
        result = minimize(rosenbrock, [0.5, 0.5])
        # All scipy OptimizeResult standard fields
        assert hasattr(result, "x")
        assert hasattr(result, "fun")
        assert hasattr(result, "jac")
        assert hasattr(result, "nit")
        assert hasattr(result, "nfev")
        assert hasattr(result, "njev")
        assert hasattr(result, "status")
        assert hasattr(result, "success")
        assert hasattr(result, "message")

    def test_x_is_ndarray(self) -> None:
        result = minimize(rosenbrock, [0.5, 0.5])
        assert isinstance(result.x, np.ndarray)

    def test_nfev_positive(self) -> None:
        result = minimize(rosenbrock, [0.5, 0.5])
        assert result.nfev > 0

    def test_status_zero_on_success(self) -> None:
        result = minimize(rosenbrock, [0.5, 0.5], options={"maxiter": 200})
        if result.success:
            assert result.status == 0
