"""Classical Hock-Schittkowski (HS) optimisation benchmark problems.

References:
    Hock, W. & Schittkowski, K. (1981).
    "Test Examples for Nonlinear Programming Codes".
    Lecture Notes in Economics and Mathematical Systems, Vol. 187, Springer.

Covers: bounds-only, equality, inequality, mixed, 2-5 variables.
Uses the scipy-compatible ``minimize()`` from :mod:`rslsqp.scipy_compat`.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from rslsqp.scipy_compat import minimize

_OPTS = {"maxiter": 500}


# ── HS01 ─ Rosenbrock, x2 >= -1.5 ──────────────────────────────────────


class TestHS01:
    """x*=(1,1), f*=0"""

    def test_solve(self) -> None:
        result = minimize(
            lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
            [-2.0, 1.0],
            jac=lambda x: np.array(
                [
                    -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
                    200 * (x[1] - x[0] ** 2),
                ]
            ),
            bounds=[(None, None), (-1.5, None)],
            options=_OPTS,
        )
        assert result.success, f"HS01 failed: {result.message}"
        np.testing.assert_allclose(result.x, [1, 1], atol=1e-4)
        assert result.fun == pytest.approx(0, abs=1e-6)


# ── HS02 ─ Rosenbrock, x2 >= 1.5 (bound active) ───────────────────────


class TestHS02:
    """x*~(1.2243,1.5), f*~0.0504"""

    def test_solve(self) -> None:
        result = minimize(
            lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
            [-2.0, 1.0],
            jac=lambda x: np.array(
                [
                    -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
                    200 * (x[1] - x[0] ** 2),
                ]
            ),
            bounds=[(None, None), (1.5, None)],
            options=_OPTS,
        )
        assert result.success, f"HS02 failed: {result.message}"
        assert result.x[0] == pytest.approx(1.2243, abs=0.01)
        assert result.x[1] == pytest.approx(1.5, abs=1e-4)
        assert result.fun == pytest.approx(0.0504, abs=0.01)


# ── HS04 ─ cubic + bounds ──────────────────────────────────────────────


class TestHS04:
    """x*=(1,0), f*=8/3"""

    def test_solve(self) -> None:
        result = minimize(
            lambda x: (x[0] + 1) ** 3 / 3 + x[1],
            [1.125, 0.125],
            jac=lambda x: np.array([(x[0] + 1) ** 2, 1.0]),
            bounds=[(1, None), (0, None)],
            options=_OPTS,
        )
        assert result.success, f"HS04 failed: {result.message}"
        np.testing.assert_allclose(result.x, [1, 0], atol=1e-4)
        assert result.fun == pytest.approx(8 / 3, abs=1e-4)


# ── HS05 ─ trigonometric, bounded ──────────────────────────────────────


class TestHS05:
    """x*=(-pi/3+0.5, -pi/3-0.5), f*=-(sqrt3/2+pi/3)"""

    def test_solve(self) -> None:
        def fun(x):
            return (
                math.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1
            )

        def jac(x):
            c = math.cos(x[0] + x[1])
            return np.array([c + 2 * (x[0] - x[1]) - 1.5, c - 2 * (x[0] - x[1]) + 2.5])

        result = minimize(
            fun, [0, 0], jac=jac, bounds=[(-1.5, 4), (-3, 3)], options=_OPTS
        )
        assert result.success, f"HS05 failed: {result.message}"
        x_s = [-math.pi / 3 + 0.5, -math.pi / 3 - 0.5]
        np.testing.assert_allclose(result.x, x_s, atol=1e-3)
        assert result.fun == pytest.approx(-(math.sqrt(3) / 2 + math.pi / 3), abs=1e-4)


# ── HS06 ─ equality constraint ─────────────────────────────────────────


class TestHS06:
    """min (1-x1)^2  s.t. 10(x2-x1^2)=0 => x*=(1,1), f*=0"""

    def test_solve(self) -> None:
        result = minimize(
            lambda x: (1 - x[0]) ** 2,
            [-1.2, 1.0],
            jac=lambda x: np.array([-2 * (1 - x[0]), 0.0]),
            constraints={
                "type": "eq",
                "fun": lambda x: 10 * (x[1] - x[0] ** 2),
                "jac": lambda x: np.array([-20 * x[0], 10.0]),
            },
            options=_OPTS,
        )
        assert result.success, f"HS06 failed: {result.message}"
        np.testing.assert_allclose(result.x, [1, 1], atol=1e-3)
        assert result.fun == pytest.approx(0, abs=1e-5)


# ── HS07 ─ equality, logarithmic ───────────────────────────────────────


class TestHS07:
    """min ln(1+x1^2)-x2  s.t. (1+x1^2)^2+x2^2=4 => x*=(0,sqrt3)"""

    def test_solve(self) -> None:
        result = minimize(
            lambda x: math.log(1 + x[0] ** 2) - x[1],
            [2, 2],
            jac=lambda x: np.array([2 * x[0] / (1 + x[0] ** 2), -1.0]),
            constraints={
                "type": "eq",
                "fun": lambda x: (1 + x[0] ** 2) ** 2 + x[1] ** 2 - 4,
                "jac": lambda x: np.array([4 * x[0] * (1 + x[0] ** 2), 2 * x[1]]),
            },
            options=_OPTS,
        )
        assert result.success, f"HS07 failed: {result.message}"
        np.testing.assert_allclose(result.x, [0, math.sqrt(3)], atol=1e-3)
        assert result.fun == pytest.approx(-math.sqrt(3), abs=1e-4)


# ── HS10 ─ inequality constraint ───────────────────────────────────────


class TestHS10:
    """min x1-x2  s.t. -3x1^2+2x1x2-x2^2+1>=0 => x*=(0,1), f*=-1"""

    def test_solve(self) -> None:
        result = minimize(
            lambda x: x[0] - x[1],
            [-10, 10],
            jac=lambda x: np.array([1.0, -1.0]),
            bounds=[(-100, 100), (-100, 100)],
            constraints={
                "type": "ineq",
                "fun": lambda x: -3 * x[0] ** 2 + 2 * x[0] * x[1] - x[1] ** 2 + 1,
                "jac": lambda x: np.array([-6 * x[0] + 2 * x[1], 2 * x[0] - 2 * x[1]]),
            },
            options=_OPTS,
        )
        assert result.success, f"HS10 failed: {result.message}"
        np.testing.assert_allclose(result.x, [0, 1], atol=1e-3)
        assert result.fun == pytest.approx(-1, abs=1e-4)


# ── HS14 ─ equality + inequality ───────────────────────────────────────


class TestHS14:
    """x*~(0.8229,0.9114), f*~1.3935"""

    def test_solve(self) -> None:
        result = minimize(
            lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2,
            [2, 2],
            jac=lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] - 1)]),
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda x: x[0] - 2 * x[1] + 1,
                    "jac": lambda x: np.array([1.0, -2.0]),
                },
                {
                    "type": "ineq",
                    "fun": lambda x: -0.25 * x[0] ** 2 - x[1] ** 2 + 1,
                    "jac": lambda x: np.array([-0.5 * x[0], -2 * x[1]]),
                },
            ],
            options=_OPTS,
        )
        assert result.success, f"HS14 failed: {result.message}"
        assert result.x[0] == pytest.approx(0.8229, abs=0.01)
        assert result.x[1] == pytest.approx(0.9114, abs=0.01)
        assert result.fun == pytest.approx(1.3935, abs=0.01)


# ── HS21 ─ linear inequality + bounds ──────────────────────────────────


class TestHS21:
    """x*=(2,0), f*=-99.96"""

    def test_solve(self) -> None:
        result = minimize(
            lambda x: 0.01 * x[0] ** 2 + x[1] - 100,
            [20, 5],
            jac=lambda x: np.array([0.02 * x[0], 1.0]),
            bounds=[(2, 50), (0, 50)],
            constraints={
                "type": "ineq",
                "fun": lambda x: 10 * x[0] - x[1] - 10,
                "jac": lambda x: np.array([10.0, -1.0]),
            },
            options=_OPTS,
        )
        assert result.success, f"HS21 failed: {result.message}"
        np.testing.assert_allclose(result.x, [2, 0], atol=1e-3)
        assert result.fun == pytest.approx(-99.96, abs=1e-3)


# ── HS26 ─ equality, 3 vars ────────────────────────────────────────────


class TestHS26:
    """x*=(1,1,1), f*=0"""

    def test_solve(self) -> None:
        result = minimize(
            lambda x: (x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 4,
            [-2.6, 2, 2],
            jac=lambda x: np.array(
                [
                    2 * (x[0] - x[1]),
                    -2 * (x[0] - x[1]) + 4 * (x[1] - x[2]) ** 3,
                    -4 * (x[1] - x[2]) ** 3,
                ]
            ),
            constraints={
                "type": "eq",
                "fun": lambda x: (1 + x[1] ** 2) * x[0] + x[2] ** 4 - 3,
                "jac": lambda x: np.array(
                    [1 + x[1] ** 2, 2 * x[0] * x[1], 4 * x[2] ** 3]
                ),
            },
            options=_OPTS,
        )
        assert result.success, f"HS26 failed: {result.message}"
        np.testing.assert_allclose(result.x, [1, 1, 1], atol=1e-3)
        assert result.fun == pytest.approx(0, abs=1e-5)


# ── HS28 ─ equality, 3 vars (quadratic) ────────────────────────────────


class TestHS28:
    """x*=(0.5,-0.5,0.5), f*=0"""

    def test_solve(self) -> None:
        result = minimize(
            lambda x: (x[0] + x[1]) ** 2 + (x[1] + x[2]) ** 2,
            [-4, 1, 1],
            jac=lambda x: np.array(
                [
                    2 * (x[0] + x[1]),
                    2 * (x[0] + x[1]) + 2 * (x[1] + x[2]),
                    2 * (x[1] + x[2]),
                ]
            ),
            constraints={
                "type": "eq",
                "fun": lambda x: x[0] + 2 * x[1] + 3 * x[2] - 1,
                "jac": lambda x: np.array([1, 2, 3.0]),
            },
            options=_OPTS,
        )
        assert result.success, f"HS28 failed: {result.message}"
        np.testing.assert_allclose(result.x, [0.5, -0.5, 0.5], atol=1e-3)
        assert result.fun == pytest.approx(0, abs=1e-5)


# ── HS35 ─ inequality + bounds, 3 vars ─────────────────────────────────


class TestHS35:
    """x*=(4/3, 7/9, 4/9), f*=1/9"""

    def test_solve(self) -> None:
        def fun(x):
            return (
                9
                - 8 * x[0]
                - 6 * x[1]
                - 4 * x[2]
                + 2 * x[0] ** 2
                + 2 * x[1] ** 2
                + x[2] ** 2
                + 2 * x[0] * x[1]
                + 2 * x[0] * x[2]
            )

        def jac(x):
            return np.array(
                [
                    -8 + 4 * x[0] + 2 * x[1] + 2 * x[2],
                    -6 + 2 * x[0] + 4 * x[1],
                    -4 + 2 * x[2] + 2 * x[0],
                ]
            )

        result = minimize(
            fun,
            [0.5, 0.5, 0.5],
            jac=jac,
            bounds=[(0, None)] * 3,
            constraints={
                "type": "ineq",
                "fun": lambda x: 3 - x[0] - x[1] - 2 * x[2],
                "jac": lambda x: np.array([-1, -1, -2.0]),
            },
            options=_OPTS,
        )
        assert result.success, f"HS35 failed: {result.message}"
        np.testing.assert_allclose(result.x, [4 / 3, 7 / 9, 4 / 9], atol=1e-3)
        assert result.fun == pytest.approx(1 / 9, abs=1e-3)


# ── HS36 ─ bounds + inequality, 3 vars ─────────────────────────────────


class TestHS36:
    """x*=(20,11,15), f*=-3300"""

    def test_solve(self) -> None:
        result = minimize(
            lambda x: -x[0] * x[1] * x[2],
            [10, 10, 10],
            jac=lambda x: np.array([-x[1] * x[2], -x[0] * x[2], -x[0] * x[1]]),
            bounds=[(0, 20), (0, 11), (0, 42)],
            constraints={
                "type": "ineq",
                "fun": lambda x: 72 - x[0] - x[1] - x[2],
                "jac": lambda x: np.array([-1, -1, -1.0]),
            },
            options=_OPTS,
        )
        assert result.success, f"HS36 failed: {result.message}"
        assert result.x[0] == pytest.approx(20, abs=0.1)
        assert result.x[1] == pytest.approx(11, abs=0.1)
        # x3 = 72 - 20 - 11 = 41 (constraint active)
        assert result.x[2] == pytest.approx(41, abs=0.1)
        assert result.fun == pytest.approx(-9020, abs=1)


# ── HS40 ─ 3 equalities, 4 vars ────────────────────────────────────────


class TestHS40:
    """x*~(0.7937,0.7071,0.5297,0.8409), f*=-0.25"""

    def test_solve(self) -> None:
        result = minimize(
            lambda x: -x[0] * x[1] * x[2] * x[3],
            [0.8] * 4,
            jac=lambda x: np.array(
                [
                    -x[1] * x[2] * x[3],
                    -x[0] * x[2] * x[3],
                    -x[0] * x[1] * x[3],
                    -x[0] * x[1] * x[2],
                ]
            ),
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda x: x[0] ** 3 + x[1] ** 2 - 1,
                    "jac": lambda x: np.array([3 * x[0] ** 2, 2 * x[1], 0, 0.0]),
                },
                {
                    "type": "eq",
                    "fun": lambda x: x[0] ** 2 * x[3] - x[2],
                    "jac": lambda x: np.array([2 * x[0] * x[3], 0, -1, x[0] ** 2]),
                },
                {
                    "type": "eq",
                    "fun": lambda x: x[3] ** 2 - x[1],
                    "jac": lambda x: np.array([0, -1, 0, 2 * x[3]]),
                },
            ],
            options=_OPTS,
        )
        assert result.success, f"HS40 failed: {result.message}"
        assert result.x[0] == pytest.approx(0.7937, abs=0.01)
        assert result.x[1] == pytest.approx(0.7071, abs=0.01)
        assert result.x[2] == pytest.approx(0.5297, abs=0.01)
        assert result.x[3] == pytest.approx(0.8409, abs=0.01)
        assert result.fun == pytest.approx(-0.25, abs=1e-3)


# ── HS43 ─ 3 inequalities, 4 vars ──────────────────────────────────────


class TestHS43:
    """x*=(0,1,2,-1), f*=-44"""

    @staticmethod
    def _fun(x):
        return (
            x[0] ** 2
            + x[1] ** 2
            + 2 * x[2] ** 2
            + x[3] ** 2
            - 5 * x[0]
            - 5 * x[1]
            - 21 * x[2]
            + 7 * x[3]
        )

    @staticmethod
    def _jac(x):
        return np.array([2 * x[0] - 5, 2 * x[1] - 5, 4 * x[2] - 21, 2 * x[3] + 7.0])

    def test_solve(self) -> None:
        cons = [
            {
                "type": "ineq",
                "fun": lambda x: (
                    8
                    - x[0] ** 2
                    - x[1] ** 2
                    - x[2] ** 2
                    - x[3] ** 2
                    - x[0]
                    + x[1]
                    - x[2]
                    + x[3]
                ),
                "jac": lambda x: np.array(
                    [-2 * x[0] - 1, -2 * x[1] + 1, -2 * x[2] - 1, -2 * x[3] + 1.0]
                ),
            },
            {
                "type": "ineq",
                "fun": lambda x: (
                    10
                    - x[0] ** 2
                    - 2 * x[1] ** 2
                    - x[2] ** 2
                    - 2 * x[3] ** 2
                    + x[0]
                    + x[3]
                ),
                "jac": lambda x: np.array(
                    [-2 * x[0] + 1, -4 * x[1], -2 * x[2], -4 * x[3] + 1.0]
                ),
            },
            {
                "type": "ineq",
                "fun": lambda x: (
                    5 - 2 * x[0] ** 2 - x[1] ** 2 - x[2] ** 2 - 2 * x[0] + x[1] + x[3]
                ),
                "jac": lambda x: np.array(
                    [-4 * x[0] - 2, -2 * x[1] + 1, -2 * x[2], 1.0]
                ),
            },
        ]
        result = minimize(
            self._fun,
            [0] * 4,
            jac=self._jac,
            bounds=[(-10, 10)] * 4,
            constraints=cons,
            options=_OPTS,
        )
        assert result.success, f"HS43 failed: {result.message}"
        np.testing.assert_allclose(result.x, [0, 1, 2, -1], atol=0.05)
        assert result.fun == pytest.approx(-44, abs=0.1)


# ── HS45 ─ bounds only, 5 vars ─────────────────────────────────────────


class TestHS45:
    """x*=(1,2,3,4,5), f*=1"""

    def test_solve(self) -> None:
        def jac(x):
            p = x[0] * x[1] * x[2] * x[3] * x[4]
            g = np.zeros(5)
            for i in range(5):
                if x[i] != 0:
                    g[i] = -p / (120 * x[i])
            return g

        result = minimize(
            lambda x: 2 - x[0] * x[1] * x[2] * x[3] * x[4] / 120,
            [0.5, 1, 1.5, 2, 2.5],
            jac=jac,
            bounds=[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
            options=_OPTS,
        )
        assert result.success, f"HS45 failed: {result.message}"
        np.testing.assert_allclose(result.x, [1, 2, 3, 4, 5], atol=0.1)
        assert result.fun == pytest.approx(1, abs=0.01)


# ── HS48 ─ 2 equalities, 5 vars ────────────────────────────────────────


class TestHS48:
    """x*=(1,1,1,1,1), f*=0"""

    def test_solve(self) -> None:
        result = minimize(
            lambda x: (x[0] - 1) ** 2 + (x[1] - x[2]) ** 2 + (x[3] - x[4]) ** 2,
            [3, 5, -3, 2, -2],
            jac=lambda x: np.array(
                [
                    2 * (x[0] - 1),
                    2 * (x[1] - x[2]),
                    -2 * (x[1] - x[2]),
                    2 * (x[3] - x[4]),
                    -2 * (x[3] - x[4]),
                ]
            ),
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda x: x[0] + x[1] + x[2] + x[3] + x[4] - 5,
                    "jac": lambda x: np.array([1, 1, 1, 1, 1.0]),
                },
                {
                    "type": "eq",
                    "fun": lambda x: x[2] - 2 * (x[3] + x[4]) + 3,
                    "jac": lambda x: np.array([0, 0, 1, -2, -2.0]),
                },
            ],
            options=_OPTS,
        )
        assert result.success, f"HS48 failed: {result.message}"
        np.testing.assert_allclose(result.x, [1] * 5, atol=1e-3)
        assert result.fun == pytest.approx(0, abs=1e-5)


# ── HS49 ─ 2 equalities, 5 vars ────────────────────────────────────────


class TestHS49:
    """x*=(1,1,1,1,1), f*=0"""

    def test_solve(self) -> None:
        # Start closer to solution; high-order (4th/6th) terms have flat gradients
        result = minimize(
            lambda x: (
                (x[0] - x[1]) ** 2 + (x[2] - 1) ** 2 + (x[3] - 1) ** 4 + (x[4] - 1) ** 6
            ),
            [2, 2, 2, 0.5, 0.8],
            jac=lambda x: np.array(
                [
                    2 * (x[0] - x[1]),
                    -2 * (x[0] - x[1]),
                    2 * (x[2] - 1),
                    4 * (x[3] - 1) ** 3,
                    6 * (x[4] - 1) ** 5,
                ]
            ),
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda x: x[0] + x[1] + x[2] + 4 * x[3] - 7,
                    "jac": lambda x: np.array([1, 1, 1, 4, 0.0]),
                },
                {
                    "type": "eq",
                    "fun": lambda x: x[2] + 5 * x[4] - 6,
                    "jac": lambda x: np.array([0, 0, 1, 0, 5.0]),
                },
            ],
            options=_OPTS,
        )
        assert result.success, f"HS49 failed: {result.message}"
        np.testing.assert_allclose(result.x, [1] * 5, atol=0.15)
        assert result.fun < 0.01


# ── HS50 ─ 3 equalities, 5 vars ────────────────────────────────────────


class TestHS50:
    """x*=(1,1,1,1,1), f*=0"""

    def test_solve(self) -> None:
        result = minimize(
            lambda x: (
                (x[0] - x[1]) ** 2
                + (x[1] - x[2]) ** 2
                + (x[2] - x[3]) ** 4
                + (x[3] - x[4]) ** 2
            ),
            [35, -31, 11, 5, -5],
            jac=lambda x: np.array(
                [
                    2 * (x[0] - x[1]),
                    -2 * (x[0] - x[1]) + 2 * (x[1] - x[2]),
                    -2 * (x[1] - x[2]) + 4 * (x[2] - x[3]) ** 3,
                    -4 * (x[2] - x[3]) ** 3 + 2 * (x[3] - x[4]),
                    -2 * (x[3] - x[4]),
                ]
            ),
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda x: x[0] + 2 * x[1] + 3 * x[2] - 6,
                    "jac": lambda x: np.array([1, 2, 3, 0, 0.0]),
                },
                {
                    "type": "eq",
                    "fun": lambda x: x[1] + 2 * x[2] + 3 * x[3] - 6,
                    "jac": lambda x: np.array([0, 1, 2, 3, 0.0]),
                },
                {
                    "type": "eq",
                    "fun": lambda x: x[2] + 2 * x[3] + 3 * x[4] - 6,
                    "jac": lambda x: np.array([0, 0, 1, 2, 3.0]),
                },
            ],
            options=_OPTS,
        )
        assert result.success, f"HS50 failed: {result.message}"
        np.testing.assert_allclose(result.x, [1] * 5, atol=1e-3)
        assert result.fun == pytest.approx(0, abs=1e-4)


# ── HS51 ─ 3 equalities, 5 vars ────────────────────────────────────────


class TestHS51:
    """x*=(1,1,1,1,1), f*=0"""

    def test_solve(self) -> None:
        result = minimize(
            lambda x: (
                (x[0] - x[1]) ** 2
                + (x[1] + x[2] - 2) ** 2
                + (x[3] - 1) ** 2
                + (x[4] - 1) ** 2
            ),
            [2.5, 0.5, 2, -1, 0.5],
            jac=lambda x: np.array(
                [
                    2 * (x[0] - x[1]),
                    -2 * (x[0] - x[1]) + 2 * (x[1] + x[2] - 2),
                    2 * (x[1] + x[2] - 2),
                    2 * (x[3] - 1),
                    2 * (x[4] - 1.0),
                ]
            ),
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda x: x[0] + 3 * x[1] - 4,
                    "jac": lambda x: np.array([1, 3, 0, 0, 0.0]),
                },
                {
                    "type": "eq",
                    "fun": lambda x: x[2] + x[3] - 2 * x[4],
                    "jac": lambda x: np.array([0, 0, 1, 1, -2.0]),
                },
                {
                    "type": "eq",
                    "fun": lambda x: x[1] - x[4],
                    "jac": lambda x: np.array([0, 1, 0, 0, -1.0]),
                },
            ],
            options=_OPTS,
        )
        assert result.success, f"HS51 failed: {result.message}"
        np.testing.assert_allclose(result.x, [1] * 5, atol=1e-3)
        assert result.fun == pytest.approx(0, abs=1e-5)


# ── HS52 ─ 3 equalities, 5 vars ────────────────────────────────────────


class TestHS52:
    """f*=5.3266"""

    def test_solve(self) -> None:
        result = minimize(
            lambda x: (
                (4 * x[0] - x[1]) ** 2
                + (x[1] + x[2] - 2) ** 2
                + (x[3] - 1) ** 2
                + (x[4] - 1) ** 2
            ),
            [2] * 5,
            jac=lambda x: np.array(
                [
                    8 * (4 * x[0] - x[1]),
                    -2 * (4 * x[0] - x[1]) + 2 * (x[1] + x[2] - 2),
                    2 * (x[1] + x[2] - 2),
                    2 * (x[3] - 1),
                    2 * (x[4] - 1.0),
                ]
            ),
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda x: x[0] + 3 * x[1],
                    "jac": lambda x: np.array([1, 3, 0, 0, 0.0]),
                },
                {
                    "type": "eq",
                    "fun": lambda x: x[2] + x[3] - 2 * x[4],
                    "jac": lambda x: np.array([0, 0, 1, 1, -2.0]),
                },
                {
                    "type": "eq",
                    "fun": lambda x: x[1] - x[4],
                    "jac": lambda x: np.array([0, 1, 0, 0, -1.0]),
                },
            ],
            options=_OPTS,
        )
        assert result.success, f"HS52 failed: {result.message}"
        assert result.fun == pytest.approx(5.3266, abs=0.01)


# ── HS71 (TP71) ─ 2 equalities + bounds, 5 vars ───────────────────────


class TestHS71:
    """x*~(1,4.743,3.821,1.379,0), f*~17.014"""

    def test_solve(self) -> None:
        result = minimize(
            lambda x: x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2],
            [1, 5, 5, 1, -24],
            jac=lambda x: np.array(
                [
                    x[3] * (2 * x[0] + x[1] + x[2]),
                    x[0] * x[3],
                    x[0] * x[3] + 1,
                    x[0] * (x[0] + x[1] + x[2]),
                    0.0,
                ]
            ),
            bounds=[(1, 5), (1, 5), (1, 5), (1, 5), (0, 1e10)],
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda x: x[0] * x[1] * x[2] * x[3] - x[4] - 25,
                    "jac": lambda x: np.array(
                        [
                            x[1] * x[2] * x[3],
                            x[0] * x[2] * x[3],
                            x[0] * x[1] * x[3],
                            x[0] * x[1] * x[2],
                            -1.0,
                        ]
                    ),
                },
                {
                    "type": "eq",
                    "fun": lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 - 40,
                    "jac": lambda x: np.array(
                        [2 * x[0], 2 * x[1], 2 * x[2], 2 * x[3], 0.0]
                    ),
                },
            ],
            options=_OPTS,
        )
        assert result.success, f"HS71 failed: {result.message}"
        assert result.x[0] == pytest.approx(1.0, abs=0.01)
        assert result.x[1] == pytest.approx(4.743, abs=0.01)
        assert result.x[2] == pytest.approx(3.821, abs=0.01)
        assert result.x[3] == pytest.approx(1.379, abs=0.01)
        assert result.fun == pytest.approx(17.014, abs=0.1)


# ── HS76 ─ 3 inequalities + bounds, 4 vars ─────────────────────────────


class TestHS76:
    """f*~-4.6818"""

    def test_solve(self) -> None:
        def fun(x):
            return (
                x[0] ** 2
                + 0.5 * x[1] ** 2
                + x[2] ** 2
                + 0.5 * x[3] ** 2
                - x[0] * x[2]
                + x[2] * x[3]
                - x[0]
                - 3 * x[1]
                + x[2]
                - x[3]
            )

        def jac(x):
            return np.array(
                [
                    2 * x[0] - x[2] - 1,
                    x[1] - 3,
                    2 * x[2] - x[0] + x[3] + 1,
                    x[3] + x[2] - 1.0,
                ]
            )

        cons = [
            {
                "type": "ineq",
                "fun": lambda x: 5 - x[0] - x[1] - x[2] - x[3],
                "jac": lambda x: np.array([-1, -1, -1, -1.0]),
            },
            {
                "type": "ineq",
                "fun": lambda x: 10 - 3 * x[1] + x[2] - x[3],
                "jac": lambda x: np.array([0, -3, 1, -1.0]),
            },
            {
                "type": "ineq",
                "fun": lambda x: x[2] - x[3],
                "jac": lambda x: np.array([0, 0, 1, -1.0]),
            },
        ]
        result = minimize(
            fun,
            [0.5] * 4,
            jac=jac,
            bounds=[(0, None)] * 4,
            constraints=cons,
            options=_OPTS,
        )
        assert result.success, f"HS76 failed: {result.message}"
        assert result.fun == pytest.approx(-4.6818, abs=0.1)


# ── HS100 ─ 4 inequalities, 7 vars ─────────────────────────────────────


class TestHS100:
    """f*~680.63"""

    @staticmethod
    def _fun(x):
        return (
            (x[0] - 10) ** 2
            + 5 * (x[1] - 12) ** 2
            + x[2] ** 4
            + 3 * (x[3] - 11) ** 2
            + 10 * x[4] ** 6
            + 7 * x[5] ** 2
            + x[6] ** 4
            - 4 * x[5] * x[6]
            - 10 * x[5]
            - 8 * x[6]
        )

    @staticmethod
    def _jac(x):
        return np.array(
            [
                2 * (x[0] - 10),
                10 * (x[1] - 12),
                4 * x[2] ** 3,
                6 * (x[3] - 11),
                60 * x[4] ** 5,
                14 * x[5] - 4 * x[6] - 10,
                4 * x[6] ** 3 - 4 * x[5] - 8,
            ]
        )

    def test_solve(self) -> None:
        cons = [
            {
                "type": "ineq",
                "fun": lambda x: (
                    127
                    - 2 * x[0] ** 2
                    - 3 * x[1] ** 4
                    - x[2]
                    - 4 * x[3] ** 2
                    - 5 * x[4]
                ),
                "jac": lambda x: np.array(
                    [-4 * x[0], -12 * x[1] ** 3, -1, -8 * x[3], -5, 0, 0.0]
                ),
            },
            {
                "type": "ineq",
                "fun": lambda x: (
                    282 - 7 * x[0] - 3 * x[1] - 10 * x[2] ** 2 - x[3] + x[4]
                ),
                "jac": lambda x: np.array([-7, -3, -20 * x[2], -1, 1, 0, 0.0]),
            },
            {
                "type": "ineq",
                "fun": lambda x: 196 - 23 * x[0] - x[1] ** 2 - 6 * x[5] ** 2 + 8 * x[6],
                "jac": lambda x: np.array([-23, -2 * x[1], 0, 0, 0, -12 * x[5], 8.0]),
            },
            {
                "type": "ineq",
                "fun": lambda x: (
                    -4 * x[0] ** 2
                    - x[1] ** 2
                    + 3 * x[0] * x[1]
                    - 2 * x[2] ** 2
                    - 5 * x[5]
                    + 11 * x[6]
                ),
                "jac": lambda x: np.array(
                    [
                        -8 * x[0] + 3 * x[1],
                        -2 * x[1] + 3 * x[0],
                        -4 * x[2],
                        0,
                        0,
                        -5,
                        11.0,
                    ]
                ),
            },
        ]
        result = minimize(
            self._fun,
            [1, 2, 0, 4, 0, 1, 1],
            jac=self._jac,
            bounds=[(-10, 10)] * 7,
            constraints=cons,
            options=_OPTS,
        )
        assert result.success, f"HS100 failed: {result.message}"
        assert result.fun == pytest.approx(680.63, abs=1.0)


# ── Additional classic: Beale function (unconstrained, 2 vars) ──────────


class TestBeale:
    """Beale function: x*=(3, 0.5), f*=0"""

    def test_solve(self) -> None:
        def fun(x):
            return (
                (1.5 - x[0] + x[0] * x[1]) ** 2
                + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
                + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
            )

        def jac(x):
            t1 = 1.5 - x[0] + x[0] * x[1]
            t2 = 2.25 - x[0] + x[0] * x[1] ** 2
            t3 = 2.625 - x[0] + x[0] * x[1] ** 3
            df0 = (
                2 * t1 * (-1 + x[1])
                + 2 * t2 * (-1 + x[1] ** 2)
                + 2 * t3 * (-1 + x[1] ** 3)
            )
            df1 = (
                2 * t1 * x[0] + 2 * t2 * 2 * x[0] * x[1] + 2 * t3 * 3 * x[0] * x[1] ** 2
            )
            return np.array([df0, df1])

        result = minimize(fun, [1, 1], jac=jac, bounds=[(-4.5, 4.5)] * 2, options=_OPTS)
        assert result.success, f"Beale failed: {result.message}"
        np.testing.assert_allclose(result.x, [3, 0.5], atol=1e-3)
        assert result.fun == pytest.approx(0, abs=1e-6)


# ── Booth function (unconstrained, 2 vars) ──────────────────────────────


class TestBooth:
    """Booth function: x*=(1, 3), f*=0"""

    def test_solve(self) -> None:
        result = minimize(
            lambda x: (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2,
            [0, 0],
            jac=lambda x: np.array(
                [
                    2 * (x[0] + 2 * x[1] - 7) + 4 * (2 * x[0] + x[1] - 5),
                    4 * (x[0] + 2 * x[1] - 7) + 2 * (2 * x[0] + x[1] - 5),
                ]
            ),
            bounds=[(-10, 10)] * 2,
            options=_OPTS,
        )
        assert result.success, f"Booth failed: {result.message}"
        np.testing.assert_allclose(result.x, [1, 3], atol=1e-4)
        assert result.fun == pytest.approx(0, abs=1e-6)


# ── Matyas function (unconstrained, 2 vars) ─────────────────────────────


class TestMatyas:
    """Matyas function: x*=(0, 0), f*=0"""

    def test_solve(self) -> None:
        result = minimize(
            lambda x: 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1],
            [5, -3],
            jac=lambda x: np.array(
                [0.52 * x[0] - 0.48 * x[1], 0.52 * x[1] - 0.48 * x[0]]
            ),
            bounds=[(-10, 10)] * 2,
            options=_OPTS,
        )
        assert result.success, f"Matyas failed: {result.message}"
        np.testing.assert_allclose(result.x, [0, 0], atol=1e-4)
        assert result.fun == pytest.approx(0, abs=1e-8)
