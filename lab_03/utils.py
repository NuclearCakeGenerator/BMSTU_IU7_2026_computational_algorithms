from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Callable, Sequence

import numpy as np


class PolynomialDegree(IntEnum):
    LINEAR = 1
    QUADRATIC = 2


@dataclass(slots=True)
class Dataset:
    dots: list[tuple[float, ...]]
    weights: list[float]

    def __init__(
        self, dots: Sequence[tuple[float, ...]], weights: Sequence[float] | None = None
    ) -> None:
        if not dots:
            raise ValueError("Dataset cannot be empty")

        first_len = len(dots[0])
        if first_len not in (2, 3):
            raise ValueError("Each dot must be (x, y) or (x, y, z)")

        self.dots = []
        for dot in dots:
            if len(dot) != first_len:
                raise ValueError("All dots must have the same dimension")
            self.dots.append(tuple(float(v) for v in dot))

        if weights is None:
            self.weights = [1.0] * len(self.dots)
        else:
            if len(weights) != len(self.dots):
                raise ValueError("Weights length must match number of dots")
            self.weights = [float(w) for w in weights]


def _design_row_1d(x: float, degree: PolynomialDegree) -> list[float]:
    if degree == PolynomialDegree.LINEAR:
        return [1.0, x]
    return [1.0, x, x * x]


def _design_row_2d(x: float, y: float, degree: PolynomialDegree) -> list[float]:
    if degree == PolynomialDegree.LINEAR:
        return [1.0, x, y]
    return [1.0, x, y, x * x, x * y, y * y]


def approximate_polynomial(
    dataset: Dataset, degree: PolynomialDegree
) -> tuple[float, ...]:
    degree = PolynomialDegree(degree)

    point_dim = len(dataset.dots[0])
    if point_dim not in (2, 3):
        raise ValueError("Only (x, y) and (x, y, z) datasets are supported")

    matrix_rows: list[list[float]] = []
    values: list[float] = []

    for dot in dataset.dots:
        if point_dim == 2:
            x, y = dot
            matrix_rows.append(_design_row_1d(x, degree))
            values.append(y)
        else:
            x, y, z = dot
            matrix_rows.append(_design_row_2d(x, y, degree))
            values.append(z)

    matrix = np.array(matrix_rows, dtype=float)
    rhs = np.array(values, dtype=float)
    weights = np.array(dataset.weights, dtype=float)

    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative")

    weight_matrix = np.diag(weights)
    normal_matrix = matrix.T @ weight_matrix @ matrix
    normal_rhs = matrix.T @ weight_matrix @ rhs

    try:
        factors = np.linalg.solve(normal_matrix, normal_rhs)
    except np.linalg.LinAlgError:
        factors = np.linalg.pinv(normal_matrix) @ normal_rhs

    return tuple(float(f) for f in factors)


def polynomial(arguments: tuple[float, ...], factors: tuple[float, ...]) -> float:
    if len(arguments) == 1:
        x = float(arguments[0])
        if len(factors) == 2:
            return factors[0] + factors[1] * x
        if len(factors) == 3:
            return factors[0] + factors[1] * x + factors[2] * x * x
        raise ValueError("Invalid number of factors for 1D polynomial")

    if len(arguments) == 2:
        x, y = float(arguments[0]), float(arguments[1])
        if len(factors) == 3:
            return factors[0] + factors[1] * x + factors[2] * y
        if len(factors) == 6:
            return (
                factors[0]
                + factors[1] * x
                + factors[2] * y
                + factors[3] * x * x
                + factors[4] * x * y
                + factors[5] * y * y
            )
        raise ValueError("Invalid number of factors for 2D polynomial")

    raise ValueError("Arguments must be (x,) or (x, y)")


def root_mean_square_error(
    points: Sequence[tuple[float, float]],
    predictor: Callable[[float], float],
) -> float:
    if not points:
        raise ValueError("Points for RMSE cannot be empty")

    sq_sum = 0.0
    for x, y in points:
        dy = predictor(x) - y
        sq_sum += dy * dy
    return math.sqrt(sq_sum / len(points))


def fit_power_model(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    transformed = []
    for x, y in points:
        if x <= 0 or y <= 0:
            raise ValueError("Power model requires x > 0 and y > 0")
        transformed.append((math.log(x), math.log(y)))

    c0, b = approximate_polynomial(Dataset(transformed), PolynomialDegree.LINEAR)
    a = math.exp(c0)
    return a, b


def fit_exponential_model(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    transformed = []
    for x, y in points:
        if y <= 0:
            raise ValueError("Exponential model requires y > 0")
        transformed.append((x, math.log(y)))

    c0, b = approximate_polynomial(Dataset(transformed), PolynomialDegree.LINEAR)
    a = math.exp(c0)
    return a, b


def fit_fraction_model(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    transformed = []
    for x, y in points:
        if x == 0:
            raise ValueError("Divizion by SERO!")
        transformed.append((1 / x, y))
    c0, c1 = approximate_polynomial(Dataset(transformed), PolynomialDegree.LINEAR)
    return c0, c1


def fit_complex_fraction_model(
    points: Sequence[tuple[float, float]],
) -> tuple[float, float]:
    transformed = []
    for x, y in points:
        if x == 0:
            raise ValueError("Divizion by SERO!")
        transformed.append((x, 1 / y))
    c0, c1 = approximate_polynomial(Dataset(transformed), PolynomialDegree.LINEAR)
    return c0, c1


def solve_boundary_problem(
    m: int,
    integration_samples: int = 200,
    convergence_tol: float = 1e-12,
    max_refinements: int = 20,
) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    """
    Solve:

        y'' + x*y' + y = 2x
        y(0)=1
        y(1)=0

    using least-squares minimization of residual integral:

        E(C) = ∫ R(x,C)^2 dx

    with basis:

        y(x) = u0(x) + Σ Ck uk(x)

    where:
        u0(x) = 1 - x
        uk(x) = x^k (1-x)

    Matrix elements are computed by numerical integration
    on a uniform grid with convergence refinement.
    """

    if m < 1:
        raise ValueError("m must be >= 1")

    def u0(x):
        return 1.0 - x

    def u0_d1(x):
        return -np.ones_like(x)

    def u0_d2(x):
        return np.zeros_like(x)

    def uk(x, k):
        return x**k * (1.0 - x)

    def uk_d1(x, k):
        return k * x ** (k - 1) - (k + 1) * x**k

    def uk_d2(x, k):
        result = -k * (k + 1) * x ** (k - 1)

        if k >= 2:
            result += k * (k - 1) * x ** (k - 2)

        return result

    def operator(y, dy, d2y, x):
        return d2y + x * dy + y - 2 * x

    def r0(x):
        return operator(
            u0(x),
            u0_d1(x),
            u0_d2(x),
            x,
        )

    def rk(x, k):
        return operator(
            uk(x, k),
            uk_d1(x, k),
            uk_d2(x, k),
            x,
        )

    def integrate_uniform(func_values, x):
        return np.trapezoid(func_values, x)

    # ============================================================
    # MATRIX ASSEMBLY
    #
    # E(C) = ∫ (r0 + Σ Ck rk)^2 dx
    #
    # Gives:
    #
    # A_ij = ∫ ri*rj
    # b_i  = -∫ r0*ri
    # ============================================================

    previous_coeffs = None

    for refinement in range(max_refinements):

        n = integration_samples * (2**refinement)

        x = np.linspace(0.0, 1.0, n)

        # evaluate residual basis functions
        residual_basis = []

        for k in range(1, m + 1):
            residual_basis.append(rk(x, k))

        residual_basis = np.array(residual_basis)

        residual_0 = r0(x)

        A = np.zeros((m, m))

        for i in range(m):
            for j in range(m):

                integrand = residual_basis[i] * residual_basis[j]

                A[i, j] = integrate_uniform(integrand, x)

        b = np.zeros(m)

        for i in range(m):

            integrand = residual_0 * residual_basis[i]

            b[i] = -integrate_uniform(integrand, x)

        try:
            coeffs = np.linalg.solve(A, b)

        except np.linalg.LinAlgError:
            coeffs = np.linalg.pinv(A) @ b

        if previous_coeffs is not None:

            delta = np.linalg.norm(coeffs - previous_coeffs)

            if delta < convergence_tol:
                break

        previous_coeffs = coeffs.copy()

    x_grid = np.linspace(0.0, 1.0, 400)

    y_grid = u0(x_grid)

    for k, c in enumerate(coeffs, start=1):
        y_grid += c * uk(x_grid, k)

    return (
        x_grid,
        y_grid,
        tuple(float(v) for v in coeffs),
    )
