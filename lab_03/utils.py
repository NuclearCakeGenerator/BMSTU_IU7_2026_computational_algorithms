from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

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
