from __future__ import annotations

import math
from typing import Callable


Integrand = Callable[[float], float]
Integrator = Callable[[Integrand, float, float, int], float]


# Keep this function separate so it is easy to redefine in source code later.
def integrate_trapezoid(
    func: Integrand,
    left: float,
    right: float,
    steps: int,
) -> float:
    if steps < 1:
        raise ValueError("Number of trapezoids must be >= 1")

    if left == right:
        return 0.0

    sign = 1.0
    a, b = left, right
    if a > b:
        a, b = b, a
        sign = -1.0

    h = (b - a) / steps
    result = 0.5 * (func(a) + func(b))

    for i in range(1, steps):
        result += func(a + i * h)

    return sign * result * h


def default_laplace_integrand(t: float) -> float:
    # Standard normal kernel.
    return math.exp(-(t * t) / 2.0)


DEFAULT_LAPLACE_SCALE = 1.0 / math.sqrt(2.0 * math.pi)


def task1_residuals(x: float, y: float) -> tuple[float, float]:
    if x - y <= 0.0:
        raise ValueError("For task 1 the condition x - y > 0 must hold")

    angle = 0.7 * (x + y)
    f1 = 20.0 * math.sin(angle) - x
    f2 = 20.0 * math.log(x - y) - 6.0 * x - y
    return f1, f2


def task1_jacobian(
    x: float,
    y: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if x - y <= 0.0:
        raise ValueError("For task 1 the condition x - y > 0 must hold")

    angle = 0.7 * (x + y)
    cos_term = math.cos(angle)
    diff_term = 1.0 / (x - y)

    return (
        (14.0 * cos_term - 1.0, 14.0 * cos_term),
        (20.0 * diff_term - 6.0, -20.0 * diff_term - 1.0),
    )


def _solve_2x2(
    a11: float,
    a12: float,
    a21: float,
    a22: float,
    b1: float,
    b2: float,
) -> tuple[float, float]:
    determinant = a11 * a22 - a12 * a21
    if abs(determinant) < 1e-15:
        raise ValueError("Jacobian matrix is singular")

    dx = (b1 * a22 - a12 * b2) / determinant
    dy = (a11 * b2 - b1 * a21) / determinant
    return dx, dy


def solve_task1_newton(
    x0: float,
    y0: float,
    eps: float,
    max_iterations: int = 50,
) -> tuple[float, float, int, float, float]:
    if eps <= 0.0:
        raise ValueError("Epsilon must be > 0")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if x0 - y0 <= 0.0:
        raise ValueError("Initial approximation must satisfy x0 - y0 > 0")

    x = float(x0)
    y = float(y0)

    for iteration in range(1, max_iterations + 1):
        f1, f2 = task1_residuals(x, y)
        jacobian = task1_jacobian(x, y)
        dx, dy = _solve_2x2(
            jacobian[0][0],
            jacobian[0][1],
            jacobian[1][0],
            jacobian[1][1],
            -f1,
            -f2,
        )

        x_next = x + dx
        y_next = y + dy

        if x_next - y_next <= 0.0:
            raise ValueError(
                "Newton iteration left the valid domain x - y > 0; adjust x0, y0"
            )

        f1_next, f2_next = task1_residuals(x_next, y_next)
        x, y = x_next, y_next

        if max(abs(dx), abs(dy)) < eps or max(abs(f1_next), abs(f2_next)) < eps:
            return x, y, iteration, f1_next, f2_next

    f1_final, f2_final = task1_residuals(x, y)
    return x, y, max_iterations, f1_final, f2_final


def laplace_function(
    x: float,
    *,
    integral_steps: int,
    integrator: Integrator = integrate_trapezoid,
    integrand: Integrand = default_laplace_integrand,
    scale: float = DEFAULT_LAPLACE_SCALE,
) -> float:
    integral = integrator(integrand, 0.0, x, integral_steps)
    return scale * integral


def bisection_root_monotonic(
    func: Callable[[float], float],
    left: float,
    right: float,
    eps: float,
    max_iterations: int,
) -> tuple[float, int]:
    if left >= right:
        raise ValueError("Left bound must be less than right bound")
    if eps <= 0.0:
        raise ValueError("Epsilon must be > 0")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    f_left = func(left)
    f_right = func(right)

    if f_left == 0.0:
        return left, 0
    if f_right == 0.0:
        return right, 0
    if f_left * f_right > 0.0:
        raise ValueError(
            "Function values at interval ends must have opposite signs "
            "for bisection"
        )

    a, b = left, right
    for iteration in range(1, max_iterations + 1):
        mid = (a + b) / 2.0
        f_mid = func(mid)

        if abs(f_mid) <= eps or (b - a) / 2.0 <= eps:
            return mid, iteration

        if f_left * f_mid < 0.0:
            b = mid
            f_right = f_mid
        else:
            a = mid
            f_left = f_mid

    return (a + b) / 2.0, max_iterations


def solve_inverse_laplace(
    target_value: float,
    left: float,
    right: float,
    *,
    eps: float,
    integral_steps: int,
    max_iterations: int,
    integrator: Integrator = integrate_trapezoid,
    integrand: Integrand = default_laplace_integrand,
    scale: float = DEFAULT_LAPLACE_SCALE,
) -> tuple[float, float, int]:
    def equation(x: float) -> float:
        return (
            laplace_function(
                x,
                integral_steps=integral_steps,
                integrator=integrator,
                integrand=integrand,
                scale=scale,
            )
            - target_value
        )

    x_root, iterations = bisection_root_monotonic(
        equation,
        left,
        right,
        eps,
        max_iterations,
    )

    phi_at_root = laplace_function(
        x_root,
        integral_steps=integral_steps,
        integrator=integrator,
        integrand=integrand,
        scale=scale,
    )

    return x_root, phi_at_root, iterations
