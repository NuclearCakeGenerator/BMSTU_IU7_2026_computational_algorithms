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
