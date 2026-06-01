from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple
import math


def integrate_trapezoid(func: Callable[[float], float], left: float, right: float, n: int) -> float:
    if n < 1:
        raise ValueError("n must be >= 1")
    h = (right - left) / n
    s = 0.5 * (func(left) + func(right))
    for i in range(1, n):
        s += func(left + i * h)
    return s * h


def integrate_simpson(func: Callable[[float], float], left: float, right: float, n: int) -> float:
    if n < 2 or n % 2 == 1:
        raise ValueError("n must be even and >= 2 for Simpson rule")
    h = (right - left) / n
    s = func(left) + func(right)
    for i in range(1, n):
        coef = 4 if i % 2 == 1 else 2
        s += coef * func(left + i * h)
    return s * h / 3.0


def _legendre(n: int, x: float) -> Tuple[float, float]:
    # Returns P_n(x) and P_n'(x)
    if n == 0:
        return 1.0, 0.0
    P0 = 1.0
    P1 = x
    for k in range(2, n + 1):
        Pk = ((2 * k - 1) * x * P1 - (k - 1) * P0) / k
        P0, P1 = P1, Pk
    # derivative via recurrence: P_n'(x) = n/(x^2-1) (x P_n(x) - P_{n-1}(x))
    Pn = P1
    Pn_1 = P0
    dPn = n * (x * Pn - Pn_1) / (x * x - 1.0)
    return Pn, dPn


def gauss_legendre_nodes_weights(n: int) -> List[Tuple[float, float]]:
    if n < 1:
        raise ValueError("n must be >= 1")
    nodes = []
    m = (n + 1) // 2
    for i in range(1, m + 1):
        # Initial guess (Chebyshev)
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(50):
            Pn, dPn = _legendre(n, x)
            dx = -Pn / dPn
            x += dx
            if abs(dx) < 1e-14:
                break
        Pn, dPn = _legendre(n, x)
        w = 2.0 / ((1.0 - x * x) * (dPn * dPn))
        nodes.append((x, w))
    # mirror
    result = []
    for x, w in nodes:
        result.append((-x, w))
    for x, w in reversed(nodes):
        result.append((x, w))
    result = sorted(result, key=lambda t: t[0])
    return result


def integrate_gauss(func: Callable[[float], float], left: float, right: float, nodes: int) -> float:
    pts = gauss_legendre_nodes_weights(nodes)
    # map from [-1,1] to [left,right]
    mid = 0.5 * (left + right)
    half = 0.5 * (right - left)
    s = 0.0
    for x, w in pts:
        s += w * func(mid + half * x)
    return half * s


def load_tabular_2d(filepath: str) -> List[Tuple[Tuple[float, float], float]]:
    p = Path(filepath)
    text = p.read_text(encoding="utf-16")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # header with x values in first line (after a label like y\x)
    header = lines[0].replace('\r','').strip()
    parts = header.split()
    # first token often like 'y\x' or similar
    x_vals = [float(tok.replace(',','.')) for tok in parts[1:]]
    data = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        y = float(parts[0].replace(',','.'))
        values = [float(v.replace(',','.')) for v in parts[1:1+len(x_vals)]]
        for xv, val in zip(x_vals, values):
            data.append(((xv, y), val))
    return data


def _lagrange_interpolate(points: List[Tuple[float, float]], target: float) -> float:
    # points: list of (x,y) ; use Lagrange formula
    if not points:
        raise ValueError("No points for interpolation")
    result = 0.0
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        term = yi
        for j in range(n):
            if j == i:
                continue
            xj = points[j][0]
            term *= (target - xj) / (xi - xj)
        result += term
    return result


def interpolate_2d(dataset: List[Tuple[Tuple[float, float], float]], target: Tuple[float, float], degree: int = 2) -> float:
    tx, ty = target
    # collect unique y values
    y_vals = sorted(set(pt[0][1] for pt in dataset))
    # for each y, collect (x,u) and interpolate in x to get u_at_x
    interp_y = []
    for y in y_vals:
        row = sorted([(pt[0][0], pt[1]) for pt in dataset if pt[0][1] == y], key=lambda p: p[0])
        if not row:
            continue
        # pick closest degree+1 points to tx
        row_sel = sorted(row, key=lambda p: abs(p[0] - tx))[: degree + 1]
        u_at_x = _lagrange_interpolate(row_sel, tx)
        interp_y.append((y, u_at_x))
    # now interpolate along y
    interp_y_sorted = sorted(interp_y, key=lambda p: abs(p[0] - ty))[: degree + 1]
    return _lagrange_interpolate(interp_y_sorted, ty)


def adaptive_simpson(func: Callable[[float], float], a: float, b: float, eps: float = 1e-6, max_recursion: int = 20) -> float:
    def simpson(f, a, b):
        c = (a + b) / 2.0
        return (b - a) * (f(a) + 4.0 * f(c) + f(b)) / 6.0

    def recursive(f, a, b, eps, S, fa, fb, fc, depth):
        c = (a + b) / 2.0
        d = (a + c) / 2.0
        e = (c + b) / 2.0
        fd = f(d)
        fe = f(e)
        Sleft = (c - a) * (fa + 4.0 * fd + fc) / 6.0
        Sright = (b - c) * (fc + 4.0 * fe + fb) / 6.0
        if depth <= 0 or abs(Sleft + Sright - S) <= 15 * eps:
            return Sleft + Sright + (Sleft + Sright - S) / 15.0
        return recursive(f, a, c, eps / 2.0, Sleft, fa, fc, fd, depth - 1) + recursive(f, c, b, eps / 2.0, Sright, fc, fb, fe, depth - 1)

    fa = func(a)
    fb = func(b)
    c = (a + b) / 2.0
    fc = func(c)
    S = (b - a) * (fa + 4.0 * fc + fb) / 6.0
    return recursive(func, a, b, eps, S, fa, fb, fc, max_recursion)


def double_integral_iterated(
    dataset: List[Tuple[Tuple[float, float], float]],
    phi: Callable[[float], float],
    psi: Callable[[float], float],
    a: float,
    b: float,
    eps: float = 1e-6,
    inner_degree: int = 2,
    outer_eps: float = 1e-6,
) -> float:
    # inner integrand for given x computes integral in y from phi(x) to psi(x)
    def inner(x: float) -> float:
        y1 = phi(x)
        y2 = psi(x)

        def f_y(y: float) -> float:
            return interpolate_2d(dataset, (x, y), degree=inner_degree)

        return adaptive_simpson(f_y, y1, y2, eps)

    return adaptive_simpson(inner, a, b, outer_eps)
