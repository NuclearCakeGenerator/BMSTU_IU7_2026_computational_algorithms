from typing import List, Tuple


def interpolate_polynomial_1d(
    dataset: List[Tuple[float, float]], target: float, degree: int = 5
) -> float:
    """
    Interpolate the value for the target point based on the provided
    1D dataset.

    Args:
        dataset: List of tuples containing 1D dataset (x, y)
        target: Target x value for interpolation
        degree: Degree of the polynomial for interpolation (default is 6)
    Returns:
        Interpolated y value at the target x
    """
    if not dataset:
        raise ValueError("Dataset cannot be empty.")

    if degree < 0:
        raise ValueError("Degree of interpolation must be at least 1.")

    if len(dataset) <= 1:
        raise ValueError(
            "Dataset must contain at least two points " "for interpolation."
        )

    degree = min(degree, len(dataset) - 1)  # Adjust degree if dataset is small

    def split_delta(args: List[Tuple[float, float]]) -> float:
        if len(args) == 1:
            return args[0][1]
        else:
            return (split_delta(args[1:]) - split_delta(args[:-1])) / (
                args[-1][0] - args[0][0]
            )

    interpolation_range = sorted(dataset, key=lambda p: abs(p[0] - target))[
        : degree + 1
    ]

    result = 0.0
    for i in range(len(interpolation_range)):
        term = split_delta(interpolation_range[: i + 1])
        for j in range(i):
            term *= target - interpolation_range[j][0]
        result += term

    return result


def interpolate_spline_1d(dataset: List[Tuple[float, float]], target: float) -> float:
    """
    Interpolate the value for the target point based on the provided
    1D dataset using a 3-degree spline interpolation.

    Args:
        dataset: List of tuples containing 1D dataset (x, y)
        target: Target x value for interpolation
    Returns:
        Interpolated y value at the target x
    """
    x = [point[0] for point in dataset]
    y = [point[1] for point in dataset]
    n = len(dataset)
    if n < 2:
        raise ValueError(
            "Dataset must contain at least two points " "for interpolation."
        )
    if target < x[0] or target > x[-1]:
        raise ValueError("Target value must be within the range of the dataset.")

    h = list()
    for i in range(n - 1):
        h.append(x[i + 1] - x[i])

    A = [0]
    for i in range(1, len(h)):
        A.append(h[i - 1])
    B = [0]
    for i in range(1, len(h)):
        B.append(2 * (h[i - 1] + h[i]))
    D = list()
    for i in range(len(h)):
        D.append(h[i])
    F = [0]
    for i in range(1, len(h) - 1):
        F.append(3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]))

    Dzeta = [0, -D[1] / B[1]]
    for i in range(2, len(h)):
        Dzeta.append(-D[i] / (B[i] + A[i] * Dzeta[i - 1]))
    Nu = [0, F[1] / B[1]]
    for i in range(2, len(h)):
        Nu.append((F[i] - A[i] * Nu[i - 1]) / (B[i] + A[i] * Dzeta[i - 1]))

    c = [0] * (n + 1)
    for i in range(n - 1, 0, -1):
        c[i] = Dzeta[i] * c[i + 1] + Nu[i]
    a = [val for val in y]
    b = list()
    for i in range(n - 1):
        b.append((a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3)
    d = list()
    for i in range(n - 1):
        d.append((c[i + 1] - c[i]) / (3 * h[i]))

    i = max(i for i in range(n - 1) if x[i] <= target < x[i + 1])
    dx = target - x[i]
    y_interpolated = a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3

    return y_interpolated
