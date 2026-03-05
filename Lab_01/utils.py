import os
from typing import List, Tuple


def _load_2d_table(filename: str) -> List[Tuple[float, float]]:
    """
    Common function to load 2D data (x, y) from a file.

    Args:
        filename: Path to the data file

    Returns:
        List of tuples (x, y)
    """
    data = []
    filepath = os.path.join("raw_data", filename)

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            parts = [i for i in parts if i]  # Remove empty strings
            if len(parts) >= 3:
                try:
                    x = float(parts[1].replace(",", "."))
                    y = float(parts[2].replace(",", "."))
                    data.append((x, y))
                except ValueError:
                    continue

    return data


def _load_3d_table(filename: str) -> List[Tuple[float, float, float, float]]:
    """
    Common function to load 3D data (T, value_at_p1, value_at_p2, value_at_p3)
    from a file.

    Args:
        filename: Path to the data file

    Returns:
        List of tuples (T, val_0.5MPa, val_1.5MPa, val_2.5MPa)
    """
    data = []
    filepath = os.path.join("raw_data", filename)

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            parts = [i for i in parts if i]  # Remove empty strings
            if len(parts) >= 4:
                try:
                    T = float(parts[0].replace(",", "."))
                    val1 = float(parts[1].replace(",", "."))
                    val2 = float(parts[2].replace(",", "."))
                    val3 = float(parts[3].replace(",", "."))
                    data.append((T, 0.5, val1))
                    data.append((T, 1.5, val2))
                    data.append((T, 2.5, val3))
                except ValueError:
                    continue

    return data


def load_current_data() -> List[Tuple[float, float]]:
    """
    Load current vs time data: I(t).

    Returns:
        List of tuples (time_s, current_A)
    """
    return _load_2d_table("current.txt")


def load_heavy_particles_concentration() -> List[Tuple[float, float, float, float]]:
    """
    Load heavy particles concentration data: Nh(T,p).

    Returns:
        List of tuples (T_K, Nh_at_0.5MPa, Nh_at_1.5MPa, Nh_at_2.5MPa) in units of 1e18 cm^-3
    """
    return _load_3d_table("heavy_particles.txt")


def load_electrical_conductivity() -> List[Tuple[float, float, float, float]]:
    """
    Load electrical conductivity data: σ(T,p).

    Returns:
        List of tuples (T_K, sigma_at_0.5MPa, sigma_at_1.5MPa, sigma_at_2.5MPa) in 1/(Ohm*cm)
    """
    return _load_3d_table("conductivity.txt")


def load_heat_capacity() -> List[Tuple[float, float, float, float]]:
    """
    Load heat capacity data: C(T,p).

    Returns:
        List of tuples (T_K, C_at_0.5MPa, C_at_1.5MPa, C_at_2.5MPa) in J/(cm^3*K)
    """
    return _load_3d_table("heat_capacity.txt")


def load_radiation_power() -> List[Tuple[float, float, float, float]]:
    """
    Load volumetric radiation power data: q(T,p).

    Returns:
        List of tuples (T_K, q_at_0.5MPa, q_at_1.5MPa, q_at_2.5MPa) in W/cm^3
    """
    return _load_3d_table("radiation.txt")


def interpolate_1d(
    dataset: List[Tuple[float, float]], target: float, degree: int = 6
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
        raise ValueError("Dataset must contain at least two points for interpolation.")

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


def interpolate_nd(dataset: List[Tuple[float]], target: Tuple[float]) -> float:
    """
    Interpolate the value for the target point based on the provided dataset.

    Args:
        dataset: List of tuples containing 2d or 3d dataset
        target: Tuple containing 1d or 2d coordinates for interpolation
    Returns:
        Interpolated value at the target point
    """

    pass
