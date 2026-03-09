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


def load_3d_data(filepath: str) -> List[Tuple[Tuple[float, float, float], float]]:
    """
    Load 3D data from file into a list of ((x, y, z), u) tuples.
    
    Args:
        filepath: Path to the data file
    Returns:
        List of tuples containing ((x, y, z), u) where u is the function value
    """
    data = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_z = None
    x_values = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this is a z-value line (e.g., "z=0")
        if line.startswith('z='):
            current_z = float(line.split('=')[1])
            continue
        
        # Check if this is a header line (e.g., "y\x	0	1	2	3	4")
        if line.startswith('y\\x') or line.startswith('0\t') and not x_values:
            # Parse x values from header
            parts = line.split('\t')
            if line.startswith('y\\x'):
                x_values = [float(x) for x in parts[1:]]
            else:
                # First data row, use indices as x values
                x_values = list(range(len(parts) - 1))
            continue
        
        # Parse data rows
        parts = line.split('\t')
        if len(parts) > 1 and current_z is not None:
            y = float(parts[0])
            values = [float(v) for v in parts[1:]]
            
            for i, u in enumerate(values):
                if i < len(x_values):
                    x = x_values[i]
                    data.append(((x, y, current_z), u))
    
    return data


def interpolate_3d(
    dataset: List[Tuple[Tuple[float, float, float], float]],
    target: Tuple[float, float, float],
    x_method: str = 'polynomial',
    y_method: str = 'polynomial',
    z_method: str = 'polynomial',
    x_degree: int = 5,
    y_degree: int = 5,
    z_degree: int = 5
) -> float:
    """
    Perform 3D interpolation using successive 1D interpolations.
    
    Args:
        dataset: List of ((x, y, z), u) tuples
        target: Target (x, y, z) point for interpolation
        x_method: Interpolation method for x dimension ('polynomial' or 'spline')
        y_method: Interpolation method for y dimension ('polynomial' or 'spline')
        z_method: Interpolation method for z dimension ('polynomial' or 'spline')
        x_degree: Polynomial degree for x (used only if x_method is 'polynomial')
        y_degree: Polynomial degree for y (used only if y_method is 'polynomial')
        z_degree: Polynomial degree for z (used only if z_method is 'polynomial')
    Returns:
        Interpolated value at target point
    """
    target_x, target_y, target_z = target
    
    # Get unique values for each dimension
    y_values = sorted(set(point[0][1] for point in dataset))
    z_values = sorted(set(point[0][2] for point in dataset))
    
    # Step 1: Interpolate along x for each (y, z) pair
    yz_pairs = [(y, z) for y in y_values for z in z_values]
    interpolated_x = []
    
    for y, z in yz_pairs:
        # Get all points with this y, z
        x_data = [(point[0][0], point[1]) for point in dataset
                  if point[0][1] == y and point[0][2] == z]
        
        if x_data:
            x_data = sorted(x_data, key=lambda p: p[0])
            
            if x_method == 'polynomial':
                u_x = interpolate_polynomial_1d(x_data, target_x, x_degree)
            else:  # spline
                u_x = interpolate_spline_1d(x_data, target_x)
            
            interpolated_x.append(((y, z), u_x))
    
    # Step 2: Interpolate along y for each z
    interpolated_y = []
    
    for z in z_values:
        # Get all interpolated points with this z
        y_data = [(point[0][0], point[1]) for point in interpolated_x
                  if point[0][1] == z]
        
        if y_data:
            y_data = sorted(y_data, key=lambda p: p[0])
            
            if y_method == 'polynomial':
                u_y = interpolate_polynomial_1d(y_data, target_y, y_degree)
            else:  # spline
                u_y = interpolate_spline_1d(y_data, target_y)
            
            interpolated_y.append((z, u_y))
    
    # Step 3: Interpolate along z
    z_data = sorted(interpolated_y, key=lambda p: p[0])
    
    if z_method == 'polynomial':
        result = interpolate_polynomial_1d(z_data, target_z, z_degree)
    else:  # spline
        result = interpolate_spline_1d(z_data, target_z)
    
    return result
