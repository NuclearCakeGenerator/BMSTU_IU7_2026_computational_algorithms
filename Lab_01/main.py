import matplotlib.pyplot as plt
import numpy as np
from utils import interpolate_1d

# Hardcoded 2D dataset
dataset = [
    (1.0, 2.1),
    (2.0, 4.2),
    (3.0, 1),
    (4.0, 8.1),
    (5.0, 4.3),
    (6.0, 11.8),
    (7.0, 27),
    (8.0, 6),
]

# Create figure
plt.figure(figsize=(10, 6))

# Plot original data points
x_data = [point[0] for point in dataset]
y_data = [point[1] for point in dataset]
plt.scatter(x_data, y_data, color="red", s=100, label="Data points", zorder=5)

# Generate interpolated curve
x_smooth = np.linspace(min(x_data), max(x_data), 100)
y_smooth = [interpolate_1d(dataset, x, degree=2) for x in x_smooth]

# Plot interpolated curve
plt.plot(x_smooth, y_smooth, "b-", linewidth=2, label="Interpolated curve")

# Labels and legend
plt.xlabel("X")
plt.ylabel("Y")
plt.title("1D Interpolation Test")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
