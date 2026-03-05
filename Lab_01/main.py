import matplotlib.pyplot as plt
import numpy as np
from utils import (
    load_current_data,
    load_heavy_particles_concentration,
    load_electrical_conductivity,
    load_heat_capacity     ,     
    load_radiation_power,
    interpolate_2d,
    interpolate_1d,
)


class configuration:
    idle_pressure: float = 0.04  # mbar
    idle_temperature: float = 300  # K
    starting_time: float = 14e-6  # s
    ending_time: float = 450e-6  # s
    starting_temperature: float = 5400  # K
    tube_length: float = 0.1  # m
    tube_radius: float = 0.02  # m
    time_step: float = 1e-6  # s


class wanted_data:
    temperature: list[tuple[float, float]] = []
    pressure: list[tuple[float, float]] = []
    conductivity: list[tuple[float, float]] = []
    volumetric_radiation: list[tuple[float, float]] = []
    resistance: list[tuple[float, float]] = []
    surface_radiation: list[tuple[float, float]] = []


class data:
    temperature: list[tuple[float, float]] = []
    pressure: list[tuple[float, float]] = []
    conductivity: list[tuple[float, float]] = []
    volumetric_radiation: list[tuple[float, float]] = []
    resistance: list[tuple[float, float]] = []
    surface_radiation: list[tuple[float, float]] = []


dataset = load_current_data()

# Create figure
plt.figure(figsize=(10, 6))

# Plot original data points
x_data = [point[0] for point in dataset]
y_data = [point[1] for point in dataset]
plt.scatter(x_data, y_data, color="red", s=100, label="Data points", zorder=5)

# Generate interpolated curve
x_smooth = np.linspace(min(x_data), max(x_data), 1000)
y_smooth = [interpolate_1d(dataset, x, degree=5) for x in x_smooth]

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


# # 2D Interpolation for 3D dataset
# def visualize_2d_interpolation(dataset_3d):
#     """
#     Visualize 2D interpolation for a 3D dataset (x, y, z).

#     Parameters:
#     dataset_3d: List of tuples (x, y, z)
#     """
#     # Extract x, y, z coordinates
#     x_pts = np.array([point[0] for point in dataset_3d])
#     y_pts = np.array([point[1] for point in dataset_3d])
#     z_pts = np.array([point[2] for point in dataset_3d])

#     # Create a grid for interpolation
#     x_min, x_max = x_pts.min(), x_pts.max()
#     y_min, y_max = y_pts.min(), y_pts.max()

#     x_grid = np.linspace(x_min, x_max, 100)
#     y_grid = np.linspace(y_min, y_max, 100)
#     X, Y = np.meshgrid(x_grid, y_grid)

#     # Interpolate z values on the grid
#     Z = np.zeros_like(X)
#     for i in range(X.shape[0]):
#         for j in range(X.shape[1]):
#             Z[i, j] = interpolate_2d(dataset_3d, (X[i, j], Y[i, j]))

#     # Create 3D plot
#     fig = plt.figure()

#     # Surface plot
#     ax1 = fig.add_subplot(121, projection="3d")
#     ax1.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8, edgecolor="none")
#     ax1.scatter(x_pts, y_pts, z_pts, color="red", s=100, label="Data points")
#     ax1.set_xlabel("X")
#     ax1.set_ylabel("Y")
#     ax1.set_zlabel("Z")
#     ax1.set_title("3D Surface Interpolation")
#     ax1.legend()

#     # # Contour plot
#     # ax2 = fig.add_subplot(122)
#     # contour = ax2.contourf(X, Y, Z, levels=20, cmap="viridis")
#     # ax2.scatter(x_pts, y_pts, color="red", s=100, zorder=5, label="Data points")
#     # ax2.set_xlabel("X")
#     # ax2.set_ylabel("Y")
#     # ax2.set_title("2D Contour Interpolation")
#     # ax2.legend()
#     # plt.colorbar(contour, ax=ax2, label="Z value")

#     # plt.tight_layout()
#     plt.show()
