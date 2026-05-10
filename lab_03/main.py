from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utils import (
    Dataset,
    PolynomialDegree,
    approximate_polynomial,
    fit_exponential_model,
    fit_fraction_model,
    fit_power_model,
    fit_complex_fraction_model,
    polynomial,
    root_mean_square_error,
    solve_boundary_problem,
)

TASK_1_DOTS: list[tuple[float, float]] = [
    (-4.0, 1.5),
    (-3.0, 4.6),
    (-2.0, 3.0),
    (-1.0, 3.1),
    (0.0, 1.2),
    (1.0, 1.6),
    (2.0, 3.9),
    (3.0, 8.2),
    (4.0, 14.6),
]
TASK_1_WEIGHTS: list[float] = [0.2, 0.2, 0.2, 0.2, 0.2, 1.2, 99, 99, 1.4]

TASK_2_DOTS: list[tuple[float, float, float]] = [
    (-2.0, -2.0, 3.9),
    (-2.0, -1.0, 4.2),
    (-2.0, 1.0, 4.3),
    (-1.5, -2.0, 5.7),
    (-1.5, -1.0, 5.9),
    (-1.5, 2.0, 5.9),
    (-1.0, -2.0, 6.8),
    (-1.0, -1.0, 6.9),
    (-1.0, 0.0, 7.1),
    (-0.5, 2.0, 6.7),
    (0.0, -2.0, 5.2),
    (0.0, -1.0, 5.0),
    (0.5, -2.0, 3.6),
    (0.5, -1.0, 3.5),
    (0.5, 0.0, 3.8),
    (0.5, 1.0, 3.7),
    (0.5, 2.0, 3.9),
    (1.0, -2.0, 2.8),
    (1.0, -1.0, 2.9),
    (1.0, 1.0, 2.7),
    (1.0, 2.0, 3.0),
    (1.5, 0.0, 3.3),
    (1.5, 1.0, 3.5),
    (2.0, 0.0, 4.8),
    (2.0, 2.0, 5.0),
]
TASK_2_WEIGHTS: list[float] = [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    100,
    100,
    100,
    100,
    100,
    100,
    100,
    100,
    100,
    100,
]

TASK_3_POINTS: list[tuple[float, float]] = [
    (0.5, 10.5),
    (1.0, 1.6),
    (1.5, 0.55),
    (2.0, 0.26),
    (2.5, 0.15),
    (3.0, 0.08),
]


class Lab03App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Lab 03 - Polynomial Approximation")
        self.root.geometry("980x640")

        self.current_task_var = tk.StringVar(value="1")

        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="Subtask:").pack(side=tk.LEFT)
        self.task_combo = ttk.Combobox(
            top_frame,
            values=["1", "2", "3", "4"],
            width=10,
            textvariable=self.current_task_var,
            state="readonly",
        )
        self.task_combo.pack(side=tk.LEFT, padx=8)
        self.task_combo.bind("<<ComboboxSelected>>", self._on_task_changed)

        self.content_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        self._render_current_task()

    def _clear_content(self) -> None:
        for child in self.content_frame.winfo_children():
            child.destroy()

    def _on_task_changed(self, _event: tk.Event) -> None:
        self._render_current_task()

    def _render_current_task(self) -> None:
        self._clear_content()
        task = self.current_task_var.get()

        if task == "1":
            self._render_task_1()
        elif task == "2":
            self._render_task_2()
        elif task == "3":
            self._render_task_3()
        elif task == "4":
            self._render_task_4()
        else:
            self._render_placeholder(task)

    def _render_placeholder(self, task_id: str) -> None:
        placeholder = ttk.Frame(self.content_frame)
        placeholder.pack(fill=tk.BOTH, expand=True)

        label = ttk.Label(
            placeholder,
            text=f"Subtask {task_id} is not implemented yet.",
            font=("Arial", 16),
            anchor="center",
            justify="center",
        )
        label.place(relx=0.5, rely=0.5, anchor="center")

    def _build_readonly_table(
        self,
        parent: ttk.Frame,
        columns: list[str],
        rows: list[tuple[float, ...]],
    ) -> None:
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill=tk.BOTH, expand=True)

        tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        v_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=v_scroll.set)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=130, anchor="center")

        for row in rows:
            tree.insert("", tk.END, values=[f"{value:.4f}" for value in row])

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _render_task_1(self) -> None:
        dataset = Dataset(TASK_1_DOTS, TASK_1_WEIGHTS)

        ttk.Label(
            self.content_frame,
            text="Task 1: 2D Dataset (x, y) with weights",
            font=("Arial", 12, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        rows = [
            (x, y, w) for (x, y), w in zip(dataset.dots, dataset.weights, strict=True)
        ]
        self._build_readonly_table(self.content_frame, ["x", "y", "weight"], rows)

        ttk.Button(
            self.content_frame,
            text="Approximate",
            command=lambda: self._plot_task_1(dataset),
        ).pack(pady=10)

    def _render_task_2(self) -> None:
        dataset = Dataset(TASK_2_DOTS, TASK_2_WEIGHTS)

        ttk.Label(
            self.content_frame,
            text="Task 2: 3D Dataset (x, y, z) with weights",
            font=("Arial", 12, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        rows = [
            (x, y, z, w)
            for (x, y, z), w in zip(dataset.dots, dataset.weights, strict=True)
        ]
        self._build_readonly_table(
            self.content_frame,
            ["x", "y", "z", "weight"],
            rows,
        )

        ttk.Button(
            self.content_frame,
            text="Approximate",
            command=lambda: self._plot_task_2(dataset),
        ).pack(pady=10)

    def _render_task_3(self) -> None:
        ttk.Label(
            self.content_frame,
            text="Task 3: Best nonlinear model selection",
            font=("Arial", 12, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        self._build_readonly_table(self.content_frame, ["x", "y"], TASK_3_POINTS)

        ttk.Button(
            self.content_frame,
            text="Find best model",
            command=self._plot_task_3,
        ).pack(pady=10)

    def _render_task_4(self) -> None:
        page = ttk.Frame(self.content_frame)
        page.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            page,
            text="Task 4: Boundary-value ODE approximation (m=2 and m=3)",
            font=("Arial", 12, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        ttk.Label(
            page,
            text=(
                "Equation in implementation: y'' + x*y' + x*y = 0, " "y(0)=1, y(1)=0"
            ),
        ).pack(anchor="w", pady=(0, 12))

        ttk.Button(
            page,
            text="Solve and plot m=2/m=3",
            command=self._plot_task_4,
        ).pack(anchor="w")

    def _plot_task_1(self, dataset: Dataset) -> None:
        dataset_unweighted = Dataset(dataset.dots)

        linear_unweighted = approximate_polynomial(
            dataset_unweighted, PolynomialDegree.LINEAR
        )
        quadratic_unweighted = approximate_polynomial(
            dataset_unweighted, PolynomialDegree.QUADRATIC
        )
        linear_weighted = approximate_polynomial(dataset, PolynomialDegree.LINEAR)
        quadratic_weighted = approximate_polynomial(dataset, PolynomialDegree.QUADRATIC)

        x_values = np.array([dot[0] for dot in dataset.dots], dtype=float)
        y_values = np.array([dot[1] for dot in dataset.dots], dtype=float)
        x_grid = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 250)

        y_linear_unweighted = [polynomial((x,), linear_unweighted) for x in x_grid]
        y_quadratic_unweighted = [
            polynomial((x,), quadratic_unweighted) for x in x_grid
        ]
        y_linear_weighted = [polynomial((x,), linear_weighted) for x in x_grid]
        y_quadratic_weighted = [polynomial((x,), quadratic_weighted) for x in x_grid]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x_values, y_values, color="black", s=40, label="Dataset dots")

        ax.plot(
            x_grid,
            y_linear_unweighted,
            color="tab:blue",
            linewidth=2,
            label="Unweighted linear",
        )
        ax.plot(
            x_grid,
            y_quadratic_unweighted,
            color="tab:green",
            linewidth=2,
            label="Unweighted quadratic",
        )
        ax.plot(
            x_grid,
            y_linear_weighted,
            color="tab:orange",
            linewidth=2,
            linestyle="--",
            label="Weighted linear",
        )
        ax.plot(
            x_grid,
            y_quadratic_weighted,
            color="tab:red",
            linewidth=2,
            linestyle="--",
            label="Weighted quadratic",
        )

        ax.set_title("Task 1: Approximation")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.25)
        ax.legend()
        plt.tight_layout()
        plt.show()

    def _plot_task_2(self, dataset: Dataset) -> None:
        dataset_unweighted = Dataset(dataset.dots)

        linear_unweighted = approximate_polynomial(
            dataset_unweighted, PolynomialDegree.LINEAR
        )
        quadratic_unweighted = approximate_polynomial(
            dataset_unweighted, PolynomialDegree.QUADRATIC
        )
        linear_weighted = approximate_polynomial(dataset, PolynomialDegree.LINEAR)
        quadratic_weighted = approximate_polynomial(dataset, PolynomialDegree.QUADRATIC)

        x_values = np.array([dot[0] for dot in dataset.dots], dtype=float)
        y_values = np.array([dot[1] for dot in dataset.dots], dtype=float)
        z_values = np.array([dot[2] for dot in dataset.dots], dtype=float)

        x_grid, y_grid = np.meshgrid(
            np.linspace(float(np.min(x_values)), float(np.max(x_values)), 30),
            np.linspace(float(np.min(y_values)), float(np.max(y_values)), 30),
        )

        z_linear_unweighted = np.vectorize(
            lambda x, y: polynomial((x, y), linear_unweighted)
        )(x_grid, y_grid)
        z_quadratic_unweighted = np.vectorize(
            lambda x, y: polynomial((x, y), quadratic_unweighted)
        )(x_grid, y_grid)
        z_linear_weighted = np.vectorize(
            lambda x, y: polynomial((x, y), linear_weighted)
        )(x_grid, y_grid)
        z_quadratic_weighted = np.vectorize(
            lambda x, y: polynomial((x, y), quadratic_weighted)
        )(x_grid, y_grid)

        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax3d: Any = ax

        ax3d.scatter(
            x_values,
            y_values,
            zs=z_values,
            color="black",
            s=25,
        )
        ax.plot_surface(
            x_grid,
            y_grid,
            z_linear_unweighted,
            alpha=0.28,
            color="tab:blue",
        )
        ax.plot_surface(
            x_grid,
            y_grid,
            z_quadratic_unweighted,
            alpha=0.28,
            color="tab:green",
        )
        ax.plot_surface(
            x_grid,
            y_grid,
            z_linear_weighted,
            alpha=0.28,
            color="tab:orange",
        )
        ax.plot_surface(
            x_grid,
            y_grid,
            z_quadratic_weighted,
            alpha=0.28,
            color="tab:red",
        )

        ax.set_title("Task 2: 3D Approximation")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="black",
                markersize=8,
                label="Dataset dots",
            ),
            Patch(
                facecolor="tab:blue",
                edgecolor="tab:blue",
                alpha=0.28,
                label="Unweighted linear",
            ),
            Patch(
                facecolor="tab:green",
                edgecolor="tab:green",
                alpha=0.28,
                label="Unweighted quadratic",
            ),
            Patch(
                facecolor="tab:orange",
                edgecolor="tab:orange",
                alpha=0.28,
                label="Weighted linear",
            ),
            Patch(
                facecolor="tab:red",
                edgecolor="tab:red",
                alpha=0.28,
                label="Weighted quadratic",
            ),
        ]
        ax.legend(handles=legend_handles, loc="upper left")

        plt.tight_layout()
        plt.show()

    def _plot_task_3(self) -> None:
        points = TASK_3_POINTS

        a_pow, b_pow = fit_power_model(points)
        a_exp, b_exp = fit_exponential_model(points)
        a_frac, b_frac = fit_fraction_model(points)
        a0_com_frac, a1_com_frac, a2_com_frac = fit_complex_fraction_model(points)

        model_predictors = {
            f"{a_pow}*x^({b_pow})": lambda x: a_pow * x**b_pow,
            f"{a_exp}*exp({b_exp}*x)": lambda x: a_exp * np.exp(b_exp * x),
            f"{a_frac} + {b_frac}/x": lambda x: a_frac + b_frac / x,
            f"{a0_com_frac} / ({a1_com_frac} + {a2_com_frac}*x)": lambda x: a0_com_frac
            / (a1_com_frac + a2_com_frac * x),
        }

        errors = {
            name: root_mean_square_error(points, predictor)
            for name, predictor in model_predictors.items()
        }
        best_name = min(errors.items(), key=lambda item: item[1])[0]

        x_values = np.array([p[0] for p in points], dtype=float)
        y_values = np.array([p[1] for p in points], dtype=float)
        x_grid = np.linspace(np.min(x_values), np.max(x_values), 280)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x_values, y_values, color="black", s=40, label="Table points")

        colors = {
            f"{a_pow}*x^({b_pow})": "tab:blue",
            f"{a_exp}*exp({b_exp}*x)": "tab:green",
            f"{a_frac} + {b_frac}/x": "tab:orange",
            f"{a0_com_frac} / ({a1_com_frac} + {a2_com_frac}*x)": "tab:red",
        }
        for name, predictor in model_predictors.items():
            y_grid = predictor(x_grid)
            ax.plot(
                x_grid,
                y_grid,
                linewidth=2,
                color=colors[name],
                label=f"{name}, RMSE={errors[name]:.4f}",
            )

        ax.set_title(f"Task 3: Best model is {best_name}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.show()

    def _plot_task_4(self) -> None:
        x2, y2, coeffs2 = solve_boundary_problem(m=2)
        x3, y3, coeffs3 = solve_boundary_problem(m=3)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x2, y2, color="tab:blue", linewidth=2, label=f"m=2, C={coeffs2}")
        ax.plot(x3, y3, color="tab:red", linewidth=2, label=f"m=3, C={coeffs3}")
        ax.scatter([0.0, 1.0], [1.0, 0.0], color="black", s=45, label="Boundary nodes")

        ax.set_title("Task 4: Approximate ODE solution")
        ax.set_xlabel("x")
        ax.set_ylabel("y(x)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.show()


def main() -> None:
    root = tk.Tk()
    Lab03App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
