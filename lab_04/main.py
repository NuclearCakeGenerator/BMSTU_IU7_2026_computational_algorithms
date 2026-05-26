from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt

from utils import solve_inverse_laplace, solve_task1_newton
from utils import solve_task3_boundary_problem


class Lab04App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Лабораторная работа 4")
        self.root.geometry("840x520")

        self.task_var = tk.StringVar(value="1")

        self.task1_x0_var = tk.DoubleVar(value=1.0)
        self.task1_y0_var = tk.DoubleVar(value=0.0)
        self.task1_max_iterations_var = tk.IntVar(value=50)

        self.task3_intervals_var = tk.IntVar(value=20)
        self.task3_eps_var = tk.DoubleVar(value=1e-6)
        self.task3_max_iterations_var = tk.IntVar(value=50)

        self.target_phi_var = tk.DoubleVar(value=0.2)
        self.left_var = tk.DoubleVar(value=0.0)
        self.right_var = tk.DoubleVar(value=2.0)
        self.eps_var = tk.DoubleVar(value=1e-6)
        self.integral_steps_var = tk.IntVar(value=400)
        self.max_iterations_var = tk.IntVar(value=100)

        self.result_var = tk.StringVar(value="")
        self.task1_summary_var = tk.StringVar(value="")
        self.task3_summary_var = tk.StringVar(value="")

        self._build_layout()
        self._render_current_task()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top = ttk.Frame(self.root, padding=12)
        top.grid(row=0, column=0, sticky="ew")

        ttk.Label(top, text="Выберите задачу:", font=("Segoe UI", 11, "bold")).pack(
            side=tk.LEFT
        )

        combo = ttk.Combobox(
            top,
            state="readonly",
            width=14,
            values=["1", "2", "3"],
            textvariable=self.task_var,
        )
        combo.pack(side=tk.LEFT, padx=10)
        combo.bind("<<ComboboxSelected>>", self._on_task_changed)

        self.content = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        self.content.grid(row=1, column=0, sticky="nsew")

    def _clear_content(self) -> None:
        for child in self.content.winfo_children():
            child.destroy()

    def _on_task_changed(self, _event: tk.Event) -> None:
        self._render_current_task()

    def _render_current_task(self) -> None:
        self._clear_content()

        task = self.task_var.get()
        if task == "1":
            self._render_task_1()
            return
        if task == "2":
            self._render_task_2()
            return
        if task == "3":
            self._render_task_3()
        else:
            self._render_placeholder(task)

    def _render_placeholder(self, task_id: str) -> None:
        frame = ttk.Frame(self.content)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame,
            text=(
                f"Задача {task_id} пока не реализована.\n"
                "Выберите задачу 1, 2 или 3 для текущей реализации."
            ),
            font=("Segoe UI", 14),
            justify="center",
            anchor="center",
        ).place(relx=0.5, rely=0.5, anchor="center")

    def _render_task_1(self) -> None:
        frame = ttk.Frame(self.content)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame,
            text="Задача 1: решение системы нелинейных уравнений методом Ньютона",
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(
            frame,
            text=(
                "Система: 20*sin(0.7x - 0.7y) + 7x + 7y = 0, 20*ln(x - y) - x - y - 6 = 0. "
                "Показывается сравнение для eps = 1e-2, 1e-4, 1e-6."
            ),
            wraplength=780,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 12))

        form = ttk.LabelFrame(frame, text="Начальное приближение", padding=10)
        form.grid(row=2, column=0, sticky="nw")

        self._add_entry(form, 0, "x0:", self.task1_x0_var)
        self._add_entry(form, 1, "y0:", self.task1_y0_var)
        self._add_entry(form, 2, "Макс. итераций:", self.task1_max_iterations_var)

        ttk.Button(
            frame,
            text="Решить задачу 1",
            command=self._solve_task_1,
        ).grid(row=3, column=0, sticky="w", pady=12)

        summary_box = ttk.LabelFrame(frame, text="Итог", padding=10)
        summary_box.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(
            summary_box,
            textvariable=self.task1_summary_var,
            justify="left",
        ).pack(anchor="w")

        table_box = ttk.LabelFrame(frame, text="Сравнение по точности", padding=10)
        table_box.grid(row=5, column=0, sticky="nsew")

        columns = ("eps", "x", "y", "iters", "|F|", "status")
        tree = ttk.Treeview(table_box, columns=columns, show="headings", height=6)
        scroll = ttk.Scrollbar(table_box, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)

        headings = {
            "eps": "eps",
            "x": "x",
            "y": "y",
            "iters": "iter",
            "|F|": "max |F|",
            "status": "status",
        }
        widths = {"eps": 90, "x": 140, "y": 140, "iters": 80, "|F|": 110, "status": 220}
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], anchor="center")

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.task1_results_tree = tree

        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(5, weight=1)

    def _render_task_3(self) -> None:
        frame = ttk.Frame(self.content)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame,
            text="Задача 3: краевая задача y'' - y^3 = x^2, y(0)=1, y(1)=3",
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(
            frame,
            text=(
                "Используются линейзация Ньютона и метод прогонки для трёхдиагональной "
                "системы. Начальное приближение берётся как линейная интерполяция "
                "между "
                "краевыми условиями."
            ),
            wraplength=780,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 12))

        form = ttk.LabelFrame(frame, text="Параметры сетки", padding=10)
        form.grid(row=2, column=0, sticky="nw")

        self._add_entry(form, 0, "Число интервалов N:", self.task3_intervals_var)
        self._add_entry(form, 1, "Точность ε:", self.task3_eps_var)
        self._add_entry(form, 2, "Макс. итераций:", self.task3_max_iterations_var)

        ttk.Button(
            frame,
            text="Решить задачу 3",
            command=self._solve_task_3,
        ).grid(row=3, column=0, sticky="w", pady=12)

        summary_box = ttk.LabelFrame(frame, text="Итог", padding=10)
        summary_box.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(
            summary_box,
            textvariable=self.task3_summary_var,
            justify="left",
        ).pack(anchor="w")

        table_box = ttk.LabelFrame(frame, text="Узлы решения", padding=10)
        table_box.grid(row=5, column=0, sticky="nsew")

        columns = ("x", "y")
        tree = ttk.Treeview(table_box, columns=columns, show="headings", height=10)
        scroll = ttk.Scrollbar(table_box, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.heading("x", text="x")
        tree.heading("y", text="y")
        tree.column("x", width=160, anchor="center")
        tree.column("y", width=180, anchor="center")
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.task3_results_tree = tree

        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(5, weight=1)

    def _render_task_2(self) -> None:
        frame = ttk.Frame(self.content)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame,
            text="Задача 2: найти x по заданному значению функции Лапласа Φ(x)",
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(
            frame,
            text=(
                "Решение: корень монотонной функции методом половинного деления. "
                "Интеграл вычисляется методом трапеций."
            ),
            wraplength=780,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 12))

        form = ttk.LabelFrame(frame, text="Входные данные", padding=10)
        form.grid(row=2, column=0, sticky="nw")

        self._add_entry(form, 0, "Целевое значение Φ(x):", self.target_phi_var)
        self._add_entry(form, 1, "Левая граница x:", self.left_var)
        self._add_entry(form, 2, "Правая граница x:", self.right_var)
        self._add_entry(form, 3, "Точность ε:", self.eps_var)
        self._add_entry(form, 4, "Число трапеций:", self.integral_steps_var)
        self._add_entry(form, 5, "Макс. итераций бисекции:", self.max_iterations_var)

        ttk.Button(
            frame,
            text="Вычислить",
            command=self._solve_task_2,
        ).grid(row=3, column=0, sticky="w", pady=12)

        result_box = ttk.LabelFrame(frame, text="Результат", padding=10)
        result_box.grid(row=4, column=0, sticky="nsew")
        ttk.Label(result_box, textvariable=self.result_var, justify="left").pack(
            anchor="w"
        )

        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(4, weight=1)

    def _add_entry(
        self,
        parent: ttk.LabelFrame,
        row: int,
        label_text: str,
        variable: tk.Variable,
    ) -> None:
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=variable, width=20).grid(
            row=row,
            column=1,
            sticky="w",
            padx=8,
            pady=4,
        )

    def _solve_task_2(self) -> None:
        try:
            target = float(self.target_phi_var.get())
            left = float(self.left_var.get())
            right = float(self.right_var.get())
            eps = float(self.eps_var.get())
            integral_steps = int(self.integral_steps_var.get())
            max_iterations = int(self.max_iterations_var.get())

            x_root, phi_at_root, iterations = solve_inverse_laplace(
                target,
                left,
                right,
                eps=eps,
                integral_steps=integral_steps,
                max_iterations=max_iterations,
            )

            self.result_var.set(
                "\n".join(
                    [
                        f"Найденный аргумент: x = {x_root:.10f}",
                        f"Проверка: Φ(x) = {phi_at_root:.10f}",
                        f"Заданное значение: Φ* = {target:.10f}",
                        f"|Φ(x) - Φ*| = {abs(phi_at_root - target):.3e}",
                        f"Итераций бисекции: {iterations}",
                    ]
                )
            )
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def _solve_task_1(self) -> None:
        tree = getattr(self, "task1_results_tree", None)
        if tree is None:
            return

        for item in tree.get_children():
            tree.delete(item)

        try:
            x0 = float(self.task1_x0_var.get())
            y0 = float(self.task1_y0_var.get())
            max_iterations = int(self.task1_max_iterations_var.get())

            eps_values = [1e-2, 1e-4, 1e-6]
            lines: list[str] = []

            for eps in eps_values:
                try:
                    x, y, iterations, f1, f2 = solve_task1_newton(
                        x0,
                        y0,
                        eps,
                        max_iterations=max_iterations,
                    )
                    residual = max(abs(f1), abs(f2))
                    status = "ok"
                    lines.append(
                        (
                            f"eps={eps:.0e}: x={x:.10f}, y={y:.10f}, "
                            f"iters={iterations}, max|F|={residual:.3e}"
                        )
                    )
                    tree.insert(
                        "",
                        tk.END,
                        values=(
                            f"{eps:.0e}",
                            f"{x:.10f}",
                            f"{y:.10f}",
                            iterations,
                            f"{residual:.3e}",
                            status,
                        ),
                    )
                except Exception as inner_exc:
                    lines.append(f"eps={eps:.0e}: error: {inner_exc}")
                    tree.insert(
                        "",
                        tk.END,
                        values=(
                            f"{eps:.0e}",
                            "-",
                            "-",
                            "-",
                            "-",
                            str(inner_exc),
                        ),
                    )

            self.task1_summary_var.set("\n".join(lines))
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def _solve_task_3(self) -> None:
        tree = getattr(self, "task3_results_tree", None)
        if tree is None:
            return

        for item in tree.get_children():
            tree.delete(item)

        try:
            intervals = int(self.task3_intervals_var.get())
            eps = float(self.task3_eps_var.get())
            max_iterations = int(self.task3_max_iterations_var.get())

            x_grid, y_grid, iterations, residual = solve_task3_boundary_problem(
                intervals,
                eps,
                max_iterations=max_iterations,
            )

            for x_value, y_value in zip(x_grid, y_grid, strict=True):
                tree.insert(
                    "",
                    tk.END,
                    values=(f"{x_value:.6f}", f"{y_value:.10f}"),
                )

            self.task3_summary_var.set(
                "\n".join(
                    [
                        f"N = {intervals}",
                        f"Итераций Ньютона: {iterations}",
                        f"Итоговая невязка/поправка: {residual:.3e}",
                    ]
                )
            )

            self._plot_task_3_solution(x_grid, y_grid)
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def _plot_task_3_solution(self, x_grid: list[float], y_grid: list[float]) -> None:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.plot(
            x_grid,
            y_grid,
            linewidth=2.2,
            color="tab:blue",
            label="Численное решение",
        )
        ax.scatter([0.0, 1.0], [1.0, 3.0], color="black", s=45, label="Границы")
        ax.set_title("Задача 3: решение краевой задачи")
        ax.set_xlabel("x")
        ax.set_ylabel("y(x)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        plt.tight_layout()
        plt.show()


def main() -> None:
    root = tk.Tk()
    Lab04App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
