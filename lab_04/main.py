from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from utils import solve_inverse_laplace


class Lab04App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Лабораторная работа 4")
        self.root.geometry("840x520")

        self.task_var = tk.StringVar(value="2")

        self.target_phi_var = tk.DoubleVar(value=0.2)
        self.left_var = tk.DoubleVar(value=0.0)
        self.right_var = tk.DoubleVar(value=2.0)
        self.eps_var = tk.DoubleVar(value=1e-6)
        self.integral_steps_var = tk.IntVar(value=400)
        self.max_iterations_var = tk.IntVar(value=100)

        self.result_var = tk.StringVar(value="")

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
        if task == "2":
            self._render_task_2()
        else:
            self._render_placeholder(task)

    def _render_placeholder(self, task_id: str) -> None:
        frame = ttk.Frame(self.content)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame,
            text=(
                f"Задача {task_id} пока не реализована.\n"
                "Выберите задачу 2 для текущей реализации."
            ),
            font=("Segoe UI", 14),
            justify="center",
            anchor="center",
        ).place(relx=0.5, rely=0.5, anchor="center")

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


def main() -> None:
    root = tk.Tk()
    Lab04App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
