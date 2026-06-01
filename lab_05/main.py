from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

from pathlib import Path
from typing import Callable

from utils import (
    integrate_trapezoid,
    integrate_simpson,
    integrate_gauss,
    load_tabular_2d,
    double_integral_iterated,
)


def analytic_abs_integral(k: float) -> float:
    return 2.0 / (k + 1.0)


class Lab05App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Лабораторная 5 — Численное интегрирование")
        self.root.geometry("760x520")

        self.tab = ttk.Notebook(self.root)
        self.tab.pack(fill=tk.BOTH, expand=True)

        self._build_task1()
        self._build_task2()

    def _build_task1(self) -> None:
        frame = ttk.Frame(self.tab, padding=10)
        self.tab.add(frame, text="Задание 1")

        ttk.Label(frame, text="Интеграл ∫_{-1}^1 |x|^k dx", font=(None, 12, "bold")).pack(anchor="w")

        form = ttk.Frame(frame)
        form.pack(anchor="w", pady=6)

        ttk.Label(form, text="k:").grid(row=0, column=0, sticky="w")
        self.k_var = tk.DoubleVar(value=1.0)
        ttk.Entry(form, textvariable=self.k_var, width=8).grid(row=0, column=1, padx=6)

        ttk.Label(form, text="Nodes (for composite rules, total nodes):").grid(row=1, column=0, sticky="w")
        self.nodes_var = tk.IntVar(value=3)
        ttk.Entry(form, textvariable=self.nodes_var, width=8).grid(row=1, column=1, padx=6)

        ttk.Button(frame, text="Compute and Compare", command=self._compute_task1).pack(pady=10)

        self.task1_result = tk.Text(frame, height=12)
        self.task1_result.pack(fill=tk.BOTH, expand=True)

    def _compute_task1(self) -> None:
        try:
            k = float(self.k_var.get())
            nodes = int(self.nodes_var.get())
            a, b = -1.0, 1.0

            # prepare function
            def f(x: float) -> float:
                return abs(x) ** k

            # composite trapezoid with nodes -> n = nodes-1
            n_trap = max(1, nodes - 1)
            val_trap = integrate_trapezoid(f, a, b, n_trap)

            # Simpson: requires even n; nodes -> n = nodes-1 must be even
            n_simp = max(2, nodes - 1)
            if n_simp % 2 == 1:
                n_simp += 1
            val_simp = integrate_simpson(f, a, b, n_simp)

            # Gauss with 3 nodes if nodes >=3 else use nodes
            nodes_gauss = 3 if nodes >= 3 else nodes
            val_gauss = integrate_gauss(f, a, b, nodes_gauss)

            analytic = analytic_abs_integral(k)

            out = []
            out.append(f"Analytic: {analytic:.12f}\n")
            out.append(f"Trapezoid (nodes={nodes}): {val_trap:.12f}, err={abs(val_trap-analytic):.3e}\n")
            out.append(f"Simpson (nodes={n_simp+1}): {val_simp:.12f}, err={abs(val_simp-analytic):.3e}\n")
            out.append(f"Gauss ({nodes_gauss} nodes): {val_gauss:.12f}, err={abs(val_gauss-analytic):.3e}\n")

            self.task1_result.delete("1.0", tk.END)
            self.task1_result.insert(tk.END, "".join(out))
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _build_task2(self) -> None:
        frame = ttk.Frame(self.tab, padding=10)
        self.tab.add(frame, text="Задание 2")

        left = ttk.Frame(frame)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=6)

        ttk.Label(left, text="Double integral over region G", font=(None, 12, "bold")).pack()

        self.alpha_var = tk.DoubleVar(value=0.25)
        self.beta_var = tk.DoubleVar(value=0.25)
        self.a_var = tk.DoubleVar(value=0.0)
        self.b_var = tk.DoubleVar(value=2.0)
        self.eps_var = tk.DoubleVar(value=1e-6)

        self._add_entry(left, "alpha (φ(x)=alpha*x^2):", self.alpha_var)
        self._add_entry(left, "beta  (ψ(x)=beta*x^2):", self.beta_var)
        self._add_entry(left, "a (left):", self.a_var)
        self._add_entry(left, "b (right):", self.b_var)
        self._add_entry(left, "eps:", self.eps_var)

        ttk.Button(left, text="Load table & Compute", command=self._compute_task2).pack(pady=8)

        right = ttk.Frame(frame)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=6)

        self.task2_result = tk.Text(right, height=18)
        self.task2_result.pack(fill=tk.BOTH, expand=True)

    def _add_entry(self, parent, label, var):
        frm = ttk.Frame(parent)
        frm.pack(anchor="w", pady=4)
        ttk.Label(frm, text=label).pack(side=tk.LEFT)
        ttk.Entry(frm, textvariable=var, width=12).pack(side=tk.LEFT, padx=6)

    def _compute_task2(self) -> None:
        try:
            alpha = float(self.alpha_var.get())
            beta = float(self.beta_var.get())
            a = float(self.a_var.get())
            b = float(self.b_var.get())
            eps = float(self.eps_var.get())

            dataset_path = Path(__file__).parent / "dataset.txt"
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")

            dataset = load_tabular_2d(str(dataset_path))

            def phi(x: float) -> float:
                return alpha * x * x

            def psi(x: float) -> float:
                return beta * x * x

            value = double_integral_iterated(dataset, phi, psi, a, b, eps=eps)

            self.task2_result.delete("1.0", tk.END)
            self.task2_result.insert(tk.END, f"Double integral value I = {value:.12f}\n")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))


def main() -> None:
    root = tk.Tk()
    Lab05App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
