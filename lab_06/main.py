import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ======================================================
# МЕТОД ПРОГОНКИ
# ======================================================

def thomas_algorithm(a, b, c, d):
    n = len(d)

    alpha = np.zeros(n)
    beta = np.zeros(n)

    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]

    for i in range(1, n):
        denom = a[i] * alpha[i - 1] + b[i]

        if i < n - 1:
            alpha[i] = -c[i] / denom

        beta[i] = (d[i] - a[i] * beta[i - 1]) / denom

    x = np.zeros(n)
    x[-1] = beta[-1]

    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    return x


# ======================================================
# ВКЛАДКА 1
# ======================================================

class DifferentiationFrame(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent)

        ttk.Label(
            self,
            text="Введите значения y через пробел"
        ).pack(anchor="w", padx=10, pady=5)

        self.entry = ttk.Entry(self, width=70)
        self.entry.pack(fill="x", padx=10)

        self.entry.insert(
            0,
            "0.571 0.889 1.091 1.231 1.333 1.412"
        )

        ttk.Button(
            self,
            text="Вычислить",
            command=self.calculate
        ).pack(pady=10)

        cols = (
            "x",
            "y",
            "one",
            "central",
            "runge",
            "align",
            "second"
        )

        self.tree = ttk.Treeview(
            self,
            columns=cols,
            show="headings",
            height=8
        )

        names = [
            "x",
            "y",
            "Одност.",
            "Центр.",
            "Рунге",
            "Выравн.",
            "2-я"
        ]

        for c, n in zip(cols, names):
            self.tree.heading(c, text=n)
            self.tree.column(c, width=100)

        self.tree.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5
        )

        self.fig = Figure(figsize=(7, 4))
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            self
        )

        self.canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

    def calculate(self):

        try:

            y = np.array(
                list(map(float,
                         self.entry.get().split()))
            )

            n = len(y)
            h = 1.0

            x = np.arange(1, n + 1)

            one = np.full(n, np.nan)
            central = np.full(n, np.nan)
            runge = np.full(n, np.nan)
            align = np.full(n, np.nan)
            second = np.full(n, np.nan)

            for i in range(n - 1):
                one[i] = (y[i + 1] - y[i]) / h

            for i in range(1, n - 1):
                central[i] = (
                    y[i + 1] - y[i - 1]
                ) / (2 * h)

            for i in range(n - 2):
                Dh = (y[i + 1] - y[i]) / h
                D2h = (y[i + 2] - y[i]) / (2 * h)

                runge[i] = 2 * Dh - D2h

            eta = 1 / y

            for i in range(1, n - 1):
                deta = (
                    eta[i + 1] - eta[i - 1]
                ) / (2 * h)

                align[i] = -y[i] ** 2 * deta

            for i in range(1, n - 1):
                second[i] = (
                    y[i + 1]
                    - 2 * y[i]
                    + y[i - 1]
                ) / h ** 2

            for row in self.tree.get_children():
                self.tree.delete(row)

            for i in range(n):

                self.tree.insert(
                    "",
                    "end",
                    values=(
                        x[i],
                        round(y[i], 6),
                        self.f(one[i]),
                        self.f(central[i]),
                        self.f(runge[i]),
                        self.f(align[i]),
                        self.f(second[i])
                    )
                )

            self.ax.clear()

            self.ax.plot(
                x,
                y,
                marker="o",
                label="y"
            )

            self.ax.plot(
                x,
                central,
                marker="s",
                label="y'"
            )

            self.ax.plot(
                x,
                second,
                marker="^",
                label="y''"
            )

            self.ax.grid(True)
            self.ax.legend()

            self.canvas.draw()

        except Exception as e:
            messagebox.showerror(
                "Ошибка",
                str(e)
            )

    def f(self, value):
        if np.isnan(value):
            return ""
        return round(value, 6)


# ======================================================
# ВКЛАДКА 2
# ======================================================

class ODEFrame(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent)

        param = ttk.LabelFrame(
            self,
            text="Параметры задачи"
        )

        param.pack(
            fill="x",
            padx=10,
            pady=5
        )

        ttk.Label(param,
                  text="Количество узлов N").grid(
            row=0,
            column=0,
            padx=5,
            pady=5
        )

        self.n_entry = ttk.Entry(param, width=10)
        self.n_entry.grid(row=0, column=1)
        self.n_entry.insert(0, "20")

        ttk.Button(
            param,
            text="Решить",
            command=self.solve
        ).grid(
            row=0,
            column=2,
            padx=10
        )

        self.tree = ttk.Treeview(
            self,
            columns=("x", "u"),
            show="headings",
            height=12
        )

        self.tree.heading("x", text="x")
        self.tree.heading("u", text="u(x)")

        self.tree.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5
        )

        self.fig = Figure(figsize=(7, 4))
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            self
        )

        self.canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

    def solve(self):

        try:

            N = int(self.n_entry.get())

            # При необходимости измените значения
            alpha = 0.0
            beta = 1.0
            gamma = 0.0

            h = 1.0 / N

            x = np.linspace(0.0, 1.0, N + 1)

            m = N + 1

            A = np.zeros(m)
            B = np.zeros(m)
            C = np.zeros(m)
            D = np.zeros(m)

            # =====================================
            # Левая граница:
            # u'(0) = alpha
            #
            # (-u0 + u1)/h = alpha
            # =====================================

            B[0] = -1.0 / h
            C[0] = 1.0 / h
            D[0] = alpha

            # =====================================
            # Внутренние узлы
            #
            # u'' - 2x²u' + 4u = 2x + e^{-x}
            #
            # u'' = (u[i+1]-2u[i]+u[i-1])/h²
            # u'  = (u[i+1]-u[i-1])/(2h)
            # =====================================

            for i in range(1, N):

                xi = x[i]

                rhs = 2 * xi + np.exp(-xi)

                A[i] = (
                    1.0 / h**2
                    + xi**2 / h
                )

                B[i] = (
                    -2.0 / h**2
                    + 4.0
                )

                C[i] = (
                    1.0 / h**2
                    - xi**2 / h
                )

                D[i] = rhs

            # =====================================
            # Правая граница:
            #
            # u'(1)=βu(1)+γ
            #
            # (uN-uN-1)/h = βuN + γ
            # =====================================

            A[N] = -1.0 / h

            B[N] = (
                1.0 / h
                - beta
            )

            D[N] = gamma

            u = thomas_algorithm(
                A,
                B,
                C,
                D
            )

            for row in self.tree.get_children():
                self.tree.delete(row)

            for xi, ui in zip(x, u):

                self.tree.insert(
                    "",
                    "end",
                    values=(
                        round(xi, 6),
                        round(ui, 10)
                    )
                )

            self.ax.clear()

            self.ax.plot(
                x,
                u,
                marker="o",
                label="u(x)"
            )

            self.ax.set_title(
                "Решение краевой задачи"
            )

            self.ax.set_xlabel("x")
            self.ax.set_ylabel("u(x)")
            self.ax.grid(True)
            self.ax.legend()

            self.canvas.draw()

        except Exception as e:
            messagebox.showerror(
                "Ошибка",
                str(e)
            )


# ======================================================
# ГЛАВНОЕ ОКНО
# ======================================================

root = tk.Tk()

root.title(
    "ЛР №6. Численное дифференцирование и метод прогонки"
)

root.geometry("1200x800")

notebook = ttk.Notebook(root)

tab1 = DifferentiationFrame(notebook)
tab2 = ODEFrame(notebook)

notebook.add(
    tab1,
    text="Задание 1"
)

notebook.add(
    tab2,
    text="Задание 2"
)

notebook.pack(
    fill="both",
    expand=True
)

root.mainloop()
