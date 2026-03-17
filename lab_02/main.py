import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from utils import load_3d_data, interpolate_3d


class Interpolation3DApp:
    def __init__(self, root):
        self.root = root
        self.root.title("3D интерполяция")
        self.root.geometry("650x700")

        # Load data
        data_file = Path(__file__).parent / "26-02-2026-Исход_данные__Лаб_работы__2.txt"
        try:
            self.dataset = load_3d_data(str(data_file))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")
            self.dataset = []

        # Variables for interpolation methods
        self.x_method = tk.StringVar(value="polynomial")
        self.y_method = tk.StringVar(value="polynomial")
        self.z_method = tk.StringVar(value="polynomial")

        # Variables for polynomial degrees
        self.x_degree = tk.IntVar(value=3)
        self.y_degree = tk.IntVar(value=3)
        self.z_degree = tk.IntVar(value=3)

        # Variables for coordinates
        self.x_coord = tk.DoubleVar(value=1.5)
        self.y_coord = tk.DoubleVar(value=1.5)
        self.z_coord = tk.DoubleVar(value=1.5)

        # Result variable
        self.result_text = tk.StringVar(value="")

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure root window to expand
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Title
        title_label = ttk.Label(
            main_frame, text="3D Интерполяция", font=("Arial", 14, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=10)

        # X dimension section
        self.create_dimension_section(main_frame, 1, "X", self.x_method, self.x_degree)

        # Y dimension section
        self.create_dimension_section(main_frame, 2, "Y", self.y_method, self.y_degree)

        # Z dimension section
        self.create_dimension_section(main_frame, 3, "Z", self.z_method, self.z_degree)

        # Separator
        separator = ttk.Separator(main_frame, orient="horizontal")
        separator.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=15)

        # Coordinates input section
        coord_frame = ttk.LabelFrame(
            main_frame, text="Координаты для интерполяции", padding="10"
        )
        coord_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(coord_frame, text="X:").grid(row=0, column=0, sticky=tk.W, padx=5)
        x_entry = ttk.Entry(coord_frame, textvariable=self.x_coord, width=15)
        x_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(coord_frame, text="Y:").grid(row=1, column=0, sticky=tk.W, padx=5)
        y_entry = ttk.Entry(coord_frame, textvariable=self.y_coord, width=15)
        y_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(coord_frame, text="Z:").grid(row=2, column=0, sticky=tk.W, padx=5)
        z_entry = ttk.Entry(coord_frame, textvariable=self.z_coord, width=15)
        z_entry.grid(row=2, column=1, padx=5, pady=5)

        # Interpolate button
        interpolate_btn = ttk.Button(
            main_frame, text="Интерполировать", command=self.perform_interpolation
        )
        interpolate_btn.grid(row=6, column=0, columnspan=3, pady=15)

        # Result section
        result_frame = ttk.LabelFrame(main_frame, text="Результат", padding="10")
        result_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        result_label = ttk.Label(
            result_frame,
            textvariable=self.result_text,
            font=("Arial", 12),
            foreground="blue",
        )
        result_label.grid(row=0, column=0, sticky=(tk.W, tk.E))

    def create_dimension_section(
        self, parent, row, dimension_name, method_var, degree_var
    ):
        # Create frame for this dimension
        dim_frame = ttk.LabelFrame(
            parent, text=f"Измерение {dimension_name}", padding="10"
        )
        dim_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        # Radio buttons for method selection
        ttk.Label(dim_frame, text="Метод:").grid(row=0, column=0, sticky=tk.W, padx=5)

        poly_radio = ttk.Radiobutton(
            dim_frame,
            text="Полиномиальная",
            variable=method_var,
            value="polynomial",
            command=lambda: self.toggle_degree_field(dimension_name, method_var),
        )
        poly_radio.grid(row=0, column=1, sticky=tk.W, padx=5)

        spline_radio = ttk.Radiobutton(
            dim_frame,
            text="Сплайн",
            variable=method_var,
            value="spline",
            command=lambda: self.toggle_degree_field(dimension_name, method_var),
        )
        spline_radio.grid(row=0, column=2, sticky=tk.W, padx=5)

        # Degree input (only for polynomial)
        ttk.Label(dim_frame, text="Степень:").grid(row=1, column=0, sticky=tk.W, padx=5)

        degree_entry = ttk.Entry(dim_frame, textvariable=degree_var, width=10)
        degree_entry.grid(row=1, column=1, padx=5, pady=5)

        # Store reference to degree entry for enabling/disabling
        setattr(self, f"{dimension_name.lower()}_degree_entry", degree_entry)

        # Initially enable/disable based on default method
        self.toggle_degree_field(dimension_name, method_var)

    def toggle_degree_field(self, dimension_name, method_var):
        degree_entry = getattr(self, f"{dimension_name.lower()}_degree_entry")
        if method_var.get() == "polynomial":
            degree_entry.config(state="normal")
        else:
            degree_entry.config(state="disabled")

    def perform_interpolation(self):
        try:
            if not self.dataset:
                messagebox.showerror("Error", "No data loaded")
                return

            # Get coordinates
            x = self.x_coord.get()
            y = self.y_coord.get()
            z = self.z_coord.get()

            # Perform interpolation
            result = interpolate_3d(
                self.dataset,
                (x, y, z),
                x_method=self.x_method.get(),
                y_method=self.y_method.get(),
                z_method=self.z_method.get(),
                x_degree=self.x_degree.get(),
                y_degree=self.y_degree.get(),
                z_degree=self.z_degree.get(),
            )

            # Display result
            self.result_text.set(f"u({x}, {y}, {z}) = {result:.6f}")

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Interpolation failed: {e}")


def main():
    root = tk.Tk()
    root.app = Interpolation3DApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
