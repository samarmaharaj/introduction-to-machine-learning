import numpy as np
from matplotlib.widgets import Slider
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import matplotlib.pyplot as plt

class LinearRegressionPlayground:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Linear Regression Playground")
        self.root.geometry("800x600")
        
        # Initialize parameters
        self.slope = 1.0
        self.intercept = 0.0
        self.noise_level = 0.1
        self.n_points = 50
        
        # Generate initial data
        self.x = np.linspace(-5, 5, self.n_points)
        self.generate_data()
        
        self.setup_ui()
        
    def generate_data(self):
        """Generate sample data with noise"""
        true_y = 2 * self.x + 1  # True relationship
        noise = np.random.normal(0, self.noise_level, len(self.x))
        self.y = true_y + noise
        
    def linear_regression(self, x, y):
        """Calculate linear regression coefficients"""
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        return slope, intercept
    
    def setup_ui(self):
        """Setup the user interface"""
        # Create main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.suptitle("Linear Regression Playground")
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Create control frame
        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        # Slope slider
        tk.Label(control_frame, text="Manual Slope:").grid(row=0, column=0, sticky="w")
        self.slope_var = tk.DoubleVar(value=self.slope)
        slope_scale = tk.Scale(control_frame, from_=-5, to=5, resolution=0.1, 
                              orient=tk.HORIZONTAL, variable=self.slope_var,
                              command=self.update_manual_plot)
        slope_scale.grid(row=0, column=1, sticky="ew")
        
        # Intercept slider
        tk.Label(control_frame, text="Manual Intercept:").grid(row=1, column=0, sticky="w")
        self.intercept_var = tk.DoubleVar(value=self.intercept)
        intercept_scale = tk.Scale(control_frame, from_=-5, to=5, resolution=0.1,
                                  orient=tk.HORIZONTAL, variable=self.intercept_var,
                                  command=self.update_manual_plot)
        intercept_scale.grid(row=1, column=1, sticky="ew")
        
        # Noise level slider
        tk.Label(control_frame, text="Noise Level:").grid(row=2, column=0, sticky="w")
        self.noise_var = tk.DoubleVar(value=self.noise_level)
        noise_scale = tk.Scale(control_frame, from_=0, to=2, resolution=0.05,
                              orient=tk.HORIZONTAL, variable=self.noise_var,
                              command=self.update_data)
        noise_scale.grid(row=2, column=1, sticky="ew")
        
        # Number of points slider
        tk.Label(control_frame, text="Number of Points:").grid(row=3, column=0, sticky="w")
        self.points_var = tk.IntVar(value=self.n_points)
        points_scale = tk.Scale(control_frame, from_=10, to=200, resolution=10,
                               orient=tk.HORIZONTAL, variable=self.points_var,
                               command=self.update_data)
        points_scale.grid(row=3, column=1, sticky="ew")
        
        # Buttons
        button_frame = tk.Frame(control_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        tk.Button(button_frame, text="Generate New Data", 
                 command=self.generate_new_data).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Reset Parameters", 
                 command=self.reset_parameters).pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        control_frame.columnconfigure(1, weight=1)
        
        # Initial plot
        self.update_plots()
        
    def update_manual_plot(self, event=None):
        """Update the manual regression plot"""
        self.slope = self.slope_var.get()
        self.intercept = self.intercept_var.get()
        self.update_plots()
        
    def update_data(self, event=None):
        """Update data based on noise and points parameters"""
        self.noise_level = self.noise_var.get()
        self.n_points = self.points_var.get()
        self.x = np.linspace(-5, 5, self.n_points)
        self.generate_data()
        self.update_plots()
        
    def generate_new_data(self):
        """Generate completely new random data"""
        self.generate_data()
        self.update_plots()
        
    def reset_parameters(self):
        """Reset all parameters to default"""
        self.slope_var.set(1.0)
        self.intercept_var.set(0.0)
        self.noise_var.set(0.1)
        self.points_var.set(50)
        self.update_data()
        
    def update_plots(self):
        """Update both plots"""
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot 1: Manual regression
        self.ax1.scatter(self.x, self.y, alpha=0.6, label='Data points')
        
        # Manual regression line
        y_manual = self.slope * self.x + self.intercept
        self.ax1.plot(self.x, y_manual, 'r-', linewidth=2, 
                     label=f'Manual: y = {self.slope:.2f}x + {self.intercept:.2f}')
        
        # Calculate and display MSE for manual line
        mse_manual = np.mean((self.y - y_manual) ** 2)
        self.ax1.set_title(f'Manual Regression (MSE: {mse_manual:.3f})')
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Plot 2: Optimal regression
        self.ax2.scatter(self.x, self.y, alpha=0.6, label='Data points')
        
        # Calculate optimal regression
        opt_slope, opt_intercept = self.linear_regression(self.x, self.y)
        y_optimal = opt_slope * self.x + opt_intercept
        self.ax2.plot(self.x, y_optimal, 'g-', linewidth=2,
                     label=f'Optimal: y = {opt_slope:.2f}x + {opt_intercept:.2f}')
        
        # Calculate and display MSE for optimal line
        mse_optimal = np.mean((self.y - y_optimal) ** 2)
        self.ax2.set_title(f'Optimal Regression (MSE: {mse_optimal:.3f})')
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Y')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        # Update canvas
        self.canvas.draw()
        
    def run(self):
        """Start the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = LinearRegressionPlayground()
    app.run()