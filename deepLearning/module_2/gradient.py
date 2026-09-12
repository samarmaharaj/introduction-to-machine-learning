import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

X = [0.5, 2.5]
Y = [0.2, 0.9]

def f(w, x, b):
    return 1 / (1 + np.exp(-np.clip(w*x + b, -500, 500)))

def loss(w, x, y, b):
    error = 0.0
    for sample_x, sample_y in zip(x, y):
        error += 0.5 * (f(w, sample_x, b) - sample_y) ** 2
    return error

def grad_b(w, x, y, b):
    return (f(w, x, b) - y) * f(w, x, b) * (1 - f(w, x, b))

def grad_w(w, x, y, b):
    return (f(w, x, b) - y) * f(w, x, b) * (1 - f(w, x, b)) * x

def do_gradient_descent():
    # Adjusted limits to match the provided image reference [-6, 6]
    w, b, eta, max_iter = 0.5, 0.0, 0.4, 1000
    visual_step = 20 
    weights, biases, errors = [], [], []
    
    plt.ion()
    figure = plt.figure(figsize=(14, 6))
    surface_axis = figure.add_subplot(121, projection="3d")
    prediction_axis = figure.add_subplot(122)

    # Creating a mesh matching the image limits [-6, 6]
    w_vals = np.linspace(-6.0, 6.0, 60)
    b_vals = np.linspace(-6.0, 6.0, 60)
    w_grid, b_grid = np.meshgrid(w_vals, b_vals)
    loss_grid = np.array([[loss(weight, X, Y, bias) for weight in w_row] for bias, w_row in zip(b_vals, w_grid)])
    
    # 1. MATCHING COLORMAP: Using 'coolwarm' to get the red-white-blue visual split
    surface_axis.plot_surface(w_grid, b_grid, loss_grid, cmap=cm.coolwarm, alpha=0.5, rcount=50, ccount=50)
    surface_axis.set_xlabel("w")
    surface_axis.set_ylabel("b")
    surface_axis.set_zlabel("error")
    surface_axis.set_title("Gradient descent on the error surface")
    
    # Lock the Z axis to match the [-1.0, 1.0] scale from your image
    z_floor = -1.0
    surface_axis.set_zlim(z_floor, 1.0)
    surface_axis.set_xlim(-6, 6)
    surface_axis.set_ylim(-6, 6)

    # Line handles for both the true surface path (red) and floor projection (black)
    path_line, = surface_axis.plot([], [], [], color="red", linewidth=2, linestyle="dashed")
    floor_line, = surface_axis.plot([], [], [], color="black", linewidth=1.5, alpha=0.7)
    
    current_dot = surface_axis.scatter([], [], [], color="red", s=30)
    floor_dot = surface_axis.scatter([], [], [], color="black", s=30)

    for i in range(max_iter):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(w, x, y, b)
            db += grad_b(w, x, y, b)
        
        w = w - eta * dw
        b = b - eta * db
        
        weights.append(w)
        biases.append(b)
        errors.append(loss(w, X, Y, b))

        if i % visual_step == 0 or i == max_iter - 1:
            # Update the 3D red path on the mesh surface
            path_line.set_data(weights, biases)
            path_line.set_3d_properties(errors)
            
            # 2. ADDING THE SHADOW: Project the exact same coordinates onto the floor (z_floor)
            floor_line.set_data(weights, biases)
            floor_line.set_3d_properties([z_floor] * len(weights))
            
            # Update tracking points
            current_dot.remove()
            floor_dot.remove()
            current_dot = surface_axis.scatter(w, b, errors[-1], color="red", s=30)
            floor_dot = surface_axis.scatter(w, b, z_floor, color="black", s=30)

            # Update prediction axis
            prediction_axis.clear()
            prediction_axis.scatter(X, Y, color="red", label="target", s=80, zorder=5)
            x_smooth = np.linspace(-1, 4, 100)
            prediction_axis.plot(x_smooth, [f(w, sx, b) for sx in x_smooth], color="blue", label="prediction")
            prediction_axis.set_xlabel("x")
            prediction_axis.set_ylabel("output")
            prediction_axis.set_ylim(-0.1, 1.1)
            prediction_axis.set_title(f"Iteration {i + 1} | Loss = {errors[-1]:.5f}")
            prediction_axis.legend()
            
            figure.tight_layout()
            plt.pause(0.001)

    print("\n--- FINAL VERIFICATION ---")
    print(f"Final Weights (w): {w}")
    print(f"Final Bias (b): {b}")
    print(f"Final Loss Value: {errors[-1]}")
    print(f"Prediction for X=0.5: {f(w, 0.5, b)} (Target: 0.2)")
    print(f"Prediction for X=2.5: {f(w, 2.5, b)} (Target: 0.9)")

    plt.ioff()
    plt.show()

do_gradient_descent()
