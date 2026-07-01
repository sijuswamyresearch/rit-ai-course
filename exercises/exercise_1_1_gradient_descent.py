"""
Exercise 1.1: Gradient Descent Visualization
Objective: Implement gradient descent from scratch and visualize optimization paths.

Requirements:
- Implement gradient descent for a simple quadratic function
- Plot the optimization path on a contour map
- Experiment with different learning rates
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Define the quadratic function: f(x, y) = x^2 + 2y^2
def quadratic_function(x, y):
    """Simple quadratic function to optimize."""
    return x**2 + 2*y**2

def gradient(x, y):
    """Compute the gradient of the quadratic function."""
    dx = 2*x
    dy = 4*y
    return np.array([dx, dy])

def gradient_descent(start_point, learning_rate, n_iterations, tolerance=1e-6):
    """
    Implement gradient descent algorithm.
    
    Parameters:
    - start_point: numpy array [x, y] - starting point
    - learning_rate: float - step size
    - n_iterations: int - maximum number of iterations
    - tolerance: float - convergence threshold
    
    Returns:
    - path: list of points visited during optimization
    - converged: bool - whether the algorithm converged
    """
    current_point = np.array(start_point, dtype=float)
    path = [current_point.copy()]
    converged = False
    
    for i in range(n_iterations):
        # Compute gradient
        grad = gradient(current_point[0], current_point[1])
        
        # Update point
        new_point = current_point - learning_rate * grad
        
        # Check convergence
        if np.linalg.norm(new_point - current_point) < tolerance:
            converged = True
            path.append(new_point.copy())
            break
        
        current_point = new_point
        path.append(current_point.copy())
    
    return np.array(path), converged

def plot_optimization_paths():
    """Visualize optimization paths for different learning rates."""
    
    # Create grid for contour plot
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = quadratic_function(X, Y)
    
    # Different learning rates to experiment with
    learning_rates = [0.01, 0.1, 0.3, 0.5]
    start_point = [2.5, 2.5]
    n_iterations = 50
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    for idx, lr in enumerate(learning_rates):
        ax = axes[idx]
        
        # Run gradient descent
        path, converged = gradient_descent(start_point, lr, n_iterations)
        
        # Plot contour
        contour = ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8)
        
        # Plot optimization path
        ax.plot(path[:, 0], path[:, 1], 'r-o', markersize=4, linewidth=2, 
                label=f'LR={lr}')
        
        # Mark start and end points
        ax.plot(start_point[0], start_point[1], 'go', markersize=10, label='Start')
        ax.plot(path[-1, 0], path[-1, 1], 'b*', markersize=15, label='End')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Gradient Descent (LR={lr})\nConverged: {converged}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gradient_descent_paths.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 3D visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    
    # Plot optimization path for medium learning rate
    path, _ = gradient_descent(start_point, 0.1, n_iterations)
    path_z = quadratic_function(path[:, 0], path[:, 1])
    ax.plot(path[:, 0], path[:, 1], path_z, 'r-o', linewidth=3, markersize=6)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.set_title('3D Visualization of Gradient Descent')
    
    plt.savefig('gradient_descent_3d.png', dpi=150, bbox_inches='tight')
    plt.show()

def experiment_with_learning_rates():
    """Experiment with different learning rates and analyze convergence."""
    
    learning_rates = np.logspace(-3, 0, 50)  # From 0.001 to 1
    start_point = [2.5, 2.5]
    n_iterations = 100
    
    final_values = []
    iterations_needed = []
    
    for lr in learning_rates:
        path, converged = gradient_descent(start_point, lr, n_iterations)
        final_values.append(quadratic_function(path[-1, 0], path[-1, 1]))
        iterations_needed.append(len(path))
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.semilogx(learning_rates, final_values, 'b-', linewidth=2)
    ax1.set_xlabel('Learning Rate (log scale)')
    ax1.set_ylabel('Final Function Value')
    ax1.set_title('Final Value vs Learning Rate')
    ax1.grid(True, alpha=0.3)
    
    ax2.semilogx(learning_rates, iterations_needed, 'r-', linewidth=2)
    ax2.set_xlabel('Learning Rate (log scale)')
    ax2.set_ylabel('Iterations to Converge')
    ax2.set_title('Iterations Needed vs Learning Rate')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_rate_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Exercise 1.1: Gradient Descent Visualization")
    print("=" * 50)
    
    print("\n1. Plotting optimization paths for different learning rates...")
    plot_optimization_paths()
    
    print("\n2. Analyzing convergence for various learning rates...")
    experiment_with_learning_rates()
    
    print("\nObservations:")
    print("- Too small learning rate: Slow convergence")
    print("- Optimal learning rate: Fast convergence")
    print("- Too large learning rate: May diverge or oscillate")
    print("\nComplete the exercise by writing a brief report on your observations.")