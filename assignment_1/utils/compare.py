"""
Utility functions for comparing spline interpolation methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from assignment_1.linear.spline import LinearSpline
from assignment_1.quadratic.spline import QuadraticSpline


def compare_methods(x, y, eval_point=None, show_derivatives=True, show_plot=True, 
                  save_plot=False, output_dir=None, timestamp=None):
    """
    Compare linear and quadratic spline interpolation on the same data.
    
    Args:
        x (array-like): X values (independent variable).
        y (array-like): Y values (dependent variable).
        eval_point (float, optional): Point at which to evaluate and compare the splines.
        show_derivatives (bool): Whether to show derivatives in the plot.
        show_plot (bool): Whether to display the plot.
        save_plot (bool): Whether to save the plot as an image file.
        output_dir (str): Directory to save the plot (if save_plot is True).
        timestamp (str): Timestamp string to include in the filename.
        
    Returns:
        dict: Dictionary of results comparing the methods.
    """
    # Ensure x and y are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Create spline interpolators
    linear_spline = LinearSpline(x, y)
    quadratic_spline = QuadraticSpline(x, y)
    
    # Create points for plotting
    x_plot = np.linspace(x[0], x[-1], 1000)
    y_linear = linear_spline(x_plot)
    y_quadratic = quadratic_spline(x_plot)
    
    # Calculate derivatives if needed
    if show_derivatives:
        # Linear spline derivative (constant within intervals)
        slopes = np.zeros_like(x_plot)
        for i, xi in enumerate(x_plot):
            idx = np.searchsorted(x, xi, side='right') - 1
            if idx >= 0 and idx < len(x) - 1:
                slopes[i] = (y[idx+1] - y[idx]) / (x[idx+1] - x[idx])
                
        # Quadratic spline derivative (analytic)
        deriv_quad = quadratic_spline.derivative(x_plot)
    
    # Evaluate at the specific point if provided
    results = {}
    if eval_point is not None and eval_point >= x[0] and eval_point <= x[-1]:
        y_linear_eval = linear_spline(eval_point)
        y_quadratic_eval = quadratic_spline(eval_point)
        
        results['eval_point'] = eval_point
        results['linear_value'] = y_linear_eval
        results['quadratic_value'] = y_quadratic_eval
        
        if show_derivatives:
            idx = np.searchsorted(x, eval_point, side='right') - 1
            deriv_linear_eval = (y[idx+1] - y[idx]) / (x[idx+1] - x[idx])
            deriv_quadratic_eval = quadratic_spline.derivative(eval_point)
            
            results['linear_derivative'] = deriv_linear_eval
            results['quadratic_derivative'] = deriv_quadratic_eval
    
    # Plot the results
    if show_plot or save_plot:
        fig, axes = plt.subplots(2 if show_derivatives else 1, 1, figsize=(12, 10 if show_derivatives else 6))
        
        # Function plot
        ax1 = axes[0] if show_derivatives else axes
        ax1.plot(x_plot, y_linear, 'b-', label='Linear Spline')
        ax1.plot(x_plot, y_quadratic, 'r-', label='Quadratic Spline')
        ax1.plot(x, y, 'ko', label='Data Points')
        
        if eval_point is not None and eval_point >= x[0] and eval_point <= x[-1]:
            ax1.plot(eval_point, y_linear_eval, 'b*', markersize=10)
            ax1.plot(eval_point, y_quadratic_eval, 'r*', markersize=10)
            ax1.axvline(x=eval_point, color='gray', linestyle='--')
            
            ax1.annotate(f'Linear: {y_linear_eval:.2f}', 
                        xy=(eval_point, y_linear_eval),
                        xytext=(eval_point + 0.5, y_linear_eval + 0.2),
                        arrowprops=dict(facecolor='blue', shrink=0.05))
                        
            ax1.annotate(f'Quadratic: {y_quadratic_eval:.2f}', 
                        xy=(eval_point, y_quadratic_eval),
                        xytext=(eval_point + 0.5, y_quadratic_eval - 0.2),
                        arrowprops=dict(facecolor='red', shrink=0.05))
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Comparison of Spline Methods')
        ax1.grid(True)
        ax1.legend()
        
        # Derivative plot
        if show_derivatives:
            ax2 = axes[1]
            ax2.plot(x_plot, slopes, 'b-', label='Linear Spline Derivative')
            ax2.plot(x_plot, deriv_quad, 'r-', label='Quadratic Spline Derivative')
            
            if eval_point is not None and eval_point >= x[0] and eval_point <= x[-1]:
                ax2.plot(eval_point, deriv_linear_eval, 'b*', markersize=10)
                ax2.plot(eval_point, deriv_quadratic_eval, 'r*', markersize=10)
                ax2.axvline(x=eval_point, color='gray', linestyle='--')
                
                ax2.annotate(f'Linear derivative: {deriv_linear_eval:.2f}', 
                            xy=(eval_point, deriv_linear_eval),
                            xytext=(eval_point + 0.5, deriv_linear_eval + 0.2),
                            arrowprops=dict(facecolor='blue', shrink=0.05))
                            
                ax2.annotate(f'Quadratic derivative: {deriv_quadratic_eval:.2f}', 
                            xy=(eval_point, deriv_quadratic_eval),
                            xytext=(eval_point + 0.5, deriv_quadratic_eval - 0.2),
                            arrowprops=dict(facecolor='red', shrink=0.05))
            
            ax2.set_xlabel('x')
            ax2.set_ylabel("Derivative")
            ax2.set_title('Comparison of Derivatives')
            ax2.grid(True)
            ax2.legend()
        
        plt.tight_layout()
        
        # Save the plot if requested
        if save_plot and output_dir:
            if timestamp is None:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            filename = f"spline_comparison_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300)
            print(f"Spline comparison plot saved to: {filepath}")
        
        if show_plot:
            plt.show()
    
    return results


def compare_rocket_example(eval_point=16):
    """
    Compare different interpolation methods using the rocket example.
    
    Args:
        eval_point (float): Point to evaluate (time in seconds).
    """
    # Rocket data: time and velocity
    t = np.array([0, 10, 15, 20, 22.5, 30])
    v = np.array([0, 227.04, 362.78, 517.35, 602.97, 901.67])
    
    # Compare splines on the rocket data
    compare_methods(t, v, eval_point, 
                   title=f"Comparison of Splines for Rocket Velocity (t = {eval_point}s)",
                   show_derivatives=True)


if __name__ == "__main__":
    compare_rocket_example() 