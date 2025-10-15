"""
Spline2D Curve Module

This module implements 2D curve interpolation using parametric splines.
A 2D spline curve consists of two splines: x(t) and y(t), each interpolating 
the x and y coordinates respectively for the same t values.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

class QuadraticSpline:
    """
    Class for quadratic spline interpolation.
    
    Quadratic splines fit a quadratic polynomial between each pair of consecutive data points,
    maintaining first derivative continuity at interior points.
    """
    
    def __init__(self, t, values):
        """
        Initialize the quadratic spline interpolator with data points.
        
        Args:
            t (array-like): Parameter values (must be strictly increasing).
            values (array-like): Values to interpolate at each parameter value.
        """
        # Convert to numpy arrays
        self.t = np.asarray(t)
        self.values = np.asarray(values)
        
        # Verify that the data is sorted by t
        if not np.all(np.diff(self.t) > 0):
            raise ValueError("Parameter values must be strictly increasing")
            
        n = len(self.t) - 1  # Number of splines
        
        # Initialize coefficients a, b, c for each spline
        self.a = np.zeros(n)
        self.b = np.zeros(n)
        self.c = np.zeros(n)
        
        # Build the equation system
        # For n splines we have 3n coefficients and need 3n equations:
        # - 2n equations: each spline passes through 2 points
        # - (n-1) equations: first derivative continuity at interior points
        # - 1 equation: we assume the first spline is linear (a[0] = 0)
        
        # Matrix of the system and right-hand side vector
        A = np.zeros((3*n, 3*n))
        B = np.zeros(3*n)
        
        # Apply conditions that each spline passes through two consecutive points
        eq = 0
        for i in range(n):
            # First point: a_i*t_i^2 + b_i*t_i + c_i = values_i
            A[eq, 3*i] = self.t[i]**2
            A[eq, 3*i+1] = self.t[i]
            A[eq, 3*i+2] = 1
            B[eq] = self.values[i]
            eq += 1
            
            # Second point: a_i*t_{i+1}^2 + b_i*t_{i+1} + c_i = values_{i+1}
            A[eq, 3*i] = self.t[i+1]**2
            A[eq, 3*i+1] = self.t[i+1]
            A[eq, 3*i+2] = 1
            B[eq] = self.values[i+1]
            eq += 1
        
        # Apply first derivative continuity conditions at interior points
        for i in range(n-1):
            # 2*a_i*t_{i+1} + b_i = 2*a_{i+1}*t_{i+1} + b_{i+1}
            A[eq, 3*i] = 2*self.t[i+1]
            A[eq, 3*i+1] = 1
            A[eq, 3*(i+1)] = -2*self.t[i+1]
            A[eq, 3*(i+1)+1] = -1
            B[eq] = 0
            eq += 1
        
        # Apply the condition that the first spline is linear (a[0] = 0)
        A[eq, 0] = 1
        B[eq] = 0
        
        # Solve the system to find the coefficients
        coeffs = np.linalg.solve(A, B)
        
        # Extract coefficients a, b, c for each spline
        for i in range(n):
            self.a[i] = coeffs[3*i]
            self.b[i] = coeffs[3*i+1]
            self.c[i] = coeffs[3*i+2]
    
    def __call__(self, t_new):
        """
        Evaluate the quadratic spline at the given parameter values.
        
        Args:
            t_new (float or array-like): Parameter values where to evaluate the spline.
            
        Returns:
            float or ndarray: Interpolated values.
        """
        t_new = np.asarray(t_new)
        scalar_input = False
        if t_new.ndim == 0:
            t_new = t_new[np.newaxis]
            scalar_input = True
            
        values_new = np.zeros_like(t_new)
        
        for i, t_val in enumerate(t_new):
            if t_val < self.t[0] or t_val > self.t[-1]:
                raise ValueError(f"Parameter value {t_val} is outside the interpolation range [{self.t[0]}, {self.t[-1]}]")
                
            # Find the corresponding interval
            idx = np.searchsorted(self.t, t_val, side='right') - 1
            
            # Make sure idx is within bounds (could happen with numerical precision issues)
            idx = max(0, min(idx, len(self.a) - 1))
            
            # Apply the corresponding quadratic polynomial
            values_new[i] = self.a[idx] * t_val**2 + self.b[idx] * t_val + self.c[idx]
            
        return values_new[0] if scalar_input else values_new
    
    def derivative(self, t_new):
        """
        Calculate the first derivative of the quadratic spline at the given parameter values.
        
        Args:
            t_new (float or array-like): Parameter values where to evaluate the derivative.
            
        Returns:
            float or ndarray: Derivative values.
        """
        t_new = np.asarray(t_new)
        scalar_input = False
        if t_new.ndim == 0:
            t_new = t_new[np.newaxis]
            scalar_input = True
            
        deriv_new = np.zeros_like(t_new)
        
        for i, t_val in enumerate(t_new):
            if t_val < self.t[0] or t_val > self.t[-1]:
                raise ValueError(f"Parameter value {t_val} is outside the interpolation range [{self.t[0]}, {self.t[-1]}]")
                
            # Find the corresponding interval
            idx = np.searchsorted(self.t, t_val, side='right') - 1
            
            # Make sure idx is within bounds (could happen with numerical precision issues)
            idx = max(0, min(idx, len(self.a) - 1))
            
            # Derivative of the quadratic polynomial: 2*a*t + b
            deriv_new[i] = 2 * self.a[idx] * t_val + self.b[idx]
            
        return deriv_new[0] if scalar_input else deriv_new


class Spline2D:
    """
    Class for 2D curve interpolation using parametric splines.
    
    A 2D spline curve consists of two splines: x(t) and y(t), each interpolating 
    the x and y coordinates respectively for the same t values.
    """
    
    def __init__(self, points, t=None):
        """
        Initialize the 2D spline curve with data points.
        
        Args:
            points (array-like): Array of 2D points to interpolate [(x1,y1), (x2,y2), ...].
            t (array-like, optional): Parameter values for each point. If None, 
                                      uniform parameterization will be used.
        """
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points must be a 2D array of shape (n, 2)")
        
        self.points = points
        
        # Create parameter values if not provided
        if t is None:
            # Use cumulative chord length parameterization
            diffs = np.diff(points, axis=0)
            segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
            cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))
            if cumulative_lengths[-1] == 0:  # Handle case where all points are the same
                t = np.linspace(0, 1, len(points))
            else:
                t = cumulative_lengths / cumulative_lengths[-1]
        else:
            t = np.asarray(t)
            if len(t) != len(points):
                raise ValueError("Number of parameter values must match number of points")
        
        self.t = t
        
        # Create x(t) and y(t) splines
        self.x_spline = QuadraticSpline(t, points[:, 0])
        self.y_spline = QuadraticSpline(t, points[:, 1])
    
    def __call__(self, t_new):
        """
        Evaluate the 2D spline curve at the given parameter values.
        
        Args:
            t_new (float or array-like): Parameter values where to evaluate the curve.
            
        Returns:
            ndarray: 2D points interpolated at the given parameter values.
        """
        t_new = np.asarray(t_new)
        scalar_input = False
        if t_new.ndim == 0:
            t_new = t_new[np.newaxis]
            scalar_input = True
        
        # Evaluate both x(t) and y(t) splines
        x_new = self.x_spline(t_new)
        y_new = self.y_spline(t_new)
        
        # Combine into 2D points
        points_new = np.column_stack((x_new, y_new))
        
        return points_new[0] if scalar_input else points_new
    
    def derivative(self, t_new):
        """
        Calculate the first derivative of the 2D spline curve at the given parameter values.
        
        Args:
            t_new (float or array-like): Parameter values where to evaluate the derivative.
            
        Returns:
            ndarray: Derivatives (dx/dt, dy/dt) at the given parameter values.
        """
        t_new = np.asarray(t_new)
        scalar_input = False
        if t_new.ndim == 0:
            t_new = t_new[np.newaxis]
            scalar_input = True
        
        # Evaluate derivatives of both x(t) and y(t) splines
        dx_dt = self.x_spline.derivative(t_new)
        dy_dt = self.y_spline.derivative(t_new)
        
        # Combine into 2D derivatives
        deriv_new = np.column_stack((dx_dt, dy_dt))
        
        return deriv_new[0] if scalar_input else deriv_new
    
    def plot(self, num_points=100, show_points=True, show_derivatives=False, ax=None):
        """
        Plot the 2D spline curve.
        
        Args:
            num_points (int): Number of points to use for the curve plot.
            show_points (bool): If True, show the original interpolation points.
            show_derivatives (bool): If True, show the derivatives at some points along the curve.
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, create new figure.
            
        Returns:
            matplotlib.axes.Axes: The axes containing the plot.
        """
        # Create a figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create parameter values for plotting
        t_plot = np.linspace(self.t[0], self.t[-1], num_points)
        
        # Evaluate the curve
        points_plot = self(t_plot)
        
        # Plot the curve
        ax.plot(points_plot[:, 0], points_plot[:, 1], 'b-', label='Spline Curve')
        
        # Show original points
        if show_points:
            ax.plot(self.points[:, 0], self.points[:, 1], 'ro', label='Control Points')
            # Number the points
            for i, (x, y) in enumerate(self.points):
                ax.text(x, y, f' {i+1}', fontsize=12)
        
        # Show derivatives
        if show_derivatives:
            # Show derivatives at a few points along the curve
            t_deriv = np.linspace(self.t[0], self.t[-1], 10)
            points_deriv = self(t_deriv)
            deriv = self.derivative(t_deriv)
            
            # Scale derivatives for visualization
            scale = 0.1 * np.max(np.linalg.norm(np.diff(self.points, axis=0), axis=1))
            dx_dt = deriv[:, 0] * scale
            dy_dt = deriv[:, 1] * scale
            
            ax.quiver(
                points_deriv[:, 0], points_deriv[:, 1], 
                dx_dt, dy_dt,
                angles='xy', scale_units='xy', scale=1,
                color='g', width=0.005, label='Tangent Vectors'
            )
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('2D Spline Curve')
        ax.grid(True)
        ax.axis('equal')
        ax.legend()
        
        return ax


def ellipse_example(save_only=False, output_dir=None, timestamp=None):
    """
    Example of a 2D spline curve interpolating points on an ellipse.
    
    The ellipse has semi-major axis a=4 and semi-minor axis b=5.
    Five random points are selected on the ellipse by choosing random parameter values.
    
    Args:
        save_only (bool): If True, only save the plot without displaying it.
        output_dir (str): Directory where to save the plot.
        timestamp (str): Timestamp to use in the filename.
    
    Returns:
        Spline2D: The created curve object.
    """
    # Ellipse parameters
    a, b = 4, 5
    
    # Generate 5 random parameter values between 0 and 2π
    np.random.seed(42)  # For reproducibility
    u = np.sort(np.random.uniform(0, 2*np.pi, 5))
    
    # Generate points on the ellipse
    points = np.array([(a*np.cos(ui), b*np.sin(ui)) for ui in u])
    
    # Create a 2D spline curve through these points
    curve = Spline2D(points, t=u)
    
    # Plot the curve and the ellipse for comparison
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the spline curve
    curve.plot(num_points=100, show_points=True, show_derivatives=True, ax=ax)
    
    # Plot the actual ellipse
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x = a * np.cos(theta)
    ellipse_y = b * np.sin(theta)
    ax.plot(ellipse_x, ellipse_y, 'r--', label='True Ellipse')
    
    ax.set_title(f'Quadratic Spline Curve Interpolating Points on Ellipse (a={a}, b={b})')
    ax.legend()
    
    plt.tight_layout()
    
    # Save the plot if output_dir is provided
    if output_dir:
        if timestamp is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"ellipse_spline_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300)
        print(f"Plot saved to: {filepath}")
    
    if not save_only:
        plt.show()
    
    return curve


if __name__ == "__main__":
    ellipse_example() 