"""
Quadratic Spline Interpolation Module

This module implements quadratic spline interpolation based on the approach described in
the "Spline Method of Interpolation" document.
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

    def __init__(self, x, y):
        """
        Initialize the quadratic spline interpolator with data points.

        Args:
            x (array-like): x-coordinates of the data points.
            y (array-like): y-coordinates of the data points.
        """
        # Convert to numpy arrays
        self.x = np.asarray(x)
        self.y = np.asarray(y)

        # Verify that the data is sorted by x
        if not np.all(np.diff(self.x) > 0):
            idx = np.argsort(self.x)
            self.x = self.x[idx]
            self.y = self.y[idx]

        n = len(self.x) - 1  # Number of splines

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
        A = np.zeros((3 * n, 3 * n))
        B = np.zeros(3 * n)

        # Apply conditions that each spline passes through two consecutive points
        eq = 0
        for i in range(n):
            # First point: a_i*x_i^2 + b_i*x_i + c_i = y_i
            A[eq, 3 * i] = self.x[i] ** 2
            A[eq, 3 * i + 1] = self.x[i]
            A[eq, 3 * i + 2] = 1
            B[eq] = self.y[i]
            eq += 1

            # Second point: a_i*x_{i+1}^2 + b_i*x_{i+1} + c_i = y_{i+1}
            A[eq, 3 * i] = self.x[i + 1] ** 2
            A[eq, 3 * i + 1] = self.x[i + 1]
            A[eq, 3 * i + 2] = 1
            B[eq] = self.y[i + 1]
            eq += 1

        # Apply first derivative continuity conditions at interior points
        for i in range(n - 1):
            # 2*a_i*x_{i+1} + b_i = 2*a_{i+1}*x_{i+1} + b_{i+1}
            A[eq, 3 * i] = 2 * self.x[i + 1]
            A[eq, 3 * i + 1] = 1
            A[eq, 3 * (i + 1)] = -2 * self.x[i + 1]
            A[eq, 3 * (i + 1) + 1] = -1
            B[eq] = 0
            eq += 1

        # Apply the condition that the first spline is linear (a[0] = 0)
        A[eq, 0] = 1
        B[eq] = 0

        # Solve the system to find the coefficients
        coeffs = np.linalg.solve(A, B)

        # Extract coefficients a, b, c for each spline
        for i in range(n):
            self.a[i] = coeffs[3 * i]
            self.b[i] = coeffs[3 * i + 1]
            self.c[i] = coeffs[3 * i + 2]

    def __call__(self, x_new):
        """
        Evaluate the quadratic spline at the given points.

        Args:
            x_new (float or array-like): Points where to evaluate the spline.

        Returns:
            float or ndarray: Interpolated values.
        """
        x_new = np.asarray(x_new)
        scalar_input = False
        if x_new.ndim == 0:
            x_new = x_new[np.newaxis]
            scalar_input = True

        y_new = np.zeros_like(x_new)

        for i, x_val in enumerate(x_new):
            if x_val < self.x[0] or x_val > self.x[-1]:
                raise ValueError(
                    f"Point {x_val} is outside the interpolation range [{self.x[0]}, {self.x[-1]}]"
                )

            # Handle the case where x_val equals the last point exactly
            if x_val == self.x[-1]:
                y_new[i] = self.y[-1]
                continue

            # Find the corresponding interval
            idx = np.searchsorted(self.x, x_val, side="right") - 1

            # Apply the corresponding quadratic polynomial
            y_new[i] = self.a[idx] * x_val**2 + self.b[idx] * x_val + self.c[idx]

        return y_new[0] if scalar_input else y_new

    def derivative(self, x_new):
        """
        Calculate the first derivative of the quadratic spline at the given points.

        Args:
            x_new (float or array-like): Points where to evaluate the derivative.

        Returns:
            float or ndarray: Derivative values.
        """
        x_new = np.asarray(x_new)
        scalar_input = False
        if x_new.ndim == 0:
            x_new = x_new[np.newaxis]
            scalar_input = True

        y_deriv = np.zeros_like(x_new)

        for i, x_val in enumerate(x_new):
            if x_val < self.x[0] or x_val > self.x[-1]:
                raise ValueError(
                    f"Point {x_val} is outside the interpolation range [{self.x[0]}, {self.x[-1]}]"
                )

            # Handle the case where x_val equals the last point exactly
            if x_val == self.x[-1]:
                # Use the last interval's coefficients to calculate the derivative at the last point
                idx = len(self.a) - 1
                y_deriv[i] = 2 * self.a[idx] * x_val + self.b[idx]
                continue

            # Find the corresponding interval
            idx = np.searchsorted(self.x, x_val, side="right") - 1

            # Derivative of the quadratic polynomial: 2*a*x + b
            y_deriv[i] = 2 * self.a[idx] * x_val + self.b[idx]

        return y_deriv[0] if scalar_input else y_deriv

    def integrate(self, a, b):
        """
        Integrate the quadratic spline over the interval [a, b].

        Args:
            a (float): Lower limit of integration.
            b (float): Upper limit of integration.

        Returns:
            float: Value of the integral.
        """
        if a < self.x[0] or b > self.x[-1]:
            raise ValueError(
                f"Integration interval [{a}, {b}] is outside the interpolation range [{self.x[0]}, {self.x[-1]}]"
            )

        # Find the subintervals containing a and b
        idx_a = np.searchsorted(self.x, a, side="right") - 1
        idx_b = np.searchsorted(self.x, b, side="right") - 1

        result = 0.0

        # Special case: a and b are in the same subinterval
        if idx_a == idx_b:
            result += self._integrate_segment(a, b, idx_a)
        else:
            # Integrate from a to the end of the first subinterval
            result += self._integrate_segment(a, self.x[idx_a + 1], idx_a)

            # Integrate complete intermediate subintervals
            for i in range(idx_a + 1, idx_b):
                result += self._integrate_segment(self.x[i], self.x[i + 1], i)

            # Integrate from the beginning of the last subinterval to b
            result += self._integrate_segment(self.x[idx_b], b, idx_b)

        return result

    def _integrate_segment(self, a, b, idx):
        """
        Integrate a segment of the quadratic spline.

        Args:
            a (float): Lower limit of integration.
            b (float): Upper limit of integration.
            idx (int): Index of the spline to integrate.

        Returns:
            float: Value of the segment integral.
        """
        # Integrate a*x^2 + b*x + c from a to b
        a_coef, b_coef, c_coef = self.a[idx], self.b[idx], self.c[idx]
        return (
            a_coef * (b**3 - a**3) / 3 + b_coef * (b**2 - a**2) / 2 + c_coef * (b - a)
        )

    def plot(self, num_points=1000, show_points=True, show_derivatives=False, ax=None):
        """
        Plot the quadratic spline and optionally its derivative.

        Args:
            num_points (int): Number of points for the plot.
            show_points (bool): If True, show the original data points.
            show_derivatives (bool): If True, show the first derivative.
            ax (matplotlib.axes.Axes, optional): Axes to plot on.

        Returns:
            matplotlib.axes.Axes: The axes used for plotting.
        """
        # Create new figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        # Create evenly spaced points for plotting the spline
        x_plot = np.linspace(self.x[0], self.x[-1], num_points)
        y_plot = self(x_plot)

        # Plot the spline
        ax.plot(x_plot, y_plot, "b-", label="Quadratic Spline")

        # Plot the data points
        if show_points:
            ax.plot(self.x, self.y, "ro", label="Data Points")

        # Plot the derivative if requested
        if show_derivatives:
            y_deriv = self.derivative(x_plot)

            # Use a twin x-axis for the derivative plot
            ax_deriv = ax.twinx()
            ax_deriv.plot(x_plot, y_deriv, "g-", label="First Derivative")
            ax_deriv.set_ylabel("Derivative")
            ax_deriv.legend(loc="upper right")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Quadratic Spline Interpolation")
        ax.grid(True)
        ax.legend(loc="upper left")

        return ax


def example_rocket(
    t_eval=16.0, show_plot=True, save_plot=False, output_dir=None, timestamp=None
):
    """
    Example of quadratic spline interpolation using rocket motion data.

    Args:
        t_eval (float): Time at which to evaluate the rocket velocity.
        show_plot (bool): Whether to display the plot.
        save_plot (bool): Whether to save the plot as an image file.
        output_dir (str): Directory to save the plot (if save_plot is True).
        timestamp (str): Timestamp string to include in the filename.

    Returns:
        dict: Dictionary containing velocity, distance, and acceleration at the specified time.
    """
    # Rocket motion data
    times = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
    heights = np.array([0, 2, 5, 9, 15, 22, 29, 37, 46, 58, 72, 88, 106, 127, 150, 176])

    # Create a quadratic spline for the rocket trajectory
    rocket_spline = QuadraticSpline(times, heights)

    # Calculate velocity at t_eval using the derivative
    velocity = rocket_spline.derivative(t_eval)

    # Calculate distance covered between t=11s and t=16s
    distance = rocket_spline.integrate(11.0, t_eval)

    # Calculate acceleration at t_eval using numerical differentiation
    dt = 0.001
    v1 = rocket_spline.derivative(t_eval - dt)
    v2 = rocket_spline.derivative(t_eval + dt)
    acceleration = (v2 - v1) / (2 * dt)

    # Store results
    results = {"velocity": velocity, "distance": distance, "acceleration": acceleration}

    # Plot the spline and its derivative
    if show_plot or save_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Plot height vs time
        rocket_spline.plot(ax=ax1, show_derivatives=False)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Height (m)")
        ax1.set_title("Rocket Height vs Time")

        # Mark the evaluation point
        ax1.plot(
            t_eval,
            rocket_spline(t_eval),
            "g*",
            markersize=10,
            label=f"Evaluation point (t={t_eval}s)",
        )

        # Shade the area for the distance integral
        t_integral = np.linspace(11.0, t_eval, 100)
        h_integral = rocket_spline(t_integral)
        ax1.fill_between(
            t_integral,
            0,
            h_integral,
            alpha=0.3,
            color="r",
            label=f"Distance: {distance:.2f} m",
        )
        ax1.legend()

        # Plot velocity vs time
        t_plot = np.linspace(times[0], times[-1], 1000)
        v_plot = rocket_spline.derivative(t_plot)

        ax2.plot(t_plot, v_plot, "r-", label="Velocity (derivative)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity (m/s)")
        ax2.set_title("Rocket Velocity vs Time")

        # Mark the evaluation point
        ax2.plot(
            t_eval,
            velocity,
            "g*",
            markersize=10,
            label=f"v(t={t_eval}s) = {velocity:.2f} m/s",
        )
        ax2.legend()
        ax2.grid(True)

        # Plot acceleration vs time (numerical 2nd derivative)
        a_plot = np.zeros_like(t_plot)
        for i, t in enumerate(t_plot):
            v1 = rocket_spline.derivative(t - dt)
            v2 = rocket_spline.derivative(t + dt)
            a_plot[i] = (v2 - v1) / (2 * dt)

        ax3.plot(t_plot, a_plot, "g-", label="Acceleration (2nd derivative)")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Acceleration (m/s²)")
        ax3.set_title("Rocket Acceleration vs Time")

        # Mark the evaluation point
        ax3.plot(
            t_eval,
            acceleration,
            "g*",
            markersize=10,
            label=f"a(t={t_eval}s) = {acceleration:.2f} m/s²",
        )
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()

        # Save the plot if requested
        if save_plot and output_dir:
            if timestamp is None:
                import datetime

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"quadratic_spline_rocket_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300)
            print(f"Quadratic spline rocket plot saved to: {filepath}")

        if show_plot:
            plt.show()

    return results


if __name__ == "__main__":
    example_rocket()
