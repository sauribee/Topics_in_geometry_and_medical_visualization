"""
Linear Spline Module for Assignment 1.

This module contains the implementation of Linear Spline interpolation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


class LinearSpline:
    """
    Linear Spline Interpolation.

    Connects consecutive data points with linear segments.
    """

    def __init__(self, x, y):
        """
        Initialize a linear spline with data points.

        Args:
            x (array-like): X values (independent variable).
            y (array-like): Y values (dependent variable).
        """
        # Ensure x and y are numpy arrays
        self.x = np.asarray(x)
        self.y = np.asarray(y)

        # Ensure x is sorted (not strictly necessary for linear splines, but good practice)
        if not np.all(np.diff(self.x) > 0):
            idx = np.argsort(self.x)
            self.x = self.x[idx]
            self.y = self.y[idx]

        # Store number of data points
        self.n = len(x)

        # Check that we have at least 2 points
        if self.n < 2:
            raise ValueError("At least 2 points are required for linear interpolation")

    def __call__(self, x_new):
        """
        Evaluate the linear spline at the given points.

        Args:
            x_new (float or array-like): Points at which to evaluate the spline.

        Returns:
            float or ndarray: Interpolated values.
        """
        # Convert x_new to array if it's a scalar
        x_new = np.asarray(x_new)
        scalar_input = False
        if x_new.ndim == 0:
            x_new = x_new[np.newaxis]
            scalar_input = True

        y_new = np.zeros_like(x_new)

        # Interpolate each point
        for i, xi in enumerate(x_new):
            # Check if xi is within bounds
            if xi < self.x[0] or xi > self.x[-1]:
                raise ValueError(
                    f"Point {xi} is outside the interpolation range [{self.x[0]}, {self.x[-1]}]"
                )

            # Find the interval that contains xi
            idx = np.searchsorted(self.x, xi, side="right") - 1

            # Apply linear interpolation formula
            x0, x1 = self.x[idx], self.x[idx + 1]
            y0, y1 = self.y[idx], self.y[idx + 1]

            y_new[i] = y0 + (y1 - y0) * (xi - x0) / (x1 - x0)

        return y_new[0] if scalar_input else y_new

    def plot(self, num_points=1000, show_data_points=True, ax=None):
        """
        Plot the linear spline.

        Args:
            num_points (int): Number of points to use for plotting the curve.
            show_data_points (bool): Whether to show the original data points.
            ax (matplotlib.axes.Axes, optional): Axes to plot on.

        Returns:
            matplotlib.axes.Axes: The axes used for plotting.
        """
        # Create new figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Create evenly spaced points for plotting the spline
        x_plot = np.linspace(self.x[0], self.x[-1], num_points)
        y_plot = self(x_plot)

        # Plot the spline
        ax.plot(x_plot, y_plot, "b-", label="Linear Spline")

        # Plot the data points
        if show_data_points:
            ax.plot(self.x, self.y, "ro", label="Data Points")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Linear Spline Interpolation")
        ax.grid(True)
        ax.legend()

        return ax


def example_rocket(
    t_eval=16.0, show_plot=True, save_plot=False, output_dir=None, timestamp=None
):
    """
    Example of linear spline interpolation using rocket motion data.

    Args:
        t_eval (float): Time at which to evaluate the rocket velocity.
        show_plot (bool): Whether to display the plot.
        save_plot (bool): Whether to save the plot as an image file.
        output_dir (str): Directory to save the plot (if save_plot is True).
        timestamp (str): Timestamp string to include in the filename.

    Returns:
        float: Velocity at the specified time.
    """
    # Rocket motion data
    times = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
    heights = np.array([0, 2, 5, 9, 15, 22, 29, 37, 46, 58, 72, 88, 106, 127, 150, 176])

    # Create a linear spline for the rocket trajectory
    rocket_spline = LinearSpline(times, heights)

    # Calculate the velocity at t_eval using finite differences
    dt = 0.001
    h1 = rocket_spline(t_eval - dt)
    h2 = rocket_spline(t_eval + dt)
    velocity = (h2 - h1) / (2 * dt)

    # Plot the spline with the original data points
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot height vs time
    rocket_spline.plot(ax=ax1)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Height (m)")
    ax1.set_title("Rocket Height vs Time")

    # Show the evaluation point on the plot
    ax1.plot(
        t_eval,
        rocket_spline(t_eval),
        "g*",
        markersize=10,
        label=f"Evaluation point (t={t_eval}s)",
    )
    ax1.legend()

    # Calculate velocity values for plotting
    t_plot = np.linspace(times[0], times[-1], 1000)
    v_plot = np.zeros_like(t_plot)

    for i, t in enumerate(t_plot):
        h1 = rocket_spline(t - dt)
        h2 = rocket_spline(t + dt)
        v_plot[i] = (h2 - h1) / (2 * dt)

    # Plot velocity vs time
    ax2.plot(t_plot, v_plot, "r-", label="Velocity (numerical derivative)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Velocity (m/s)")
    ax2.set_title("Rocket Velocity vs Time")

    # Show the evaluation point on the velocity plot
    ax2.plot(
        t_eval,
        velocity,
        "g*",
        markersize=10,
        label=f"v(t={t_eval}s) = {velocity:.2f} m/s",
    )
    ax2.legend()

    ax2.grid(True)
    plt.tight_layout()

    # Save the plot if requested
    if save_plot and output_dir:
        if timestamp is None:
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"linear_spline_rocket_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300)
        print(f"Linear spline rocket plot saved to: {filepath}")

    if show_plot:
        plt.show()

    return velocity


if __name__ == "__main__":
    example_rocket()
