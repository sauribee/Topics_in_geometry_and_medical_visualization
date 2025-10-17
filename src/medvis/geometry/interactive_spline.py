"""
Interactive script for adjusting control points in an ellipse spline.

This script allows interactive movement of control points
to find the best spline estimation for an ellipse.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from assignment_2.curves.spline2d.curve import Spline2D


class InteractiveSpline:
    def __init__(self):
        # Ellipse parameters
        self.a, self.b = 4, 5

        # Set up figure and axes
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.2)

        # Generate initial points on the ellipse (same as in the original example)
        np.random.seed(42)  # For reproducibility
        self.u = np.sort(np.random.uniform(0, 2 * np.pi, 5))
        self.points = np.array(
            [(self.a * np.cos(ui), self.b * np.sin(ui)) for ui in self.u]
        )

        # Create initial spline
        self.create_spline()

        # Create buttons
        self.ax_reset = plt.axes([0.7, 0.05, 0.2, 0.075])
        self.btn_reset = Button(self.ax_reset, "Reset Points")
        self.btn_reset.on_clicked(self.reset_points)

        self.ax_save = plt.axes([0.4, 0.05, 0.2, 0.075])
        self.btn_save = Button(self.ax_save, "Save Image")
        self.btn_save.on_clicked(self.save_plot)

        # Save important data for interaction
        self.selected_point = None
        self.point_artists = []

        # Initialize the plot
        self.update_plot()

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def create_spline(self):
        """Create a new spline with the current points"""
        self.curve = Spline2D(self.points, t=self.u)

    def update_plot(self):
        """Update the plot with current points and curves"""
        self.ax.clear()

        # Draw the spline using the correct range of t parameters
        t_plot = np.linspace(self.u[0], self.u[-1] - 1e-10, 100)
        curve_points = self.curve(t_plot)
        self.ax.plot(curve_points[:, 0], curve_points[:, 1], "b-", label="Spline")

        # Draw the real ellipse
        theta = np.linspace(0, 2 * np.pi, 100)
        ellipse_x = self.a * np.cos(theta)
        ellipse_y = self.b * np.sin(theta)
        self.ax.plot(ellipse_x, ellipse_y, "r--", label="Real Ellipse")

        # Draw the control points
        self.point_artists = []
        for i, (x, y) in enumerate(self.points):
            (point,) = self.ax.plot(x, y, "ro", markersize=10, picker=5)
            self.point_artists.append(point)
            self.ax.text(x, y, f" {i+1}", fontsize=12)

        # Calculate the error (average distance between the spline and the real ellipse)
        # For error calculation, we sample the ellipse and find nearby points on the spline
        theta_samples = np.linspace(0, 2 * np.pi, 100)
        ellipse_points = np.column_stack(
            (self.a * np.cos(theta_samples), self.b * np.sin(theta_samples))
        )

        # Map theta values to t parameters (approximate)
        # First normalize theta between 0 and 1
        normalized_theta = (theta_samples - theta_samples[0]) / (
            theta_samples[-1] - theta_samples[0]
        )
        # Then scale to the range of u
        mapped_t = self.u[0] + normalized_theta * (self.u[-1] - self.u[0])

        spline_points = self.curve(mapped_t)
        error = np.mean(np.sqrt(np.sum((spline_points - ellipse_points) ** 2, axis=1)))

        self.ax.set_title(f"Interactive Spline Fitting to Ellipse (Error: {error:.4f})")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.grid(True)
        self.ax.axis("equal")
        self.ax.legend()

        self.fig.canvas.draw_idle()

    def on_press(self, event):
        """Handle mouse button press event"""
        if event.inaxes != self.ax:
            return

        # Check if a point was clicked
        for i, (x, y) in enumerate(self.points):
            if abs(event.xdata - x) < 0.2 and abs(event.ydata - y) < 0.2:
                self.selected_point = i
                break

    def on_motion(self, event):
        """Handle mouse movement event"""
        if self.selected_point is None or event.inaxes != self.ax:
            return

        # Update the position of the selected point
        self.points[self.selected_point] = [event.xdata, event.ydata]

        # Recalculate the t parameter value for this point
        # (this is optional, you could keep the original t values)
        new_t = np.arctan2(event.ydata / self.b, event.xdata / self.a)
        if new_t < 0:
            new_t += 2 * np.pi

        # Only update the t value if the new value maintains the order
        if (self.selected_point == 0 or new_t > self.u[self.selected_point - 1]) and (
            self.selected_point == len(self.u) - 1
            or new_t < self.u[self.selected_point + 1]
        ):
            self.u[self.selected_point] = new_t

        # Recreate the spline and update the plot
        self.create_spline()
        self.update_plot()

    def on_release(self, event):
        """Handle mouse button release event"""
        self.selected_point = None

    def reset_points(self, event):
        """Reset points to their original positions"""
        np.random.seed(42)
        self.u = np.sort(np.random.uniform(0, 2 * np.pi, 5))
        self.points = np.array(
            [(self.a * np.cos(ui), self.b * np.sin(ui)) for ui in self.u]
        )
        self.create_spline()
        self.update_plot()

    def save_plot(self, event):
        """Save the current plot as an image"""
        import os
        import datetime

        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimized_ellipse_spline_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)

        # Save image
        self.fig.savefig(filepath, dpi=300)
        print(f"Image saved to: {filepath}")


def main():
    """Main function to run the interactive script"""
    _interactive_spline = InteractiveSpline()
    plt.show()


if __name__ == "__main__":
    main()
