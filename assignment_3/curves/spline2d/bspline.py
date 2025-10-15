"""
B-Spline implementation for 2D curve interpolation.

This module provides classes for quadratic B-spline interpolation
of 2D points, with special focus on non-uniform parameterization.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev


class QuadraticBSpline:
    """
    Class for quadratic B-spline interpolation.
    
    Quadratic B-splines provide C1 continuity across the entire curve
    and are suitable for interpolating medical image contours.
    """
    
    def __init__(self, points, closed=True, smoothing=0.0, auto_correct=False, monotone_parameterization=True):
        """
        Initialize the quadratic B-spline interpolator with data points.
        
        Args:
            points (array-like): Array of 2D points to interpolate [(x1,y1), (x2,y2), ...].
            closed (bool): Whether the curve should be closed (True) or open (False).
            smoothing (float): Smoothing parameter (0.0 = exact interpolation, >0 = smoother curve).
            auto_correct (bool): Whether to automatically correct points to avoid self-intersections.
            monotone_parameterization (bool): Whether to ensure monotone parameterization along a main direction.
        """
        self.points = np.asarray(points)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("Points must be a 2D array of shape (n, 2)")
        
        self.closed = closed
        self.n_points = len(self.points)
        self.smoothing = smoothing
        self.auto_correct = auto_correct
        self.monotone_parameterization = monotone_parameterization
        
        # Apply pre-processing if needed
        if self.auto_correct:
            self._correct_points()
        
        # Create parameterization and fit the spline
        self._create_parameterization()
        self._fit_spline()
    
    def _correct_points(self):
        """
        Pre-process points to remove potential issues like sharp turns or self-intersections.
        This is useful for anatomical contours that should have a generally smooth shape.
        """
        if self.n_points < 4:
            print("Too few points for correction.")
            return
        
        # Find the general progression direction
        if self.closed:
            # Calculate center of mass
            center = np.mean(self.points, axis=0)
            print(f"Center of mass: {center}")
            
            # Calculate main axis using the covariance matrix (simplified PCA)
            cov_matrix = np.cov(self.points.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            # Main axis is the eigenvector with the largest eigenvalue
            main_axis = eigenvectors[:, np.argmax(eigenvalues)]
            print(f"Main contour direction: {main_axis}")
        
        # Check for sharp angles (potential areas of self-intersection)
        angles = []
        for i in range(self.n_points):
            prev_idx = (i - 1) % self.n_points
            next_idx = (i + 1) % self.n_points
            
            # Vectors to previous and next points
            v_prev = self.points[prev_idx] - self.points[i]
            v_next = self.points[next_idx] - self.points[i]
            
            # Normalize vectors
            v_prev_norm = np.linalg.norm(v_prev)
            v_next_norm = np.linalg.norm(v_next)
            
            if v_prev_norm > 1e-8 and v_next_norm > 1e-8:
                v_prev = v_prev / v_prev_norm
                v_next = v_next / v_next_norm
                
                # Calculate angle
                cos_angle = np.clip(np.dot(v_prev, v_next), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angles.append(angle)
            else:
                angles.append(np.pi)  # Default angle if points are too close
        
        # Find sharp turns (small angles indicate sharp turns)
        # We use radians: pi = 180 degrees, pi/2 = 90 degrees, etc.
        sharp_turn_threshold = np.pi/6  # 30 degrees
        sharp_turns = [i for i, angle in enumerate(angles) if angle < sharp_turn_threshold]
        
        if sharp_turns:
            print(f"Sharp turns detected at points: {sharp_turns}")
            
            # Smooth out sharp turns by adjusting point positions
            new_points = self.points.copy()
            for i in sharp_turns:
                prev_idx = (i - 1) % self.n_points
                next_idx = (i + 1) % self.n_points
                
                # Replace problematic point with average of neighbors plus a small outward offset
                avg_point = (self.points[prev_idx] + self.points[next_idx]) / 2
                
                # Calculate vector from the center of mass to average point (pushes outward)
                center = np.mean(self.points, axis=0)
                outward_vector = avg_point - center
                outward_vector_norm = np.linalg.norm(outward_vector)
                
                if outward_vector_norm > 1e-8:
                    outward_vector = outward_vector / outward_vector_norm
                    # Add a small outward push to prevent self-intersection
                    new_points[i] = avg_point + outward_vector * 2.0
                else:
                    new_points[i] = avg_point
                    
                print(f"Adjusted point {i} from {self.points[i]} to {new_points[i]}")
            
            self.points = new_points
            print("Points corrected to reduce sharp turns.")
        else:
            print("No sharp turns detected.")
    
    def _create_parameterization(self):
        """
        Create a non-uniform parameterization based on the chord length between points.
        For anatomical contours, this works well because it respects the distribution of points.
        """
        # Calculate distances between consecutive points
        if self.closed:
            # For closed curves, include the distance from last to first point
            points_loop = np.vstack((self.points, self.points[0:1]))
            diffs = np.diff(points_loop, axis=0)
        else:
            diffs = np.diff(self.points, axis=0)
        
        chord_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        
        # Handle case where some points might be identical
        chord_lengths = np.maximum(chord_lengths, 1e-10)
        
        # Apply additional weighting based on curvature to give more importance to areas of high curvature
        if self.n_points > 4:
            # Estimate curvature at each point
            curvature_weights = np.ones_like(chord_lengths)
            
            # Apply curvature-based weighting
            for i in range(self.n_points):
                prev_idx = (i - 1) % self.n_points
                curr_idx = i
                next_idx = (i + 1) % self.n_points
                
                # Skip if we're at a boundary of an open curve
                if not self.closed and (i == 0 or i == self.n_points - 1):
                    continue
                
                # Compute vectors to adjacent points
                v1 = self.points[curr_idx] - self.points[prev_idx]
                v2 = self.points[next_idx] - self.points[curr_idx]
                
                # Compute turning angle
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                
                if v1_norm > 1e-10 and v2_norm > 1e-10:
                    v1 = v1 / v1_norm
                    v2 = v2 / v2_norm
                    # Calculate curvature (approximation using turning angle)
                    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
                    turning_angle = np.arccos(dot_product)
                    
                    # Higher turning angle = higher curvature = more parameter space
                    curvature_weights[curr_idx % len(curvature_weights)] = 1.0 + turning_angle / np.pi
            
            # Apply the curvature weights to chord lengths (more parameter space for high curvature)
            chord_lengths = chord_lengths * curvature_weights
            print(f"Applied curvature-based parameterization weights")
        
        # Cumulative chord length parameterization
        cumulative = np.concatenate(([0], np.cumsum(chord_lengths)))
        
        # Normalize to [0, 1]
        if cumulative[-1] > 0:
            self.params = cumulative / cumulative[-1]
        else:
            # Fallback to uniform parameterization if all points are identical
            self.params = np.linspace(0, 1, self.n_points)
            
        # If requested, ensure monotone progression in the dominant direction
        if self.monotone_parameterization and self.n_points > 3:
            # Calculate main axis using covariance matrix (simplified PCA)
            cov_matrix = np.cov(self.points.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            # Main axis is the eigenvector with the largest eigenvalue
            main_axis = eigenvectors[:, np.argmax(eigenvalues)]
            main_axis = main_axis / np.linalg.norm(main_axis)
            
            # Find farthest points along the main axis (simplification of convex hull diameter)
            projections = np.array([np.dot(p, main_axis) for p in self.points])
            min_idx = np.argmin(projections)
            max_idx = np.argmax(projections)
            
            # Check if points follow a reasonably monotonic path
            # Project each point onto the main axis
            indices = np.arange(self.n_points)
            ordered_indices = indices[np.argsort(projections)]
            
            # Check if the ordering is close to the current ordering
            # If not, sort points by their projection
            is_monotonic = True
            for i in range(1, len(ordered_indices)):
                if abs(ordered_indices[i] - ordered_indices[i-1]) > 3 and \
                   abs(ordered_indices[i] - ordered_indices[i-1]) < self.n_points - 3:
                    is_monotonic = False
                    break
            
            if not is_monotonic:
                print("Reordering points for monotone progression along main axis.")
                sort_indices = np.argsort(projections)
                self.points = self.points[sort_indices]
                
                # Recalculate parameterization for new point ordering
                self._create_parameterization()
                return
        
        # Store the chord lengths for later use
        self.chord_lengths = chord_lengths
        
        # Print detailed information for debugging
        print(f"Parameter values: min={min(self.params):.3f}, max={max(self.params):.3f}, count={len(self.params)}")
        print(f"Chord lengths: min={min(chord_lengths):.3f}, max={max(chord_lengths):.3f}, avg={np.mean(chord_lengths):.3f}")
    
    def _fit_spline(self):
        """
        Fit a quadratic B-spline to the points using the calculated parameterization.
        """
        try:
            # For closed curves, we need to handle the periodicity
            if self.closed:
                # Add points at the beginning and end to ensure proper closure
                # We duplicate some points to maintain the intended shape
                points_extended = np.vstack((self.points[-2:], self.points, self.points[:2]))
                
                # Adjust the parameter values accordingly
                param_range = self.params[-1] - self.params[0]
                params_start = self.params[0] - (self.params[1] - self.params[0])
                params_end = self.params[-1] + (self.params[-1] - self.params[-2])
                
                params_extended = np.concatenate((
                    [params_start, params_start + (self.params[0] - params_start)/2],
                    self.params,
                    [params_end - (params_end - self.params[-1])/2, params_end]
                ))
                
                # Ensure the parameters are strictly increasing
                params_extended = np.linspace(0, 1, len(points_extended))
                
                # Use scipy's splprep for fitting B-splines
                # k=2 for quadratic splines, s=smoothing parameter
                self.tck, self.u = splprep(
                    [points_extended[:, 0], points_extended[:, 1]], 
                    u=params_extended, 
                    k=3,  # Increased to cubic spline for better smoothness
                    s=self.smoothing,  # Allow some smoothing if specified
                    per=1  # Periodic spline
                )
            else:
                # For open curves, we use the standard approach
                # Prevent duplicate parameter values by ensuring they're unique
                if len(np.unique(self.params)) < len(self.params):
                    print("Warning: Duplicate parameter values detected. Using uniform parameterization.")
                    self.params = np.linspace(0, 1, self.n_points)
                
                # Use scipy's splprep for fitting B-splines
                self.tck, self.u = splprep(
                    [self.points[:, 0], self.points[:, 1]], 
                    u=self.params, 
                    k=3,  # Increased to cubic spline for better smoothness
                    s=self.smoothing  # Allow some smoothing if specified
                )
            
            # Extract knot vector and control points
            self.knots = self.tck[0]
            self.c_x = self.tck[1][0]
            self.c_y = self.tck[1][1]
            
            print("Spline fit successful")
            print(f"Number of knots: {len(self.knots)}")
            print(f"Number of control points: {len(self.c_x)}")
            
            # Verify that the spline doesn't self-intersect
            self._check_self_intersections()
            
        except Exception as e:
            print(f"Error fitting spline: {e}")
            # Fallback to a simpler approach if the standard fitting fails
            print("Using a simplified fitting approach...")
            
            # For curves with few points, we might need to reduce the degree
            degree = min(3, self.n_points - 1)  # Maximum degree possible, preferring cubic
            
            # If too few points for a cubic, adjust the parameters
            if degree < 3:
                print(f"Warning: Too few points ({self.n_points}) for cubic spline. Using degree {degree}.")
            
            if self.closed:
                # For closed curves with few points
                # Duplicate the first point at the end to close the curve
                points_closed = np.vstack((self.points, self.points[0:1]))
                params_closed = np.linspace(0, 1, len(points_closed))
                
                self.tck, self.u = splprep(
                    [points_closed[:, 0], points_closed[:, 1]], 
                    u=params_closed, 
                    k=degree,
                    s=self.smoothing
                )
            else:
                # For open curves with few points
                params_uniform = np.linspace(0, 1, self.n_points)
                
                self.tck, self.u = splprep(
                    [self.points[:, 0], self.points[:, 1]], 
                    u=params_uniform, 
                    k=degree,
                    s=self.smoothing
                )
            
            # Extract knot vector and control points
            self.knots = self.tck[0]
            self.c_x = self.tck[1][0]
            self.c_y = self.tck[1][1]
            
            print("Fallback spline fit successful")
    
    def _check_self_intersections(self, n_check_points=100):
        """
        Check for potential self-intersections in the spline.
        This is important for anatomical contours that should not self-intersect.
        """
        # Evaluate the spline at many points
        check_points = self.evaluate(n_points=n_check_points)
        
        # Approximate check based on distance between non-adjacent points
        min_allowed_distance = 8.0  # Increased minimum allowed distance between non-adjacent points
        
        # Count intersections to avoid too many warnings
        intersection_count = 0
        max_warnings = 10  # Maximum number of warnings to display
        
        for i in range(n_check_points):
            for j in range(i+3, n_check_points-1):  # Increased minimum distance between points
                # Skip adjacent points
                if abs(i-j) <= 3 or (self.closed and (n_check_points - abs(i-j)) <= 3):
                    continue
                
                # Calculate distance between points
                d = np.linalg.norm(check_points[i] - check_points[j])
                if d < min_allowed_distance:
                    intersection_count += 1
                    if intersection_count <= max_warnings:
                        print(f"Warning: Potential self-intersection detected between points {i} and {j} (distance={d:.2f})")
                        print(f"Positions: {check_points[i]} and {check_points[j]}")
        
        if intersection_count > max_warnings:
            print(f"... and {intersection_count - max_warnings} more potential self-intersections")
        
        if intersection_count == 0:
            print("No self-intersections detected.")
        else:
            print(f"Total potential self-intersections: {intersection_count}")
    
    def evaluate(self, u_values=None, n_points=100):
        """
        Evaluate the B-spline at the given parameter values.
        
        Args:
            u_values (array-like, optional): Parameter values where to evaluate the spline.
                                            If None, n_points evenly spaced values are used.
            n_points (int): Number of points to evaluate if u_values is None.
            
        Returns:
            ndarray: 2D points along the spline curve.
        """
        if u_values is None:
            if self.closed:
                # For closed curves, ensure we complete the loop
                u_values = np.linspace(0, 1, n_points, endpoint=True)
            else:
                # For open curves, stay within the parameter range
                u_values = np.linspace(0, 1, n_points)
        
        # Evaluate the spline
        x_new, y_new = splev(u_values, self.tck)
        return np.column_stack((x_new, y_new))
    
    def derivative(self, u_values=None, n_points=100):
        """
        Calculate the first derivative of the B-spline at the given parameter values.
        
        Args:
            u_values (array-like, optional): Parameter values where to evaluate the derivative.
                                            If None, n_points evenly spaced values are used.
            n_points (int): Number of points to evaluate if u_values is None.
            
        Returns:
            ndarray: Derivative vectors (dx/du, dy/du) at the given parameter values.
        """
        if u_values is None:
            if self.closed:
                # For closed curves, ensure we complete the loop
                u_values = np.linspace(0, 1, n_points, endpoint=True)
            else:
                # For open curves, stay within the parameter range
                u_values = np.linspace(0, 1, n_points)
        
        # Evaluate the derivative (der=1 for first derivative)
        dx_du, dy_du = splev(u_values, self.tck, der=1)
        return np.column_stack((dx_du, dy_du))
    
    def curvature(self, u_values=None, n_points=100):
        """
        Calculate the curvature of the B-spline at the given parameter values.
        
        Args:
            u_values (array-like, optional): Parameter values where to evaluate the curvature.
                                            If None, n_points evenly spaced values are used.
            n_points (int): Number of points to evaluate if u_values is None.
            
        Returns:
            ndarray: Curvature values at the given parameter values.
        """
        if u_values is None:
            if self.closed:
                # For closed curves, ensure we complete the loop
                u_values = np.linspace(0, 1, n_points, endpoint=True)
            else:
                # For open curves, stay within the parameter range
                u_values = np.linspace(0, 1, n_points)
        
        # Calculate first and second derivatives
        dx_du, dy_du = splev(u_values, self.tck, der=1)
        d2x_du2, d2y_du2 = splev(u_values, self.tck, der=2)
        
        # Calculate curvature: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = np.abs(dx_du * d2y_du2 - dy_du * d2x_du2)
        denominator = np.power(dx_du**2 + dy_du**2, 1.5)
        
        # Handle zero denominators
        mask = denominator > 1e-10
        curvature = np.zeros_like(u_values)
        curvature[mask] = numerator[mask] / denominator[mask]
        
        return curvature
    
    def plot(self, ax=None, show_points=True, show_derivatives=False, show_curvature=False,
             curve_color='blue', point_color='red', der_color='green', curvature_color='purple',
             n_curve_points=100, n_der_points=10, der_scale=0.02, curvature_scale=0.01,
             highlight_self_intersections=True, min_intersection_distance=8.0):
        """
        Plot the B-spline curve.
        
        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure is created.
            show_points (bool): Whether to show the interpolation points.
            show_derivatives (bool): Whether to show tangent vectors along the curve.
            show_curvature (bool): Whether to show curvature vectors along the curve.
            curve_color (str): Color for the curve.
            point_color (str): Color for the interpolation points.
            der_color (str): Color for the derivative vectors.
            curvature_color (str): Color for the curvature indicators.
            n_curve_points (int): Number of points to use for plotting the curve.
            n_der_points (int): Number of derivatives to show along the curve.
            der_scale (float): Scale factor for the derivative vectors.
            curvature_scale (float): Scale factor for the curvature indicators.
            highlight_self_intersections (bool): Whether to highlight potential self-intersections.
            min_intersection_distance (float): Minimum distance to consider as a self-intersection.
            
        Returns:
            matplotlib.axes.Axes: The axes containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the original points
        if show_points:
            ax.scatter(self.points[:, 0], self.points[:, 1], 
                       color=point_color, s=30, label='Puntos originales')
            
            # Optionally, add point indices for better reference
            for i, point in enumerate(self.points):
                ax.text(point[0], point[1], str(i), fontsize=8, 
                        ha='center', va='center', color='black', 
                        bbox=dict(facecolor='white', alpha=0.5, pad=0))
        
        # Plot the curve
        curve_points = self.evaluate(n_points=n_curve_points)
        ax.plot(curve_points[:, 0], curve_points[:, 1], color=curve_color, 
                linewidth=2, label='B-spline cúbico')
        
        # Check for self-intersections
        if highlight_self_intersections:
            intersection_count = 0
            for i in range(n_curve_points):
                for j in range(i+3, n_curve_points-1):
                    # Skip adjacent points
                    if abs(i-j) <= 3 or (self.closed and (n_curve_points - abs(i-j)) <= 3):
                        continue
                    
                    # Calculate distance between points
                    d = np.linalg.norm(curve_points[i] - curve_points[j])
                    if d < min_intersection_distance:  # Using the provided threshold
                        ax.plot([curve_points[i, 0], curve_points[j, 0]], 
                                [curve_points[i, 1], curve_points[j, 1]], 
                                'r--', linewidth=1, alpha=0.5)
                        intersection_count += 1
            
            if intersection_count > 0:
                print(f"Highlighted {intersection_count} potential self-intersections on the plot")
        
        # Plot derivatives if requested
        if show_derivatives:
            # Evaluate at evenly spaced parameters
            if self.closed:
                u_der = np.linspace(0, 1, n_der_points, endpoint=False)
            else:
                u_der = np.linspace(0, 1, n_der_points)
                
            curve_points_der = self.evaluate(u_der)
            derivatives = self.derivative(u_der)
            
            # Normalize and scale derivatives for visualization
            magnitudes = np.sqrt(np.sum(derivatives**2, axis=1))
            nonzero_mask = magnitudes > 1e-10
            normalized_ders = np.zeros_like(derivatives)
            normalized_ders[nonzero_mask] = derivatives[nonzero_mask] / magnitudes[nonzero_mask, np.newaxis]
            
            # Plot the derivatives as arrows
            for i in range(len(u_der)):
                if nonzero_mask[i]:
                    ax.arrow(curve_points_der[i, 0], curve_points_der[i, 1],
                             der_scale * normalized_ders[i, 0], der_scale * normalized_ders[i, 1],
                             head_width=der_scale/3, head_length=der_scale/2,
                             fc=der_color, ec=der_color)
        
        # Plot curvature if requested
        if show_curvature:
            if self.closed:
                u_curv = np.linspace(0, 1, n_der_points, endpoint=False)
            else:
                u_curv = np.linspace(0, 1, n_der_points)
                
            curve_points_curv = self.evaluate(u_curv)
            derivatives = self.derivative(u_curv)
            curvatures = self.curvature(u_curv)
            
            # Normalize derivatives to create normal vectors
            magnitudes = np.sqrt(np.sum(derivatives**2, axis=1))
            nonzero_mask = magnitudes > 1e-10
            normals = np.zeros_like(derivatives)
            # Normal is perpendicular to the tangent, so swap x,y and negate one
            normals[nonzero_mask, 0] = -derivatives[nonzero_mask, 1] / magnitudes[nonzero_mask]
            normals[nonzero_mask, 1] = derivatives[nonzero_mask, 0] / magnitudes[nonzero_mask]
            
            # Plot curvature indicators
            for i in range(len(u_curv)):
                if nonzero_mask[i] and curvatures[i] > 1e-10:
                    # Scale normal by curvature
                    n_scaled = normals[i] * curvatures[i] * curvature_scale
                    ax.arrow(curve_points_curv[i, 0], curve_points_curv[i, 1],
                            n_scaled[0], n_scaled[1],
                            head_width=curvature_scale/4, head_length=curvature_scale/3,
                            fc=curvature_color, ec=curvature_color)
        
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        ax.set_title('Interpolación con B-spline cúbico')
        
        return ax 