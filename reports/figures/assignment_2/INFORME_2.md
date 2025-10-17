# Report: Spline Optimization for Ellipse Approximation

## Introduction

This report details the optimization process for finding the best position of control points for a quadratic spline that approximates an ellipse. The goal was to improve the approximation by moving the points (not adding more), as part of the applied geometry assignment.

## Methodology

### The Original Problem

The original code generated a spline that passes through 5 points randomly positioned on an ellipse with semi-axes a=4 and b=5. Although the spline passed through these points exactly, the approximation to the complete ellipse was not optimal, as can be seen in the original image `ellipse_spline_20250505_160043.png`.

### Optimization Approach

To optimize the approximation:

1. We developed a random search algorithm that:
   - Generates different configurations of points on the ellipse
   - Evaluates each configuration by calculating the error (average distance) between the spline and the real ellipse
   - Selects the configuration with the lowest error

2. Error metrics:
   - For each point on the ellipse, we find the closest point on the spline
   - We calculate the root mean square of the distances (RMSE)

### Implementation

Three scripts were created:

1. `optimize_spline.py`: Searches for the best configuration of points by testing thousands of random configurations
2. `apply_optimized_points.py`: Applies the best configuration found and generates a final image
3. `interactive_spline.py`: Interactive version (for systems with graphical interface)

## Results

### Quantitative Improvement

| Configuration | RMSE Error | Improvement |
|---------------|------------|------------|
| Original spline | 1.016781 | - |
| Optimized spline | 0.329119 | 67.63% |

### Optimal Configuration Found

The optimized points (coordinates and angles) are:

| Point | Coordinates (x, y) | Angle (rad) |
|-------|-------------------|--------------|
| 1 | (3.9273, 0.9488) | 0.1909 |
| 2 | (2.3147, 4.0778) | 0.9537 |
| 3 | (-1.9238, 4.3837) | 2.0725 |
| 4 | (-3.5511, -2.3014) | 3.6199 |
| 5 | (3.8178, -1.4920) | 5.9802 |

### Visualization

The following images were generated:
- `optimized_ellipse_final_*.png`: Final image with the optimized spline
- `optimized_ellipse_spline_*.png`: Comparison images between the original and optimized spline

## Analysis

### Why Does It Work Better?

The optimized configuration:
1. Distributes the points more strategically around the ellipse
2. Places more points in regions of greater curvature
3. Optimizes to minimize the global error, not just to pass through specific points

### Limitations

- Random search does not guarantee finding the global optimum
- Using only 5 points limits the maximum achievable accuracy
- The error metric is an approximation

## Conclusions

This exercise demonstrates the importance of the strategic location of control points in splines. A spline with the same number of points (5) can approximate an ellipse much better if these points are optimally positioned.

The 67.63% improvement in error shows that it is not always necessary to increase the number of points to improve the approximation; sometimes it is enough to place them better.

## Future Work

- Implement other optimization algorithms (e.g., genetic algorithms)
- Study the optimal theoretical distribution of points for approximating conic curves
- Extend the approach to other more complex curve types 