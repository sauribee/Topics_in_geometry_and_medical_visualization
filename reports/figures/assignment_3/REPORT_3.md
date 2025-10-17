# Report: Quadratic B-Splines for Contours in Medical Images

## Introduction

This report details the implementation of quadratic B-splines for the interpolation of contours in medical images, specifically for MRI scans of the shoulder rotator cuff. The need to represent these anatomical contours accurately is crucial for the diagnosis and analysis of possible muscle tears, a common pathology in Colombia.

## Medical Context

The rotator cuff is a group of muscles and tendons that surround the shoulder joint, providing stability and allowing a wide range of motion. Injuries in this area are frequent, especially in people who perform repetitive arm movements.

Shoulder MRIs provide axial slices (typically 15-20) that allow visualization of the condition of these tissues. The precise delineation of the rotator cuff contour in these images is fundamental for:

1. Quantifying the degree of injury
2. Planning treatments
3. Monitoring post-treatment evolution
4. Performing comparative analyses

## Theoretical Foundations

### Quadratic B-Splines

B-splines are piecewise parametric functions that provide a high degree of continuity. A quadratic B-spline (degree 2) guarantees C¹ continuity throughout the curve, which means that both the curve and its first derivative are continuous.

The general formula for a B-spline is:

$$S(t) = \sum_{i=0}^{n} P_i N_{i,k}(t)$$

Where:
- $P_i$ are the control points
- $N_{i,k}(t)$ are the B-spline basis functions of degree $k$
- $t$ is the parameter that traverses the curve

For a quadratic B-spline, the basis functions are defined recursively, providing a smooth interpolation between control points.

### Chord Length Parameterization

A crucial aspect in the interpolation of anatomical contours is the non-uniform distribution of points. To address this problem, we implement chord length parameterization, which assigns parameter values proportional to the distance between consecutive points.

The parameterization is calculated as:

$$t_i = \frac{\sum_{j=1}^{i} d_j}{\sum_{j=1}^{n} d_j}$$

Where:
- $d_j$ is the Euclidean distance between points $j-1$ and $j$
- $t_i$ is the parameter value assigned to point $i$

This parameterization ensures that the curve better adapts to the spatial distribution of points, allocating more parametric "time" to regions where points are further apart.

## Implementation

### Code Structure

The implementation consists of the following main components:

1. **QuadraticBSpline**: Class that implements interpolation with quadratic B-splines.
2. **Data loading functions**: To import points from MATLAB files.
3. **Visualization functions**: To graphically represent the results.

### Interpolation Algorithm

The interpolation process follows these steps:

1. **Point loading**: The contour points are imported from a MATLAB file.
2. **Parameterization calculation**: Chord length parameterization is implemented.
3. **B-spline construction**: `scipy.interpolate.splprep` is used to construct the B-spline.
4. **Curve evaluation**: The B-spline is evaluated at a dense set of parameter values.
5. **Visualization**: The curve is graphically represented along with the original points.

### Handling Closed Curves

An important aspect is the handling of closed curves, since anatomical contours are usually closed structures. For this:

1. Additional points are added at the beginning and end to ensure continuity.
2. Periodicity conditions are used in the B-spline fitting.
3. It is ensured that the curve closes smoothly, without discontinuities.

### Calculation and Visualization of Derivatives

The implementation allows calculating and visualizing derivatives (tangents) along the curve, which provides additional information about the shape and smoothness of the contour. Derivatives are normalized for visualization as unit vectors.

## Results

### Contour Visualization

The main result is a B-spline curve that accurately interpolates the points of the rotator cuff contour, providing a smooth and continuous representation. The curve adapts to the non-uniform distribution of points thanks to the chord length parameterization.

### Parameterization Analysis

The chord length parameterization distributes the parameter values proportionally to the distances between consecutive points. This is reflected in the calculated values:

```
Parameter values: [0.0, 0.043, 0.096, 0.150, ..., 0.967, 1.0]
Chord lengths: [18.38, 22.20, 22.83, 20.10, ..., 11.70, 14.04]
```

This non-uniform distribution of the parameter allows the curve to better adapt to regions with different point densities.

## Advantages of B-Splines for Medical Contours

1. **Smooth representation**: B-splines provide smooth curves that better represent natural anatomical structures.
2. **Guaranteed continuity**: C¹ continuity ensures smooth transitions, eliminating artifacts that could be misinterpreted as anomalies.
3. **Computational efficiency**: The parametric representation is computationally efficient for subsequent operations such as area calculation or curvature analysis.
4. **Adaptability**: Chord length parameterization allows the curve to adapt to the spatial distribution of points.
5. **Robustness**: The implemented algorithm includes error handling mechanisms and special cases, such as very close points or problematic configurations.

## Limitations and Future Work

### Current Limitations

1. **Dependence on the number of points**: The quality of interpolation depends on the number and distribution of the original points.
2. **Local vs. global fitting**: B-splines provide a fit that may not be optimal from a global perspective.

### Future Work

1. **Point optimization**: Similar to the work done in Assignment 2, an algorithm could be implemented to optimize the position of control points.
2. **Higher degree splines**: Implement higher degree B-splines for applications requiring greater smoothness.
3. **Curvature analysis**: Develop tools to analyze curvature along the contour, which could provide additional diagnostic information.
4. **Automatic segmentation**: Integrate this technique with automatic medical image segmentation algorithms.

## Conclusions

The implementation of quadratic B-splines with chord length parameterization provides an effective solution for the representation of contours in medical images. The continuity and smoothness guaranteed by B-splines, together with the adaptability provided by non-uniform parameterization, make this technique particularly suitable for medical applications where precision and naturalness are crucial.

This approach lays the groundwork for future developments in medical image analysis, providing a mathematically robust representation that can be used for quantification, comparison, and advanced visualization of anatomical structures.

## References

1. de Boor, C. (1978). A Practical Guide to Splines. Springer-Verlag.
2. Piegl, L., & Tiller, W. (1997). The NURBS Book. Springer-Verlag.
3. Bartels, R. H., Beatty, J. C., & Barsky, B. A. (1987). An Introduction to Splines for Use in Computer Graphics and Geometric Modeling. Morgan Kaufmann.
4. Lee, E. T. Y. (1989). Choosing nodes in parametric curve interpolation. Computer-Aided Design, 21(6), 363-370. 