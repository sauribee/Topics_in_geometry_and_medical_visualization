# Assignment 1: Spline Interpolation (1D)

This assignment implements linear and quadratic spline interpolation methods described in the "Spline Method of Interpolation" document.

## Description

The implementation provides:

1. **Linear Spline Interpolation** - Connects consecutive data points with straight lines. Simple but effective.
2. **Quadratic Spline Interpolation** - Fits quadratic polynomials between pairs of consecutive points, maintaining first derivative continuity at interior points.

It includes examples based on rocket motion, demonstrating how to calculate:
- Velocity at a specific point through interpolation
- Distance covered through spline integration
- Acceleration through spline differentiation

## Project Structure

```
assignment_1/
│
├── linear/                  # Linear Spline Implementation
│   ├── __init__.py
│   └── spline.py            # LinearSpline class and example_rocket
│
├── quadratic/               # Quadratic Spline Implementation
│   ├── __init__.py
│   └── spline.py            # QuadraticSpline class and example_rocket
│
├── utils/                   # Utilities for the project
│   ├── __init__.py
│   └── compare.py           # Functions to compare interpolation methods
│
├── __init__.py
└── main.py                  # Main script to run examples
```

## Usage

### Run all rocket examples

```bash
python -m assignment_1.main
```

### Run only a specific method

```bash
# Only Linear Spline
python -m assignment_1.main --method linear

# Only Quadratic Spline
python -m assignment_1.main --method quadratic

# Comparison of both methods
python -m assignment_1.main --method compare
```

### Change the evaluation point

```bash
python -m assignment_1.main --eval_point 18
```

### Use custom data

```bash
python -m assignment_1.main --custom --x 0 1 2 3 4 --y 0 1 4 9 16
```

### Runge Phenomenon Demonstration

To demonstrate why splines are preferable to high-order polynomials for interpolation, you can uncomment the Runge example execution in `main.py` and run:

```bash
python -m assignment_1.main
```

## Technical Details

### Linear Spline

Linear spline interpolation implements:
- Connecting points with straight lines
- Evaluating the interpolated function at any point within the range
- Visualizing the resulting spline

### Quadratic Spline

Quadratic spline interpolation implements:
- Fitting quadratic polynomials between points
- Maintaining first derivative continuity
- Calculating derivatives (acceleration) at any point
- Calculating integrals (distance) between any two points
- Visualizing the spline and its derivative

## Included Examples

1. **Example 1**: Calculation of rocket velocity at t = 16s using Linear Spline
2. **Example 2**: Complete rocket analysis using Quadratic Spline:
   - Velocity at t = 16s
   - Distance covered between t = 11s and t = 16s
   - Acceleration at t = 16s 