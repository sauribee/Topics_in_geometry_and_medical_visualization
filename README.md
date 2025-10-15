# Geometric Methods Project

This repository contains implementations of different geometric methods developed for the Applied Geometry course. Each assignment is contained in its own directory with specific implementations and examples.

## Assignments

The repository currently includes:

1. **Assignment 1: Spline Interpolation (1D)** - Implementation of linear and quadratic splines for 1D data interpolation with applications in rocket motion.
2. **Assignment 2: Parametric Spline Curves (2D)** - Implementation of 2D parametric curves using quadratic splines with applications in curve design.

Each assignment directory contains its own README with detailed instructions and usage examples.

## Requirements

The project requires:

- Python 3.6+
- NumPy
- Matplotlib
- SciPy

## Installation

### 1. Clone the repository:

```bash
git clone https://github.com/yourusername/geometric_project.git
cd geometric_project
```

### 2. Create a virtual environment (recommended):

```bash
python -m venv venv
```

#### On Windows:
```bash
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
source venv/bin/activate
```

### 3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Code

Each assignment has specific instructions for running its examples. Please refer to:

- [Assignment 1 Instructions](assignment_1/README.md)
- [Assignment 2 Instructions](assignment_2/README.md)

### Generate Plots in Headless Environment

If working in a headless environment (like a server without GUI), all examples have been updated to save plots as image files in each assignment's output directory:

```bash
python -m assignment_2.main --example ellipse --save_only
```

The saved plots will be available in the respective assignment's output directory.

## Repository Structure

```
geometric_project/
│
├── assignment_1/          # 1D Spline Interpolation
│   ├── README.md          # Specific instructions for Assignment 1
│   ├── linear/            # Linear spline implementation 
│   ├── quadratic/         # Quadratic spline implementation
│   └── ... 
│
├── assignment_2/          # 2D Parametric Spline Curves
│   ├── README.md          # Specific instructions for Assignment 2
│   ├── curves/            # Implementation of 2D curves
│   ├── output/            # Saved plot images
│   └── ...
│
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## License

This project is part of the Applied Geometry course and is provided for educational purposes. 