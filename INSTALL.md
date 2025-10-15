# Installation Guide

This guide will help you set up and run the Spline Interpolation project.

## Requirements

- Python 3.6 or higher
- Git (optional, for cloning the repository)

## Setting Up the Environment

### 1. Clone or download the repository (optional)

```bash
git clone <repository-url>
cd geometric_project
```

### 2. Set up the virtual environment

The project includes a virtual environment and setup script for easy installation.

#### On Linux/MacOS:

```bash
# Create the virtual environment (already done if you're using the provided repository)
python3 -m venv geometric_project_env

# Activate the environment and install dependencies using the script
./setup.sh
```

#### On Windows:

```bash
# Create the virtual environment (already done if you're using the provided repository)
python -m venv geometric_project_env

# Activate the environment
geometric_project_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Examples

After setting up the environment, you can run the examples:

### Basic Examples

```bash
# Run all examples
python -m spline_interpolation.main

# Run only the linear spline example
python -m spline_interpolation.main --method linear

# Run only the quadratic spline example
python -m spline_interpolation.main --method quadratic

# Run comparison between methods
python -m spline_interpolation.main --method compare
```

### Custom Data

You can also provide your own data points:

```bash
python -m spline_interpolation.main --custom --x 0 1 2 3 4 --y 0 1 4 9 16
```

### Runge Phenomenon Example

To run the Runge phenomenon example that demonstrates the advantages of splines over high-order polynomials, you need to uncomment the last line in `spline_interpolation/main.py`:

1. Open `spline_interpolation/main.py`
2. Uncomment the line: `# run_test_runge()`
3. Run the main script:

```bash
python -m spline_interpolation.main
```

## Deactivating the Environment

When you're done, you can deactivate the virtual environment:

```bash
deactivate
``` 