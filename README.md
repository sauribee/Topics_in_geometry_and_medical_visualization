# Topics in Geometry and Medical Visualization (MedVis)

A modular Python project for **geometric modeling with Bézier/B‑splines**, **medical contour analysis**, **numerical stability in curve fitting**, and **2D/3D visualization**.

______________________________________________________________________

## Repository Structure

```bash
Topics_in_geometry_and_medical_visualization/
├── src/medvis/                    # Python package (installable via pip install -e .)
│   ├── __init__.py
│   ├── io/                        # I/O: DICOM (SimpleITK/pydicom), NIfTI (NiBabel)
│   │   ├── dicom_series.py
│   │   ├── dicom_utils.py
│   │   └── nifti.py
│   ├── preprocess/                # Preprocessing and filtering
│   │   ├── outliers.py            # Outlier detection and removal
│   │   └── resample.py            # Resampling utilities
│   ├── geometry/                  # Geometric modeling (Bézier, B-splines, contours)
│   │   ├── bezier.py              # Bézier curves with numerical stability
│   │   ├── bezier_piecewise.py    # Piecewise cubic Bézier fitting
│   │   ├── bspline.py             # Open uniform B-spline interpolation
│   │   ├── contour2d.py           # 2D contour processing
│   │   ├── contour_fit.py         # High-level contour fitting
│   │   ├── contour_slice_io.py    # Contour I/O utilities
│   │   ├── contour_slice_runner.py # Batch contour processing
│   │   └── loft3d.py              # 3D lofting from 2D contours
│   ├── metrics/                   # Metrics and evaluation
│   ├── viz/                       # Visualization (PyVista/VTK, matplotlib)
│   │   ├── volume_viz.py          # Volume rendering utilities
│   │   └── mesh_viz.py            # Mesh visualization utilities
│   └── cli.py                     # CLI for reproducible pipelines
├── scripts/                       # Analysis and reporting scripts
│   ├── bezier_arc_chord_report.py              # Arc-chord parameterization analysis
│   ├── bezier_interpolation_curves_report.py   # Bézier interpolation (circle, ellipse, etc.)
│   ├── bezier_skull_approximation_report.py    # Skull LSQ approximation report
│   ├── bezier_skull_interpolation_report.py    # Skull Bézier interpolation report (single slice)
│   ├── bspline_skull_interpolation_report.py   # Skull B-spline interpolation report (single slice)
│   ├── bspline_skull_slices_batch_report.py    # 🆕 B-spline batch analysis (10 slices)
│   ├── generate_synthetic_contours_report.py   # Comprehensive synthetic data analysis
│   └── synthetic_ellipse_report.py             # Ellipse fitting comparison
├── data/                          # Data directory
│   ├── skull/                     # Skull contour data (single slice - original)
│   │   ├── borde_craneo_parte_izquierda_Eje_X.txt
│   │   ├── borde_craneo_parte_izquierda_Eje_Y.txt
│   │   ├── borde_craneo_parte_derecha_Eje_X.txt
│   │   └── borde_craneo_parte_derecha_Eje_Y.txt
│   ├── skull_edges/               # 🆕 Multi-slice skull data (10 axial slices)
│   │   ├── corte0/ ... corte9/    # Each contains: x.txt, y.txt, img.txt, shape.txt
│   │   └── [418-828 points per slice]
│   ├── external/                  # External datasets
│   ├── interim/                   # Intermediate processing results
│   ├── processed/                 # Final processed data
│   └── raw/                       # Raw data (DICOM/NIfTI if available)
├── reports/                       # Generated reports and outputs
│   ├── figures/                   # Generated plots and visualizations
│   │   ├── bezier_bsplines_reports/         # Bézier and B-spline analysis
│   │   │   ├── arc_chord_parameterization/  # Arc-chord parameterization study
│   │   │   ├── bezier_approximation_analysis/ # LSQ approximation comparisons
│   │   │   ├── bezier_interpolation_analysis/ # Interpolation error analysis
│   │   │   └── bezier_interpolation_curves/   # Geometric shapes (circle, ellipse, etc.)
│   │   ├── skull_reports/                   # Skull contour analysis
│   │   │   ├── bezier_skull_approximation/  # Skull LSQ Bézier approximation
│   │   │   ├── bezier_skull_interpolation/  # Skull Bézier interpolation
│   │   │   ├── bspline_skull_interpolation/ # Skull B-spline interpolation
│   │   │   └── skull_slices_bspline/        # 🆕 Multi-slice B-spline (10 slices)
│   │   └── synthetic_reports/               # Synthetic data analysis
│   │       ├── synthetic_report_00/         # Circle analysis
│   │       ├── synthetic_report_01/         # Ellipse analysis
│   │       ├── synthetic_report_02/         # Parabola fragment
│   │       ├── synthetic_report_03/         # Lemniscate
│   │       ├── synthetic_report_04/         # Noisy perturbed curves
│   │       └── synthetic_report_05/         # Additional synthetic shapes
│   ├── csv/                       # Exported CSV data (contours, samples)
│   └── models/                    # Serialized curve models (JSON)
├── tests/                         # Unit and functional tests (pytest)
│   ├── test_bezier_param.py       # Bézier parameterization tests
│   ├── test_import.py             # Import smoke tests
│   ├── test_io_synthetic.py       # I/O tests with synthetic data
│   ├── test_viz_synthetic.py      # Visualization tests
│   └── text_contour_slice_runner.py  # Contour processing tests
├── .gitattributes                 # Git attributes (LFS patterns)
├── .gitignore                     # Git ignore patterns
├── .pre-commit-config.yaml        # Pre-commit hooks (Ruff, Black)
├── pyproject.toml                 # Build metadata (project: "medvis")
├── requirements.txt               # Runtime dependencies
└── README.md                      # This file
```

______________________________________________________________________

## Quickstart

### Prerequisites

- **Python 3.12+** (tested on 3.12)
- Recommended: a virtual environment (`venv` or conda)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Topics_in_geometry_and_medical_visualization.git
cd Topics_in_geometry_and_medical_visualization

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in editable mode
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### Running Analysis Scripts

The `scripts/` folder contains ready-to-use analysis pipelines:

#### 1. Skull Bézier Interpolation Report

Generates exact interpolation of skull contour and protuberance using Bézier curves:

```bash
python scripts/bezier_skull_interpolation_report.py \
    --n-points-skull 12 \
    --n-points-prot 6 \
    --y-threshold 52
```

**Output:** 3 figures (full skull, protuberance, comparison grid)
**Method:** Exact Bézier interpolation (degree = N-1)

#### 2. Skull Bézier LSQ Approximation Report

Generates smooth approximation using least-squares Bézier curves:

```bash
python scripts/bezier_skull_approximation_report.py \
    --n-points-skull 14 \
    --n-points-prot 8 \
    --degree-skull 8 \
    --degree-prot 5 \
    --y-threshold 52
```

**Output:** 3 figures with error metrics
**Method:** LSQ approximation with fixed low degree (no oscillations)

#### 3. Skull B-spline Interpolation Report

Generates exact interpolation using B-splines (local control):

```bash
python scripts/bspline_skull_interpolation_report.py \
    --n-points-skull 14 \
    --n-points-prot 7 \
    --degree 3 \
    --y-threshold 52
```

**Output:** 3 figures (full skull, protuberance, comparison grid)
**Method:** B-spline interpolation with local control (degree 3 = cubic)

#### 4. 🆕 Multi-Slice B-spline Batch Report (10 Axial Slices)

Processes 10 axial skull slices with B-spline interpolation and protuberance detection:

```bash
python scripts/bspline_skull_slices_batch_report.py \
    --slice-start 0 \
    --slice-end 9 \
    --n-samples-full 20 \
    --n-samples-prot 10 \
    --degree 3 \
    --y-threshold 50
```

**Output:**

- 10 individual slice reports
- Multi-slice comparison grid (2×5)
- Metrics summary plots
- CSV file with statistics

**Method:** B-spline batch processing with automatic protuberance detection

**Key Features:**

- Consistent methodology across all slices
- Protuberance detection: 7/10 slices (cortes 0-6)
- Average 568 points per slice
- Error: ~10⁻¹³ (machine precision)

#### 5. Bézier Interpolation on Geometric Shapes

Analyzes Bézier interpolation on circle, ellipse, parabola, and lemniscate:

```bash
python scripts/bezier_interpolation_curves_report.py \
    --n-points 8 \
    --output-dir reports/figures/bezier_bsplines_reports/bezier_interpolation_curves
```

**Output:** Individual figures for each shape + comparison grid

#### 5. Arc-Chord Parameterization Analysis

Compares different parameterization methods (uniform, chord-length, centripetal, arc-length):

```bash
python scripts/bezier_arc_chord_report.py
```

**Output:** Detailed comparison of parameterization effects on curve fitting

#### 6. Comprehensive Synthetic Contours Report

Generates detailed analysis of various synthetic contours (6 different reports):

```bash
python scripts/generate_synthetic_contours_report.py
```

**Output:** Multiple report folders in `reports/figures/synthetic_reports/`

#### 7. Synthetic Ellipse Analysis

Quick comparison of Bézier interpolation, LSQ, and B-spline on synthetic ellipse:

```bash
python scripts/synthetic_ellipse_report.py
```

**Output:** Single comparative analysis of the three main methods

### Data Structure

The project includes skull contour data in `data/skull/`:

- Left and right skull borders (X and Y coordinates)
- Text format, space-separated values
- Used for demonstrating Bézier and B-spline fitting

______________________________________________________________________

## Key Features

### Geometric Modeling

- **Bézier Curves** (`geometry/bezier.py`)

  - Numerically stable Bernstein matrix computation (log-space for high degrees)
  - Chord-length, arc-chord, and centripetal parameterization
  - Interpolation (for N ≤ 10 points) and LSQ approximation (stable for any N)
  - Automatic warnings for ill-conditioned high-degree fitting
  - De Casteljau evaluation and subdivision

- **Piecewise Cubic Bézier** (`geometry/bezier_piecewise.py`)

  - Adaptive curve subdivision with error control
  - C1 continuity enforcement at joints
  - Robust for large point sets (no numerical instability)

- **B-splines** (`geometry/bspline.py`)

  - Open uniform B-spline interpolation
  - Averaged knot vector generation
  - Cox-de Boor evaluation

### Contour Processing

- 2D contour smoothing, resampling, and filtering (`geometry/contour2d.py`)
- Outlier detection and removal (`preprocess/outliers.py`)
- Arc-length parameterization for uniform sampling
- 3D lofting from multiple 2D contours (`geometry/loft3d.py`)

### Numerical Stability

**Critical improvements for high-degree curve fitting:**

- Log-space computation prevents overflow in binomial coefficients (n > 20)
- SVD-based least-squares solving for ill-conditioned systems
- User warnings when degree > 15 (interpolation) or > 20 (LSQ)
- Condition number monitoring for Bernstein matrices

**Best practices implemented:**

- Use LSQ approximation (degree 5-15) instead of high-degree interpolation
- Piecewise cubic Bézier for complex shapes (automatic segmentation)
- Arc-length uniform sampling instead of index-based sampling

______________________________________________________________________

## Development

### Testing

Run all tests:

```bash
pytest -q
```

Run specific test modules:

```bash
pytest tests/test_bezier_param.py -v
pytest tests/test_import.py -v
```

Tests use synthetic data for reproducibility (no external dependencies).

### Code Quality

Pre-commit hooks are configured for automatic formatting:

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run
```

**Tools configured:**

- **Ruff**: Fast Python linter (replaces flake8, isort)
- **Black**: Code formatter
- **nbstripout**: Remove notebook outputs before commit

### Project Structure Best Practices

- `src/` layout for installable package
- Separate `scripts/` for analysis pipelines
- `reports/` for reproducible outputs (figures, CSV, models)
- `tests/` with pytest for unit/integration tests

______________________________________________________________________

## Dependencies

### Runtime (Core)

Defined in `requirements.txt`:

- **Numerical:** `numpy>=2.1`, `scipy>=1.14`
- **Geometry:** `bezier>=2024.6.20`
- **Image Processing:** `scikit-image>=0.25.2`, `opencv-python-headless`
- **Visualization:** `matplotlib`, `pyvista`, `vtk`
- **Medical I/O:** `SimpleITK`, `pydicom`, `nibabel`

### Development Tools

Included in `requirements.txt`:

- **Testing:** `pytest`
- **Code Quality:** `pre-commit`, `ruff`, `black`, `jupytext`, `nbstripout`

### Installation (Core)

All dependencies are installed via:

```bash
pip install -e .
```

This installs both runtime and development dependencies from `requirements.txt`.

______________________________________________________________________
