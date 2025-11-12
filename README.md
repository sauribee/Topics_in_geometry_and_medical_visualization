# Topics in Geometry and Medical Visualization (MedVis)

A modular Python project for **geometric modeling with Bézier/B‑splines**, **medical contour analysis**, **numerical stability in curve fitting**, and **2D/3D visualization**. The repository follows a modern `src/` layout and is designed for **reproducibility** (tests, pre‑commit hooks, automated analysis scripts).

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
│   ├── bezier_bspline_interpolation_report.py  # Compare Bézier/B-spline methods
│   ├── generate_synthetic_contours_report.py   # Generate synthetic data reports
│   ├── skull_lemniscate_interpolation.py       # Skull lemniscate analysis
│   ├── skull_protuberance_analysis.py          # Protuberance approximation (6 configs)
│   └── synthetic_ellipse_report.py             # Ellipse fitting report
├── data/                          # Data directory
│   ├── skull/                     # Skull contour data (left/right borders)
│   │   ├── borde_craneo_parte_izquierda_Eje_X.txt
│   │   ├── borde_craneo_parte_izquierda_Eje_Y.txt
│   │   ├── borde_craneo_parte_derecha_Eje_X.txt
│   │   └── borde_craneo_parte_derecha_Eje_Y.txt
│   ├── external/                  # External datasets
│   ├── interim/                   # Intermediate processing results
│   ├── processed/                 # Final processed data
│   └── raw/                       # Raw data (DICOM/NIfTI if available)
├── reports/                       # Generated reports and outputs
│   ├── figures/                   # Generated plots and visualizations
│   │   ├── bezier_bspline_interpolation/  # Bézier vs B-spline comparisons
│   │   ├── skull_lemniscate/      # Lemniscate fitting results
│   │   ├── skull_protuberance/    # Protuberance analysis (LSQ deg 5-10)
│   │   └── synthetic_report_*/    # Synthetic data analysis results
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

#### 1. Skull Protuberance Analysis (Bézier LSQ Fitting)

Analyzes skull contour protuberance with 6 different degree configurations (5-10):

```bash
python scripts/skull_protuberance_analysis.py \
    --data-dir data/skull \
    --out-dir reports/figures/skull_protuberance \
    --y-threshold 50
```

**Output:** 2x3 grid comparing LSQ degrees 5, 6, 7, 8, 9, 10

#### 2. Skull Lemniscate Interpolation

Fits lemniscate-shaped curves to skull boundary:

```bash
python scripts/skull_lemniscate_interpolation.py \
    --data-dir data/skull \
    --out-dir reports/figures/skull_lemniscate \
    --n-samples 30
```

#### 3. Synthetic Ellipse Analysis

Compares Bézier interpolation, LSQ, and B-spline methods on synthetic ellipse data:

```bash
python scripts/synthetic_ellipse_report.py
```

#### 4. Comprehensive Synthetic Contours Report

Generates detailed analysis of various synthetic contours:

```bash
python scripts/generate_synthetic_contours_report.py
```

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

## Output Examples

### Generated Reports

All scripts generate structured outputs in `reports/`:

- **Figures** (`reports/figures/`): PNG plots with matplotlib/PyVista

  - Comparison grids (2x3, 1x3 layouts)
  - Overlay plots with original data + fitted curves
  - Error visualizations

- **CSV Data** (`reports/csv/`): Exported contour points, samples, and parameters

  - Original contour points
  - Sampled points for visualization
  - Curve evaluation points

- **Models** (`reports/models/`): JSON-serialized curve models

  - Control points
  - Degree information
  - Parameterization metadata

### Example Output Structure

```bash
reports/figures/skull_protuberance/
├── protuberance_comparison.png      # 2x3 grid (degrees 5-10)
├── protuberance_data.csv            # Original 56 points
├── samples_5pts.csv                 # Arc-length sampled points
├── samples_6pts.csv
├── ...
├── samples_10pts.csv
└── approximation_summary.txt        # Numerical error report
```

______________________________________________________________________

## Project Philosophy

### Numerical Stability First

This project emphasizes **robust numerical methods** over naive implementations:

- High-degree polynomial interpolation is **inherently unstable** (Runge's phenomenon)
- LSQ approximation with moderate degree (5-15) provides better results
- Piecewise methods avoid global ill-conditioning
- Arc-length sampling ensures geometric uniformity

### Reproducibility

- All analysis scripts are self-contained
- Outputs are versioned and timestamped
- Tests ensure consistent behavior across environments
- Pre-commit hooks maintain code quality

### Educational Value

The codebase serves as a reference for:

- Numerical methods in geometric modeling
- Best practices for curve fitting
- Scientific Python project structure
- Reproducible computational workflows

______________________________________________________________________

## Contributing

Contributions are welcome! Please:

1. Fork the repository
1. Create a feature branch (`git checkout -b feature/amazing-feature`)
1. Make your changes with tests
1. Run pre-commit hooks (`pre-commit run --all-files`)
1. Commit your changes (`git commit -m 'Add amazing feature'`)
1. Push to the branch (`git push origin feature/amazing-feature`)
1. Open a Pull Request

### Code Style

- Follow PEP 8 (enforced by Ruff and Black)
- Add docstrings to public functions (NumPy style preferred)
- Include type hints where appropriate
- Write tests for new functionality

______________________________________________________________________

## License

MIT License - see LICENSE file for details.

______________________________________________________________________

## Contact

For questions, issues, or collaboration:

- Open an issue on GitHub
- See `pyproject.toml` for project metadata

______________________________________________________________________

## Acknowledgments

This project builds upon:

- **Numerical methods**: De Casteljau algorithm, Cox-de Boor recursion
- **Python ecosystem**: NumPy, SciPy, matplotlib, PyVista
- **Medical imaging**: SimpleITK, pydicom, NiBabel
- **Best practices**: `src/` layout, pytest, pre-commit, reproducible science

Special thanks to the scientific Python community for excellent tools and documentation.
