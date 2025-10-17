# Topics in Geometry and Medical Visualization (MedVis)

A modular Python project for **medical image I/O (DICOM/NIfTI)**, **basic preprocessing**, **geometric modeling with Bézier/B‑splines and equidistant sampling**, and **interactive 3D visualization** (PyVista/VTK). The repository follows a modern `src/` layout and is designed for **reproducibility** (tests, pre‑commit hooks, CI, Git‑LFS for large binaries).

---

## Repository Structure

```
Topics_in_geometry_and_medical_visualization/
├── src/medvis/              # Python package (installable)
│   ├── __init__.py
│   ├── io/                  # I/O: DICOM (SimpleITK/pydicom), NIfTI (NiBabel)
│   ├── preprocess/          # Filters, thresholding (HU), masks
│   ├── geometry/            # Bézier, B-splines, arc-length sampling
│   ├── viz/                 # Volume & surface rendering (PyVista/VTK)
│   └── cli.py               # Reproducible CLI pipelines
├── notebooks/               # Exploratory notebooks (paired with scripts if desired)
├── data/
│   ├── raw/                 # Raw medical data (DICOM/NIfTI) → tracked with Git‑LFS
│   │   ├── dicom/
│   │   └── nifti/
│   ├── interim/             # Intermediate artifacts (masks, labels)
│   └── processed/           # Derived meshes/tables (PLY/STL/CSV/NPZ) → Git‑LFS
├── reports/
│   └── figures/             # Plots, renders, exported images/PDFs
├── tests/                   # Unit/functional tests (pytest)
├── .gitattributes           # Git‑LFS patterns for medical/mesh files
├── .pre-commit-config.yaml  # Ruff/Black/nbstripout hooks
├── pyproject.toml           # Build metadata (project name: "medvis")
├── requirements.txt         # Base runtime requirements (optional if using pyproject)
├── requirements-dev.txt     # Dev tooling (pytest, pre-commit, ruff, black, jupytext)
└── environment.yml          # Optional conda env (mirrors requirements)
```

---

## Quickstart

### Prerequisites
- **Python 3.12** (coexists with 3.13 via `py.exe` on Windows or via Conda/Miniforge).  
- Recommended: a virtual environment (`venv`) or a conda env.

### Install (editable)
```bash
pip install -e .
pre-commit install
```
> If you prefer conda, create an environment and then run `pip install -e .` inside it.

### Data layout
Place anonymized medical datasets here:
```
data/
  raw/
    dicom/<study_or_series>/ ... .dcm
    nifti/<case>/ ... .nii[.gz]
```
Large files (e.g., `.dcm`, `.nii*`, `.stl`, `.ply`, `.obj`) are tracked by **Git‑LFS** via `.gitattributes`.

### CLI example
Process a DICOM series to produce orthogonal slices and an isosurface mesh:
```bash
python -m medvis.cli --dicom-dir data/raw/dicom/<your-series> \
                     --out-dir reports/figures/quick \
                     --level 300
```
Outputs:
- `reports/figures/quick/axial.nii.gz`, `coronal.nii.gz`, `sagital.nii.gz`
- `reports/figures/quick/isosurface.ply`

### Notebooks
Open `notebooks/00_quickstart_medvis.ipynb` for a hands‑on tour: reading a DICOM series, plotting axial/coronal/sagittal slices, and extracting a surface via marching cubes.

---

## Development

### Tests
```bash
pytest -q
```
Use synthetic volumes for robust, data‑independent tests (see `tests/test_viz_synthetic.py`).

### Lint & format
Hooks are configured with **pre‑commit**:
```bash
pre-commit run --all-files
```
Tools: **Ruff** (lint/format), **Black**, **nbstripout**.

### Continuous Integration
The GitHub Actions workflow installs the package, runs pre‑commit hooks, and executes tests on Python 3.12. Cache for pip is enabled to speed up runs.

---

## Dependencies
Runtime (core): `numpy`, `scipy`, `matplotlib`, `scikit-image`, `pydicom`, `SimpleITK`, `nibabel`, `pyvista`, `vtk`, `bezier>=2024.6.20`, `opencv-python-headless`.

Dev tooling: `pytest`, `pre-commit`, `ruff`, `black`, `jupytext`, `nbstripout` (see `requirements-dev.txt`).

> If you keep a conda environment, ensure `environment.yml` mirrors the above.

---

## License
Specify your license of choice (e.g., MIT/Apache‑2.0) in `LICENSE` and reference it here.

---

## Citation
If you publish results based on this repository, please cite this project. You can add a `CITATION.cff` file in the root so GitHub exposes a “Cite this repository” button.

---

## Acknowledgments
This project borrows best practices from modern Python packaging (`src/` layout), scientific Python ecosystem (ITK/SimpleITK, PyVista/VTK), and reproducible research workflows (pytest, pre‑commit, CI, Git‑LFS).

