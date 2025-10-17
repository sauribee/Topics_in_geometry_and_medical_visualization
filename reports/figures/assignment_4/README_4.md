# Humerus Detection and Modeling

This project implements an advanced system for the detection and 3D modeling of the humerus from computerized tomography images.

## Project Structure

The project has been completely modularized to facilitate its maintenance and extension:

```
assignment_4/
├── humerus_detection/          # Main module for detection
│   ├── io/                    # Module for input/output operations
│   ├── preprocessing/         # Module for image preprocessing
│   ├── contour/               # Module for contour detection and manipulation
│   ├── spline/                # Module for B-spline fitting
│   └── visualization/         # Module for results visualization
├── run_detection.py           # Script to execute contour detection
├── run_visualization.py       # Script to generate 3D visualization
└── axial_sections/            # Directory with DICOM images of axial sections
```

## Requirements

This project requires the following Python libraries:

```
numpy
matplotlib
scipy
scikit-image
pydicom
imageio
```

You can install all dependencies with:

```bash
pip install numpy matplotlib scipy scikit-image pydicom imageio
```

## Usage

### Contour Detection

To run the humerus contour detection:

```bash
python assignment_4/run_detection.py --dicom_dir path/to/dicom/files --output_dir path/to/output
```

Parameters:
- `--dicom_dir`: Directory containing DICOM files (default: "assignment_4/axial_sections")
- `--output_dir`: Directory to save the results (default: "assignment_4/advanced_results")
- `--show`: Display images during processing (optional)

### 3D Visualization

To generate the 3D model from the detected contours:

```bash
python assignment_4/run_visualization.py --dicom_dir path/to/dicom/files --results_dir path/to/results --output_dir path/to/3d/model
```

Parameters:
- `--dicom_dir`: Directory containing the original DICOM files (default: "assignment_4/axial_sections")
- `--results_dir`: Directory with detection results (default: "assignment_4/advanced_results")
- `--output_dir`: Directory to save the 3D model (default: "assignment_4/3d_model")

## Main Features

1. **Robust Detection**: Uses Hough transform for a first approximation and active contours (snake) for fine adjustment.
2. **Spatial Continuity**: Maintains coherence between consecutive slices.
3. **Handling of Problematic Cases**: Special algorithms for images where detection is difficult.
4. **Natural Termination**: Correctly models the termination of the humerus in the final images.
5. **3D Visualization**: Generates a 3D model from the detected contours.

## Results

- Images with contours are saved in the output directory.
- The 3D model is saved as a static image (.png) and as an animated GIF with rotation.

## Algorithm

The algorithm implements the following stages:

1. **Preprocessing**: Normalization, contrast enhancement, and noise reduction.
2. **Contour detection**:
   - Approximate circle detection with Hough transform.
   - Precise contour search within the region of interest.
   - Fine adjustment with active contours (snake).
3. **B-spline Modeling**: Smoothing and regularization of contours.
4. **Visualization**: Generation of 2D and 3D visualizations.

## References

1. Lorensen, W. E., & Cline, H. E. (1987). Marching cubes: A high resolution 3D surface construction algorithm. ACM SIGGRAPH Computer Graphics, 21(4), 163-169.
2. Kass, M., Witkin, A., & Terzopoulos, D. (1988). Snakes: Active contour models. International Journal of Computer Vision, 1(4), 321-331.
3. Lee, E. T. Y. (1989). Choosing nodes in parametric curve interpolation. Computer-Aided Design, 21(6), 363-370.
4. Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62-66. 