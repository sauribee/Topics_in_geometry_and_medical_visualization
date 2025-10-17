from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import SimpleITK as sitk
except Exception:  # pragma: no cover
    sitk = None  # type: ignore[assignment]

from .contour_slice_runner import (
    SliceMeta,
    RunnerConfig,
    SliceFitArtifacts,
    process_slice,
    save_slice_plot,
)

ArrayF = NDArray[np.float64]
ArrayU8 = NDArray[np.uint8]

__all__ = [
    "load_mask2d_from_nifti",
    "load_mask2d_from_sitk",
    "save_samples_csv",
    "save_bezier_json",
    "save_bspline_json",
    "process_and_save",
]


# -----------------------------------------------------------------------------
# Loading 2D masks from NIfTI / SimpleITK
# -----------------------------------------------------------------------------


def _assert_sitk_available() -> None:
    if sitk is None:
        raise RuntimeError(
            "SimpleITK is not available. Install it to enable NIfTI/DICOM IO."
        )


def load_mask2d_from_sitk(
    img: object,  # use object to avoid type checker errors when sitk is None
    *,
    slice_index: Optional[int] = None,
    threshold: float = 0.5,
) -> Tuple[ArrayU8, SliceMeta]:
    """
    Extract a 2D binary mask from a SimpleITK image (2D or 3D axial slice).

    Assumptions
    -----------
    - We assume an **axial** slice if the image is 3D.
    - We assume identity (or near-identity) direction cosines. If your image
      uses non-trivial orientation, a full 3D→2D physical mapping should be
      applied (out of scope here, can be added later).

    Parameters
    ----------
    img : sitk.Image
        Input image. For label masks, values > threshold are considered True.
    slice_index : int, optional
        Axial slice index if `img` is 3D. If None, uses the middle slice.
    threshold : float
        Threshold for binarization when the image is numeric (not boolean labels).

    Returns
    -------
    mask2d : (H,W) uint8
        Binary mask (0/1).
    meta : SliceMeta
        Physical mapping for the (x,y) plane (spacing_xy, origin_xy).
    """
    _assert_sitk_available()

    dim = img.GetDimension()
    spacing = img.GetSpacing()  # (sx, sy, [sz])
    origin = img.GetOrigin()  # (ox, oy, [oz])

    if dim == 2:
        arr = sitk.GetArrayFromImage(img)  # (H, W) as numpy
        # SimpleITK GetArrayFromImage returns (row-major) yx order; interpret as (H,W)
        mask = (arr.astype(np.float64) > threshold).astype(np.uint8)
        sx, sy = float(spacing[0]), float(spacing[1])
        ox, oy = float(origin[0]), float(origin[1])
        meta = SliceMeta(spacing_xy=(sx, sy), origin_xy=(ox, oy), slice_id=None)
        return mask, meta

    if dim == 3:
        # Array shape is (Z, Y, X)
        vol = sitk.GetArrayFromImage(img)
        Z = vol.shape[0]
        k = slice_index if slice_index is not None else (Z // 2)
        if not (0 <= k < Z):
            raise IndexError(f"slice_index {k} out of range [0, {Z-1}]")
        arr2d = vol[k, :, :]  # (Y, X) ≡ (H, W)
        mask = (arr2d.astype(np.float64) > threshold).astype(np.uint8)
        sx, sy = float(spacing[0]), float(spacing[1])
        ox, oy = float(origin[0]), float(origin[1])
        # Note: we ignore z-origin here; this is a 2D mapping for the slice.
        meta = SliceMeta(spacing_xy=(sx, sy), origin_xy=(ox, oy), slice_id=f"z{k}")
        return mask, meta

    raise ValueError(f"Unsupported image dimension: {dim}")


def load_mask2d_from_nifti(
    nifti_path: str | Path,
    *,
    slice_index: Optional[int] = None,
    threshold: float = 0.5,
) -> Tuple[ArrayU8, SliceMeta]:
    """
    Read a NIfTI file using SimpleITK and return a 2D mask (binary) with metadata.

    Parameters
    ----------
    nifti_path : str | Path
        Path to a NIfTI file (.nii, .nii.gz).
    slice_index : int, optional
        Axial slice index for 3D images; if None, uses the middle slice.
    threshold : float
        Threshold for binarization when the image is numeric.

    Returns
    -------
    (mask2d, meta)
    """
    _assert_sitk_available()
    p = Path(nifti_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    img = sitk.ReadImage(str(p))
    return load_mask2d_from_sitk(img, slice_index=slice_index, threshold=threshold)


# -----------------------------------------------------------------------------
# Saving artifacts (CSV / JSON / PNG)
# -----------------------------------------------------------------------------


def save_samples_csv(path: str | Path, samples_xy: ArrayF) -> Path:
    """
    Save (x,y) samples into a CSV with header.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, samples_xy, delimiter=",", header="x,y", comments="", fmt="%.9f")
    return out


def save_bezier_json(
    path: str | Path,
    artifacts: SliceFitArtifacts,
) -> Path:
    """
    Save the piecewise Bézier model as JSON:
      - knots (global [0,1])
      - segments: control_points (4x2), per segment
      - metrics: max_error, mean_error
      - samples_xy: omitted here to keep file small (use CSV for samples)
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, object] = {
        "model": {
            "knots": artifacts.bezier.knots.tolist(),
            "segments": [
                {"control_points": seg.control_points.tolist()}
                for seg in artifacts.bezier.model.segments
            ],
        },
        "metrics": {
            "max_error": artifacts.bezier.metrics.max_error,
            "mean_error": artifacts.bezier.metrics.mean_error,
        },
        "slice": {
            "contour_points": int(artifacts.contour_xy.shape[0]),
            "bezier_samples": int(artifacts.bezier.samples_xy.shape[0]),
        },
    }
    out.write_text(json.dumps(data, indent=2))
    return out


def save_bspline_json(
    path: str | Path,
    artifacts: SliceFitArtifacts,
) -> Path:
    """
    Save the closed B-spline (SciPy tck) as JSON:
      - t: knot vector
      - c: control points (x_coeffs,y_coeffs); SciPy stores them separately
      - k: spline degree
      - periodic: True
      - metrics
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    tck = artifacts.bspline.tck
    t, c, k = tck  # t: (m,), c: [cx, cy], k: int
    data: Dict[str, object] = {
        "model": {
            "t": np.asarray(t, dtype=np.float64).tolist(),
            "cx": np.asarray(c[0], dtype=np.float64).tolist(),
            "cy": np.asarray(c[1], dtype=np.float64).tolist(),
            "k": int(k),
            "periodic": bool(artifacts.bspline.periodic),
        },
        "metrics": {
            "max_error": artifacts.bspline.metrics.max_error,
            "mean_error": artifacts.bspline.metrics.mean_error,
        },
        "slice": {
            "contour_points": int(artifacts.contour_xy.shape[0]),
            "bspline_samples": int(artifacts.bspline.samples_xy.shape[0]),
        },
    }
    out.write_text(json.dumps(data, indent=2))
    return out


# -----------------------------------------------------------------------------
# High-level: process a slice and save all outputs
# -----------------------------------------------------------------------------


def process_and_save(
    mask2d: np.ndarray,
    meta: SliceMeta,
    cfg: RunnerConfig,
    out_dir: str | Path,
    base_name: str = "slice",
    *,
    save_overlay: bool = True,
) -> Dict[str, Path]:
    """
    Run the full 2D slice pipeline and save:
      - contour.csv
      - bezier_samples.csv
      - bezier_model.json
      - bspline_samples.csv
      - bspline_model.json
      - overlay.png (optional)

    Parameters
    ----------
    mask2d : (H,W) array-like
        Binary/numeric mask. Foreground is > 0.5 if numeric.
    meta : SliceMeta
        Physical mapping (spacing_xy, origin_xy).
    cfg : RunnerConfig
        Extraction and fitting configuration.
    out_dir : str | Path
        Output directory where files will be stored.
    base_name : str
        Prefix for filenames.
    save_overlay : bool
        If True, save an overlay figure (mask background + curve overlays).

    Returns
    -------
    paths : Dict[str, Path]
        Mapping of artifact name → saved file path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run
    artifacts = process_slice(mask2d, meta, cfg)

    # Save plain contours (data) and samples
    p_contour = save_samples_csv(
        out_dir / f"{base_name}_contour.csv", artifacts.contour_xy
    )
    p_bz_samp = save_samples_csv(
        out_dir / f"{base_name}_bezier_samples.csv", artifacts.bezier.samples_xy
    )
    p_bs_samp = save_samples_csv(
        out_dir / f"{base_name}_bspline_samples.csv", artifacts.bspline.samples_xy
    )

    # Save models (JSON)
    p_bz_json = save_bezier_json(out_dir / f"{base_name}_bezier_model.json", artifacts)
    p_bs_json = save_bspline_json(
        out_dir / f"{base_name}_bspline_model.json", artifacts
    )

    # Optional overlay PNG (foreground mask)
    png_path: Optional[Path] = None
    if save_overlay:
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            # Draw the mask in physical coordinates
            H, W = mask2d.shape
            x0 = meta.origin_xy[0]
            y0 = meta.origin_xy[1]
            sx, sy = meta.spacing_xy
            extent = [x0, x0 + W * sx, y0 + H * sy, y0]
            ax.imshow(mask2d, cmap="gray", origin="upper", extent=extent)
            # Overlay curves
            from .contour_slice_runner import plot_slice_fit

            plot_slice_fit(
                artifacts, ax=ax, title=meta.slice_id or base_name, legend=True
            )
            ax.set_aspect("equal", adjustable="box")
            fig.tight_layout()
            png_path = out_dir / f"{base_name}_overlay.png"
            fig.savefig(png_path, dpi=120)
            plt.close(fig)
        except Exception:
            png_path = save_slice_plot(
                artifacts,
                out_dir / f"{base_name}_overlay.png",
                dpi=120,
                title=meta.slice_id or base_name,
            )

    paths: Dict[str, Path] = {
        "contour_csv": p_contour,
        "bezier_samples_csv": p_bz_samp,
        "bezier_model_json": p_bz_json,
        "bspline_samples_csv": p_bs_samp,
        "bspline_model_json": p_bs_json,
    }
    if png_path is not None:
        paths["overlay_png"] = png_path
    return paths
