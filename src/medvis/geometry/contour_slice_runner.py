from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
from numpy.typing import NDArray

from .contour2d import (
    ContourExtractionConfig,
    extract_primary_contour,
)
from .contour_fit import (
    BezierPWConfig,
    BSplineConfig,
    BezierPWResult,
    BSplineResult,
)

ArrayF = NDArray[np.float64]


__all__ = [
    "SliceMeta",
    "RunnerConfig",
    "SliceFitArtifacts",
    "process_slice",
    "plot_slice_fit",
    "save_slice_plot",
]


# ---------------------------------------------------------------------------
# Metadata and configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SliceMeta:
    """
    Minimal per-slice metadata for physical mapping.

    spacing_xy : (sx, sy) in mm/pixel along (x=columns, y=rows)
    origin_xy  : (ox, oy) in mm at pixel (row=0, col=0)
    """

    spacing_xy: Tuple[float, float]
    origin_xy: Tuple[float, float]
    slice_id: str
    direction_2x2: Optional[np.ndarray] = None


@dataclass(frozen=True)
class RunnerConfig:
    """
    End-to-end configuration for a 2D slice: extraction and fitting.

    contour : parameters for binary→contour extraction
    bezier  : parameters for piecewise cubic Bézier fitting
    bspline : parameters for closed B-spline fitting
    """

    contour: ContourExtractionConfig = ContourExtractionConfig()
    bezier: BezierPWConfig = BezierPWConfig()
    bspline: BSplineConfig = BSplineConfig()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


def process_slice(mask2d: np.ndarray, meta: SliceMeta, cfg) -> Dict[str, Any]:
    """
    Orquesta: extracción de contorno -> ajuste -> métricas/plots.
    Pasa la matriz de dirección in-plane al extractor.
    """
    contour_xy = extract_primary_contour(
        mask=mask2d,
        config=cfg.contour if hasattr(cfg, "contour") else None,
        direction_2x2=meta.direction_2x2,
    )

    out: Dict[str, Any] = {"contour_xy": contour_xy}
    return out


@dataclass(frozen=True)
class SliceFitArtifacts:
    """
    Artifacts returned by processing a 2D slice.

    contour_xy : (N,2) closed polygon in physical coordinates (x,y)
    bezier     : fitted piecewise cubic Bézier result
    bspline    : fitted closed B-spline result
    """

    contour_xy: ArrayF
    bezier: BezierPWResult
    bspline: BSplineResult


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _closed_for_plot(P: ArrayF) -> ArrayF:
    """Return a polyline with first vertex appended for closed plotting."""
    return np.vstack([P, P[0]])


def plot_slice_fit(
    art: SliceFitArtifacts, ax=None, title: Optional[str] = None, legend: bool = True
):
    """
    Plot original contour (dots) and both fitted curves (lines).

    Notes
    -----
    - Uses matplotlib only when this function is called.
    - Intended for quick QA; production plots can be done elsewhere.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Original contour (downsample for readability if very dense)
    C = art.contour_xy
    step = max(1, C.shape[0] // 400)
    ax.plot(C[::step, 0], C[::step, 1], ".", alpha=0.6, label="contour (data)")

    # Bézier piecewise (evenly-spaced samples)
    B = art.bezier.samples_xy
    ax.plot(
        _closed_for_plot(B)[:, 0],
        _closed_for_plot(B)[:, 1],
        "-",
        lw=1.5,
        label="piecewise Bézier",
    )

    # B-spline closed (evenly-spaced samples)
    S = art.bspline.samples_xy
    ax.plot(
        _closed_for_plot(S)[:, 0],
        _closed_for_plot(S)[:, 1],
        "-",
        lw=1.5,
        label="closed B-spline",
    )

    # Cosmetics
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    return ax


def save_slice_plot(
    art: SliceFitArtifacts,
    out_path: str | Path,
    dpi: int = 120,
    title: Optional[str] = None,
) -> Path:
    """
    Save a PNG figure overlaying contour and fitted curves.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)  # safe for headless environments
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_slice_fit(art, ax=ax, title=title, legend=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
