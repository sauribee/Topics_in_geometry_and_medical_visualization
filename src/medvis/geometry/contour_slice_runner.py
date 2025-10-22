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


# --- dentro de contour_slice_runner.py ---


def _closed_for_plot(P: ArrayF) -> ArrayF:
    """Return a polyline with first vertex appended for closed plotting."""
    return np.vstack([P, P[0]])


def plot_slice_fit(
    art: SliceFitArtifacts,
    ax=None,
    title: Optional[str] = None,
    legend: bool = True,
    *,
    show_control_points: bool = True,
    show_control_polygon: bool = False,
    highlight_segments: bool = True,
    cmap_name: str = "viridis",
):
    """
    Grafica:
      - contorno de datos (puntos),
      - ajuste Bézier por tramos (segmentos coloreados en orden de generación),
      - ajuste B-spline cerrado (línea),
      - puntos de control (rojos) y P0 inicial (negro).

    Parámetros
    ----------
    show_control_points : dibuja puntos de control de cada segmento (rojo).
    show_control_polygon: si True, une los puntos de control con línea punteada tenue.
    highlight_segments  : si True, colorea cada segmento según su índice (progresión).
    cmap_name           : nombre de colormap de Matplotlib para los segmentos.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    # 1) Contorno original (discreto)
    C = art.contour_xy
    step = max(1, C.shape[0] // 400)
    ax.plot(
        C[::step, 0], C[::step, 1], ".", alpha=0.6, label="contorno (datos)", zorder=1
    )

    # 2) Bézier por tramos: resaltar segmento por segmento
    pw = art.bezier.model
    K = len(pw.segments)
    cmap = plt.get_cmap(cmap_name)

    # P0 global (primer punto de control del primer segmento)
    P0 = pw.segments[0].control_points[0]
    ax.scatter(
        [P0[0]],
        [P0[1]],
        s=48,
        c="k",
        marker="o",
        edgecolors="k",
        linewidths=0.6,
        zorder=5,
        label="P₀ inicial",
    )

    # Trazo por segmentos
    first_label_drawn = False
    for i, seg in enumerate(pw.segments):
        poly = seg.to_polyline(samples=160)

        if highlight_segments:
            tcolor = cmap(0.0 if K == 1 else i / (K - 1))
            lw = 2.2
            alpha = 0.95
        else:
            tcolor = "C1"
            lw = 1.6
            alpha = 1.0

        ax.plot(
            poly[:, 0],
            poly[:, 1],
            "-",
            lw=lw,
            color=tcolor,
            alpha=alpha,
            label=("Bézier (segmentos)" if not first_label_drawn else None),
            zorder=2,
        )
        first_label_drawn = True

        if show_control_points or show_control_polygon:
            cps = seg.control_points  # (4,2)
            if show_control_polygon:
                ax.plot(
                    cps[:, 0],
                    cps[:, 1],
                    "--",
                    color="red",
                    alpha=0.35,
                    lw=1.0,
                    zorder=3,
                )
            if show_control_points:
                ax.scatter(cps[:, 0], cps[:, 1], c="red", s=30, zorder=4)

    # 3) B-spline cerrado (muestra continua)
    S = art.bspline.samples_xy
    ax.plot(
        _closed_for_plot(S)[:, 0],
        _closed_for_plot(S)[:, 1],
        "-",
        lw=1.4,
        color="C2",
        alpha=0.9,
        label="B-spline cerrado",
        zorder=2,
    )

    # 4) Estética
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (unid.)")
    ax.set_ylabel("y (unid.)")
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
    **plot_kwargs,
) -> Path:
    """
    Guarda un PNG con la superposición pedida. Los kwargs se pasan a plot_slice_fit.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_slice_fit(art, ax=ax, title=title, legend=True, **plot_kwargs)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
