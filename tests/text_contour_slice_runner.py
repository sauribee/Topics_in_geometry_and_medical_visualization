from pathlib import Path

import numpy as np

# Configure non-interactive backend BEFORE importing pyplot anywhere in the test session
import matplotlib

matplotlib.use("Agg", force=True)

from medvis.geometry.contour_slice_runner import (
    SliceMeta,
    RunnerConfig,
    process_slice,
    save_slice_plot,
)
from medvis.geometry.contour2d import ContourExtractionConfig
from medvis.geometry.contour_fit import BezierPWConfig, BSplineConfig


def _synthetic_disk(
    H: int = 256, W: int = 256, r: int = 70, cx: int = 128, cy: int = 128
):
    yy, xx = np.mgrid[0:H, 0:W]
    return (((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r).astype(np.uint8)


def test_slice_runner_plot_tmp_png(tmp_path: Path):
    # Synthetic mask (disk) with physical spacing
    mask = _synthetic_disk(H=220, W=240, r=70, cx=120, cy=110)
    meta = SliceMeta(spacing_xy=(0.5, 0.5), origin_xy=(0.0, 0.0), slice_id="synthetic")

    # Tight configs for a quick test
    contour_cfg = ContourExtractionConfig(
        min_points=300,
        ensure_ccw=True,
        pad_on_boundary=True,
        pad_width=1,
        spacing_xy=meta.spacing_xy,
        origin_xy=meta.origin_xy,
    )
    bezier_cfg = BezierPWConfig(max_error=0.5, sample_n=160)  # relaxed error for speed
    bspline_cfg = BSplineConfig(s=0.001, k=3, sample_n=160)

    artifacts = process_slice(
        mask,
        meta,
        RunnerConfig(contour=contour_cfg, bezier=bezier_cfg, bspline=bspline_cfg),
    )

    # Basic shape checks
    assert artifacts.contour_xy.ndim == 2 and artifacts.contour_xy.shape[1] == 2
    assert artifacts.bezier.samples_xy.shape == (bezier_cfg.sample_n, 2)
    assert artifacts.bspline.samples_xy.shape == (bspline_cfg.sample_n, 2)
    assert len(artifacts.bezier.model.segments) >= 1

    # Save a PNG to tmp
    out_png = tmp_path / "overlay.png"
    p = save_slice_plot(artifacts, out_png, dpi=110, title="synthetic slice")
    assert p.exists() and p.stat().st_size > 0
