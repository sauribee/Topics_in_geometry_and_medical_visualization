#!/usr/bin/env python
# scripts/synthetic_ellipse_report.py

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

# Importa desde el paquete 'geometry'
from medvis.geometry.contour_fit import (
    BezierPWConfig,
    BSplineConfig,
    fit_contour_bezier_piecewise,
    fit_contour_bspline_closed,
)
from medvis.geometry.contour_slice_runner import SliceFitArtifacts
from medvis.geometry.contour_slice_runner import save_slice_plot
from medvis.geometry.contour_slice_io import (
    save_samples_csv,
    save_bezier_json,
    save_bspline_json,
)


def synthetic_ellipse_cloud(
    n: int = 360,
    a: float = 40.0,
    b: float = 25.0,
    rot_deg: float = 25.0,
    cx: float = 0.0,
    cy: float = 0.0,
    warp: float = 0.12,
    noise_sigma: float = 0.6,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Nube cerrada (x,y) ~ elipse rotada + deformación radial suave + ruido.
    Retorna un polígono cerrado implícito (N,2), sin repetir el primer punto al final.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)

    # elipse base
    x0 = a * np.cos(t)
    y0 = b * np.sin(t)

    # ligera deformación (armónicos) para asimetría anatómica
    r_warp = 1.0 + warp * np.cos(3 * t) + 0.5 * warp * np.sin(5 * t)
    x0 *= r_warp
    y0 *= r_warp

    # rotación
    th = np.deg2rad(rot_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], float)
    XY = np.column_stack([x0, y0]) @ R.T

    # traslación + ruido
    XY[:, 0] += cx
    XY[:, 1] += cy
    XY += rng.normal(scale=noise_sigma, size=XY.shape)
    return XY


def main():
    ap = argparse.ArgumentParser(
        description="Genera una nube elíptica sintética, ajusta (Bézier/B-spline) y guarda reporte."
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="reports",
        help="Directorio base de salida (por defecto: reports)",
    )
    ap.add_argument(
        "--base-name",
        type=str,
        default="ellipse_synth",
        help="Prefijo para los archivos de salida",
    )
    ap.add_argument(
        "--n", type=int, default=360, help="Número de puntos del contorno sintético"
    )
    ap.add_argument("--a", type=float, default=40.0, help="Semieje mayor de la elipse")
    ap.add_argument("--b", type=float, default=25.0, help="Semieje menor de la elipse")
    ap.add_argument("--rot", type=float, default=25.0, help="Rotación en grados")
    ap.add_argument(
        "--warp", type=float, default=0.12, help="Magnitud de deformación armónica"
    )
    ap.add_argument(
        "--noise",
        type=float,
        default=0.6,
        help="Desv. estándar del ruido (unidades de XY)",
    )
    ap.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")

    # Tolerancia de ajuste (en unidades de XY). Ajusta según escala de a,b.
    ap.add_argument(
        "--bezier-max-error",
        type=float,
        default=1.2,
        help="Error máximo (distancia vértice→polilínea de la curva) para el ajuste por tramos",
    )
    ap.add_argument(
        "--bezier-alpha",
        type=float,
        default=0.5,
        help="Parametrización: 1.0=longitud de cuerda, 0.5=centrípeta",
    )
    ap.add_argument(
        "--bezier-sample-n",
        type=int,
        default=200,
        help="Muestras uniformes del modelo Bézier cerrado para exportar",
    )
    ap.add_argument(
        "--spline-s",
        type=float,
        default=0.0,
        help="Parámetro de suavizado de splprep (0=interpolación)",
    )
    ap.add_argument(
        "--spline-k", type=int, default=3, help="Grado del B-spline (3=cúbico)"
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    csv_dir = out_dir / "csv"
    json_dir = out_dir / "models"

    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    # 1) Nube sintética
    XY = synthetic_ellipse_cloud(
        n=args.n,
        a=args.a,
        b=args.b,
        rot_deg=args.rot,
        warp=args.warp,
        noise_sigma=args.noise,
        seed=args.seed,
    )

    # 2) Ajustes
    bz_cfg = BezierPWConfig(
        max_error=float(args.bezier_max_error),
        parameterization_alpha=float(args.bezier_alpha),
        c1_enforce=True,
        sample_n=int(args.bezier_sample_n),
    )
    bs_cfg = BSplineConfig(
        s=float(args.spline_s),
        k=int(args.spline_k),
        sample_n=int(args.bezier_sample_n),
    )

    bz_res = fit_contour_bezier_piecewise(XY, cfg=bz_cfg)
    bs_res = fit_contour_bspline_closed(XY, cfg=bs_cfg)

    art = SliceFitArtifacts(contour_xy=XY, bezier=bz_res, bspline=bs_res)

    # 3) Guardados
    base = args.base_name

    save_samples_csv(csv_dir / f"{base}_contour.csv", XY)
    save_samples_csv(csv_dir / f"{base}_bezier_samples.csv", bz_res.samples_xy)
    save_samples_csv(csv_dir / f"{base}_bspline_samples.csv", bs_res.samples_xy)

    save_bezier_json(json_dir / f"{base}_bezier_model.json", art)
    save_bspline_json(json_dir / f"{base}_bspline_model.json", art)

    # 4) Figura con puntos de control (rojos), P0 (negro) y segmentos resaltados
    save_slice_plot(
        art,
        fig_dir / f"{base}_overlay.png",
        dpi=140,
        title="Ajuste de contorno sintético (Bézier por tramos / B-spline)",
        show_control_points=True,
        show_control_polygon=True,
        highlight_segments=True,
        cmap_name="viridis",
    )

    print(f"[OK] Reporte guardado en:\n  {fig_dir}\n  {csv_dir}\n  {json_dir}")


if __name__ == "__main__":
    main()
