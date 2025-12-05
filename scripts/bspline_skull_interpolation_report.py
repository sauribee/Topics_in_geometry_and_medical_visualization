#!/usr/bin/env python3
"""
Reporte de Interpolación con B-splines: Cráneo y Protuberancia
===============================================================

Genera visualizaciones de interpolación con B-splines para:
1. Contorno completo del cráneo
2. Protuberancia occipital (Y < threshold)
3. Grid comparativo de ambas

Los B-splines ofrecen interpolación suave sin las oscilaciones
de Bézier de alto grado.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from medvis.geometry.bspline import (
    fit_bspline_interpolate,
    evaluate_bspline,
)
from medvis.geometry.bezier import chord_parameterization

# Configuración de estilo
plt.style.use("seaborn-v0_8-darkgrid")


def load_skull_data(data_dir: Path) -> dict:
    """
    Carga los datos del contorno del cráneo.

    Parameters
    ----------
    data_dir : Path
        Directorio con los archivos de datos

    Returns
    -------
    dict
        Diccionario con 'left', 'right' y 'full' contours
    """
    left_x = np.loadtxt(data_dir / "borde_craneo_parte_izquierda_Eje_X.txt")
    left_y = np.loadtxt(data_dir / "borde_craneo_parte_izquierda_Eje_Y.txt")
    right_x = np.loadtxt(data_dir / "borde_craneo_parte_derecha_Eje_X.txt")
    right_y = np.loadtxt(data_dir / "borde_craneo_parte_derecha_Eje_Y.txt")

    left_pts = np.column_stack([left_x, left_y])
    right_pts = np.column_stack([right_x, right_y])

    # Combinar para contorno completo
    full_contour = np.vstack([left_pts, right_pts[::-1]])

    return {
        "left": left_pts,
        "right": right_pts,
        "full": full_contour,
    }


def extract_protuberance(contour: np.ndarray, y_threshold: float = 50.0) -> np.ndarray:
    """
    Extrae los puntos de la protuberancia occipital como una curva abierta continua.

    Parameters
    ----------
    contour : (n, 2) array
        Puntos del contorno completo
    y_threshold : float
        Umbral en Y para identificar protuberancia (Y < threshold)

    Returns
    -------
    (m, 2) array
        Puntos de la protuberancia ordenados de izquierda a derecha
    """
    # Filtrar puntos por umbral de Y
    mask = contour[:, 1] < y_threshold
    prot_points = contour[mask].copy()

    if len(prot_points) == 0:
        return prot_points

    # Ordenar de izquierda a derecha (por coordenada X)
    # Esto garantiza que sea una curva abierta continua
    sorted_indices = np.argsort(prot_points[:, 0])
    prot_points = prot_points[sorted_indices]

    return prot_points


def sample_uniform_arclength(points: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Muestrea puntos uniformemente por longitud de arco.

    Parameters
    ----------
    points : (n, 2) array
        Puntos originales
    n_samples : int
        Número de puntos a muestrear

    Returns
    -------
    (n_samples, 2) array
        Puntos muestreados uniformemente
    """
    if len(points) < 2:
        return points

    # Calcular longitudes de arco acumuladas
    seg_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc_length = np.concatenate(([0], np.cumsum(seg_lengths)))

    # Normalizar
    if arc_length[-1] == 0:
        return points[:n_samples]

    arc_length_norm = arc_length / arc_length[-1]

    # Muestrear uniformemente
    target_s = np.linspace(0, 1, n_samples)

    # Interpolar para obtener nuevos puntos
    sampled_x = np.interp(target_s, arc_length_norm, points[:, 0])
    sampled_y = np.interp(target_s, arc_length_norm, points[:, 1])

    return np.column_stack([sampled_x, sampled_y])


def plot_bspline_interpolation(
    all_points: np.ndarray,
    sampled_points: np.ndarray,
    degree: int,
    title: str,
    out_path: Path,
) -> None:
    """
    Genera visualización de interpolación con B-splines y nube de puntos tenue.

    Parameters
    ----------
    all_points : (n, 2) array
        Todos los puntos del contorno (nube de puntos)
    sampled_points : (m, 2) array
        Puntos muestreados para interpolar
    degree : int
        Grado del B-spline
    title : str
        Título de la figura
    out_path : Path
        Ruta de salida para la figura
    """
    n_points = sampled_points.shape[0]

    # Interpolar con B-spline
    knots, control_points, p = fit_bspline_interpolate(sampled_points, degree=degree)

    # Calcular parámetros
    u = chord_parameterization(sampled_points, alpha=1.0, normalize=True)

    # Calcular error de interpolación
    errors = []
    for i, pt in enumerate(sampled_points):
        pt_interp = evaluate_bspline(knots, control_points, p, np.array([u[i]]))
        error = np.linalg.norm(pt - pt_interp[0])
        errors.append(error)

    max_error = max(errors)
    mean_error = np.mean(errors)

    # Evaluar curva interpolada en muchos puntos
    t_dense = np.linspace(0, 1, 500)
    curve_dense = evaluate_bspline(knots, control_points, p, t_dense)

    # Crear figura: 1x2
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.35, width_ratios=[1.3, 1])

    # Colores
    color_curve = "#6A1B9A"  # Morado para B-splines
    color_points = "#00897B"  # Verde azulado para puntos
    color_cloud = "#BDBDBD"  # Gris para nube de puntos

    # Panel 1: Curva interpolada con B-spline
    ax1 = fig.add_subplot(gs[0, 0])

    # Nube de puntos en el fondo (TENUE)
    ax1.scatter(
        all_points[:, 0],
        all_points[:, 1],
        s=8,
        c=color_cloud,
        alpha=0.25,
        label=f"Nube de puntos (N={len(all_points)})",
        zorder=1,
    )

    # Curva B-spline interpolada
    ax1.plot(
        curve_dense[:, 0],
        curve_dense[:, 1],
        "-",
        lw=3,
        color=color_curve,
        label=f"B-spline (grado {degree})",
        zorder=3,
    )

    # Puntos muestreados para interpolar
    ax1.plot(
        sampled_points[:, 0],
        sampled_points[:, 1],
        "o",
        ms=10,
        color=color_points,
        markeredgewidth=2,
        markeredgecolor="white",
        label=f"Puntos interpolados (N={n_points})",
        zorder=5,
    )

    # Marcar inicio y fin
    ax1.plot(
        sampled_points[0, 0],
        sampled_points[0, 1],
        "o",
        ms=14,
        color="green",
        markeredgewidth=2.5,
        markeredgecolor="white",
        zorder=6,
    )
    ax1.plot(
        sampled_points[-1, 0],
        sampled_points[-1, 1],
        "s",
        ms=14,
        color="red",
        markeredgewidth=2.5,
        markeredgecolor="white",
        zorder=6,
    )

    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("X", fontweight="bold")
    ax1.set_ylabel("Y", fontweight="bold")
    ax1.set_title(
        f"{title}\nInterpolación con B-spline (N={n_points} pts, grado {degree})",
        fontweight="bold",
        fontsize=12,
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(
        loc="upper left", bbox_to_anchor=(0, -0.12), fontsize=9, ncol=3, frameon=True
    )

    # Invertir eje Y para que sea como coordenadas de imagen
    ax1.invert_yaxis()

    # Panel 2: Información del sistema
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")

    # Calcular número de nudos internos
    n_internal_knots = len(knots) - 2 * (degree + 1)

    info_text = f"""
INTERPOLACIÓN CON B-SPLINES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Método: Open Uniform B-splines
con nudos promediados

• N puntos a interpolar: {n_points}
• Grado del B-spline: {degree}
• Puntos de control: {len(control_points)}
• Vector de nudos: {len(knots)} valores
• Nudos internos: {n_internal_knots}

CARACTERÍSTICAS B-SPLINES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Control LOCAL (no global)
✓ Mayor ESTABILIDAD numérica
✓ Curvas MÁS SUAVES
✓ Ideal para interpolación

ERROR DE INTERPOLACIÓN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Error máximo: {max_error:.2e}
• Error medio: {mean_error:.2e}

{"✓ Interpolación precisa" if mean_error < 1e-6 else "⚡ Error aceptable"}

VENTAJAS vs BÉZIER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• NO sufre de Runge phenomenon
• Control local (cambiar 1 pt
  afecta solo una región)
• Mejor para muchos puntos
• Curvas más naturales

DATOS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Nube total: {len(all_points)} puntos
• Muestreados: {n_points} puntos
• Método: Arc-length uniforme
• Parametrización: Chord-length
    """

    ax2.text(
        0.05,
        0.95,
        info_text,
        transform=ax2.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lavender", alpha=0.3),
    )

    fig.suptitle(
        f"Interpolación B-spline: {title}",
        fontsize=14,
        fontweight="bold",
        y=0.96,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Figura guardada: {out_path.name}")


def plot_comparison_grid(
    skull_data: dict,
    full_sampled: np.ndarray,
    prot_sampled: np.ndarray,
    degree: int,
    out_path: Path,
) -> None:
    """
    Crea un grid comparativo 1x2 con cráneo completo y protuberancia.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    color_curve = "#6A1B9A"  # Morado
    color_points = "#00897B"  # Verde azulado
    color_cloud = "#BDBDBD"

    # Panel 1: Cráneo completo
    ax1 = axes[0]

    # Nube de puntos tenue
    ax1.scatter(
        skull_data["full"][:, 0],
        skull_data["full"][:, 1],
        s=6,
        c=color_cloud,
        alpha=0.25,
        label=f"Nube (N={len(skull_data['full'])})",
        zorder=1,
    )

    # Interpolar con B-spline
    knots1, C1, p1 = fit_bspline_interpolate(full_sampled, degree=degree)
    t_dense = np.linspace(0, 1, 500)
    curve1_dense = evaluate_bspline(knots1, C1, p1, t_dense)

    ax1.plot(
        curve1_dense[:, 0],
        curve1_dense[:, 1],
        "-",
        lw=3,
        color=color_curve,
        label=f"B-spline (grado {degree})",
        zorder=3,
    )
    ax1.plot(
        full_sampled[:, 0],
        full_sampled[:, 1],
        "o",
        ms=8,
        color=color_points,
        markeredgewidth=2,
        markeredgecolor="white",
        label=f"Interpolados (N={len(full_sampled)})",
        zorder=5,
    )

    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title("Cráneo Completo", fontweight="bold", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(
        loc="upper left", bbox_to_anchor=(0, -0.1), fontsize=9, ncol=3, frameon=True
    )
    ax1.set_xlabel("X", fontweight="bold")
    ax1.set_ylabel("Y", fontweight="bold")
    ax1.invert_yaxis()

    # Panel 2: Protuberancia
    ax2 = axes[1]

    # Extraer nube de puntos de la protuberancia
    prot_cloud = extract_protuberance(skull_data["full"], y_threshold=50.0)

    # Nube de puntos tenue
    ax2.scatter(
        prot_cloud[:, 0],
        prot_cloud[:, 1],
        s=6,
        c=color_cloud,
        alpha=0.25,
        label=f"Nube (N={len(prot_cloud)})",
        zorder=1,
    )

    # Interpolar con B-spline
    knots2, C2, p2 = fit_bspline_interpolate(prot_sampled, degree=degree)
    curve2_dense = evaluate_bspline(knots2, C2, p2, t_dense)

    ax2.plot(
        curve2_dense[:, 0],
        curve2_dense[:, 1],
        "-",
        lw=3,
        color=color_curve,
        label=f"B-spline (grado {degree})",
        zorder=3,
    )
    ax2.plot(
        prot_sampled[:, 0],
        prot_sampled[:, 1],
        "o",
        ms=8,
        color=color_points,
        markeredgewidth=2,
        markeredgecolor="white",
        label=f"Interpolados (N={len(prot_sampled)})",
        zorder=5,
    )

    ax2.set_aspect("equal", adjustable="box")
    ax2.set_title("Protuberancia Occipital (Y < 50)", fontweight="bold", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(
        loc="upper left", bbox_to_anchor=(0, -0.1), fontsize=9, ncol=3, frameon=True
    )
    ax2.set_xlabel("X", fontweight="bold")
    ax2.set_ylabel("Y", fontweight="bold")
    ax2.invert_yaxis()

    fig.suptitle(
        "Comparación: Interpolación con B-splines en Contorno de Cráneo",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Grid comparativo guardado: {out_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Genera reporte de interpolación con B-splines para cráneo y protuberancia"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/skull",
        help="Directorio con los archivos de datos del cráneo",
    )
    parser.add_argument(
        "--n-points-skull",
        type=int,
        default=20,
        help="Número de puntos para interpolar el cráneo completo (default: 20)",
    )
    parser.add_argument(
        "--n-points-prot",
        type=int,
        default=15,
        help="Número de puntos para interpolar la protuberancia (default: 15)",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=3,
        help="Grado del B-spline (típico: 3 para cúbico) (default: 3)",
    )
    parser.add_argument(
        "--y-threshold",
        type=float,
        default=50.0,
        help="Umbral en Y para identificar protuberancia (default: 50.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/figures/bspline_skull_interpolation",
        help="Directorio de salida para las figuras",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("REPORTE: INTERPOLACIÓN CON B-SPLINES - CRÁNEO Y PROTUBERANCIA")
    print("=" * 70)
    print()

    # Cargar datos
    data_dir = Path(args.data_dir)
    print(f"📁 Cargando datos desde: {data_dir}")

    if not data_dir.exists():
        print(f"❌ Error: El directorio {data_dir} no existe")
        return

    skull_data = load_skull_data(data_dir)
    print(f"  ✓ Cargados {len(skull_data['full'])} puntos del contorno completo")
    print()

    # Extraer protuberancia
    print(f"🔍 Extrayendo protuberancia occipital (Y < {args.y_threshold})...")
    protuberance = extract_protuberance(skull_data["full"], args.y_threshold)
    print(f"  ✓ Encontrados {len(protuberance)} puntos en la protuberancia")
    print()

    # Muestrear puntos uniformemente
    print("📐 Muestreando puntos uniformemente por arc-length...")
    full_sampled = sample_uniform_arclength(skull_data["full"], args.n_points_skull)
    prot_sampled = sample_uniform_arclength(protuberance, args.n_points_prot)
    print(f"  ✓ Cráneo: {args.n_points_skull} puntos")
    print(f"  ✓ Protuberancia: {args.n_points_prot} puntos")
    print()

    # Directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Figura del cráneo completo
    print("1. Generando figura del cráneo completo...")
    plot_bspline_interpolation(
        skull_data["full"],
        full_sampled,
        args.degree,
        "Contorno Completo del Cráneo",
        output_dir / "01_craneo_completo_bspline.png",
    )
    print()

    # 2. Figura de la protuberancia
    print("2. Generando figura de la protuberancia...")
    plot_bspline_interpolation(
        protuberance,
        prot_sampled,
        args.degree,
        "Protuberancia Occipital",
        output_dir / "02_protuberancia_bspline.png",
    )
    print()

    # 3. Grid comparativo
    print("3. Generando grid comparativo...")
    plot_comparison_grid(
        skull_data,
        full_sampled,
        prot_sampled,
        args.degree,
        output_dir / "00_comparison_grid.png",
    )
    print()

    print("=" * 70)
    print("✅ REPORTE COMPLETADO")
    print("=" * 70)
    print()
    print(f"📁 Figuras generadas en: {output_dir}")
    print()
    print("📊 Total de figuras: 3")
    print()
    print("💡 CARACTERÍSTICAS:")
    print(f"   1. Nube de puntos original: {len(skull_data['full'])} puntos")
    print(
        f"   2. Cráneo interpolado: {args.n_points_skull} puntos con B-spline grado {args.degree}"
    )
    print(
        f"   3. Protuberancia interpolada: {args.n_points_prot} puntos con B-spline grado {args.degree}"
    )
    print(f"   4. Umbral protuberancia: Y < {args.y_threshold}")
    print()
    print("🎯 VENTAJAS DE B-SPLINES:")
    print("   • Control local (no global como Bézier)")
    print("   • Mayor estabilidad numérica")
    print("   • Curvas más suaves y naturales")
    print("   • Ideal para interpolación con muchos puntos")
    print()


if __name__ == "__main__":
    main()
