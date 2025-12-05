#!/usr/bin/env python3
"""
Reporte de Aproximación de Bézier: Cráneo y Protuberancia
==========================================================

Genera visualizaciones de APROXIMACIÓN LSQ de Bézier para:
1. Contorno completo del cráneo
2. Protuberancia occipital (Y < threshold)
3. Grid comparativo de ambas

Usa aproximación de mínimos cuadrados con grados bajos para evitar
oscilaciones, en lugar de interpolación exacta.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from medvis.geometry.bezier import (
    chord_parameterization,
    fit_bezier_lsq,
)

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


def plot_approximation(
    all_points: np.ndarray,
    sampled_points: np.ndarray,
    degree: int,
    title: str,
    out_path: Path,
) -> None:
    """
    Genera visualización de aproximación LSQ con nube de puntos tenue.

    Parameters
    ----------
    all_points : (n, 2) array
        Todos los puntos del contorno (nube de puntos)
    sampled_points : (m, 2) array
        Puntos muestreados para aproximar
    degree : int
        Grado de la curva de Bézier
    title : str
        Título de la figura
    out_path : Path
        Ruta de salida para la figura
    """
    n_points = sampled_points.shape[0]

    # Aproximar con Bézier LSQ (grado fijo bajo)
    curve = fit_bezier_lsq(sampled_points, degree=degree, parameterization_alpha=0.5)
    control_points = curve.control_points

    # Calcular parámetros y errores
    t = chord_parameterization(sampled_points, alpha=0.5, normalize=True)

    # Calcular error de aproximación
    errors = []
    for i, p in enumerate(sampled_points):
        p_approx = curve.evaluate(t[i])
        error = np.linalg.norm(p - p_approx)
        errors.append(error)

    max_error = max(errors)
    mean_error = np.mean(errors)

    # Evaluar curva aproximada en muchos puntos
    t_dense = np.linspace(0, 1, 500)
    curve_dense = curve.evaluate_batch(t_dense)

    # Crear figura: 1x2
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.35, width_ratios=[1.3, 1])

    # Colores
    color_curve = "#E65100"  # Naranja oscuro para aproximación
    color_points = "#1565C0"  # Azul para puntos
    color_cloud = "#BDBDBD"  # Gris para nube de puntos

    # Panel 1: Curva aproximada
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

    # Curva aproximada
    ax1.plot(
        curve_dense[:, 0],
        curve_dense[:, 1],
        "-",
        lw=3,
        color=color_curve,
        label=f"Curva Bézier LSQ (grado {degree})",
        zorder=3,
    )

    # Puntos muestreados para aproximar
    ax1.plot(
        sampled_points[:, 0],
        sampled_points[:, 1],
        "o",
        ms=10,
        color=color_points,
        markeredgewidth=2,
        markeredgecolor="white",
        label=f"Puntos a aproximar (N={n_points})",
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
        f"{title}\nAproximación de Bézier LSQ (N={n_points} pts, grado {degree})",
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

    info_text = f"""
APROXIMACIÓN POR MÍNIMOS CUADRADOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sistema: minimizar ||A×C - P||²

• N puntos a aproximar: {n_points}
• Grado de Bézier: {degree}
• Puntos de control: {len(control_points)}

DIFERENCIA CON INTERPOLACIÓN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Grado MENOR que N-1
✓ Curva NO pasa exactamente por pts
✓ Minimiza error cuadrático global
✓ MÁS SUAVE, sin oscilaciones

ERROR DE APROXIMACIÓN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Error máximo: {max_error:.4f}
• Error medio: {mean_error:.4f}

{"✓ Error aceptable" if mean_error < 5.0 else "⚠️  Error alto"}

VENTAJAS LSQ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• NO sufre de Runge phenomenon
• Curvas más suaves
• Mejor para datos con ruido
• Grados bajos (5-7 típico)

DATOS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Nube total: {len(all_points)} puntos
• Muestreados: {n_points} puntos
• Método: Arc-length uniforme
• Parametrización: Chord-length α=0.5
    """

    ax2.text(
        0.05,
        0.95,
        info_text,
        transform=ax2.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.3),
    )

    fig.suptitle(
        f"Aproximación LSQ de Bézier: {title}",
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
    degree_skull: int,
    degree_prot: int,
    out_path: Path,
) -> None:
    """
    Crea un grid comparativo 1x2 con cráneo completo y protuberancia.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    color_curve = "#E65100"  # Naranja
    color_points = "#1565C0"  # Azul
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

    # Aproximar
    curve1 = fit_bezier_lsq(
        full_sampled, degree=degree_skull, parameterization_alpha=0.5
    )
    t_dense = np.linspace(0, 1, 500)
    curve1_dense = curve1.evaluate_batch(t_dense)

    ax1.plot(
        curve1_dense[:, 0],
        curve1_dense[:, 1],
        "-",
        lw=3,
        color=color_curve,
        label=f"Bézier LSQ (grado {degree_skull})",
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
        label=f"A aproximar (N={len(full_sampled)})",
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

    # Aproximar
    curve2 = fit_bezier_lsq(
        prot_sampled, degree=degree_prot, parameterization_alpha=0.5
    )
    curve2_dense = curve2.evaluate_batch(t_dense)

    ax2.plot(
        curve2_dense[:, 0],
        curve2_dense[:, 1],
        "-",
        lw=3,
        color=color_curve,
        label=f"Bézier LSQ (grado {degree_prot})",
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
        label=f"A aproximar (N={len(prot_sampled)})",
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
        "Comparación: Aproximación LSQ de Bézier en Contorno de Cráneo",
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
        description="Genera reporte de aproximación LSQ de Bézier para cráneo y protuberancia"
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
        help="Número de puntos para aproximar el cráneo completo (default: 20)",
    )
    parser.add_argument(
        "--n-points-prot",
        type=int,
        default=15,
        help="Número de puntos para aproximar la protuberancia (default: 15)",
    )
    parser.add_argument(
        "--degree-skull",
        type=int,
        default=7,
        help="Grado de Bézier para el cráneo (default: 7)",
    )
    parser.add_argument(
        "--degree-prot",
        type=int,
        default=4,
        help="Grado de Bézier para la protuberancia (default: 4)",
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
        default="reports/figures/bezier_skull_approximation",
        help="Directorio de salida para las figuras",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("REPORTE: APROXIMACIÓN LSQ DE BÉZIER - CRÁNEO Y PROTUBERANCIA")
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
    plot_approximation(
        skull_data["full"],
        full_sampled,
        args.degree_skull,
        "Contorno Completo del Cráneo",
        output_dir / "01_craneo_completo_approximation.png",
    )
    print()

    # 2. Figura de la protuberancia
    print("2. Generando figura de la protuberancia...")
    plot_approximation(
        protuberance,
        prot_sampled,
        args.degree_prot,
        "Protuberancia Occipital",
        output_dir / "02_protuberancia_approximation.png",
    )
    print()

    # 3. Grid comparativo
    print("3. Generando grid comparativo...")
    plot_comparison_grid(
        skull_data,
        full_sampled,
        prot_sampled,
        args.degree_skull,
        args.degree_prot,
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
        f"   2. Cráneo aproximado: {args.n_points_skull} puntos → grado {args.degree_skull}"
    )
    print(
        f"   3. Protuberancia aproximada: {args.n_points_prot} puntos → grado {args.degree_prot}"
    )
    print(f"   4. Umbral protuberancia: Y < {args.y_threshold}")
    print()
    print("🎯 VENTAJAS DE LSQ:")
    print("   • Sin oscilaciones (Runge phenomenon)")
    print("   • Curvas más suaves")
    print("   • Grados bajos (estables)")
    print("   • Mejor para datos con ruido")
    print()


if __name__ == "__main__":
    main()
