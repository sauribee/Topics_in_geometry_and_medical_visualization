#!/usr/bin/env python3
"""
B-spline Protuberance Analysis Report
======================================

Analiza específicamente las protuberancias occipitales de los cortes
0-4 del cráneo usando interpolación B-spline.

Restricción: Y < 60 para identificar la protuberancia.
"""

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from medvis.geometry.bezier import chord_parameterization
from medvis.geometry.bspline import evaluate_bspline, fit_bspline_interpolate

# Configuración de estilo
plt.style.use("seaborn-v0_8-darkgrid")


def load_slice_points(data_dir: Path, slice_id: int) -> dict:
    """Carga los puntos de un corte específico."""
    slice_dir = data_dir / f"corte{slice_id}"

    if not slice_dir.exists():
        return None

    try:
        x_file = slice_dir / f"corte{slice_id}_x.txt"
        y_file = slice_dir / f"corte{slice_id}_y.txt"

        x = np.array(ast.literal_eval(x_file.read_text().strip()))
        y = np.array(ast.literal_eval(y_file.read_text().strip()))

        points = np.column_stack([x, y])

        return {"id": slice_id, "points": points, "n_points": len(x)}
    except Exception as e:
        print(f"  ❌ Error cargando corte {slice_id}: {e}")
        return None


def extract_protuberance(points: np.ndarray, y_threshold: float = 60.0) -> np.ndarray:
    """
    Extrae la protuberancia occipital.

    Parameters
    ----------
    points : (n, 2) array
        Puntos del contorno completo
    y_threshold : float
        Umbral Y (Y < threshold identifica protuberancia)

    Returns
    -------
    (m, 2) array
        Puntos de la protuberancia ordenados de izquierda a derecha
    """
    # Filtrar por umbral Y
    mask = points[:, 1] < y_threshold
    prot_points = points[mask].copy()

    if len(prot_points) == 0:
        return prot_points

    # Ordenar de izquierda a derecha (curva abierta continua)
    sorted_indices = np.argsort(prot_points[:, 0])
    prot_points = prot_points[sorted_indices]

    return prot_points


def sample_uniform_arclength(points: np.ndarray, n_samples: int) -> np.ndarray:
    """Muestrea puntos uniformemente por longitud de arco."""
    if len(points) < 2:
        return points

    if len(points) <= n_samples:
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

    # Interpolar
    sampled_x = np.interp(target_s, arc_length_norm, points[:, 0])
    sampled_y = np.interp(target_s, arc_length_norm, points[:, 1])

    return np.column_stack([sampled_x, sampled_y])


def analyze_protuberance(
    slice_data: dict,
    y_threshold: float = 60.0,
    n_samples: int = 10,
    degree: int = 3,
) -> dict:
    """
    Analiza la protuberancia de un corte con B-spline.

    Returns
    -------
    dict
        {
            'slice_id': int,
            'original_full': np.ndarray,
            'prot_original': np.ndarray,
            'prot_sampled': np.ndarray,
            'knots': np.ndarray,
            'control_points': np.ndarray,
            'degree': int,
            'curve_dense': np.ndarray,
            'error_mean': float,
            'success': bool
        }
    """
    points = slice_data["points"]
    slice_id = slice_data["id"]

    # Extraer protuberancia
    prot_points = extract_protuberance(points, y_threshold)

    if len(prot_points) < degree + 2:
        print(
            f"  ⚠️  Corte {slice_id}: Solo {len(prot_points)} puntos (mínimo: {degree+2})"
        )
        return {
            "slice_id": slice_id,
            "original_full": points,
            "prot_original": prot_points,
            "prot_sampled": prot_points,
            "knots": None,
            "control_points": None,
            "degree": None,
            "curve_dense": prot_points,
            "error_mean": 0,
            "success": False,
        }

    # Muestrear
    prot_sampled = sample_uniform_arclength(prot_points, n_samples)

    try:
        # Interpolar con B-spline
        knots, C, p = fit_bspline_interpolate(prot_sampled, degree=degree)

        # Evaluar curva densa
        t_dense = np.linspace(0, 1, 500)
        curve_dense = evaluate_bspline(knots, C, p, t_dense)

        # Calcular error
        u = chord_parameterization(prot_sampled, alpha=1.0, normalize=True)
        errors = []
        for i, pt in enumerate(prot_sampled):
            pt_interp = evaluate_bspline(knots, C, p, np.array([u[i]]))
            errors.append(np.linalg.norm(pt - pt_interp[0]))

        return {
            "slice_id": slice_id,
            "original_full": points,
            "prot_original": prot_points,
            "prot_sampled": prot_sampled,
            "knots": knots,
            "control_points": C,
            "degree": p,
            "curve_dense": curve_dense,
            "error_mean": np.mean(errors),
            "success": True,
        }

    except Exception as e:
        print(f"  ❌ Error en B-spline corte {slice_id}: {e}")
        return {
            "slice_id": slice_id,
            "original_full": points,
            "prot_original": prot_points,
            "prot_sampled": prot_sampled,
            "knots": None,
            "control_points": None,
            "degree": None,
            "curve_dense": prot_sampled,
            "error_mean": 0,
            "success": False,
        }


def plot_protuberance(analysis: dict, output_path: Path):
    """Genera visualización de la protuberancia de un corte."""
    slice_id = analysis["slice_id"]

    fig, ax1 = plt.subplots(figsize=(9, 9))

    color_cloud = "#BDBDBD"  # Gris
    color_curve = "#E65100"  # Naranja (protuberancia)
    color_points = "#00897B"  # Verde azulado

    # Nube de puntos completa (tenue)
    full_pts = analysis["original_full"]
    ax1.scatter(
        full_pts[:, 0],
        full_pts[:, 1],
        s=4,
        c=color_cloud,
        alpha=0.2,
        label=f"Contorno completo (N={len(full_pts)})",
        zorder=1,
    )

    # Puntos de protuberancia original
    prot_orig = analysis["prot_original"]
    if len(prot_orig) > 0:
        ax1.scatter(
            prot_orig[:, 0],
            prot_orig[:, 1],
            s=8,
            c=color_cloud,
            alpha=0.4,
            label=f"Protuberancia original (N={len(prot_orig)})",
            zorder=2,
        )

    # Curva B-spline
    if analysis["success"]:
        curve = analysis["curve_dense"]
        ax1.plot(
            curve[:, 0],
            curve[:, 1],
            "-",
            lw=3,
            color=color_curve,
            label=f"B-spline (grado {analysis['degree']})",
            zorder=4,
        )

        # Puntos muestreados
        sampled = analysis["prot_sampled"]
        ax1.plot(
            sampled[:, 0],
            sampled[:, 1],
            "o",
            ms=10,
            color=color_points,
            markeredgewidth=2,
            markeredgecolor="white",
            label=f"Puntos interpolados (N={len(sampled)})",
            zorder=5,
        )

        # Marcar inicio y fin
        ax1.plot(
            sampled[0, 0],
            sampled[0, 1],
            "o",
            ms=14,
            color="green",
            markeredgewidth=2.5,
            markeredgecolor="white",
            zorder=6,
        )
        ax1.plot(
            sampled[-1, 0],
            sampled[-1, 1],
            "s",
            ms=14,
            color="red",
            markeredgewidth=2.5,
            markeredgecolor="white",
            zorder=6,
        )

    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title(
        f"Corte {slice_id}: Protuberancia Occipital (Y < 60)",
        fontweight="bold",
        fontsize=14,
    )
    ax1.set_xlabel("X", fontweight="bold")
    ax1.set_ylabel("Y", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=10, frameon=True)
    ax1.invert_yaxis()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_comparison_grid(all_analyses: list, output_path: Path):
    """Grid comparativo de todas las protuberancias."""
    n = len(all_analyses)

    fig, axes = plt.subplots(1, n, figsize=(n * 4, 4.5))
    if n == 1:
        axes = [axes]

    color_cloud = "#BDBDBD"
    color_curve = "#E65100"

    for ax, analysis in zip(axes, all_analyses):
        # Contorno completo tenue
        full_pts = analysis["original_full"]
        ax.scatter(full_pts[:, 0], full_pts[:, 1], s=2, c=color_cloud, alpha=0.15)

        # Curva B-spline de protuberancia
        if analysis["success"]:
            curve = analysis["curve_dense"]
            ax.plot(curve[:, 0], curve[:, 1], lw=3, c=color_curve)

            # Puntos muestreados
            sampled = analysis["prot_sampled"]
            ax.plot(
                sampled[:, 0],
                sampled[:, 1],
                "o",
                ms=6,
                color="white",
                markeredgecolor=color_curve,
                markeredgewidth=2,
            )

        ax.set_aspect("equal")
        ax.set_title(
            f"Corte {analysis['slice_id']}\n(Prot: {len(analysis['prot_original'])} pts)",
            fontsize=10,
            fontweight="bold",
        )
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Protuberancias Occipitales: Cortes 0-4 (Y < 60)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Análisis de protuberancias occipitales con B-splines"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/skull_edges", help="Directorio de datos"
    )
    parser.add_argument(
        "--slices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="IDs de cortes a procesar (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--y-threshold",
        type=float,
        default=60.0,
        help="Umbral Y para protuberancia (default: 60)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Número de puntos a muestrear (default: 10)",
    )
    parser.add_argument(
        "--degree", type=int, default=3, help="Grado B-spline (default: 3)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/figures/protuberance_analysis",
        help="Directorio de salida",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ANÁLISIS DE PROTUBERANCIAS OCCIPITALES CON B-SPLINES")
    print("=" * 70)
    print()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print(f"📁 Directorio de datos: {data_dir}")
    print(f"🔍 Cortes a procesar: {args.slices}")
    print(f"📐 Restricción: Y < {args.y_threshold}")
    print(f"🎯 Parámetros: n_samples={args.n_samples}, grado={args.degree}")
    print()

    # Cargar y analizar
    print("📊 Cargando y analizando cortes...")
    all_analyses = []

    for slice_id in args.slices:
        print(f"\n  Procesando corte {slice_id}...")
        slice_data = load_slice_points(data_dir, slice_id)

        if slice_data is None:
            print(f"    ⚠️  Corte {slice_id} no encontrado")
            continue

        print(f"    ✓ Cargado: {slice_data['n_points']} puntos totales")

        analysis = analyze_protuberance(
            slice_data,
            y_threshold=args.y_threshold,
            n_samples=args.n_samples,
            degree=args.degree,
        )

        print(
            f"    {'✅' if analysis['success'] else '⚠️ '} Protuberancia: {len(analysis['prot_original'])} pts → {len(analysis['prot_sampled'])} muestreados"
        )

        all_analyses.append(analysis)

        # Graficar individual
        output_path = output_dir / "individual" / f"corte{slice_id}_protuberance.png"
        plot_protuberance(analysis, output_path)
        print(f"    ✓ Figura guardada: {output_path.name}")

    # Grid comparativo
    if all_analyses:
        print("\n📐 Generando grid comparativo...")
        grid_path = output_dir / "protuberances_comparison_grid.png"
        plot_comparison_grid(all_analyses, grid_path)
        print(f"  ✓ Grid guardado: {grid_path.name}")

    # Resumen
    print("\n" + "=" * 70)
    print("✅ ANÁLISIS COMPLETADO")
    print("=" * 70)
    print()
    print(f"📁 Directorio de salida: {output_dir}")
    print()
    print("📊 Resumen:")
    successful = sum(1 for a in all_analyses if a["success"])
    print(f"  • Cortes procesados: {len(all_analyses)}")
    print(f"  • Interpolaciones exitosas: {successful}/{len(all_analyses)}")
    print()
    print("📁 Archivos generados:")
    print(f"  • {len(all_analyses)} reportes individuales")
    print("  • 1 grid comparativo")
    print()


if __name__ == "__main__":
    main()
