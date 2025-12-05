#!/usr/bin/env python3
"""
B-spline Skull Sides Connection Report
=======================================

Contornea los lados izquierdo y derecho de los cortes 6-9 con B-splines
y los une con una curva suave en la parte inferior.

Criterios:
- Lado izquierdo: X < 150
- Lado derecho: X > 250
- Conexión inferior: Y alrededor de 450

"""

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def separate_sides(
    points: np.ndarray, x_left: float = 150, x_right: float = 250
) -> dict:
    """
    Separa los puntos en lado izquierdo y derecho.

    Parameters
    ----------
    points : (n, 2) array
        Puntos del contorno
    x_left : float
        Umbral X para lado izquierdo (X < x_left)
    x_right : float
        Umbral X para lado derecho (X > x_right)

    Returns
    -------
    dict
        {
            'left': np.ndarray - puntos lado izquierdo,
            'right': np.ndarray - puntos lado derecho,
            'middle': np.ndarray - puntos intermedios (opcional)
        }
    """
    mask_left = points[:, 0] < x_left
    mask_right = points[:, 0] > x_right
    mask_middle = (~mask_left) & (~mask_right)

    left_points = points[mask_left].copy()
    right_points = points[mask_right].copy()
    middle_points = points[mask_middle].copy()

    # Ordenar lado izquierdo de arriba a abajo (por Y)
    left_points = left_points[np.argsort(left_points[:, 1])]

    # Ordenar lado derecho de arriba a abajo (por Y)
    right_points = right_points[np.argsort(right_points[:, 1])]

    return {
        "left": left_points,
        "right": right_points,
        "middle": middle_points,
    }


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


def create_bottom_connection(
    left_curve: np.ndarray,
    right_curve: np.ndarray,
    n_points: int = 20,
) -> np.ndarray:
    """
    Crea una curva suave tipo Bézier cúbica para conectar el lado izquierdo
    con el derecho en la parte inferior, garantizando continuidad suave (C1).

    Parameters
    ----------
    left_curve : (n, 2) array
        Curva del lado izquierdo completa
    right_curve : (m, 2) array
        Curva del lado derecho completa
    n_points : int
        Número de puntos para la conexión

    Returns
    -------
    (n_points, 2) array
        Puntos de conexión suave
    """
    # Punto inicial: final del lado izquierdo
    P0 = left_curve[-1]

    # Punto final: final del lado derecho
    P3 = right_curve[-1]

    # Calcular vector tangente al final del lado izquierdo
    # (dirección de los últimos puntos)
    if len(left_curve) > 5:
        tangent_left = left_curve[-1] - left_curve[-5]
        tangent_left = tangent_left / np.linalg.norm(tangent_left)
    else:
        tangent_left = np.array([0, 1])  # Dirección hacia abajo

    # Calcular vector tangente al final del lado derecho
    # (dirección de los últimos puntos)
    if len(right_curve) > 5:
        tangent_right = right_curve[-1] - right_curve[-5]
        tangent_right = tangent_right / np.linalg.norm(tangent_right)
    else:
        tangent_right = np.array([0, 1])

    # Calcular puntos de control para Bézier cúbica
    # P1: desde P0 en dirección tangente_left
    dist = np.linalg.norm(P3 - P0)
    P1 = P0 + tangent_left * (dist * 0.35)

    # P2: desde P3 en dirección opuesta a tangent_right
    P2 = P3 - tangent_right * (dist * 0.35)

    # Evaluar curva de Bézier cúbica
    t = np.linspace(0, 1, n_points)
    t = t.reshape(-1, 1)

    # Fórmula de Bézier cúbica
    bezier_curve = (
        (1 - t) ** 3 * P0
        + 3 * (1 - t) ** 2 * t * P1
        + 3 * (1 - t) * t**2 * P2
        + t**3 * P3
    )

    return bezier_curve


def analyze_skull_sides(
    slice_data: dict,
    x_left: float = 150,
    x_right: float = 250,
    n_samples_side: int = 20,
    n_samples_bottom: int = 15,
    degree: int = 3,
) -> dict:
    """
    Analiza y contornea los lados del cráneo con B-splines.

    Returns
    -------
    dict
        Análisis completo con curvas para cada lado y conexión
    """
    points = slice_data["points"]
    slice_id = slice_data["id"]

    # Separar lados
    sides = separate_sides(points, x_left, x_right)

    print(
        f"    Lado izquierdo: {len(sides['left'])} pts, Derecho: {len(sides['right'])} pts"
    )

    results = {
        "slice_id": slice_id,
        "original": points,
        "left_original": sides["left"],
        "right_original": sides["right"],
    }

    # LADO IZQUIERDO
    if len(sides["left"]) > degree + 1:
        left_sampled = sample_uniform_arclength(sides["left"], n_samples_side)

        try:
            knots_l, C_l, p_l = fit_bspline_interpolate(left_sampled, degree=degree)
            t_dense = np.linspace(0, 1, 300)
            left_curve = evaluate_bspline(knots_l, C_l, p_l, t_dense)

            results["left_sampled"] = left_sampled
            results["left_curve"] = left_curve
            results["left_success"] = True
        except Exception as e:
            print(f"    ⚠️  Error en lado izquierdo: {e}")
            results["left_sampled"] = left_sampled
            results["left_curve"] = left_sampled
            results["left_success"] = False
    else:
        results["left_sampled"] = sides["left"]
        results["left_curve"] = sides["left"]
        results["left_success"] = False

    # LADO DERECHO
    if len(sides["right"]) > degree + 1:
        right_sampled = sample_uniform_arclength(sides["right"], n_samples_side)

        try:
            knots_r, C_r, p_r = fit_bspline_interpolate(right_sampled, degree=degree)
            t_dense = np.linspace(0, 1, 300)
            right_curve = evaluate_bspline(knots_r, C_r, p_r, t_dense)

            results["right_sampled"] = right_sampled
            results["right_curve"] = right_curve
            results["right_success"] = True
        except Exception as e:
            print(f"    ⚠️  Error en lado derecho: {e}")
            results["right_sampled"] = right_sampled
            results["right_curve"] = right_sampled
            results["right_success"] = False
    else:
        results["right_sampled"] = sides["right"]
        results["right_curve"] = sides["right"]
        results["right_success"] = False

    # CONEXIÓN INFERIOR
    if results["left_success"] and results["right_success"]:
        # Crear conexión suave con Bézier cúbica (continuidad C1)
        # usando las curvas completas para calcular tangentes
        bottom_curve = create_bottom_connection(
            results["left_curve"], results["right_curve"], n_samples_bottom
        )

        results["bottom_sampled"] = bottom_curve
        results["bottom_curve"] = bottom_curve
        results["bottom_success"] = True
    else:
        results["bottom_sampled"] = np.array([])
        results["bottom_curve"] = np.array([])
        results["bottom_success"] = False

    return results


def plot_skull_sides(analysis: dict, output_path: Path):
    """Visualiza los lados contorneados y la conexión."""
    slice_id = analysis["slice_id"]

    fig, ax = plt.subplots(figsize=(10, 10))

    color_cloud = "#BDBDBD"  # Gris
    color_left = "#1E88E5"  # Azul
    color_right = "#E53935"  # Rojo
    color_bottom = "#43A047"  # Verde
    color_points = "#FFA726"  # Naranja

    # Nube de puntos original (tenue)
    original = analysis["original"]
    ax.scatter(
        original[:, 0],
        original[:, 1],
        s=3,
        c=color_cloud,
        alpha=0.2,
        label=f"Original (N={len(original)})",
        zorder=1,
    )

    # LADO IZQUIERDO
    if analysis["left_success"]:
        left_curve = analysis["left_curve"]
        ax.plot(
            left_curve[:, 0],
            left_curve[:, 1],
            "-",
            lw=3,
            color=color_left,
            label="Lado izquierdo (X < 150)",
            zorder=4,
        )

        left_sampled = analysis["left_sampled"]
        ax.plot(
            left_sampled[:, 0],
            left_sampled[:, 1],
            "o",
            ms=7,
            color=color_points,
            markeredgewidth=1.5,
            markeredgecolor="white",
            zorder=5,
        )

    # LADO DERECHO
    if analysis["right_success"]:
        right_curve = analysis["right_curve"]
        ax.plot(
            right_curve[:, 0],
            right_curve[:, 1],
            "-",
            lw=3,
            color=color_right,
            label="Lado derecho (X > 250)",
            zorder=4,
        )

        right_sampled = analysis["right_sampled"]
        ax.plot(
            right_sampled[:, 0],
            right_sampled[:, 1],
            "o",
            ms=7,
            color=color_points,
            markeredgewidth=1.5,
            markeredgecolor="white",
            zorder=5,
        )

    # CONEXIÓN INFERIOR
    if analysis["bottom_success"]:
        bottom_curve = analysis["bottom_curve"]
        ax.plot(
            bottom_curve[:, 0],
            bottom_curve[:, 1],
            "-",
            lw=3,
            color=color_bottom,
            label="Conexión inferior (Y ~ 450)",
            zorder=4,
        )

        bottom_sampled = analysis["bottom_sampled"]
        ax.plot(
            bottom_sampled[:, 0],
            bottom_sampled[:, 1],
            "s",
            ms=6,
            color=color_points,
            markeredgewidth=1.5,
            markeredgecolor="white",
            zorder=5,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"Corte {slice_id}: Contorno por Lados con B-splines",
        fontweight="bold",
        fontsize=14,
    )
    ax.set_xlabel("X", fontweight="bold")
    ax.set_ylabel("Y", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10, frameon=True)
    ax.invert_yaxis()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_comparison_grid(all_analyses: list, output_path: Path):
    """Grid comparativo de todos los cortes."""
    n = len(all_analyses)

    # Configurar layout según número de cortes
    if n <= 3:
        fig, axes = plt.subplots(1, n, figsize=(n * 5, 5))
        if n == 1:
            axes = [axes]
    else:
        ncols = 2
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6 * nrows))
        axes = axes.flatten()

    color_cloud = "#BDBDBD"
    color_left = "#1E88E5"
    color_right = "#E53935"
    color_bottom = "#43A047"

    for ax, analysis in zip(axes, all_analyses):
        # Nube original
        original = analysis["original"]
        ax.scatter(original[:, 0], original[:, 1], s=2, c=color_cloud, alpha=0.15)

        # Curvas
        if analysis["left_success"]:
            ax.plot(
                analysis["left_curve"][:, 0],
                analysis["left_curve"][:, 1],
                lw=2.5,
                c=color_left,
            )

        if analysis["right_success"]:
            ax.plot(
                analysis["right_curve"][:, 0],
                analysis["right_curve"][:, 1],
                lw=2.5,
                c=color_right,
            )

        if analysis["bottom_success"]:
            ax.plot(
                analysis["bottom_curve"][:, 0],
                analysis["bottom_curve"][:, 1],
                lw=2.5,
                c=color_bottom,
            )

        ax.set_aspect("equal")
        ax.set_title(f"Corte {analysis['slice_id']}", fontweight="bold", fontsize=11)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

    # Título dinámico basado en los cortes procesados
    slice_ids = [a["slice_id"] for a in all_analyses]
    if len(slice_ids) > 0:
        slice_range = f"{min(slice_ids)}-{max(slice_ids)}"
    else:
        slice_range = "N/A"

    fig.suptitle(
        f"Contorno por Lados con B-splines: Cortes {slice_range}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Contornea lados izquierdo/derecho con B-splines y conexión inferior"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/skull_edges", help="Directorio de datos"
    )
    parser.add_argument(
        "--slices",
        type=int,
        nargs="+",
        default=[6, 7, 8],
        help="IDs de cortes (default: 6 7 8, excluye 9 por irregularidad)",
    )
    parser.add_argument(
        "--x-left", type=float, default=150, help="Umbral X izquierdo (default: 150)"
    )
    parser.add_argument(
        "--x-right", type=float, default=250, help="Umbral X derecho (default: 250)"
    )
    parser.add_argument(
        "--n-samples-side",
        type=int,
        default=20,
        help="Puntos a muestrear por lado (default: 20)",
    )
    parser.add_argument(
        "--n-samples-bottom",
        type=int,
        default=15,
        help="Puntos para conexión inferior (default: 15)",
    )
    parser.add_argument(
        "--degree", type=int, default=3, help="Grado B-spline (default: 3)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/figures/skull_sides_connection",
        help="Directorio de salida",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("CONTORNO POR LADOS CON B-SPLINES Y CONEXIÓN INFERIOR")
    print("=" * 70)
    print()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print(f"📁 Directorio de datos: {data_dir}")
    print(f"🔍 Cortes a procesar: {args.slices}")
    print(f"📐 Lado izquierdo: X < {args.x_left}")
    print(f"📐 Lado derecho: X > {args.x_right}")
    print("📐 Conexión inferior: Y ~ 450")
    print(
        f"🎯 Parámetros: n_side={args.n_samples_side}, n_bottom={args.n_samples_bottom}, grado={args.degree}"
    )
    print()

    # Procesar cortes
    print("📊 Procesando cortes...")
    all_analyses = []

    for slice_id in args.slices:
        print(f"\n  Corte {slice_id}:")
        slice_data = load_slice_points(data_dir, slice_id)

        if slice_data is None:
            continue

        print(f"    ✓ Cargado: {slice_data['n_points']} puntos")

        analysis = analyze_skull_sides(
            slice_data,
            x_left=args.x_left,
            x_right=args.x_right,
            n_samples_side=args.n_samples_side,
            n_samples_bottom=args.n_samples_bottom,
            degree=args.degree,
        )

        status_left = "✅" if analysis["left_success"] else "⚠️"
        status_right = "✅" if analysis["right_success"] else "⚠️"
        status_bottom = "✅" if analysis["bottom_success"] else "⚠️"

        print(
            f"    {status_left} Izquierdo | {status_right} Derecho | {status_bottom} Conexión"
        )

        all_analyses.append(analysis)

        # Graficar individual
        output_path = output_dir / "individual" / f"corte{slice_id}_sides.png"
        plot_skull_sides(analysis, output_path)
        print(f"    ✓ Figura guardada: {output_path.name}")

    # Grid comparativo
    if all_analyses:
        print("\n📐 Generando grid comparativo...")
        grid_path = output_dir / "skull_sides_comparison_grid.png"
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
    print(f"  • Cortes procesados: {len(all_analyses)}")
    successful = sum(
        1
        for a in all_analyses
        if a["left_success"] and a["right_success"] and a["bottom_success"]
    )
    print(f"  • Completamente exitosos: {successful}/{len(all_analyses)}")
    print()


if __name__ == "__main__":
    main()
