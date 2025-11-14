"""
Reporte Visual: Interpolación de Curvas Geométricas con Bézier

Este script demuestra cómo interpolar curvas geométricas clásicas usando
curvas de Bézier. Dado un conjunto de N puntos sobre una curva, encuentra
los N puntos de control de una curva de Bézier de grado N-1 que pasa
exactamente por esos puntos.

Curvas demostradas:
- Círculo
- Elipse
- Fragmento de parábola
- Lemniscata

Para cada curva se muestra:
- Puntos originales
- Curva de Bézier interpolada
- Polígono de control
- Información del sistema (matriz de Bernstein, condición)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from medvis.geometry.bezier import (
    fit_bezier_interpolate,
    bernstein_matrix,
    chord_parameterization,
)


# Configurar estilo de matplotlib
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9


def generate_circle_points(n: int = 8, radius: float = 2.0) -> np.ndarray:
    """
    Genera N puntos sobre un círculo.

    Parameters
    ----------
    n : int
        Número de puntos
    radius : float
        Radio del círculo

    Returns
    -------
    points : (n, 2) array
        Puntos sobre el círculo
    """
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y])


def generate_ellipse_points(n: int = 8, a: float = 3.0, b: float = 1.5) -> np.ndarray:
    """
    Genera N puntos sobre una elipse.

    Parameters
    ----------
    n : int
        Número de puntos
    a : float
        Semieje mayor
    b : float
        Semieje menor

    Returns
    -------
    points : (n, 2) array
        Puntos sobre la elipse
    """
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    return np.column_stack([x, y])


def generate_parabola_points(n: int = 8, x_range: float = 3.0) -> np.ndarray:
    """
    Genera N puntos sobre un fragmento de parábola y = x^2.

    Parameters
    ----------
    n : int
        Número de puntos
    x_range : float
        Rango en x: [-x_range, x_range]

    Returns
    -------
    points : (n, 2) array
        Puntos sobre la parábola
    """
    x = np.linspace(-x_range, x_range, n)
    y = x**2
    return np.column_stack([x, y])


def generate_lemniscate_points(n: int = 16, a: float = 2.0) -> np.ndarray:
    """
    Genera N puntos sobre una lemniscata de Bernoulli.

    Ecuación: (x^2 + y^2)^2 = a^2(x^2 - y^2)
    Forma paramétrica: x = a*cos(t)/(1+sin^2(t)), y = a*sin(t)*cos(t)/(1+sin^2(t))

    Parameters
    ----------
    n : int
        Número de puntos
    a : float
        Parámetro de escala

    Returns
    -------
    points : (n, 2) array
        Puntos sobre la lemniscata
    """
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    denom = 1 + np.sin(theta) ** 2
    x = a * np.cos(theta) / denom
    y = a * np.sin(theta) * np.cos(theta) / denom
    return np.column_stack([x, y])


def plot_curve_interpolation(
    points: np.ndarray,
    curve_name: str,
    out_path: Path,
    show_details: bool = True,
) -> None:
    """
    Genera visualización simplificada de la interpolación de Bézier.

    Parameters
    ----------
    points : (n, 2) array
        Puntos a interpolar
    curve_name : str
        Nombre de la curva
    out_path : Path
        Ruta de salida para la figura
    show_details : bool
        No usado, mantenido por compatibilidad
    """
    n_points = points.shape[0]

    # Interpolar con Bézier
    curve = fit_bezier_interpolate(points, parameterization_alpha=0.5)
    control_points = curve.control_points

    # Calcular parámetros y matriz
    t = chord_parameterization(points, alpha=0.5, normalize=True)
    A = bernstein_matrix(curve.degree, t, stable=True)
    cond = np.linalg.cond(A)

    # Evaluar curva interpolada en muchos puntos
    t_dense = np.linspace(0, 1, 500)
    curve_dense = curve.evaluate_batch(t_dense)

    # Crear figura simple: 1x2
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.35, width_ratios=[1.3, 1])

    # Colores
    color_curve = "#2E7D32"
    color_points = "#D32F2F"

    # Panel 1: Curva interpolada (SIN polígono de control)
    ax1 = fig.add_subplot(gs[0, 0])

    # Curva interpolada
    ax1.plot(
        curve_dense[:, 0],
        curve_dense[:, 1],
        "-",
        lw=3,
        color=color_curve,
        label=f"Curva Bézier (grado {curve.degree})",
        zorder=3,
    )

    # Puntos a interpolar
    ax1.plot(
        points[:, 0],
        points[:, 1],
        "o",
        ms=10,
        color=color_points,
        markeredgewidth=2,
        markeredgecolor="white",
        label=f"Puntos a interpolar (N={n_points})",
        zorder=5,
    )

    # Marcar inicio y fin
    ax1.plot(
        points[0, 0],
        points[0, 1],
        "o",
        ms=14,
        color="green",
        markeredgewidth=2.5,
        markeredgecolor="white",
        zorder=6,
    )
    ax1.plot(
        points[-1, 0],
        points[-1, 1],
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
        f"{curve_name}\nInterpolación de Bézier (N={n_points} puntos)",
        fontweight="bold",
        fontsize=12,
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(
        loc="upper left", bbox_to_anchor=(0, -0.12), fontsize=10, ncol=2, frameon=True
    )

    # Panel 2: Información del sistema (simplificada)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")

    # Mostrar solo las primeras 5 filas/columnas para claridad
    n_show = min(5, A.shape[0])
    m_show = min(5, A.shape[1])
    A_sample = A[:n_show, :m_show]

    info_text = f"""
SISTEMA DE INTERPOLACIÓN
━━━━━━━━━━━━━━━━━━━━━━━━━━

Sistema lineal: A × C = P

• Matriz A: {A.shape[0]}×{A.shape[1]} (Bernstein)
• Vector C: {control_points.shape[0]} puntos control
• Vector P: {n_points} puntos a interpolar

MATRIZ DE BERNSTEIN (muestra {n_show}×{m_show}):

A = """

    # Formatear la matriz sample
    for i in range(n_show):
        row_str = "    [ "
        for j in range(m_show):
            row_str += f"{A_sample[i, j]:7.4f} "
        row_str += "]"
        info_text += f"\n{row_str}"

    info_text += f"""

CONDICIONAMIENTO:
━━━━━━━━━━━━━━━━━━━━━━━━━━
Número de condición: {cond:.2e}
"""
    if cond < 100:
        info_text += "\n✓ Excelente (< 100)"
    elif cond < 1e4:
        info_text += "\n⚡ Aceptable (< 10⁴)"
    else:
        info_text += "\n⚠️  Mal condicionado (> 10⁴)"

    info_text += f"""

GRADO DE LA CURVA:
━━━━━━━━━━━━━━━━━━━━━━━━━━
Grado n = N-1 = {curve.degree}

Para interpolar N={n_points} puntos,
se requiere Bézier de grado {curve.degree}
con {control_points.shape[0]} puntos de control.
    """

    ax2.text(
        0.05,
        0.95,
        info_text,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
    )

    fig.suptitle(
        f"Interpolación de Bézier: {curve_name}",
        fontsize=14,
        fontweight="bold",
        y=0.96,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Figura guardada: {out_path.name}")


def plot_comparison_grid(curves_dict: dict, out_path: Path) -> None:
    """
    Crea un grid comparativo de todas las curvas interpoladas (sin polígono de control).
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    color_curve = "#2E7D32"
    color_points = "#D32F2F"

    for idx, (name, points) in enumerate(curves_dict.items()):
        ax = axes[idx]

        # Interpolar
        curve = fit_bezier_interpolate(points, parameterization_alpha=0.5)

        # Evaluar
        t_dense = np.linspace(0, 1, 500)
        curve_dense = curve.evaluate_batch(t_dense)

        # Plot (SIN polígono de control)
        ax.plot(
            curve_dense[:, 0],
            curve_dense[:, 1],
            "-",
            lw=3,
            color=color_curve,
            label=f"Bézier (grado {curve.degree})",
        )
        ax.plot(
            points[:, 0],
            points[:, 1],
            "o",
            ms=10,
            color=color_points,
            markeredgewidth=2,
            markeredgecolor="white",
            label=f"Puntos (N={len(points)})",
            zorder=5,
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{name}", fontweight="bold", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0, -0.12),
            fontsize=9,
            ncol=2,
            frameon=True,
        )
        ax.set_xlabel("X", fontweight="bold")
        ax.set_ylabel("Y", fontweight="bold")

    fig.suptitle(
        "Comparación: Interpolación de Bézier en Curvas Geométricas",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Grid comparativo guardado: {out_path.name}")


def main():
    ap = argparse.ArgumentParser(
        description="Generar reporte visual de interpolación de curvas con Bézier"
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/figures/bezier_interpolation_curves"),
        help="Directorio de salida para figuras",
    )
    ap.add_argument(
        "--n-points",
        type=int,
        default=8,
        help="Número de puntos para interpolar (default: 8)",
    )
    args = ap.parse_args()

    print("=" * 70)
    print("REPORTE: INTERPOLACIÓN DE CURVAS GEOMÉTRICAS CON BÉZIER")
    print("=" * 70)

    # Generar curvas
    n = args.n_points
    print(f"\n📐 Generando curvas con N={n} puntos...")

    curves = {
        "Círculo": generate_circle_points(n=n, radius=2.0),
        "Elipse": generate_ellipse_points(n=n, a=3.0, b=1.5),
        "Parábola": generate_parabola_points(n=n, x_range=2.5),
        "Lemniscata": generate_lemniscate_points(n=max(n, 12), a=2.0),
    }

    # Análisis individual
    print("\n1. Generando análisis individual de cada curva...")
    for i, (name, points) in enumerate(curves.items(), 1):
        print(f"\n   {i}. {name} (N={len(points)} puntos)")
        safe_name = name.lower().replace(" ", "_")
        plot_curve_interpolation(
            points,
            name,
            args.out_dir / f"{i:02d}_{safe_name}_interpolation.png",
            show_details=True,
        )

    # Grid comparativo
    print("\n2. Generando grid comparativo...")
    plot_comparison_grid(curves, args.out_dir / "00_comparison_grid.png")

    # Resumen
    print(f"\n{'='*70}")
    print("✅ REPORTE COMPLETADO")
    print(f"{'='*70}")
    print(f"\n📁 Figuras generadas en: {args.out_dir}")
    print(f"\n📊 Total de figuras: {len(curves) + 1}")
    print("\n💡 CONCEPTOS DEMOSTRADOS:")
    print("   1. Interpolación de Bézier: curva pasa exactamente por N puntos")
    print("   2. Grado de la curva: n = N-1 (para N puntos)")
    print("   3. Puntos de control: resueltos via sistema lineal A×C = P")
    print("   4. Matriz de Bernstein: base para el sistema de interpolación")
    print(f"   5. Para N={n}: curva de grado {n-1} con {n} puntos de control")
    print()


if __name__ == "__main__":
    main()
