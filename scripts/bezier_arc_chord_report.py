"""
Reporte Visual: Arc-Chord Parameterization en Curvas de Bézier

Este script genera un reporte completo con visualizaciones que explican:
- Qué es la parametrización arc-chord
- Por qué es importante para curvas de Bézier
- Comparación con parametrización uniforme
- Ejemplos con diferentes curvas

Genera figuras bonitas y bien organizadas para documentación.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from medvis.geometry.bezier import (
    BezierCurve,
    arc_chord_parameterization,
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


def create_line_curve() -> BezierCurve:
    """Curva lineal (grado 1)."""
    control_points = np.array([[0.0, 0.0], [5.0, 3.0]])
    return BezierCurve(control_points)


def create_quadratic_curve() -> BezierCurve:
    """Curva cuadrática simple (grado 2)."""
    control_points = np.array([[0.0, 0.0], [2.5, 4.0], [5.0, 0.0]])
    return BezierCurve(control_points)


def create_cubic_curve() -> BezierCurve:
    """Curva cúbica S-shape (grado 3)."""
    control_points = np.array([[0.0, 0.0], [1.0, 3.0], [4.0, -1.0], [5.0, 2.0]])
    return BezierCurve(control_points)


def create_complex_cubic() -> BezierCurve:
    """Curva cúbica con velocidad variable (grado 3)."""
    control_points = np.array([[0.0, 0.0], [0.5, 0.1], [4.5, 2.9], [5.0, 3.0]])
    return BezierCurve(control_points)


def plot_single_curve_analysis(
    curve: BezierCurve,
    curve_name: str,
    out_path: Path,
) -> None:
    """
    Analiza una sola curva mostrando:
    - La curva con sus puntos de control
    - Parametrización uniforme vs arc-chord
    - Distribución de puntos con cada método
    - Perfil de velocidad
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Colores consistentes
    color_curve = "#2E7D32"
    color_uniform = "#D32F2F"
    color_arclength = "#1976D2"
    color_control = "#FF6F00"

    # 1. Curva con puntos de control
    ax1 = fig.add_subplot(gs[0, 0])

    t_dense = np.linspace(0, 1, 500)
    curve_dense = curve.evaluate_batch(t_dense)

    ax1.plot(
        curve_dense[:, 0],
        curve_dense[:, 1],
        "-",
        lw=3,
        color=color_curve,
        label="Curva de Bézier",
        zorder=3,
    )

    # Polígono de control
    cp = curve.control_points
    ax1.plot(
        cp[:, 0],
        cp[:, 1],
        "o--",
        ms=8,
        lw=1.5,
        color=color_control,
        alpha=0.6,
        label="Polígono de control",
        zorder=4,
    )

    # Marcar inicio y fin
    ax1.plot(
        cp[0, 0],
        cp[0, 1],
        "o",
        ms=12,
        color="green",
        markeredgewidth=2,
        markeredgecolor="white",
        zorder=5,
    )
    ax1.text(
        cp[0, 0], cp[0, 1] - 0.3, "Inicio", ha="center", fontweight="bold", fontsize=10
    )

    ax1.plot(
        cp[-1, 0],
        cp[-1, 1],
        "s",
        ms=12,
        color="red",
        markeredgewidth=2,
        markeredgecolor="white",
        zorder=5,
    )
    ax1.text(
        cp[-1, 0], cp[-1, 1] + 0.3, "Fin", ha="center", fontweight="bold", fontsize=10
    )

    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("X", fontweight="bold")
    ax1.set_ylabel("Y", fontweight="bold")
    ax1.set_title(
        f"{curve_name}\nCurva de Bézier (grado {curve.degree})",
        fontweight="bold",
        fontsize=12,
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    # 2. Parametrización uniforme
    ax2 = fig.add_subplot(gs[0, 1])

    n_samples = 10
    t_uniform = np.linspace(0, 1, n_samples)
    points_uniform = curve.evaluate_batch(t_uniform)

    ax2.plot(
        curve_dense[:, 0],
        curve_dense[:, 1],
        "-",
        lw=2,
        color=color_curve,
        alpha=0.5,
        zorder=1,
    )
    ax2.plot(
        points_uniform[:, 0],
        points_uniform[:, 1],
        "o",
        ms=10,
        color=color_uniform,
        label="Puntos uniformes en t",
        zorder=3,
    )

    # Conectar puntos
    for i in range(len(points_uniform) - 1):
        ax2.plot(
            [points_uniform[i, 0], points_uniform[i + 1, 0]],
            [points_uniform[i, 1], points_uniform[i + 1, 1]],
            "--",
            lw=1,
            color=color_uniform,
            alpha=0.4,
        )

    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("X", fontweight="bold")
    ax2.set_ylabel("Y", fontweight="bold")
    ax2.set_title(
        "Parametrización Uniforme\n(espaciado constante en t)", fontweight="bold"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    # 3. Parametrización arc-length
    ax3 = fig.add_subplot(gs[0, 2])

    from medvis.geometry.bezier import arclength_parameterization

    t_arclength = arclength_parameterization(curve, n_samples, samples=2000)
    points_arclength = curve.evaluate_batch(t_arclength)

    ax3.plot(
        curve_dense[:, 0],
        curve_dense[:, 1],
        "-",
        lw=2,
        color=color_curve,
        alpha=0.5,
        zorder=1,
    )
    ax3.plot(
        points_arclength[:, 0],
        points_arclength[:, 1],
        "s",
        ms=10,
        color=color_arclength,
        label="Puntos por arc-length",
        zorder=3,
    )

    # Conectar puntos
    for i in range(len(points_arclength) - 1):
        ax3.plot(
            [points_arclength[i, 0], points_arclength[i + 1, 0]],
            [points_arclength[i, 1], points_arclength[i + 1, 1]],
            "--",
            lw=1,
            color=color_arclength,
            alpha=0.4,
        )

    ax3.set_aspect("equal", adjustable="box")
    ax3.set_xlabel("X", fontweight="bold")
    ax3.set_ylabel("Y", fontweight="bold")
    ax3.set_title(
        "Parametrización Arc-Length\n(espaciado uniforme en distancia)",
        fontweight="bold",
    )
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best")

    # 4. Gráfica t vs arc-length
    ax4 = fig.add_subplot(gs[1, 0])

    t_analysis, s_analysis = arc_chord_parameterization(
        curve, samples=1000, normalize=True
    )

    ax4.plot(
        t_analysis,
        s_analysis,
        "-",
        lw=3,
        color=color_arclength,
        label="s(t): arc-length acumulada",
    )
    ax4.plot(
        [0, 1],
        [0, 1],
        "--",
        lw=2,
        color="gray",
        alpha=0.5,
        label="Línea ideal (velocidad constante)",
    )

    ax4.set_xlabel("Parámetro t", fontweight="bold")
    ax4.set_ylabel("Arc-length normalizada s", fontweight="bold")
    ax4.set_title(
        "Relación t → s\n(curvatura de la línea indica velocidad variable)",
        fontweight="bold",
    )
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="best")
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])

    # 5. Distancias entre puntos consecutivos
    ax5 = fig.add_subplot(gs[1, 1])

    # Distancias con parametrización uniforme
    dists_uniform = []
    for i in range(len(points_uniform) - 1):
        d = np.linalg.norm(points_uniform[i + 1] - points_uniform[i])
        dists_uniform.append(d)

    # Distancias con arc-length
    dists_arclength = []
    for i in range(len(points_arclength) - 1):
        d = np.linalg.norm(points_arclength[i + 1] - points_arclength[i])
        dists_arclength.append(d)

    x = np.arange(len(dists_uniform))
    width = 0.35

    ax5.bar(
        x - width / 2,
        dists_uniform,
        width,
        label="Uniforme",
        color=color_uniform,
        alpha=0.7,
    )
    ax5.bar(
        x + width / 2,
        dists_arclength,
        width,
        label="Arc-length",
        color=color_arclength,
        alpha=0.7,
    )

    # Línea de referencia (distancia ideal)
    ideal_dist = np.mean(dists_arclength)
    ax5.axhline(
        y=ideal_dist,
        color="green",
        linestyle="--",
        lw=2,
        alpha=0.5,
        label=f"Ideal: {ideal_dist:.3f}",
    )

    ax5.set_xlabel("Segmento i → i+1", fontweight="bold")
    ax5.set_ylabel("Distancia euclidiana", fontweight="bold")
    ax5.set_title(
        "Distancias entre Puntos Consecutivos\n(arc-length más uniforme)",
        fontweight="bold",
    )
    ax5.set_xticks(x)
    ax5.set_xticklabels(
        [f"{i}→{i+1}" for i in range(len(dists_uniform))], rotation=45, ha="right"
    )
    ax5.legend(loc="best")
    ax5.grid(True, alpha=0.3, axis="y")

    # 6. Información y métricas
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    # Calcular métricas
    std_uniform = np.std(dists_uniform)
    std_arclength = np.std(dists_arclength)
    cv_uniform = std_uniform / np.mean(dists_uniform) * 100
    cv_arclength = std_arclength / np.mean(dists_arclength) * 100

    total_length = curve.length(samples=2000)

    info_text = f"""
INFORMACIÓN DE LA CURVA

Tipo: {curve_name}
Grado: {curve.degree}
Puntos de control: {curve.control_points.shape[0]}
Longitud total: {total_length:.4f}

PARAMETRIZACIÓN UNIFORME:
━━━━━━━━━━━━━━━━━━━━━━━━
• Distancia media: {np.mean(dists_uniform):.4f}
• Desviación estándar: {std_uniform:.4f}
• Coef. variación: {cv_uniform:.2f}%
• {"⚠️  IRREGULAR" if cv_uniform > 20 else "✓ Aceptable"}

PARAMETRIZACIÓN ARC-LENGTH:
━━━━━━━━━━━━━━━━━━━━━━━━
• Distancia media: {np.mean(dists_arclength):.4f}
• Desviación estándar: {std_arclength:.4f}
• Coef. variación: {cv_arclength:.2f}%
• {"✓ UNIFORME" if cv_arclength < 10 else "⚠️  Moderado"}

MEJORA:
━━━━━━━━━━━━━━━━━━━━━━━━
Reducción de variabilidad: {(1 - cv_arclength/cv_uniform)*100:.1f}%

CONCLUSIÓN:
━━━━━━━━━━━━━━━━━━━━━━━━
Arc-length produce puntos más
uniformemente distribuidos a lo
largo de la curva.
    """

    ax6.text(
        0.05,
        0.95,
        info_text,
        transform=ax6.transAxes,
        fontsize=9,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
    )

    fig.suptitle(
        f"Análisis de Parametrización Arc-Chord: {curve_name}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Figura guardada: {out_path.name}")


def plot_comparison_grid(curves_dict: dict, out_path: Path) -> None:
    """
    Crea una comparación en grid de todas las curvas mostrando
    la diferencia entre parametrización uniforme y arc-length.
    """
    n_curves = len(curves_dict)
    fig, axes = plt.subplots(n_curves, 3, figsize=(16, 4.5 * n_curves))

    if n_curves == 1:
        axes = axes.reshape(1, -1)

    color_curve = "#2E7D32"
    color_uniform = "#D32F2F"
    color_arclength = "#1976D2"

    for idx, (name, curve) in enumerate(curves_dict.items()):
        # Evaluar curva
        t_dense = np.linspace(0, 1, 500)
        curve_dense = curve.evaluate_batch(t_dense)

        n_samples = 10
        t_uniform = np.linspace(0, 1, n_samples)
        points_uniform = curve.evaluate_batch(t_uniform)

        from medvis.geometry.bezier import arclength_parameterization

        t_arclength = arclength_parameterization(curve, n_samples, samples=2000)
        points_arclength = curve.evaluate_batch(t_arclength)

        # Columna 1: Curva base
        ax = axes[idx, 0]
        ax.plot(curve_dense[:, 0], curve_dense[:, 1], "-", lw=3, color=color_curve)
        cp = curve.control_points
        ax.plot(cp[:, 0], cp[:, 1], "o--", ms=6, lw=1, color="orange", alpha=0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{name}", fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Columna 2: Uniforme
        ax = axes[idx, 1]
        ax.plot(
            curve_dense[:, 0],
            curve_dense[:, 1],
            "-",
            lw=2,
            color=color_curve,
            alpha=0.4,
        )
        ax.plot(
            points_uniform[:, 0], points_uniform[:, 1], "o", ms=8, color=color_uniform
        )
        for i in range(len(points_uniform) - 1):
            ax.plot(
                [points_uniform[i, 0], points_uniform[i + 1, 0]],
                [points_uniform[i, 1], points_uniform[i + 1, 1]],
                "--",
                lw=1,
                color=color_uniform,
                alpha=0.3,
            )
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Parametrización Uniforme", fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Columna 3: Arc-length
        ax = axes[idx, 2]
        ax.plot(
            curve_dense[:, 0],
            curve_dense[:, 1],
            "-",
            lw=2,
            color=color_curve,
            alpha=0.4,
        )
        ax.plot(
            points_arclength[:, 0],
            points_arclength[:, 1],
            "s",
            ms=8,
            color=color_arclength,
        )
        for i in range(len(points_arclength) - 1):
            ax.plot(
                [points_arclength[i, 0], points_arclength[i + 1, 0]],
                [points_arclength[i, 1], points_arclength[i + 1, 1]],
                "--",
                lw=1,
                color=color_arclength,
                alpha=0.3,
            )
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Parametrización Arc-Length", fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Comparación de Parametrizaciones en Curvas de Bézier",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Grid comparativo guardado: {out_path.name}")


def create_concept_diagram(out_path: Path) -> None:
    """
    Crea un diagrama conceptual explicando qué es arc-chord parameterization.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Crear una curva de ejemplo
    curve = create_cubic_curve()
    t_dense = np.linspace(0, 1, 500)
    curve_dense = curve.evaluate_batch(t_dense)

    # Panel 1: Concepto de parametrización uniforme (problema)
    ax = axes[0]
    ax.plot(
        curve_dense[:, 0],
        curve_dense[:, 1],
        "-",
        lw=4,
        color="#2E7D32",
        label="Curva de Bézier",
    )

    t_uniform = np.linspace(0, 1, 6)
    points_uniform = curve.evaluate_batch(t_uniform)

    for i, (p, t) in enumerate(zip(points_uniform, t_uniform)):
        ax.plot(p[0], p[1], "o", ms=12, color="#D32F2F")
        ax.annotate(
            f"t={t:.2f}",
            xy=(p[0], p[1]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

    # Mostrar distancias desiguales
    for i in range(len(points_uniform) - 1):
        mid = (points_uniform[i] + points_uniform[i + 1]) / 2
        dist = np.linalg.norm(points_uniform[i + 1] - points_uniform[i])
        ax.text(
            mid[0],
            mid[1] - 0.4,
            f"d={dist:.2f}",
            ha="center",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "PROBLEMA: Parametrización Uniforme\n" "(Δt constante ≠ Δdistancia constante)",
        fontweight="bold",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    # Panel 2: Solución con arc-length (solución)
    ax = axes[1]
    ax.plot(
        curve_dense[:, 0],
        curve_dense[:, 1],
        "-",
        lw=4,
        color="#2E7D32",
        label="Curva de Bézier",
    )

    from medvis.geometry.bezier import arclength_parameterization

    t_arclength = arclength_parameterization(curve, 6, samples=2000)
    points_arclength = curve.evaluate_batch(t_arclength)

    for i, (p, t) in enumerate(zip(points_arclength, t_arclength)):
        ax.plot(p[0], p[1], "s", ms=12, color="#1976D2")
        ax.annotate(
            f"t={t:.2f}",
            xy=(p[0], p[1]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
        )

    # Mostrar distancias más uniformes
    for i in range(len(points_arclength) - 1):
        mid = (points_arclength[i] + points_arclength[i + 1]) / 2
        dist = np.linalg.norm(points_arclength[i + 1] - points_arclength[i])
        ax.text(
            mid[0],
            mid[1] - 0.4,
            f"d={dist:.2f}",
            ha="center",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "SOLUCIÓN: Parametrización Arc-Length\n" "(Distancias aproximadamente iguales)",
        fontweight="bold",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    fig.suptitle(
        "Concepto: Arc-Chord Parameterization en Curvas de Bézier",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Diagrama conceptual guardado: {out_path.name}")


def main():
    ap = argparse.ArgumentParser(
        description="Generar reporte visual de arc-chord parameterization"
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/figures/arc_chord_parameterization"),
        help="Directorio de salida para figuras",
    )
    args = ap.parse_args()

    print("=" * 70)
    print("REPORTE: ARC-CHORD PARAMETERIZATION EN CURVAS DE BÉZIER")
    print("=" * 70)

    # Crear curvas de ejemplo
    curves = {
        "Línea Recta": create_line_curve(),
        "Cuadrática": create_quadratic_curve(),
        "Cúbica S-Shape": create_cubic_curve(),
        "Cúbica Compleja": create_complex_cubic(),
    }

    print(f"\n📐 Analizando {len(curves)} curvas...")

    # 1. Diagrama conceptual
    print("\n1. Generando diagrama conceptual...")
    create_concept_diagram(args.out_dir / "00_concepto_arc_chord.png")

    # 2. Análisis individual de cada curva
    print("\n2. Analizando curvas individualmente...")
    for i, (name, curve) in enumerate(curves.items(), 1):
        print(f"\n   {i}. {name} (grado {curve.degree}):")
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        plot_single_curve_analysis(
            curve, name, args.out_dir / f"{i:02d}_{safe_name}_analysis.png"
        )

    # 3. Grid comparativo
    print("\n3. Generando grid comparativo...")
    plot_comparison_grid(curves, args.out_dir / "99_comparison_grid.png")

    # Resumen final
    print(f"\n{'='*70}")
    print("✅ REPORTE COMPLETADO")
    print(f"{'='*70}")
    print(f"\n📁 Figuras generadas en: {args.out_dir}")
    print(f"\n📊 Total de figuras: {len(curves) + 2}")
    print("\n💡 CONCLUSIONES DEL REPORTE:")
    print(
        "   1. Parametrización uniforme (Δt constante) produce puntos irregularmente espaciados"
    )
    print(
        "   2. Arc-chord parametrization distribuye puntos uniformemente por distancia"
    )
    print("   3. Crucial para: muestreo, ajuste de curvas, animación, renderizado")
    print("   4. Especialmente importante en curvas con velocidad variable")
    print()


if __name__ == "__main__":
    main()
