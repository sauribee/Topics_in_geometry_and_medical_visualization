"""
Análisis de Aproximación de Bézier por Mínimos Cuadrados (LSQ)

Este script demuestra cómo encontrar los puntos de control de una curva de
Bézier que APROXIMA (no interpola) una secuencia de puntos dados usando LSQ.

Teoría:
-------
Dada una secuencia de N puntos P = [p_0, p_1, ..., p_{N-1}], queremos encontrar
una curva de Bézier de grado n < N-1 con n+1 puntos de control C = [c_0, ..., c_n]
que minimice el error de aproximación.

Sistema LSQ (sobredeterminado):
    B(t_i) = Σ B_j^n(t_i) * c_j ≈ p_i    para i = 0, ..., N-1

    En forma matricial: A * C ≈ P

donde A[i,j] = B_j^n(t_i) es la matriz de Bernstein (N x (n+1), N > n+1).

Ventajas sobre interpolación:
- Evita oscilaciones (fenómeno de Runge) de alto grado
- Curvas más suaves y estables
- Mejor condicionamiento numérico

Grado de la curva: n = 5-7 (independiente de N)
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
    chord_parameterization,
    bernstein_matrix,
    fit_bezier_lsq,
)


def generate_ellipse_points(n_points: int, noise_level: float = 0.0) -> np.ndarray:
    """Generar puntos de una elipse con ruido opcional."""
    a, b = 5.0, 3.0  # Semi-ejes
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = a * np.cos(theta)
    y = b * np.sin(theta)

    points = np.column_stack([x, y])

    if noise_level > 0:
        noise = np.random.randn(n_points, 2) * noise_level
        points += noise

    return points


def generate_circle_points(n_points: int, noise_level: float = 0.0) -> np.ndarray:
    """Generar puntos de un círculo con ruido opcional."""
    r = 4.0
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    points = np.column_stack([x, y])

    if noise_level > 0:
        noise = np.random.randn(n_points, 2) * noise_level
        points += noise

    return points


def generate_parabola_points(n_points: int, noise_level: float = 0.0) -> np.ndarray:
    """Generar puntos de una parábola con ruido opcional."""
    x = np.linspace(-3, 3, n_points)
    y = 0.5 * x**2

    points = np.column_stack([x, y])

    if noise_level > 0:
        noise = np.random.randn(n_points, 2) * noise_level
        points += noise

    return points


def generate_lemniscate_points(n_points: int, noise_level: float = 0.0) -> np.ndarray:
    """Generar puntos de una lemniscata (figura de 8)."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    scale = 3.0
    x = scale * np.cos(t) / (1 + np.sin(t) ** 2)
    y = scale * np.sin(t) * np.cos(t) / (1 + np.sin(t) ** 2)

    points = np.column_stack([x, y])

    if noise_level > 0:
        noise = np.random.randn(n_points, 2) * noise_level
        points += noise

    return points


def solve_bezier_approximation(
    points: np.ndarray,
    degree: int = 5,
    param_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Resolver el sistema de mínimos cuadrados para aproximación de Bézier.
    Usa grado MENOR que N-1 para evitar oscilaciones.

    Parameters
    ----------
    points : (N, d) array
        Puntos a aproximar
    degree : int
        Grado de la curva (debe ser < N-1 para LSQ)
    param_alpha : float
        Parámetro de chord-length (0=uniforme, 0.5=centripetal, 1.0=chord)

    Returns
    -------
    control_points : (degree+1, d) array
        Puntos de control de la curva de Bézier
    A : (N, degree+1) array
        Matriz de Bernstein
    t : (N,) array
        Parámetros usados
    cond : float
        Número de condición de la matriz A
    approx_error : float
        Error RMS de aproximación
    """
    N = points.shape[0]

    if degree >= N:
        raise ValueError(f"Degree {degree} must be < N={N} for LSQ approximation")

    # 1. Calcular parámetros t_i usando chord-length
    t = chord_parameterization(points, alpha=param_alpha, normalize=True)

    # 2. Ajustar curva usando LSQ
    curve = fit_bezier_lsq(
        points,
        degree=degree,
        parameterization_alpha=param_alpha,
    )

    control_points = curve.control_points

    # 3. Construir matriz de Bernstein para análisis
    A = bernstein_matrix(degree, t, stable=True)

    # 4. Calcular número de condición
    cond = np.linalg.cond(A)

    # 5. Calcular error de aproximación
    residuals = []
    for i, p in enumerate(points):
        p_approx = curve.evaluate(t[i])
        residuals.append(np.linalg.norm(p - p_approx))
    approx_error = np.sqrt(np.mean(np.array(residuals) ** 2))

    return control_points, A, t, cond, approx_error


def plot_interpolation_analysis(
    points: np.ndarray,
    control_points: np.ndarray,
    curve: BezierCurve,
    A: np.ndarray,
    t: np.ndarray,
    cond: float,
    curve_name: str,
    n_points: int,
    noise_level: float,
    out_path: Path,
) -> None:
    """Generar visualización simplificada del análisis de aproximación."""

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.4, width_ratios=[1.2, 1])

    # 1. Curva aproximada (sin polígono de control)
    ax1 = fig.add_subplot(gs[0, 0])

    # Evaluar curva en muchos puntos
    t_eval = np.linspace(0, 1, 500)
    curve_eval = curve.evaluate_batch(t_eval)

    ax1.plot(
        points[:, 0],
        points[:, 1],
        "o",
        ms=10,
        color="blue",
        label=f"Puntos a aproximar (N={n_points})",
        zorder=5,
    )
    ax1.plot(
        curve_eval[:, 0],
        curve_eval[:, 1],
        "-",
        lw=3,
        color="green",
        label=f"Aproximación LSQ (grado {curve.degree})",
        zorder=3,
    )

    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("X", fontsize=11)
    ax1.set_ylabel("Y", fontsize=11)
    ax1.set_title(
        f"{curve_name} - Aproximación Bézier LSQ Grado {curve.degree}",
        fontsize=12,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best", fontsize=10)

    # 2. Información del sistema
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")

    info_text = f"""
INFORMACIÓN DEL SISTEMA SOBREDETERMINADO

Curva: {curve_name}
Ruido: {noise_level:.3f}

APROXIMACIÓN BÉZIER (LSQ):
━━━━━━━━━━━━━━━━━━━━━━━━
• N puntos a aproximar: {n_points}
• Grado de la curva: n = {curve.degree}
• Puntos de control: {control_points.shape[0]}

SISTEMA LSQ:  A × C ≈ P
━━━━━━━━━━━━━━━━━━━━━━━━
• Matriz A: {A.shape[0]}×{A.shape[1]} (Bernstein)
• Vector P: {n_points}×{points.shape[1]} (puntos a aproximar)
• Vector C: {control_points.shape[0]}×{points.shape[1]} (puntos de control)

ESTABILIDAD NUMÉRICA:
━━━━━━━━━━━━━━━━━━━━━━━━
• Número de condición: {cond:.2e}
• Estado: {"ILL-CONDITIONED" if cond > 1e10 else "BIEN CONDICIONADO"}

PARÁMETROS:
━━━━━━━━━━━━━━━━━━━━━━━━
• Método: Chord-length (alpha=0.5)
• t_min = {t.min():.6f}
• t_max = {t.max():.6f}
• Δt promedio = {np.mean(np.diff(t)):.6f}
    """

    ax2.text(
        0.05,
        0.95,
        info_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_data_to_csv(
    points: np.ndarray,
    control_points: np.ndarray,
    t: np.ndarray,
    A: np.ndarray,
    curve_name: str,
    n_points: int,
    out_dir: Path,
) -> None:
    """Guardar datos del análisis en CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{curve_name.lower().replace(' ', '_')}_n{n_points}"

    # Puntos a aproximar
    np.savetxt(
        out_dir / f"{prefix}_points.csv",
        points,
        delimiter=",",
        header="x,y",
        comments="",
        fmt="%.6f",
    )

    # Puntos de control
    np.savetxt(
        out_dir / f"{prefix}_control_points.csv",
        control_points,
        delimiter=",",
        header="x,y",
        comments="",
        fmt="%.6f",
    )

    # Parámetros
    np.savetxt(
        out_dir / f"{prefix}_parameters.csv",
        t,
        delimiter=",",
        header="t",
        comments="",
        fmt="%.6f",
    )

    # Matriz de Bernstein
    np.savetxt(
        out_dir / f"{prefix}_bernstein_matrix.csv",
        A,
        delimiter=",",
        comments="",
        fmt="%.6f",
    )


def main():
    ap = argparse.ArgumentParser(
        description="Análisis de aproximación de Bézier con sistemas sobredeterminados"
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/figures/bezier_approximation_analysis"),
        help="Directorio de salida para figuras",
    )
    ap.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("reports/csv/bezier_approximation_analysis"),
        help="Directorio de salida para CSVs",
    )
    ap.add_argument(
        "--noise",
        type=float,
        default=0.05,
        help="Nivel de ruido para perturbar los puntos",
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad"
    )
    args = ap.parse_args()

    # Fijar semilla para reproducibilidad
    np.random.seed(args.seed)

    print("=" * 70)
    print("ANÁLISIS DE APROXIMACIÓN BÉZIER (LSQ)")
    print("Resolución de Sistema Sobredeterminado: A × C ≈ P")
    print("=" * 70)

    # Configuraciones: (nombre, generador) - Solo N=10 para todas
    configurations = [
        ("Elipse", generate_ellipse_points),
        ("Círculo", generate_circle_points),
        ("Parábola", generate_parabola_points),
        ("Lemniscata", generate_lemniscate_points),
    ]

    n_points = 10  # Puntos a aproximar
    degree = 5  # Grado de la curva (MENOR que N-1 para LSQ)

    for curve_name, generator in configurations:
        print(f"\n{'─'*70}")
        print(f"📐 Curva: {curve_name} (N={n_points} puntos, grado={degree} LSQ)")
        print(f"{'─'*70}")

        # Generar puntos con ruido
        points = generator(n_points, noise_level=args.noise)

        # Resolver sistema LSQ para aproximación (NO interpolación)
        control_points, A, t, cond, approx_error = solve_bezier_approximation(
            points, degree=degree, param_alpha=0.5
        )

        # Crear curva de Bézier
        curve = BezierCurve(control_points)

        # Información
        print(f"  Grado de la curva: {curve.degree}")
        print(f"  Puntos de control: {control_points.shape[0]}")
        print(f"  Número de condición: {cond:.2e}", end="")
        if cond > 1e10:
            print(" ⚠️  ILL-CONDITIONED")
        elif cond > 1e6:
            print(" ⚡ MODERADO")
        else:
            print(" ✓ BIEN CONDICIONADO")

        # Error de aproximación LSQ
        print(f"  Error RMS de aproximación: {approx_error:.4f}")

        # Calcular error máximo
        errors = []
        for i, p in enumerate(points):
            p_approx = curve.evaluate(t[i])
            error = np.linalg.norm(p - p_approx)
            errors.append(error)

        max_err = max(errors)
        mean_err = np.mean(errors)
        print(f"  Error puntual: max={max_err:.4f}, mean={mean_err:.4f}")

        # Generar visualización con grado en el nombre
        out_path = (
            args.out_dir
            / f"{curve_name.lower().replace(' ', '_')}_grado{curve.degree}.png"
        )
        plot_interpolation_analysis(
            points,
            control_points,
            curve,
            A,
            t,
            cond,
            curve_name,
            n_points,
            args.noise,
            out_path,
        )
        print(f"  ✓ Figura guardada: {out_path}")

        # Guardar CSVs
        save_data_to_csv(
            points, control_points, t, A, curve_name, n_points, args.csv_dir
        )
        print(f"  ✓ CSVs guardados en: {args.csv_dir}")


if __name__ == "__main__":
    main()
