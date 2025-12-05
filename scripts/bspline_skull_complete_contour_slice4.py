#!/usr/bin/env python3
"""
B-spline Complete Skull Contour - Slice 4
==========================================

Genera un contorno completo del corte 4 del cráneo integrando:
1. Protuberancia occipital (Y < 50) - parte superior
2. Lado izquierdo (X < 120) - lateral izquierdo
3. Lado derecho (X > 150) - lateral derecho
4. Conexión inferior suave (Y ~ 240)
5. Conexiones superiores suaves con la protuberancia

Todas las conexiones garantizan continuidad C1 (suavidad).
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


def create_smooth_connection(
    curve_from: np.ndarray, curve_to: np.ndarray, n_points: int = 15
) -> np.ndarray:
    """
    Crea una conexión suave tipo Bézier cúbica entre dos curvas.
    Garantiza continuidad C1 usando vectores tangentes.

    Parameters
    ----------
    curve_from : (n, 2) array
        Curva de origen (usa el punto final)
    curve_to : (m, 2) array
        Curva de destino (usa el punto inicial)
    n_points : int
        Número de puntos de la conexión

    Returns
    -------
    (n_points, 2) array
        Curva de conexión suave
    """
    # Puntos extremos
    P0 = curve_from[-1]  # Final de la curva origen
    P3 = curve_to[0]  # Inicio de la curva destino

    # Vector tangente al final de curve_from
    if len(curve_from) > 5:
        tangent_from = curve_from[-1] - curve_from[-5]
        tangent_from = tangent_from / np.linalg.norm(tangent_from)
    else:
        tangent_from = np.array([1, 0])

    # Vector tangente al inicio de curve_to
    if len(curve_to) > 5:
        tangent_to = curve_to[5] - curve_to[0]
        tangent_to = tangent_to / np.linalg.norm(tangent_to)
    else:
        tangent_to = np.array([1, 0])

    # Puntos de control para Bézier cúbica
    dist = np.linalg.norm(P3 - P0)
    P1 = P0 + tangent_from * (dist * 0.35)
    P2 = P3 - tangent_to * (dist * 0.35)

    # Evaluar curva de Bézier cúbica
    t = np.linspace(0, 1, n_points)
    t = t.reshape(-1, 1)

    bezier_curve = (
        (1 - t) ** 3 * P0
        + 3 * (1 - t) ** 2 * t * P1
        + 3 * (1 - t) * t**2 * P2
        + t**3 * P3
    )

    return bezier_curve


def analyze_complete_contour(
    slice_data: dict,
    y_prot: float = 50,
    x_left: float = 120,
    x_right: float = 150,
    y_bottom: float = 240,
    n_samples_prot: int = 15,
    n_samples_side: int = 25,
    n_samples_conn: int = 12,
    degree: int = 3,
) -> dict:
    """
    Genera el contorno completo del cráneo con todas las regiones.

    Returns
    -------
    dict
        Diccionario con todas las curvas y puntos
    """
    points = slice_data["points"]
    slice_id = slice_data["id"]

    results = {"slice_id": slice_id, "original": points}

    print(f"\n  📊 Análisis del Corte {slice_id}:")
    print(f"    Total de puntos: {len(points)}")

    # 1. PROTUBERANCIA OCCIPITAL (Y < y_prot)
    mask_prot = points[:, 1] < y_prot
    prot_points = points[mask_prot].copy()

    if len(prot_points) > degree + 1:
        # Ordenar por X (izquierda a derecha)
        prot_points = prot_points[np.argsort(prot_points[:, 0])]
        prot_sampled = sample_uniform_arclength(prot_points, n_samples_prot)

        try:
            knots_p, C_p, p_p = fit_bspline_interpolate(prot_sampled, degree=degree)
            t_dense = np.linspace(0, 1, 200)
            prot_curve = evaluate_bspline(knots_p, C_p, p_p, t_dense)

            results["prot_original"] = prot_points
            results["prot_sampled"] = prot_sampled
            results["prot_curve"] = prot_curve
            results["prot_success"] = True
            print(
                f"    ✅ Protuberancia: {len(prot_points)} pts → {len(prot_sampled)} muestreados"
            )
        except Exception as e:
            print(f"    ⚠️  Error en protuberancia: {e}")
            results["prot_success"] = False
    else:
        results["prot_success"] = False

    # 2. LADO IZQUIERDO (X < x_left, Y >= y_prot, Y <= y_bottom)
    mask_left = (
        (points[:, 0] < x_left) & (points[:, 1] >= y_prot) & (points[:, 1] <= y_bottom)
    )
    left_points = points[mask_left].copy()

    if len(left_points) > degree + 1:
        # Ordenar por Y (arriba a abajo)
        left_points = left_points[np.argsort(left_points[:, 1])]
        left_sampled = sample_uniform_arclength(left_points, n_samples_side)

        try:
            knots_l, C_l, p_l = fit_bspline_interpolate(left_sampled, degree=degree)
            t_dense = np.linspace(0, 1, 300)
            left_curve = evaluate_bspline(knots_l, C_l, p_l, t_dense)

            results["left_original"] = left_points
            results["left_sampled"] = left_sampled
            results["left_curve"] = left_curve
            results["left_success"] = True
            print(
                f"    ✅ Lado izquierdo: {len(left_points)} pts → {len(left_sampled)} muestreados"
            )
        except Exception as e:
            print(f"    ⚠️  Error en lado izquierdo: {e}")
            results["left_success"] = False
    else:
        results["left_success"] = False

    # 3. LADO DERECHO (X > x_right, Y >= y_prot, Y <= y_bottom)
    mask_right = (
        (points[:, 0] > x_right) & (points[:, 1] >= y_prot) & (points[:, 1] <= y_bottom)
    )
    right_points = points[mask_right].copy()

    if len(right_points) > degree + 1:
        # Ordenar por Y (arriba a abajo)
        right_points = right_points[np.argsort(right_points[:, 1])]
        right_sampled = sample_uniform_arclength(right_points, n_samples_side)

        try:
            knots_r, C_r, p_r = fit_bspline_interpolate(right_sampled, degree=degree)
            t_dense = np.linspace(0, 1, 300)
            right_curve = evaluate_bspline(knots_r, C_r, p_r, t_dense)

            results["right_original"] = right_points
            results["right_sampled"] = right_sampled
            results["right_curve"] = right_curve
            results["right_success"] = True
            print(
                f"    ✅ Lado derecho: {len(right_points)} pts → {len(right_sampled)} muestreados"
            )
        except Exception as e:
            print(f"    ⚠️  Error en lado derecho: {e}")
            results["right_success"] = False
    else:
        results["right_success"] = False

    # 4. CONEXIONES SUAVES
    if results["left_success"] and results["right_success"]:
        # Conexión inferior: lado izquierdo → lado derecho
        bottom_conn = create_smooth_connection(
            results["left_curve"], results["right_curve"][::-1], n_samples_conn
        )
        results["bottom_conn"] = bottom_conn
        results["bottom_success"] = True
        print(f"    ✅ Conexión inferior: {len(bottom_conn)} pts")
    else:
        results["bottom_success"] = False

    if results["prot_success"] and results["left_success"]:
        # Conexión superior izquierda: protuberancia → lado izquierdo
        left_top_conn = create_smooth_connection(
            results["prot_curve"][:1], results["left_curve"], n_samples_conn
        )
        results["left_top_conn"] = left_top_conn
        results["left_top_success"] = True
        print(f"    ✅ Conexión superior izquierda: {len(left_top_conn)} pts")
    else:
        results["left_top_success"] = False

    if results["prot_success"] and results["right_success"]:
        # Conexión superior derecha: lado derecho → protuberancia
        right_top_conn = create_smooth_connection(
            results["right_curve"][::-1], results["prot_curve"][-1:], n_samples_conn
        )
        results["right_top_conn"] = right_top_conn
        results["right_top_success"] = True
        print(f"    ✅ Conexión superior derecha: {len(right_top_conn)} pts")
    else:
        results["right_top_success"] = False

    return results


def plot_complete_contour(analysis: dict, output_path: Path):
    """Visualiza el contorno completo del cráneo en un grid de 2 paneles."""
    slice_id = analysis["slice_id"]

    # Crear figura con 2 subplots (1 fila, 2 columnas)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

    # Colores
    color_cloud = "#BDBDBD"
    color_prot = "#E65100"  # Naranja
    color_left = "#1E88E5"  # Azul
    color_right = "#E53935"  # Rojo
    color_bottom = "#43A047"  # Verde
    color_top = "#8E24AA"  # Púrpura
    color_points = "#FFA726"  # Amarillo para puntos

    # Nube original (muy tenue) - en ambos paneles
    original = analysis["original"]
    for ax in [ax1, ax2]:
        ax.scatter(
            original[:, 0],
            original[:, 1],
            s=2,
            c=color_cloud,
            alpha=0.15,
            label=f"Original (N={len(original)})",
            zorder=1,
        )

    # ===== PANEL 1 (IZQUIERDO): CON PUNTOS =====
    # PROTUBERANCIA
    if analysis["prot_success"]:
        prot_curve = analysis["prot_curve"]
        ax1.plot(
            prot_curve[:, 0],
            prot_curve[:, 1],
            "-",
            lw=4,
            color=color_prot,
            label="Protuberancia (Y < 50)",
            zorder=5,
        )
        prot_sampled = analysis["prot_sampled"]
        ax1.plot(
            prot_sampled[:, 0],
            prot_sampled[:, 1],
            "o",
            ms=7,
            color=color_points,
            markeredgewidth=1.5,
            markeredgecolor="white",
            label="Puntos de control",
            zorder=6,
        )

    # LADO IZQUIERDO
    if analysis["left_success"]:
        left_curve = analysis["left_curve"]
        ax1.plot(
            left_curve[:, 0],
            left_curve[:, 1],
            "-",
            lw=4,
            color=color_left,
            label="Lado izquierdo (X < 120)",
            zorder=5,
        )
        left_sampled = analysis["left_sampled"]
        ax1.plot(
            left_sampled[:, 0],
            left_sampled[:, 1],
            "o",
            ms=7,
            color=color_points,
            markeredgewidth=1.5,
            markeredgecolor="white",
            zorder=6,
        )

    # LADO DERECHO
    if analysis["right_success"]:
        right_curve = analysis["right_curve"]
        ax1.plot(
            right_curve[:, 0],
            right_curve[:, 1],
            "-",
            lw=4,
            color=color_right,
            label="Lado derecho (X > 150)",
            zorder=5,
        )
        right_sampled = analysis["right_sampled"]
        ax1.plot(
            right_sampled[:, 0],
            right_sampled[:, 1],
            "o",
            ms=7,
            color=color_points,
            markeredgewidth=1.5,
            markeredgecolor="white",
            zorder=6,
        )

    # CONEXIÓN INFERIOR
    if analysis["bottom_success"]:
        bottom_conn = analysis["bottom_conn"]
        ax1.plot(
            bottom_conn[:, 0],
            bottom_conn[:, 1],
            "-",
            lw=4,
            color=color_bottom,
            label="Conexión inferior (Y ~ 240)",
            zorder=5,
        )
        ax1.plot(
            bottom_conn[:, 0],
            bottom_conn[:, 1],
            "s",
            ms=6,
            color=color_points,
            markeredgewidth=1.5,
            markeredgecolor="white",
            zorder=6,
        )

    # CONEXIONES SUPERIORES (con label en la leyenda)
    if analysis["left_top_success"]:
        left_top_conn = analysis["left_top_conn"]
        ax1.plot(
            left_top_conn[:, 0],
            left_top_conn[:, 1],
            "-",
            lw=4,
            color=color_top,
            label="Conexiones superiores",
            zorder=5,
        )
        ax1.plot(
            left_top_conn[:, 0],
            left_top_conn[:, 1],
            "^",
            ms=6,
            color=color_points,
            markeredgewidth=1.5,
            markeredgecolor="white",
            zorder=6,
        )

    if analysis["right_top_success"]:
        right_top_conn = analysis["right_top_conn"]
        ax1.plot(
            right_top_conn[:, 0],
            right_top_conn[:, 1],
            "-",
            lw=4,
            color=color_top,
            zorder=5,
        )
        ax1.plot(
            right_top_conn[:, 0],
            right_top_conn[:, 1],
            "^",
            ms=6,
            color=color_points,
            markeredgewidth=1.5,
            markeredgecolor="white",
            zorder=6,
        )

    # ===== PANEL 2 (DERECHO): SOLO CURVAS SUAVES =====
    # PROTUBERANCIA
    if analysis["prot_success"]:
        prot_curve = analysis["prot_curve"]
        ax2.plot(
            prot_curve[:, 0],
            prot_curve[:, 1],
            "-",
            lw=4,
            color=color_prot,
            label="Protuberancia (Y < 50)",
            zorder=5,
        )

    # LADO IZQUIERDO
    if analysis["left_success"]:
        left_curve = analysis["left_curve"]
        ax2.plot(
            left_curve[:, 0],
            left_curve[:, 1],
            "-",
            lw=4,
            color=color_left,
            label="Lado izquierdo (X < 120)",
            zorder=5,
        )

    # LADO DERECHO
    if analysis["right_success"]:
        right_curve = analysis["right_curve"]
        ax2.plot(
            right_curve[:, 0],
            right_curve[:, 1],
            "-",
            lw=4,
            color=color_right,
            label="Lado derecho (X > 150)",
            zorder=5,
        )

    # CONEXIÓN INFERIOR
    if analysis["bottom_success"]:
        bottom_conn = analysis["bottom_conn"]
        ax2.plot(
            bottom_conn[:, 0],
            bottom_conn[:, 1],
            "-",
            lw=4,
            color=color_bottom,
            label="Conexión inferior (Y ~ 240)",
            zorder=5,
        )

    # CONEXIONES SUPERIORES (con label en la leyenda)
    if analysis["left_top_success"]:
        left_top_conn = analysis["left_top_conn"]
        ax2.plot(
            left_top_conn[:, 0],
            left_top_conn[:, 1],
            "-",
            lw=4,
            color=color_top,
            label="Conexiones superiores",
            zorder=5,
        )

    if analysis["right_top_success"]:
        right_top_conn = analysis["right_top_conn"]
        ax2.plot(
            right_top_conn[:, 0],
            right_top_conn[:, 1],
            "-",
            lw=4,
            color=color_top,
            zorder=5,
        )

    # Configurar ambos paneles
    for ax in [ax1, ax2]:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X", fontweight="bold", fontsize=11)
        ax.set_ylabel("Y", fontweight="bold", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

    # Títulos específicos
    ax1.set_title(
        "Con Puntos de Control",
        fontweight="bold",
        fontsize=13,
    )
    ax2.set_title(
        "Curvas Suaves (sin puntos)",
        fontweight="bold",
        fontsize=13,
    )

    # Leyendas en posición central con estilo mejorado
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
        ncol=2,
        borderaxespad=0,
    )
    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
        ncol=2,
        borderaxespad=0,
    )

    # Título general
    fig.suptitle(
        f"Corte {slice_id}: Contorno Completo con B-splines",
        fontweight="bold",
        fontsize=15,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Genera contorno completo del corte 4 con B-splines y conexiones suaves"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/skull_edges", help="Directorio de datos"
    )
    parser.add_argument(
        "--slice-id", type=int, default=4, help="ID del corte (default: 4)"
    )
    parser.add_argument(
        "--y-prot", type=float, default=50, help="Umbral Y protuberancia (default: 50)"
    )
    parser.add_argument(
        "--x-left", type=float, default=120, help="Umbral X izquierdo (default: 120)"
    )
    parser.add_argument(
        "--x-right", type=float, default=150, help="Umbral X derecho (default: 150)"
    )
    parser.add_argument(
        "--y-bottom", type=float, default=240, help="Umbral Y inferior (default: 240)"
    )
    parser.add_argument(
        "--n-samples-prot",
        type=int,
        default=15,
        help="Puntos protuberancia (default: 15)",
    )
    parser.add_argument(
        "--n-samples-side",
        type=int,
        default=25,
        help="Puntos por lado (default: 25)",
    )
    parser.add_argument(
        "--n-samples-conn",
        type=int,
        default=12,
        help="Puntos por conexión (default: 12)",
    )
    parser.add_argument(
        "--degree", type=int, default=3, help="Grado B-spline (default: 3)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/figures/skull_complete_contour",
        help="Directorio de salida",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("CONTORNO COMPLETO DEL CRÁNEO CON B-SPLINES")
    print("=" * 70)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print(f"\n📁 Directorio de datos: {data_dir}")
    print(f"🔍 Corte a procesar: {args.slice_id}")
    print("\n📐 Criterios de segmentación:")
    print(f"   • Protuberancia: Y < {args.y_prot}")
    print(f"   • Lado izquierdo: X < {args.x_left}")
    print(f"   • Lado derecho: X > {args.x_right}")
    print(f"   • Conexión inferior: Y ~ {args.y_bottom}")
    print("\n🎯 Parámetros B-spline:")
    print(f"   • Grado: {args.degree}")
    print(f"   • Puntos protuberancia: {args.n_samples_prot}")
    print(f"   • Puntos lados: {args.n_samples_side}")
    print(f"   • Puntos conexiones: {args.n_samples_conn}")

    # Cargar datos
    slice_data = load_slice_points(data_dir, args.slice_id)

    if slice_data is None:
        print(f"\n❌ Error: No se pudo cargar el corte {args.slice_id}")
        return

    # Analizar
    analysis = analyze_complete_contour(
        slice_data,
        y_prot=args.y_prot,
        x_left=args.x_left,
        x_right=args.x_right,
        y_bottom=args.y_bottom,
        n_samples_prot=args.n_samples_prot,
        n_samples_side=args.n_samples_side,
        n_samples_conn=args.n_samples_conn,
        degree=args.degree,
    )

    # Verificar éxito
    total_success = sum(
        [
            analysis.get("prot_success", False),
            analysis.get("left_success", False),
            analysis.get("right_success", False),
            analysis.get("bottom_success", False),
            analysis.get("left_top_success", False),
            analysis.get("right_top_success", False),
        ]
    )

    print(f"\n  📊 Componentes exitosos: {total_success}/6")

    # Graficar
    output_path = output_dir / f"corte{args.slice_id}_complete_contour.png"
    plot_complete_contour(analysis, output_path)
    print(f"\n  ✅ Figura guardada: {output_path}")

    # Resumen final
    print("\n" + "=" * 70)
    print("✅ ANÁLISIS COMPLETADO")
    print("=" * 70)
    print(f"\n📁 Archivo generado: {output_path}")
    print(f"📊 Componentes del contorno: {total_success}/6")
    print("\n✨ Contorno completo con conexiones suaves generado exitosamente!\n")


if __name__ == "__main__":
    main()
