#!/usr/bin/env python3
"""
Batch B-spline Analysis: Multi-Slice Skull with Protuberance Detection
========================================================================

Procesa múltiples cortes axiales de cráneo aplicando:
1. Interpolación B-spline del contorno completo
2. Detección y ajuste de protuberancia occipital por corte
3. Visualización 2D individual por corte
4. Grid comparativo multi-corte
5. Análisis de métricas entre cortes

Este script extiende el análisis de B-spline de un solo corte
a una serie completa de cortes axiales para análisis 3D.
"""

import argparse
import ast
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from medvis.geometry.bezier import chord_parameterization
from medvis.geometry.bspline import evaluate_bspline, fit_bspline_interpolate

# Configuración de estilo
plt.style.use("seaborn-v0_8-darkgrid")


def load_slice_data(data_dir: Path, slice_id: int) -> Optional[Dict]:
    """
    Carga los datos de un corte específico.

    Parameters
    ----------
    data_dir : Path
        Directorio raíz que contiene las carpetas corteN
    slice_id : int
        ID del corte (0-9)

    Returns
    -------
    dict or None
        Diccionario con datos del corte:
        {
            'id': int,
            'x': np.ndarray,
            'y': np.ndarray,
            'points': np.ndarray (N, 2),
            'n_points': int
        }
        None si el corte no existe
    """
    slice_dir = data_dir / f"corte{slice_id}"

    if not slice_dir.exists():
        print(f"  ⚠️  Corte {slice_id} no encontrado en {slice_dir}")
        return None

    try:
        # Leer archivos de coordenadas (formato: [x1, x2, ..., xn])
        x_file = slice_dir / f"corte{slice_id}_x.txt"
        y_file = slice_dir / f"corte{slice_id}_y.txt"

        x_str = x_file.read_text().strip()
        y_str = y_file.read_text().strip()

        # Parse lista Python usando ast.literal_eval (más seguro que eval)
        x = np.array(ast.literal_eval(x_str))
        y = np.array(ast.literal_eval(y_str))

        if len(x) != len(y):
            print(f"  ⚠️  Corte {slice_id}: longitudes X e Y no coinciden")
            return None

        points = np.column_stack([x, y])

        return {
            "id": slice_id,
            "x": x,
            "y": y,
            "points": points,
            "n_points": len(x),
        }

    except Exception as e:
        print(f"  ❌ Error cargando corte {slice_id}: {e}")
        return None


def extract_protuberance(
    contour: np.ndarray, y_threshold: float = 50.0, method: str = "absolute"
) -> np.ndarray:
    """
    Extrae los puntos de la protuberancia occipital como una curva abierta continua.

    Parameters
    ----------
    contour : (n, 2) array
        Puntos del contorno completo
    y_threshold : float
        Umbral para identificar protuberancia
    method : str
        'absolute': Y < y_threshold (valor fijo)
        'percentile': Y < percentil(y_threshold) (adaptativo)

    Returns
    -------
    (m, 2) array
        Puntos de la protuberancia ordenados de izquierda a derecha
    """
    if method == "absolute":
        mask = contour[:, 1] < y_threshold
    elif method == "percentile":
        threshold_val = np.percentile(contour[:, 1], y_threshold)
        mask = contour[:, 1] < threshold_val
    else:
        raise ValueError(f"Método no válido: {method}")

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

    # Interpolar para obtener nuevos puntos
    sampled_x = np.interp(target_s, arc_length_norm, points[:, 0])
    sampled_y = np.interp(target_s, arc_length_norm, points[:, 1])

    return np.column_stack([sampled_x, sampled_y])


def analyze_slice(
    slice_data: Dict,
    n_samples_full: int = 20,
    n_samples_prot: int = 10,
    degree: int = 3,
    y_threshold: float = 50.0,
    protuberance_method: str = "absolute",
) -> Dict:
    """
    Análisis completo de un corte con B-splines.

    Parameters
    ----------
    slice_data : dict
        Datos del corte (de load_slice_data)
    n_samples_full : int
        Número de puntos para muestrear el contorno completo
    n_samples_prot : int
        Número de puntos para muestrear la protuberancia
    degree : int
        Grado del B-spline
    y_threshold : float
        Umbral para detección de protuberancia
    protuberance_method : str
        Método de detección ('absolute' o 'percentile')

    Returns
    -------
    dict
        Análisis completo con datos de contorno y protuberancia
    """
    points = slice_data["points"]

    # 1. CONTORNO COMPLETO
    full_sampled = sample_uniform_arclength(points, n_samples_full)

    try:
        knots_f, C_f, p_f = fit_bspline_interpolate(full_sampled, degree=degree)

        # Evaluar curva densa
        t_dense = np.linspace(0, 1, 500)
        curve_f_dense = evaluate_bspline(knots_f, C_f, p_f, t_dense)

        # Calcular error de interpolación
        u = chord_parameterization(full_sampled, alpha=1.0, normalize=True)
        errors_f = []
        for i, pt in enumerate(full_sampled):
            pt_interp = evaluate_bspline(knots_f, C_f, p_f, np.array([u[i]]))
            errors_f.append(np.linalg.norm(pt - pt_interp[0]))

        full_result = {
            "original": points,
            "sampled": full_sampled,
            "knots": knots_f,
            "control_points": C_f,
            "degree": p_f,
            "curve_dense": curve_f_dense,
            "error_max": max(errors_f) if errors_f else 0,
            "error_mean": np.mean(errors_f) if errors_f else 0,
            "success": True,
        }
    except Exception as e:
        print(f"    ⚠️  Error en contorno completo corte {slice_data['id']}: {e}")
        full_result = {
            "original": points,
            "sampled": full_sampled,
            "knots": None,
            "control_points": None,
            "degree": None,
            "curve_dense": full_sampled,
            "error_max": 0,
            "error_mean": 0,
            "success": False,
        }

    # 2. PROTUBERANCIA
    prot_points = extract_protuberance(points, y_threshold, protuberance_method)

    if len(prot_points) > degree + 1:
        prot_sampled = sample_uniform_arclength(prot_points, n_samples_prot)

        try:
            knots_p, C_p, p_p = fit_bspline_interpolate(prot_sampled, degree=degree)

            # Evaluar curva densa
            curve_p_dense = evaluate_bspline(knots_p, C_p, p_p, t_dense)

            # Calcular error
            u_p = chord_parameterization(prot_sampled, alpha=1.0, normalize=True)
            errors_p = []
            for i, pt in enumerate(prot_sampled):
                pt_interp = evaluate_bspline(knots_p, C_p, p_p, np.array([u_p[i]]))
                errors_p.append(np.linalg.norm(pt - pt_interp[0]))

            prot_result = {
                "original": prot_points,
                "sampled": prot_sampled,
                "knots": knots_p,
                "control_points": C_p,
                "degree": p_p,
                "curve_dense": curve_p_dense,
                "error_max": max(errors_p) if errors_p else 0,
                "error_mean": np.mean(errors_p) if errors_p else 0,
                "success": True,
            }
        except Exception as e:
            print(f"    ⚠️  Error en protuberancia corte {slice_data['id']}: {e}")
            prot_result = {
                "original": prot_points,
                "sampled": prot_sampled,
                "knots": None,
                "control_points": None,
                "degree": None,
                "curve_dense": prot_sampled,
                "error_max": 0,
                "error_mean": 0,
                "success": False,
            }
    else:
        # Muy pocos puntos para B-spline
        prot_result = {
            "original": prot_points,
            "sampled": prot_points,
            "knots": None,
            "control_points": None,
            "degree": None,
            "curve_dense": prot_points if len(prot_points) > 0 else np.array([]),
            "error_max": 0,
            "error_mean": 0,
            "success": False,
        }

    return {
        "slice_id": slice_data["id"],
        "full": full_result,
        "protuberance": prot_result,
    }


def plot_slice_report(analysis: Dict, output_path: Path) -> None:
    """
    Genera reporte visual de un corte individual: full + protuberance.

    Parameters
    ----------
    analysis : dict
        Resultado de analyze_slice
    output_path : Path
        Ruta de salida para la figura
    """
    slice_id = analysis["slice_id"]

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, wspace=0.35, width_ratios=[1.3, 1.3, 1])

    # Colores
    color_curve = "#6A1B9A"  # Morado B-spline
    color_points = "#00897B"  # Verde azulado
    color_cloud = "#BDBDBD"  # Gris
    color_prot = "#E65100"  # Naranja

    # Panel 1: Contorno completo
    ax1 = fig.add_subplot(gs[0, 0])

    # Nube de puntos tenue
    full_orig = analysis["full"]["original"]
    ax1.scatter(
        full_orig[:, 0],
        full_orig[:, 1],
        s=6,
        c=color_cloud,
        alpha=0.25,
        label=f"Nube (N={len(full_orig)})",
        zorder=1,
    )

    # Curva B-spline
    curve_f = analysis["full"]["curve_dense"]
    ax1.plot(
        curve_f[:, 0],
        curve_f[:, 1],
        "-",
        lw=3,
        color=color_curve,
        label=f"B-spline (grado {analysis['full']['degree']})",
        zorder=3,
    )

    # Puntos muestreados
    sampled_f = analysis["full"]["sampled"]
    ax1.plot(
        sampled_f[:, 0],
        sampled_f[:, 1],
        "o",
        ms=8,
        color=color_points,
        markeredgewidth=2,
        markeredgecolor="white",
        label=f"Interpolados (N={len(sampled_f)})",
        zorder=5,
    )

    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title(
        f"Corte {slice_id}: Contorno Completo", fontweight="bold", fontsize=12
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(
        loc="upper left", bbox_to_anchor=(0, -0.1), fontsize=9, ncol=3, frameon=True
    )
    ax1.set_xlabel("X", fontweight="bold")
    ax1.set_ylabel("Y", fontweight="bold")
    ax1.invert_yaxis()

    # Panel 2: Protuberancia
    ax2 = fig.add_subplot(gs[0, 1])

    prot_orig = analysis["protuberance"]["original"]

    if len(prot_orig) > 0 and analysis["protuberance"]["success"]:
        # Nube de puntos tenue
        ax2.scatter(
            prot_orig[:, 0],
            prot_orig[:, 1],
            s=6,
            c=color_cloud,
            alpha=0.25,
            label=f"Nube (N={len(prot_orig)})",
            zorder=1,
        )

        # Curva B-spline
        curve_p = analysis["protuberance"]["curve_dense"]
        ax2.plot(
            curve_p[:, 0],
            curve_p[:, 1],
            "-",
            lw=3,
            color=color_prot,
            label=f"B-spline (grado {analysis['protuberance']['degree']})",
            zorder=3,
        )

        # Puntos muestreados
        sampled_p = analysis["protuberance"]["sampled"]
        ax2.plot(
            sampled_p[:, 0],
            sampled_p[:, 1],
            "o",
            ms=8,
            color=color_points,
            markeredgewidth=2,
            markeredgecolor="white",
            label=f"Interpolados (N={len(sampled_p)})",
            zorder=5,
        )

        ax2.set_aspect("equal", adjustable="box")
        ax2.set_title(
            f"Corte {slice_id}: Protuberancia Occipital",
            fontweight="bold",
            fontsize=12,
        )
        ax2.grid(True, alpha=0.3)
        ax2.legend(
            loc="upper left",
            bbox_to_anchor=(0, -0.1),
            fontsize=9,
            ncol=3,
            frameon=True,
        )
        ax2.invert_yaxis()
    else:
        ax2.text(
            0.5,
            0.5,
            "Sin protuberancia\ndetectada",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
        )
        ax2.axis("off")

    ax2.set_xlabel("X", fontweight="bold")
    ax2.set_ylabel("Y", fontweight="bold")

    # Panel 3: Información del sistema
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")

    info_text = f"""
CORTE {slice_id}: B-SPLINE
{'=' * 30}

CONTORNO COMPLETO:
  • Pts originales: {len(full_orig)}
  • Pts muestreados: {len(sampled_f)}
  • Grado B-spline: {analysis['full']['degree'] or 'N/A'}
  • Pts control: {len(analysis['full']['control_points']) if analysis['full']['control_points'] is not None else 'N/A'}
  • Error máx: {analysis['full']['error_max']:.2e}
  • Error medio: {analysis['full']['error_mean']:.2e}
  • Status: {'✅ OK' if analysis['full']['success'] else '⚠️  Error'}

PROTUBERANCIA:
  • Pts originales: {len(prot_orig)}
  • Pts muestreados: {len(analysis['protuberance']['sampled'])}
  • Grado B-spline: {analysis['protuberance']['degree'] or 'N/A'}
  • Error máx: {analysis['protuberance']['error_max']:.2e}
  • Error medio: {analysis['protuberance']['error_mean']:.2e}
  • Status: {'✅ OK' if analysis['protuberance']['success'] else '⚠️  No detectada'}

VENTAJAS B-SPLINE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ Control LOCAL
  ✓ Grado bajo fijo
  ✓ Estabilidad numérica
  ✓ Ideal para médico
    """

    ax3.text(
        0.05,
        0.95,
        info_text,
        transform=ax3.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lavender", alpha=0.3),
    )

    fig.suptitle(
        f"Análisis B-spline: Corte {slice_id}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_multi_slice_grid(
    all_analyses: List[Dict], output_path: Path, title: str = None
) -> None:
    """
    Crea grid comparativo mostrando todos los cortes.

    Parameters
    ----------
    all_analyses : list of dict
        Lista de análisis de cada corte
    output_path : Path
        Ruta de salida
    title : str, optional
        Título personalizado
    """
    n_slices = len(all_analyses)

    # Determinar layout del grid
    if n_slices <= 6:
        nrows, ncols = 2, 3
    elif n_slices <= 10:
        nrows, ncols = 2, 5
    else:
        nrows = int(np.ceil(n_slices / 5))
        ncols = 5

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    color_full = "#6A1B9A"  # Morado
    color_prot = "#E65100"  # Naranja
    color_cloud = "#BDBDBD"

    for idx, analysis in enumerate(all_analyses):
        ax = axes[idx]

        # Plotear contorno completo
        full_orig = analysis["full"]["original"]
        full_curve = analysis["full"]["curve_dense"]

        ax.scatter(
            full_orig[:, 0],
            full_orig[:, 1],
            s=3,
            c=color_cloud,
            alpha=0.2,
            zorder=1,
        )
        ax.plot(full_curve[:, 0], full_curve[:, 1], lw=2.5, c=color_full, zorder=3)

        # Plotear protuberancia si existe
        if (
            len(analysis["protuberance"]["original"]) > 0
            and analysis["protuberance"]["success"]
        ):
            prot_curve = analysis["protuberance"]["curve_dense"]
            ax.plot(
                prot_curve[:, 0],
                prot_curve[:, 1],
                lw=2,
                c=color_prot,
                linestyle="--",
                zorder=4,
                label="Protuberancia",
            )

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"Corte {analysis['slice_id']} (N={len(full_orig)})",
            fontweight="bold",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)

    # Ocultar ejes vacíos
    for idx in range(n_slices, len(axes)):
        axes[idx].axis("off")

    if title is None:
        title = f"B-spline Multi-Corte: {n_slices} Cortes Axiales de Cráneo"

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Grid multi-corte guardado: {output_path.name}")


def compute_metrics(all_analyses: List[Dict]) -> pd.DataFrame:
    """
    Calcula métricas por corte.

    Returns
    -------
    pd.DataFrame
        Tabla con métricas de cada corte
    """
    rows = []
    for analysis in all_analyses:
        row = {
            "slice_id": analysis["slice_id"],
            "n_points_full": len(analysis["full"]["original"]),
            "n_sampled_full": len(analysis["full"]["sampled"]),
            "degree_full": analysis["full"]["degree"],
            "error_max_full": analysis["full"]["error_max"],
            "error_mean_full": analysis["full"]["error_mean"],
            "success_full": analysis["full"]["success"],
            "n_points_prot": len(analysis["protuberance"]["original"]),
            "n_sampled_prot": len(analysis["protuberance"]["sampled"]),
            "degree_prot": analysis["protuberance"]["degree"],
            "error_max_prot": analysis["protuberance"]["error_max"],
            "error_mean_prot": analysis["protuberance"]["error_mean"],
            "has_protuberance": analysis["protuberance"]["success"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def plot_metrics_summary(df: pd.DataFrame, output_path: Path) -> None:
    """
    Genera gráficas de resumen de métricas.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Número de puntos por corte
    ax1 = axes[0, 0]
    ax1.bar(df["slice_id"], df["n_points_full"], color="#6A1B9A", alpha=0.7)
    ax1.set_xlabel("Corte ID", fontweight="bold")
    ax1.set_ylabel("Número de Puntos", fontweight="bold")
    ax1.set_title("Puntos Originales por Corte", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # 2. Error medio de interpolación
    ax2 = axes[0, 1]
    ax2.plot(
        df["slice_id"],
        df["error_mean_full"],
        "o-",
        lw=2,
        ms=8,
        color="#6A1B9A",
        label="Contorno completo",
    )
    ax2.plot(
        df[df["has_protuberance"]]["slice_id"],
        df[df["has_protuberance"]]["error_mean_prot"],
        "s--",
        lw=2,
        ms=8,
        color="#E65100",
        label="Protuberancia",
    )
    ax2.set_xlabel("Corte ID", fontweight="bold")
    ax2.set_ylabel("Error Medio", fontweight="bold")
    ax2.set_title("Error de Interpolación B-spline", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    # 3. Detección de protuberancia
    ax3 = axes[1, 0]
    has_prot = df["has_protuberance"].sum()
    no_prot = len(df) - has_prot
    ax3.bar(
        ["Con Protuberancia", "Sin Protuberancia"],
        [has_prot, no_prot],
        color=["#E65100", "#BDBDBD"],
        alpha=0.7,
    )
    ax3.set_ylabel("Número de Cortes", fontweight="bold")
    ax3.set_title("Detección de Protuberancia Occipital", fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Puntos de protuberancia por corte
    ax4 = axes[1, 1]
    df_prot = df[df["has_protuberance"]]
    if len(df_prot) > 0:
        ax4.bar(
            df_prot["slice_id"],
            df_prot["n_points_prot"],
            color="#E65100",
            alpha=0.7,
        )
        ax4.set_xlabel("Corte ID", fontweight="bold")
        ax4.set_ylabel("Número de Puntos", fontweight="bold")
        ax4.set_title("Puntos en Protuberancia por Corte", fontweight="bold")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(
            0.5,
            0.5,
            "No se detectaron\nprotuberancias",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax4.axis("off")

    fig.suptitle("Métricas de Análisis Multi-Corte", fontsize=16, fontweight="bold")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Resumen de métricas guardado: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch B-spline analysis para múltiples cortes de cráneo"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/skull_edges",
        help="Directorio con carpetas corteN",
    )
    parser.add_argument(
        "--slice-start",
        type=int,
        default=0,
        help="ID del primer corte a procesar (default: 0)",
    )
    parser.add_argument(
        "--slice-end",
        type=int,
        default=9,
        help="ID del último corte a procesar (default: 9)",
    )
    parser.add_argument(
        "--n-samples-full",
        type=int,
        default=20,
        help="Número de puntos para contorno completo (default: 20)",
    )
    parser.add_argument(
        "--n-samples-prot",
        type=int,
        default=10,
        help="Número de puntos para protuberancia (default: 10)",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=3,
        help="Grado del B-spline (default: 3 = cúbico)",
    )
    parser.add_argument(
        "--y-threshold",
        type=float,
        default=50.0,
        help="Umbral Y para protuberancia (default: 50.0)",
    )
    parser.add_argument(
        "--protuberance-method",
        type=str,
        choices=["absolute", "percentile"],
        default="absolute",
        help="Método de detección de protuberancia (default: absolute)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/figures/skull_slices_bspline",
        help="Directorio de salida",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("BATCH B-SPLINE ANALYSIS: MULTI-SLICE SKULL")
    print("=" * 70)
    print()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        print(f"❌ Error: {data_dir} no existe")
        return

    # 1. CARGAR TODOS LOS CORTES
    print(f"📁 Cargando cortes desde: {data_dir}")
    print(f"   Rango: corte{args.slice_start} - corte{args.slice_end}")
    print()

    all_slice_data = []
    for slice_id in range(args.slice_start, args.slice_end + 1):
        slice_data = load_slice_data(data_dir, slice_id)
        if slice_data is not None:
            all_slice_data.append(slice_data)
            print(f"  ✓ Corte {slice_id}: {slice_data['n_points']} puntos cargados")
        else:
            print(f"  ⚠️  Corte {slice_id}: no se pudo cargar")

    if len(all_slice_data) == 0:
        print("\n❌ No se cargó ningún corte. Terminando.")
        return

    print(f"\n  Total: {len(all_slice_data)} cortes cargados")
    print()

    # 2. ANALIZAR CADA CORTE
    print("🔬 Analizando cortes con B-splines...")
    print(
        f"   Parámetros: grado={args.degree}, n_full={args.n_samples_full}, n_prot={args.n_samples_prot}"
    )
    print()

    all_analyses = []
    for slice_data in all_slice_data:
        print(f"  Procesando corte {slice_data['id']}...")
        analysis = analyze_slice(
            slice_data,
            n_samples_full=args.n_samples_full,
            n_samples_prot=args.n_samples_prot,
            degree=args.degree,
            y_threshold=args.y_threshold,
            protuberance_method=args.protuberance_method,
        )
        all_analyses.append(analysis)
        print(
            f"    ✓ Completado (Full: {'OK' if analysis['full']['success'] else 'Error'}, Prot: {'OK' if analysis['protuberance']['success'] else 'No detectada'})"
        )

    print()

    # 3. GENERAR REPORTES INDIVIDUALES
    print("📊 Generando reportes individuales...")
    individual_dir = output_dir / "individual_slices"
    individual_dir.mkdir(parents=True, exist_ok=True)

    for analysis in all_analyses:
        slice_id = analysis["slice_id"]
        output_path = individual_dir / f"corte{slice_id}_bspline.png"
        plot_slice_report(analysis, output_path)
        print(f"  ✓ Corte {slice_id} guardado: {output_path.name}")

    print()

    # 4. GRID MULTI-CORTE
    print("📐 Generando grid comparativo multi-corte...")
    grid_path = output_dir / "00_multi_slice_grid.png"
    plot_multi_slice_grid(all_analyses, grid_path)
    print()

    # 5. MÉTRICAS
    print("📈 Calculando métricas...")
    df_metrics = compute_metrics(all_analyses)

    # Guardar CSV
    csv_path = output_dir / "slice_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)
    print(f"  ✓ Métricas guardadas: {csv_path.name}")

    # Gráficas de resumen
    metrics_plot_path = output_dir / "01_metrics_summary.png"
    plot_metrics_summary(df_metrics, metrics_plot_path)
    print()

    # 6. RESUMEN FINAL
    print("=" * 70)
    print("✅ ANÁLISIS COMPLETADO")
    print("=" * 70)
    print()
    print(f"📁 Directorio de salida: {output_dir}")
    print()
    print("📊 Estadísticas:")
    print(f"   • Cortes procesados: {len(all_analyses)}")
    print(f"   • Cortes con protuberancia: {df_metrics['has_protuberance'].sum()}")
    print(f"   • Puntos promedio por corte: {df_metrics['n_points_full'].mean():.0f}")
    print(f"   • Error medio interpolación: {df_metrics['error_mean_full'].mean():.2e}")
    print()
    print("📁 Archivos generados:")
    print(f"   • {len(all_analyses)} reportes individuales")
    print("   • 1 grid comparativo")
    print("   • 1 resumen de métricas")
    print("   • 1 archivo CSV con datos")
    print()


if __name__ == "__main__":
    main()
