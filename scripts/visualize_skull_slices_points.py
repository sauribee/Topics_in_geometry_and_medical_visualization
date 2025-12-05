#!/usr/bin/env python3
"""
Visualize Skull Slices Points
=============================

Script sencillo para visualizar la nube de puntos original de cada corte
axial del cráneo, sin aplicar interpolación ni procesamiento adicional.
"""

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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

        return {"id": slice_id, "x": x, "y": y, "n_points": len(x)}
    except Exception as e:
        print(f"Error cargando corte {slice_id}: {e}")
        return None


def plot_slice_points(slice_data: dict, output_path: Path):
    """Grafica la nube de puntos de un solo corte."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        slice_data["x"],
        slice_data["y"],
        s=10,
        c="#1565C0",  # Azul
        alpha=0.6,
        label=f"Puntos (N={slice_data['n_points']})",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Corte {slice_data['id']}", fontweight="bold", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.invert_yaxis()  # Coordenadas de imagen
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_all_slices_grid(all_slices: list, output_path: Path):
    """Crea un grid comparativo de todas las nubes de puntos."""
    n_slices = len(all_slices)
    rows = int(np.ceil(n_slices / 5))
    cols = 5

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    axes = axes.flatten()

    for i, slice_data in enumerate(all_slices):
        ax = axes[i]
        ax.scatter(slice_data["x"], slice_data["y"], s=2, c="#1565C0", alpha=0.5)
        ax.set_aspect("equal")
        ax.set_title(
            f"Corte {slice_data['id']}\n(N={slice_data['n_points']})", fontsize=10
        )
        ax.invert_yaxis()
        ax.axis("off")

    # Ocultar ejes vacíos
    for i in range(n_slices, len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        "Nubes de Puntos: Cortes Axiales de Cráneo", fontsize=16, fontweight="bold"
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualizar nubes de puntos de cortes de cráneo"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/skull_edges", help="Directorio de datos"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/figures/skull_points_viz",
        help="Directorio de salida",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print(f"📁 Cargando datos desde: {data_dir}")

    all_slices = []
    for i in range(10):
        data = load_slice_points(data_dir, i)
        if data:
            all_slices.append(data)
            print(f"  ✓ Corte {i}: {data['n_points']} puntos")

            # Graficar individual
            plot_slice_points(data, output_dir / "individual" / f"corte{i}_points.png")

    # Graficar grid
    if all_slices:
        plot_all_slices_grid(all_slices, output_dir / "all_slices_grid.png")
        print(f"\n✅ Visualizaciones guardadas en: {output_dir}")


if __name__ == "__main__":
    main()
