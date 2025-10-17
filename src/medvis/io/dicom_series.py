# src/medvis/io/dicom_series.py
from __future__ import annotations
from pathlib import Path
import os
from typing import Tuple
import SimpleITK as sitk

__all__ = [
    "find_first_series",
    "read_series_to_volume",
    "read_from_path",
    "get_spacing_xyz",
]


def find_first_series(dicom_dir: str | os.PathLike) -> list[str]:
    """Devuelve la lista de archivos DICOM (una serie) encontrada en `dicom_dir`."""
    dicom_dir = Path(dicom_dir)
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise FileNotFoundError(f"No DICOM series found: {dicom_dir}")
    file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        str(dicom_dir), series_ids[0]
    )
    if not file_names:
        raise FileNotFoundError(
            f"Series exists but no files were returned: {dicom_dir}"
        )
    return list(file_names)


def read_series_to_volume(dicom_dir: str | os.PathLike) -> sitk.Image:
    """Lee la PRIMERA serie detectada en un directorio y devuelve un volumen SimpleITK."""
    files = find_first_series(dicom_dir)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    return reader.Execute()


def _series_dirs_under(root: Path) -> list[Path]:
    """Encuentra recursivamente directorios que contengan al menos una serie DICOM."""
    found: list[Path] = []
    if not root.is_dir():
        return found

    # El propio root
    try:
        if sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(root)):
            found.append(root)
    except Exception:
        pass

    # Subdirectorios
    for sub in root.rglob("*"):
        if not sub.is_dir():
            continue
        try:
            if sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(sub)):
                found.append(sub)
        except Exception:
            continue
    return found


def read_from_path(dicom_path: str | os.PathLike) -> sitk.Image:
    """
    Carga un volumen DICOM a partir de:
      - una carpeta que contenga la serie,
      - una carpeta raíz con subcarpetas de series (búsqueda recursiva),
      - o un archivo DICOM (usa su carpeta padre).
    """
    p = Path(dicom_path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")

    # Si es archivo, usar su carpeta
    if p.is_file():
        p = p.parent

    # Intento directo en p
    try:
        return read_series_to_volume(p)
    except FileNotFoundError:
        pass

    # Búsqueda recursiva bajo p
    for d in _series_dirs_under(p):
        try:
            return read_series_to_volume(d)
        except Exception:
            continue

    raise FileNotFoundError(f"No DICOM series found at or under: {p}")


def get_spacing_xyz(img: sitk.Image) -> Tuple[float, float, float]:
    """Devuelve el spacing (x, y, z) en mm tal como lo reporta SimpleITK."""
    sx, sy, sz = img.GetSpacing()
    return float(sx), float(sy), float(sz)
