from __future__ import annotations
from typing import Sequence, Tuple
import SimpleITK as sitk
import pydicom

__all__ = [
    "find_first_series",
    "read_series_to_volume",
    "get_spacing_xyz",
    "get_pixel_spacing_xy",
]


def find_first_series(dicom_dir: str) -> Sequence[str]:
    reader = sitk.ImageSeriesReader()
    series = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series:
        raise FileNotFoundError(f"No DICOM series found: {dicom_dir}")
    files: Sequence[str] = reader.GetGDCMSeriesFileNames(dicom_dir, series[0])
    return files


def read_series_to_volume(dicom_dir: str, *, load_private: bool = False) -> sitk.Image:
    files = find_first_series(dicom_dir)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    if load_private:
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
    return reader.Execute()


def get_pixel_spacing_xy(dicom_path: str) -> Tuple[float, float]:
    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
    ps = getattr(ds, "PixelSpacing", [1.0, 1.0])
    try:
        return float(ps[0]), float(ps[1])
    except Exception:
        return 1.0, 1.0


def get_spacing_xyz(
    img: sitk.Image, *, fallback: float = 1.0
) -> Tuple[float, float, float]:
    sp = img.GetSpacing()
    sx = float(sp[0]) if len(sp) > 0 else fallback
    sy = float(sp[1]) if len(sp) > 1 else fallback
    sz = float(sp[2]) if len(sp) > 2 else fallback
    return sx, sy, sz
