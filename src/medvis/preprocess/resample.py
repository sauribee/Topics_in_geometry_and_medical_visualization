import SimpleITK as sitk
from typing import Tuple


def resample_isotropic(
    img: sitk.Image, new_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> sitk.Image:
    """Resamplea el volumen a espaciado isotrópico dado (mm)."""
    old_spacing = img.GetSpacing()  # (sx, sy, sz)
    old_size = img.GetSize()  # (Nx, Ny, Nz) en ITK (ojo: ITK usa (x,y,z))
    new_size = [
        int(round(old_size[i] * (old_spacing[i] / new_spacing[i]))) for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)  # para intensidades CT
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(img)
