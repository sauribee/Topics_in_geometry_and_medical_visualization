import numpy as np
import SimpleITK as sitk
from medvis.io.dicom_series import get_spacing_xyz


def test_spacing_xyz_synthetic():
    vol = (np.random.rand(8, 16, 24) * 255).astype(np.float32)  # (z,y,x)
    img = sitk.GetImageFromArray(vol)
    assert get_spacing_xyz(img) == (1.0, 1.0, 1.0)
