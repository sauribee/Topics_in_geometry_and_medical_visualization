import numpy as np
import SimpleITK as sitk
from medvis.viz.volume_viz import isosurface_mesh


def test_isosurface_on_blob():
    Z, Y, X = 32, 32, 32
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X]
    cx, cy, cz, r = X / 2, Y / 2, Z / 2, 8
    vol = (((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) <= r**2).astype(float)
    img = sitk.GetImageFromArray(vol)
    mesh = isosurface_mesh(img, level=0.5)
    assert mesh.n_points > 0 and mesh.n_cells > 0
