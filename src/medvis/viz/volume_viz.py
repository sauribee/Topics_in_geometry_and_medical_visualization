from __future__ import annotations
import numpy as np
import SimpleITK as sitk
import pyvista as pv
from skimage.measure import marching_cubes

__all__ = ["orthogonal_slices", "isosurface_mesh"]


def orthogonal_slices(img: sitk.Image):
    vol = sitk.GetArrayFromImage(img)  # (z,y,x)
    z, y, x = vol.shape
    axial = vol[z // 2, :, :]
    coronal = vol[:, y // 2, :]
    sagittal = vol[:, :, x // 2]
    return axial, coronal, sagittal


def isosurface_mesh(img: sitk.Image, level: float) -> pv.PolyData:
    vol = sitk.GetArrayFromImage(img)  # (z,y,x)
    verts, faces, _, _ = marching_cubes(vol, level=level)
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    mesh = pv.PolyData(verts[:, ::-1], faces_pv)  # (x,y,z)
    return mesh
