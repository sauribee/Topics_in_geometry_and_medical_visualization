from __future__ import annotations
import argparse
import os
import SimpleITK as sitk
from medvis.io.dicom_series import read_series_to_volume, get_spacing_xyz
from medvis.viz.volume_viz import orthogonal_slices, isosurface_mesh


def main() -> None:
    p = argparse.ArgumentParser(description="DICOM → volume → slices & isosurface")
    p.add_argument(
        "--dicom-dir", required=True, help="Folder containing a DICOM series"
    )
    p.add_argument("--out-dir", default="reports/figures/cli", help="Output folder")
    p.add_argument("--level", type=float, default=300.0, help="Isosurface level")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    img = read_series_to_volume(args.dicom_dir)
    sx, sy, sz = get_spacing_xyz(img)
    print(f"Spacing (mm): {(sx, sy, sz)}")

    axial, coronal, sagital = orthogonal_slices(img)
    sitk.WriteImage(
        sitk.GetImageFromArray(axial), os.path.join(args.out_dir, "axial.nii.gz")
    )
    sitk.WriteImage(
        sitk.GetImageFromArray(coronal), os.path.join(args.out_dir, "coronal.nii.gz")
    )
    sitk.WriteImage(
        sitk.GetImageFromArray(sagital), os.path.join(args.out_dir, "sagittal.nii.gz")
    )

    mesh = isosurface_mesh(img, level=args.level)
    mesh.save(os.path.join(args.out_dir, "isosurface.ply"))
    print("Done.")


if __name__ == "__main__":
    main()
