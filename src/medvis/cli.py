from __future__ import annotations
import argparse
import os
import csv
import SimpleITK as sitk
import pyvista as pv
from medvis.io.dicom_series import read_from_path, get_spacing_xyz
from medvis.preprocess.resample import resample_isotropic
from medvis.viz.volume_viz import orthogonal_slices, isosurface_mesh
from medvis.metrics.mesh_metrics import surface_area, volume_if_closed


def main() -> None:
    p = argparse.ArgumentParser(
        description="DICOM (file/dir) → slices, isosurface, metrics"
    )
    p.add_argument("--dicom-path", required=True, help="DICOM file or directory")
    p.add_argument("--out-dir", default="reports/figures/cli", help="Output folder")
    p.add_argument("--level", type=float, default=300.0, help="Isosurface level")
    p.add_argument(
        "--iso-spacing",
        type=float,
        default=None,
        help="If set (e.g., 1.0), resample to isotropic spacing (mm)",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="Save an off-screen PNG preview of the mesh",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    img = read_from_path(args.dicom_path)
    if args.iso_spacing is not None:
        img = resample_isotropic(img, (args.iso_spacing,) * 3)

    sx, sy, sz = get_spacing_xyz(img)
    print(f"Spacing (mm): {(sx, sy, sz)}")

    axial, coronal, sagittal = orthogonal_slices(img)
    sitk.WriteImage(
        sitk.GetImageFromArray(axial), os.path.join(args.out_dir, "axial.nii.gz")
    )
    sitk.WriteImage(
        sitk.GetImageFromArray(coronal), os.path.join(args.out_dir, "coronal.nii.gz")
    )
    sitk.WriteImage(
        sitk.GetImageFromArray(sagittal), os.path.join(args.out_dir, "sagittal.nii.gz")
    )

    mesh = isosurface_mesh(img, level=args.level)
    ply_path = os.path.join(args.out_dir, "isosurface.ply")
    mesh.save(ply_path)

    area = surface_area(mesh)
    vol = volume_if_closed(mesh)
    print(f"Area (mm^2): {area:.3f}  |  Volume (mm^3): {vol:.3f}")

    with open(os.path.join(args.out_dir, "mesh_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "ply_path",
                "area_mm2",
                "volume_mm3",
                "spacing_x",
                "spacing_y",
                "spacing_z",
                "level",
            ]
        )
        w.writerow([ply_path, area, vol, sx, sy, sz, args.level])

    if args.preview:
        pvt = pv.Plotter(off_screen=True)
        pvt.add_mesh(mesh, show_edges=False)
        pvt.add_axes()
        pvt.show_bounds(grid="back", location="outer")
        png = os.path.join(args.out_dir, "isosurface_preview.png")
        pvt.show(screenshot=png)
        print(f"Preview: {png}")

    print("Done.")


if __name__ == "__main__":
    main()
