from __future__ import annotations
import pyvista as pv


def surface_area(mesh: pv.PolyData) -> float:
    # requiere malla triangulada
    return mesh.triangulate().area


def volume_if_closed(mesh: pv.PolyData) -> float:
    # usa vtkMassProperties; devuelve 0.0 si no es cerrada
    tri = mesh.triangulate()
    if tri.n_cells == 0:
        return 0.0
    # comprobar bordes abiertos
    edges = tri.extract_feature_edges(boundary_edges=True)
    if edges.n_points > 0:
        return 0.0
    mp = pv._vtk.vtkMassProperties()
    mp.SetInputData(tri)
    mp.Update()
    return float(mp.GetVolume())
