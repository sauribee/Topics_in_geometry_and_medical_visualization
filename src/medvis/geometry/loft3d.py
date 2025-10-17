from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .contour2d import resample_closed_polyline

ArrayF = NDArray[np.float64]
ArrayI = NDArray[np.int64]


__all__ = [
    "ContourRing",
    "LoftParams",
    "LoftMesh",
    "align_rings_by_min_l2",
    "loft_triangulated",
    "save_ply_ascii",
    "to_pyvista_polydata",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContourRing:
    """
    A closed planar ring at z = const, given in physical coordinates (x, y).

    The ring is implicitly closed; the first vertex is not repeated at the end.
    """

    z: float
    xy: ArrayF  # shape (N, 2), closed implicitly

    def as_xyz(self) -> ArrayF:
        """Return (N,3) array by stacking z as third coordinate."""
        N = self.xy.shape[0]
        zcol = np.full((N, 1), float(self.z), dtype=np.float64)
        return np.hstack([np.asarray(self.xy, dtype=np.float64), zcol])


@dataclass(frozen=True)
class LoftParams:
    """
    Parameters controlling ring resampling, alignment and meshing.
    """

    samples_per_ring: int = 180  # common N for all rings after resampling
    cap_ends: bool = False  # add triangle fans capping first and last rings
    reverse_allowed: bool = True  # allow flipping orientation to minimize twist


@dataclass(frozen=True)
class LoftMesh:
    """
    Triangle mesh from lofting a stack of rings.
    """

    vertices: ArrayF  # (M, 3)
    faces: ArrayI  # (K, 3) triangle indices into 'vertices'

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])


# ---------------------------------------------------------------------------
# Alignment utilities
# ---------------------------------------------------------------------------


def _ring_cost_shift(A: ArrayF, B: ArrayF) -> float:
    """
    Sum of squared distances between A[k] and B[k] for k=0..N-1 (no shift).
    """
    dif = A - B
    return float(np.sum(np.einsum("ij,ij->i", dif, dif)))


def _best_circular_shift(A: ArrayF, B: ArrayF) -> Tuple[int, float]:
    """
    Find circular shift s that minimizes L2 distance between A and roll(B, s).

    Returns
    -------
    (s, cost)
    """
    N = A.shape[0]
    best_s = 0
    best_cost = np.inf
    for s in range(N):
        cost = _ring_cost_shift(A, np.roll(B, shift=s, axis=0))
        if cost < best_cost:
            best_cost = cost
            best_s = s
    return best_s, best_cost


def align_rings_by_min_l2(
    prev_xy: ArrayF,
    curr_xy: ArrayF,
    *,
    allow_reverse: bool = True,
) -> ArrayF:
    """
    Align current ring to previous one by (i) optional reversal and (ii) optimal
    circular shift that minimizes sum of squared vertex distances.

    Both inputs must be sampled with the same N and be closed implicitly.
    """
    A = np.asarray(prev_xy, dtype=np.float64)
    B = np.asarray(curr_xy, dtype=np.float64)
    if A.shape != B.shape:
        raise ValueError("Rings must have the same shape for alignment.")

    # Try normal orientation
    s_fwd, c_fwd = _best_circular_shift(A, B)

    if not allow_reverse:
        return np.roll(B, shift=s_fwd, axis=0)

    # Try reversed orientation
    B_rev = B[::-1].copy()
    s_rev, c_rev = _best_circular_shift(A, B_rev)

    if c_rev < c_fwd:
        return np.roll(B_rev, shift=s_rev, axis=0)
    else:
        return np.roll(B, shift=s_fwd, axis=0)


# ---------------------------------------------------------------------------
# Lofting core
# ---------------------------------------------------------------------------


def _resample_rings(rings: List[ContourRing], N: int) -> List[ContourRing]:
    """
    Resample each ring to exactly N points (even arc-length spacing).
    """
    out: List[ContourRing] = []
    for r in rings:
        xyN = resample_closed_polyline(r.xy, n=N)
        out.append(ContourRing(z=float(r.z), xy=xyN))
    return out


def _align_stack(rings: List[ContourRing], allow_reverse: bool) -> List[ContourRing]:
    """
    Align each ring to the previous one (minimize twist).
    """
    if len(rings) <= 1:
        return rings
    aligned: List[ContourRing] = [rings[0]]
    for i in range(1, len(rings)):
        prev = aligned[-1]
        curr = rings[i]
        xy_aligned = align_rings_by_min_l2(
            prev.xy, curr.xy, allow_reverse=allow_reverse
        )
        aligned.append(ContourRing(z=curr.z, xy=xy_aligned))
    return aligned


def _connect_rings_faces(n_rings: int, N: int) -> ArrayI:
    """
    Create triangle faces connecting consecutive rings with N vertices per ring.

    For ring i and i+1, we add two triangles per edge k:
      (i,k) - (i,k+1) - (i+1,k)   and   (i,k+1) - (i+1,k+1) - (i+1,k)
    """
    faces: List[Tuple[int, int, int]] = []
    for i in range(n_rings - 1):
        base_i = i * N
        base_j = (i + 1) * N
        for k in range(N):
            k1 = (k + 1) % N
            a = base_i + k
            b = base_i + k1
            c = base_j + k
            d = base_j + k1
            faces.append((a, b, c))
            faces.append((b, d, c))
    return np.asarray(faces, dtype=np.int64)


def _cap_ring_faces(base: int, N: int, reverse: bool) -> ArrayI:
    """
    Create triangle fan faces capping a single ring starting at index 'base'.
    A new center vertex must be appended by the caller at index 'base_center'.
    If reverse=True, reverse triangle orientation.
    """
    faces: List[Tuple[int, int, int]] = []
    # triangle fan: (center, base+k, base+k+1)
    center = base  # convention: caller provides 'center' as first index in a block
    ring0 = base + 1
    for k in range(N):
        a = center
        b = ring0 + k
        c = ring0 + ((k + 1) % N)
        if reverse:
            faces.append((a, c, b))
        else:
            faces.append((a, b, c))
    return np.asarray(faces, dtype=np.int64)


def loft_triangulated(
    rings_in: List[ContourRing],
    params: Optional[LoftParams] = None,
) -> LoftMesh:
    """
    Loft a stack of closed rings (x,y at z=const) into a watertight triangle mesh.

    Steps
    -----
    1) Resample each ring to a common number of points N.
    2) Align every ring to the previous to minimize twist and seam mismatch.
    3) Build vertex array (stack z) and triangular faces connecting rings.
    4) Optionally add triangle-fan caps at the first and last rings.

    Notes
    -----
    - The method assumes one *outer* ring per slice (no holes). If you need
      inner rings (medullary cavity), consider generating separate surfaces or
      using a true surface loft algorithm with topology.
    """
    if params is None:
        params = LoftParams()

    if len(rings_in) < 2:
        raise ValueError("At least two rings are required for lofting.")

    N = int(params.samples_per_ring)
    if N < 3:
        raise ValueError("samples_per_ring must be >= 3.")

    # 1) Resample
    rings = _resample_rings(rings_in, N=N)

    # 2) Align stack
    rings = _align_stack(rings, allow_reverse=bool(params.reverse_allowed))

    # 3) Assemble vertices
    verts_list = [r.as_xyz() for r in rings]
    V: ArrayF = np.vstack(verts_list)  # shape = (n_rings*N, 3)
    n_rings = len(rings)

    # 4) Side faces
    F_sides = _connect_rings_faces(n_rings, N)

    if not params.cap_ends:
        return LoftMesh(vertices=V, faces=F_sides)

    # 5) Caps: we add two centers and fans
    # First ring cap (at z of rings[0])
    center0 = np.mean(rings[0].as_xyz(), axis=0, keepdims=True)  # (1,3)
    # Last ring cap
    center1 = np.mean(rings[-1].as_xyz(), axis=0, keepdims=True)

    # New vertex array: [center0, ring0, ..., ring_{last}, center1]
    V_cap0 = np.vstack([center0, rings[0].as_xyz()])
    V_mid = (
        np.vstack([r.as_xyz() for r in rings[1:-1]])
        if n_rings > 2
        else np.zeros((0, 3))
    )
    V_cap1 = np.vstack([rings[-1].as_xyz(), center1])
    V_all = np.vstack([V_cap0, V_mid, V_cap1])

    # Indexing:
    # cap0 block starts at 0, length = 1 + N
    # ring blocks from previous 'V' are re-indexed here; recompute faces accordingly.
    # For simplicity, rebuild all faces with the new indexing:
    #   Block layout:
    #   [ (center0, ring0), ring1, ring2, ..., ring_{last-1}, (ring_last, center1) ]
    blocks_starts: List[int] = []
    offset = 0
    blocks_starts.append(offset)  # cap0 block
    offset += 1 + N
    for _ in range(1, n_rings - 1):
        blocks_starts.append(offset)  # ring i block
        offset += N
    blocks_starts.append(offset)  # last block (ring_last + center1)
    # Sanity
    expected_total = (1 + N) + max(0, (n_rings - 2) * N) + (N + 1)
    assert V_all.shape[0] == expected_total

    # Rebuild side faces with new indices
    faces: List[Tuple[int, int, int]] = []
    # Between cap0-ring0 block and ring1 (if exists)
    if n_rings >= 2:
        base_i = 1  # first ring starts after center0 in cap0 block
        base_j = 1 + N  # next block start (ring1) if exists
        if n_rings == 2:
            # special: next block is the last block where ring_last starts at 'last_block_start'
            base_j = blocks_starts[-1]
        for k in range(N):
            k1 = (k + 1) % N
            a = base_i + k
            b = base_i + k1
            c = base_j + k
            d = base_j + k1
            faces.append((a, b, c))
            faces.append((b, d, c))
    # Between middle rings (if any)
    for i in range(1, n_rings - 2):
        base_i = blocks_starts[i]
        base_j = blocks_starts[i + 1]
        for k in range(N):
            k1 = (k + 1) % N
            a = base_i + k
            b = base_i + k1
            c = base_j + k
            d = base_j + k1
            faces.append((a, b, c))
            faces.append((b, d, c))
    # Between last middle ring and last block (ring_last + center1)
    if n_rings >= 3:
        base_i = blocks_starts[-2]
        base_j = blocks_starts[-1]
        for k in range(N):
            k1 = (k + 1) % N
            a = base_i + k
            b = base_i + k1
            c = base_j + k
            d = base_j + k1
            faces.append((a, b, c))
            faces.append((b, d, c))

    # Caps
    F_cap0 = _cap_ring_faces(base=blocks_starts[0], N=N, reverse=False)  # outward-ish
    # Last block: ring_last starts at blocks_starts[-1], center1 is at the end (index = start+N)
    # We arranged the last block as [ring_last (N), center1 (1)], so create a temporary
    # layout [center1, ring_last] to reuse _cap_ring_faces:
    # Build faces manually to keep correctness:
    last_block_start = blocks_starts[-1]
    center1_idx = last_block_start + N
    for k in range(N):
        b = last_block_start + k
        c = last_block_start + (k + 1) % N
        # orientation: aim for outward considering +z is "up"
        faces.append((center1_idx, c, b))

    F = np.vstack([np.asarray(faces, dtype=np.int64), F_cap0])
    return LoftMesh(vertices=V_all, faces=F)


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------


def save_ply_ascii(path: str | Path, mesh: LoftMesh) -> Path:
    """
    Save a triangle mesh to PLY ASCII (vertices + faces only).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    with p.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {V.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {F.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in V:
            f.write(f"{v[0]:.9f} {v[1]:.9f} {v[2]:.9f}\n")
        for tri in F:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")
    return p


def to_pyvista_polydata(mesh: LoftMesh):
    """
    Convert to pyvista.PolyData if pyvista is available; otherwise returns None.
    """
    try:
        import pyvista as pv  # type: ignore
    except Exception:
        return None
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    # PyVista expects a 'faces' array with a leading 3 per triangle.
    faces_pv = np.hstack([np.full((F.shape[0], 1), 3, dtype=np.int64), F]).ravel()
    return pv.PolyData(V, faces_pv)
