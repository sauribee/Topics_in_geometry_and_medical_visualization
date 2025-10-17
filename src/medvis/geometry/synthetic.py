from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

ArrayB = NDArray[np.bool_]
ArrayU8 = NDArray[np.uint8]
ArrayF = NDArray[np.float64]

__all__ = ["elliptical_cortex_mask"]


def _rotated_extents(a: float, b: float, theta: float) -> Tuple[float, float]:
    """
    Half-extent (Rx, Ry) of an ellipse with semi-axes (a, b) rotated by angle theta (radians).
    Used to ensure the outer ellipse fits inside the image with a safety margin.
    """
    c, s = np.cos(theta), np.sin(theta)
    # Projection of rotated ellipse onto image axes:
    # Rx = sqrt((a cosθ)^2 + (b sinθ)^2), Ry = sqrt((a sinθ)^2 + (b cosθ)^2)
    Rx = float(np.hypot(a * c, b * s))
    Ry = float(np.hypot(a * s, b * c))
    return Rx, Ry


def elliptical_cortex_mask(
    H: int,
    W: int,
    *,
    cx: float | None = None,
    cy: float | None = None,
    a_out: float = 80.0,
    b_out: float = 50.0,
    angle_deg: float = 25.0,
    t_major: float = 12.0,
    t_minor: float = 8.0,
    inner_dx: float = 6.0,
    inner_dy: float = -4.0,
    notch_enable: bool = True,
    notch_radius: float = 6.0,
    notch_phi_deg: float = 0.0,
    margin_px: int = 6,
) -> ArrayU8:
    """
    Generate a binary mask emulating a bone cross-section:
    a rotated elliptical cortex (outer ellipse minus inner ellipse, possibly eccentric),
    with an optional cortical notch.

    The function **guarantees** the outer ellipse fits inside the image by scaling it
    if necessary to keep a `margin_px` to the image borders. This avoids degenerate
    cases where the contour becomes the image rectangle.

    Parameters
    ----------
    H, W : int
        Image height and width (pixels).
    cx, cy : float, optional
        Center in pixels. Defaults to image center.
    a_out, b_out : float
        Outer ellipse semi-axes (pixels).
    angle_deg : float
        Rotation angle of the ellipse in degrees.
    t_major, t_minor : float
        Approximate cortical thickness along major/minor axes (pixels).
        Inner semi-axes are computed as (a_out - t_major, b_out - t_minor), clamped to >= 1.
    inner_dx, inner_dy : float
        Eccentricity of the medullary cavity (shift of inner ellipse center) in LAB frame (pixels).
    notch_enable : bool
        Whether to cut a small notch on the outer cortex.
    notch_radius : float
        Radius of the notch (pixels).
    notch_phi_deg : float
        Angular position of the notch on the outer ellipse (degrees) in the ellipse frame.
    margin_px : int
        Safety margin to borders (pixels). The outer ellipse is scaled down if needed
        to keep this margin after rotation.

    Returns
    -------
    mask : (H,W) uint8
        1 where cortex, 0 elsewhere.
    """
    if cx is None:
        cx = (W - 1) * 0.5
    if cy is None:
        cy = (H - 1) * 0.5

    # Ensure the outer ellipse fits within the image with a margin.
    theta = float(np.deg2rad(angle_deg))
    Rx, Ry = _rotated_extents(a_out, b_out, theta)

    # Available half-room to borders along x/y for the chosen center:
    room_x = float(min(cx, (W - 1) - cx)) - margin_px
    room_y = float(min(cy, (H - 1) - cy)) - margin_px
    room_x = max(room_x, 1.0)
    room_y = max(room_y, 1.0)

    # Scale factor so that Rx <= room_x and Ry <= room_y
    scale = float(min(room_x / max(Rx, 1e-6), room_y / max(Ry, 1e-6), 1.0))
    if scale < 1.0:
        a_out *= scale
        b_out *= scale
        Rx, Ry = _rotated_extents(a_out, b_out, theta)

    # Inner semi-axes
    a_in = max(a_out - t_major, 1.0)
    b_in = max(b_out - t_minor, 1.0)

    # Grid
    yy, xx = np.mgrid[0:H, 0:W]
    x = xx - cx
    y = yy - cy

    # Rotation matrices: LAB → ellipse frame (R(-θ)) and back (R(θ))
    cth, sth = np.cos(theta), np.sin(theta)
    xp = cth * x + sth * y
    yp = -sth * x + cth * y

    # Outer ellipse
    outer = ((xp / a_out) ** 2 + (yp / b_out) ** 2) <= 1.0

    # Inner ellipse (eccentric): shift in LAB then rotate
    x_in = x - inner_dx
    y_in = y - inner_dy
    xp_in = cth * x_in + sth * y_in
    yp_in = -sth * x_in + cth * y_in
    inner = ((xp_in / a_in) ** 2 + (yp_in / b_in) ** 2) <= 1.0

    ring = outer & (~inner)

    if notch_enable and notch_radius > 0:
        # Notch center on outer ellipse at angle φ in ellipse frame
        phi = float(np.deg2rad(notch_phi_deg))
        p_e = np.array([a_out * np.cos(phi), b_out * np.sin(phi)])  # ellipse frame
        # Map to LAB and translate by center
        R = np.array([[cth, -sth], [sth, cth]])
        notch_center = (R @ p_e) + np.array([cx, cy])
        nx = xx - notch_center[0]
        ny = yy - notch_center[1]
        notch = (nx * nx + ny * ny) <= (notch_radius**2)
        ring = ring & (~notch)

    return ring.astype(np.uint8)
