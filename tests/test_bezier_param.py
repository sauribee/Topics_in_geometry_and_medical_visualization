import numpy as np
from medvis.geometry.bezier import (
    BezierCurve,
    arc_chord_parameterization,
    arclength_parameterization,
)


def test_arc_chord_monotone_and_bounds():
    cps = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 2.0], [4.0, 0.0]], dtype=float)
    curve = BezierCurve(cps)
    ts, s = arc_chord_parameterization(curve, samples=1024, normalize=True)
    assert np.isclose(ts[0], 0.0) and np.isclose(ts[-1], 1.0)
    assert np.isclose(s[0], 0.0) and np.isclose(s[-1], 1.0)
    assert np.all(np.diff(ts) > 0)
    assert np.all(np.diff(s) >= 0)


def test_arclength_uniform_line():
    cps = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    curve = BezierCurve(cps)
    n = 17
    tvals = arclength_parameterization(curve, n=n, samples=2048)
    uniform = np.linspace(0.0, 1.0, n)
    assert np.allclose(tvals, uniform, rtol=1e-6, atol=1e-7)


def test_arclength_equal_chords_cubic():
    cps = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, -1.0], [4.0, 0.0]], dtype=float)
    curve = BezierCurve(cps)
    n = 64
    tvals = arclength_parameterization(curve, n=n, samples=4096)
    P = curve.evaluate_batch(tvals)
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    m = float(np.mean(d))
    assert m > 0
    cv = float(np.std(d) / m)
    assert cv < 0.1


def test_arclength_degenerate_curve_uniform():
    cps = np.array([[2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0]], dtype=float)
    curve = BezierCurve(cps)
    n = 10
    tvals = arclength_parameterization(curve, n=n, samples=64)
    uniform = np.linspace(0.0, 1.0, n)
    assert np.allclose(tvals, uniform, rtol=1e-12, atol=1e-12)
    P = curve.evaluate_batch(tvals)
    assert np.allclose(P, P[0][None, :])
