"""Track geometry: loads a track CSV and projects car positions onto the centerline.

CSV columns: x_m, y_m, w_left_m, w_right_m (one centerline point every ~10 cm,
counter-clockwise, first point on the start line, closed loop - the last point
connects back to the first).

The centerline is a periodic cubic spline x(s), y(s) parametrised by arc length s
(metres, 0 = start line, period L = lap length). Units: m, rad, left positive.

Agreed interface:
    s, e_c, e_psi, kappa = track.project(x, y, psi, s_prev)
    x_c, y_c = track.point_at(s)
"""
from __future__ import annotations

import math
import os

import numpy as np

TRACK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tracks")
TRACK_FILES = {
    "training": os.path.join(TRACK_DIR, "training_track.csv"),
    "hidden": os.path.join(TRACK_DIR, "hidden_track.csv"),
}


def wrap_angle(a):
    """Wrap an angle (or array of angles) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def _periodic_spline_second_derivs(h, y):
    """Second derivatives M_i of a periodic cubic spline through knots y_i.

    h[i] = s_{i+1} - s_i (h[-1] closes the loop). Solves the cyclic
    tridiagonal system densely (N ~ 1000, done once at load time).
    """
    n = len(y)
    A = np.zeros((n, n))
    rhs = np.empty(n)
    for i in range(n):
        hp = h[i - 1]          # interval before knot i (wraps)
        hn = h[i]              # interval after knot i
        A[i, (i - 1) % n] += hp / 6.0
        A[i, i] += (hp + hn) / 3.0
        A[i, (i + 1) % n] += hn / 6.0
        rhs[i] = (y[(i + 1) % n] - y[i]) / hn - (y[i] - y[i - 1]) / hp
    return np.linalg.solve(A, rhs)


class Track:
    """Closed race track with arc-length centerline spline.

    Public attributes:
        name, length (lap length L, m), s_knots, x_knots, y_knots,
        w_left, w_right (per knot, m).
    """

    def __init__(self, csv_path: str, name: str | None = None, search_back: float = 1.0,
                 search_ahead: float = 3.0):
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        if data.ndim != 2 or data.shape[1] < 4:
            raise ValueError(f"{csv_path}: expected columns x_m, y_m, w_left_m, w_right_m")
        # Drop a duplicated closing point if the file repeats the first point.
        if np.hypot(*(data[-1, :2] - data[0, :2])) < 1e-6:
            data = data[:-1]

        self.name = name or os.path.splitext(os.path.basename(csv_path))[0]
        self.csv_path = csv_path
        self.x_knots = data[:, 0].copy()
        self.y_knots = data[:, 1].copy()
        self.w_left = data[:, 2].copy()
        self.w_right = data[:, 3].copy()
        self.n = len(data)

        # Arc length by chord lengths (10 cm spacing -> error ~1e-5 relative).
        dx = np.roll(self.x_knots, -1) - self.x_knots
        dy = np.roll(self.y_knots, -1) - self.y_knots
        self._h = np.hypot(dx, dy)
        self.s_knots = np.concatenate(([0.0], np.cumsum(self._h)[:-1]))
        self.length = float(np.sum(self._h))
        self._mean_h = self.length / self.n

        self._Mx = _periodic_spline_second_derivs(self._h, self.x_knots)
        self._My = _periodic_spline_second_derivs(self._h, self.y_knots)

        # Projection search window (metres of arc length around s_prev).
        self.search_back = float(search_back)
        self.search_ahead = float(search_ahead)

    # ------------------------------------------------------------------ spline
    def _seg(self, s):
        """Segment index i and local coordinate u in [0, h_i) for arc length s."""
        s = s % self.length
        i = int(np.searchsorted(self.s_knots, s, side="right") - 1)
        i = min(max(i, 0), self.n - 1)
        return i, s - self.s_knots[i]

    def _eval(self, s):
        """(x, y, dx/ds, dy/ds, d2x/ds2, d2y/ds2) at scalar arc length s."""
        i, u = self._seg(s)
        j = (i + 1) % self.n
        h = self._h[i]
        a = (h - u) / h
        b = u / h
        out = []
        for P, M in ((self.x_knots, self._Mx), (self.y_knots, self._My)):
            p = (a * P[i] + b * P[j]
                 + ((a ** 3 - a) * M[i] + (b ** 3 - b) * M[j]) * h * h / 6.0)
            dp = ((P[j] - P[i]) / h
                  + (-(3 * a * a - 1) * M[i] + (3 * b * b - 1) * M[j]) * h / 6.0)
            ddp = a * M[i] + b * M[j]
            out.append((p, dp, ddp))
        (x, dx, ddx), (y, dy, ddy) = out
        return x, y, dx, dy, ddx, ddy

    def point_at(self, s):
        """Centerline point (x, y) at progress s (agreed interface; wraps around
        the lap, so s + lookahead past the finish line is fine)."""
        x, y, *_ = self._eval(s)
        return x, y

    point = point_at  # short alias used inside the sim code

    def heading(self, s):
        """Centerline tangent angle at s (rad, CCW from +x)."""
        _, _, dx, dy, _, _ = self._eval(s)
        return math.atan2(dy, dx)

    def curvature(self, s):
        """Signed centerline curvature at s (1/m, left turn positive)."""
        _, _, dx, dy, ddx, ddy = self._eval(s)
        return (dx * ddy - dy * ddx) / (dx * dx + dy * dy) ** 1.5

    def width(self, s):
        """Road half-widths (w_left, w_right) at s, linearly interpolated."""
        i, u = self._seg(s)
        j = (i + 1) % self.n
        b = u / self._h[i]
        return ((1 - b) * self.w_left[i] + b * self.w_left[j],
                (1 - b) * self.w_right[i] + b * self.w_right[j])

    def frenet_to_cartesian(self, s, e_c=0.0, e_psi=0.0):
        """Pose (x, y, psi) of a car at progress s, lateral offset e_c, heading error e_psi."""
        x, y, dx, dy, _, _ = self._eval(s)
        th = math.atan2(dy, dx)
        return (x - e_c * math.sin(th), y + e_c * math.cos(th), float(wrap_angle(th + e_psi)))

    # -------------------------------------------------------------- projection
    def project(self, x, y, psi, s_prev=None):
        """Project the car onto the centerline (agreed interface).

        Args:
            x, y:    car position (m).
            psi:     car heading (rad) - needed for the heading error.
            s_prev:  previous progress (m). The search only looks within
                     [s_prev - search_back, s_prev + search_ahead], so it never
                     jumps to a nearby section of track. None = search the whole
                     lap (use only at reset).

        Returns:
            s      progress along the lap, in [0, L)                  (m)
            e_c    lateral offset, left of centerline positive        (m)
            e_psi  heading error psi - road heading, in [-pi, pi)     (rad)
            kappa  centerline curvature at s, left turn positive      (1/m)
        """
        px, py = float(x), float(y)
        if s_prev is None:
            idx = np.arange(self.n)
        else:
            i0 = int(math.floor((s_prev - self.search_back) / self._mean_h))
            i1 = int(math.ceil((s_prev + self.search_ahead) / self._mean_h))
            idx = np.arange(i0, i1 + 1) % self.n

        # 1) closest point on the polyline segments in the window
        xi, yi = self.x_knots[idx], self.y_knots[idx]
        jdx = (idx + 1) % self.n
        sx, sy = self.x_knots[jdx] - xi, self.y_knots[jdx] - yi
        h2 = sx * sx + sy * sy
        t = np.clip(((px - xi) * sx + (py - yi) * sy) / h2, 0.0, 1.0)
        d2 = (xi + t * sx - px) ** 2 + (yi + t * sy - py) ** 2
        k = int(np.argmin(d2))
        s = self.s_knots[idx[k]] + t[k] * self._h[idx[k]]

        # 2) Newton refinement on the smooth spline: (c(s)-p) . c'(s) = 0
        for _ in range(3):
            cx, cy, dx, dy, ddx, ddy = self._eval(s)
            rx, ry = cx - px, cy - py
            f = rx * dx + ry * dy
            fp = dx * dx + dy * dy + rx * ddx + ry * ddy
            if fp <= 1e-9:
                break
            step = max(-0.1, min(0.1, f / fp))
            s -= step
            if abs(step) < 1e-9:
                break
        s %= self.length

        cx, cy, dx, dy, ddx, ddy = self._eval(s)
        nrm = math.hypot(dx, dy)
        tx, ty = dx / nrm, dy / nrm
        e_c = tx * (py - cy) - ty * (px - cx)
        th = math.atan2(dy, dx)
        e_psi = float(wrap_angle(psi - th))
        kappa = (dx * ddy - dy * ddx) / nrm ** 3
        return float(s), float(e_c), e_psi, float(kappa)

    # ------------------------------------------------------------ road ahead
    def lookahead(self, s, distances):
        """Curvature of the road at s + d for each d in `distances` (1/m)."""
        return np.array([self.curvature(s + d) for d in distances])

    def heading_ahead(self, s, distances, psi):
        """Road direction at s + d relative to the car heading psi, wrapped (rad).
        Positive = the road turns left of where the car points."""
        return wrap_angle(np.array([self.heading(s + d) for d in distances]) - psi)

    def lookahead_points(self, s, distances, x=None, y=None, psi=None):
        """Centerline points at s + d. If x, y, psi are given, returned in the
        car frame (x forward, y left); otherwise in world frame. Shape (N, 2)."""
        pts = np.array([self.point(s + d) for d in distances])
        if x is None:
            return pts
        c, sn = math.cos(psi), math.sin(psi)
        rel = pts - np.array([x, y])
        return np.stack([c * rel[:, 0] + sn * rel[:, 1], -sn * rel[:, 0] + c * rel[:, 1]], axis=1)

    # -------------------------------------------------------------- for MPCC
    def centerline_table(self, ds=0.1, s_min=None, s_max=None):
        """Sampled centerline for building your own interpolant.

        Returns dict of arrays: s, x, y, theta (unwrapped heading), kappa,
        w_left, w_right. By default covers [-5 m, L + 20 m] so an MPCC horizon
        can run past the finish line without wrapping inside the solver.
        """
        s_min = -5.0 if s_min is None else s_min
        s_max = self.length + 20.0 if s_max is None else s_max
        s = np.arange(s_min, s_max + 0.5 * ds, ds)
        ev = np.array([self._eval(si) for si in s])
        theta = np.unwrap(np.arctan2(ev[:, 3], ev[:, 2]))
        kappa = (ev[:, 2] * ev[:, 5] - ev[:, 3] * ev[:, 4]) / np.hypot(ev[:, 2], ev[:, 3]) ** 3
        wl, wr = np.array([self.width(si) for si in s]).T
        return {"s": s, "x": ev[:, 0], "y": ev[:, 1], "theta": theta, "kappa": kappa,
                "w_left": wl, "w_right": wr}

    def casadi_centerline(self, ds=0.1, s_min=None, s_max=None):
        """Smooth centerline as CasADi B-spline interpolants (requires `casadi`).

        Returns dict of casadi.Function, each mapping s (m) -> value:
        'x', 'y', 'theta', 'kappa'. The splines cover s in [s_min, s_max] only
        (default [-5, L + 20], i.e. one lap plus 20 m past the finish line).
        """
        try:
            import casadi as ca
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("casadi_centerline() needs `pip install casadi`") from exc
        tab = self.centerline_table(ds, s_min, s_max)
        grid = [tab["s"].tolist()]
        return {k: ca.interpolant(f"track_{k}", "bspline", grid, tab[k].tolist())
                for k in ("x", "y", "theta", "kappa")}


def load_track(track="training", **kwargs) -> Track:
    """Load 'training', 'hidden', or a path to a track CSV."""
    path = TRACK_FILES.get(track, track)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Track '{track}' not found (looked for {path})")
    name = track if track in TRACK_FILES else None
    return Track(path, name=name, **kwargs)
