"""Offline racing line + speed profile for the MPCC (computed once per track).

1. Minimum-curvature line: choose a lateral offset n(s) from the centerline (left
   positive, |n| <= half width - margin) that minimises the squared second
   difference of the path points - a standard bounded least-squares
   approximation of "minimise total curvature" (e.g. Heilmeier et al. 2019).
2. Speed profile along that line: v = sqrt(a_lat / |kappa|) capped at v_max,
   then a backward pass (braking) and a forward pass (motor) on its arc length.

Uses only the public track API (point_at, heading, width, length), so it works
on any track the env loads - nothing is tuned per track.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear


def _periodic_d2(n):
    D = np.zeros((n, n))
    idx = np.arange(n)
    D[idx, (idx - 1) % n] = 1.0
    D[idx, idx] = -2.0
    D[idx, (idx + 1) % n] = 1.0
    return D


def _smooth(v, passes):
    for _ in range(passes):
        v = 0.25 * np.roll(v, 1) + 0.5 * v + 0.25 * np.roll(v, -1)
    return v


class RacingLine:
    """Lateral offset n(s), path curvature and target speed, all looked up by centerline s."""

    def __init__(self, track, car_params, *, margin=0.30, ds=0.25, lat_frac=0.92,
                 brake_frac=0.75, accel_frac=1.0, v_max=None, min_curvature=True):
        L = float(track.length)
        M = int(round(L / ds))
        s = np.linspace(0.0, L, M, endpoint=False)
        c = np.array([track.point_at(v) for v in s])
        th = np.unwrap(np.array([track.heading(v) for v in s]))
        nrm = np.stack([-np.sin(th), np.cos(th)], axis=1)          # left normal
        w = np.array([track.width(v) for v in s])                    # (w_left, w_right)
        lo = -(w[:, 1] - margin)
        hi = w[:, 0] - margin
        if min_curvature:
            D = _periodic_d2(M)
            A = np.vstack([D * nrm[:, 0], D * nrm[:, 1]])
            b = -np.concatenate([D @ c[:, 0], D @ c[:, 1]])
            n = lsq_linear(A, b, bounds=(lo, hi), method="bvls").x
            n = np.clip(_smooth(n, 3), lo, hi)
        else:
            n = np.zeros(M)
        p = c + n[:, None] * nrm
        # derivatives w.r.t. centerline s on the periodic grid
        d1 = (np.roll(p, -1, 0) - np.roll(p, 1, 0)) / (2 * ds)
        d2 = (np.roll(p, -1, 0) - 2 * p + np.roll(p, 1, 0)) / ds**2
        speed = np.linalg.norm(d1, axis=1)                         # d(path length)/ds
        kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(speed, 1e-6) ** 3
        kappa = _smooth(kappa, 2)
        v_top = float(v_max if v_max is not None else car_params["v_max"])
        a_lat = lat_frac * car_params["a_lat_max"]
        v = np.minimum(np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-4)), v_top)
        dl = speed * ds                                              # path length per cell
        a_up = accel_frac * car_params["a_max"]
        a_dn = brake_frac * car_params["b_max"]
        for _ in range(3):                                           # periodic: a few sweeps
            for j in range(M - 1, -1, -1):
                k = (j + 1) % M
                v[j] = min(v[j], np.sqrt(v[k] ** 2 + 2 * a_dn * dl[j]))
            for j in range(M):
                k = (j - 1) % M
                v[j] = min(v[j], np.sqrt(v[k] ** 2 + 2 * a_up * dl[k]))
        self.length = L
        self.s, self.n, self.kappa, self.v, self.path_rate = s, n, kappa, v, speed
        self.points = p
        self.lap_estimate = float(np.sum(dl / np.maximum(v, 0.1)))

    def _lookup(self, arr, s):
        ss = np.mod(np.asarray(s, dtype=float), self.length)
        return np.interp(ss, np.r_[self.s, self.length], np.r_[arr, arr[0]])

    def offset(self, s):
        return self._lookup(self.n, s)

    def speed(self, s):
        return self._lookup(self.v, s)

    def rate(self, s):
        """d(racing-line length)/d(centerline s)."""
        return self._lookup(self.path_rate, s)
