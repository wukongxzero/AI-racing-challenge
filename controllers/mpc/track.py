"""Track geometry for the AI Racing Challenge MPCC controller.

CSV columns: x_m, y_m, w_left_m, w_right_m. Centerline direction
is the positive driving direction; left/right are defined accordingly.
"""
from pathlib import Path
import argparse
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree


class Track:
    def __init__(self, csv_path):
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
        required = {"x_m", "y_m", "w_left_m", "w_right_m"}
        if not required.issubset(data.dtype.names or ()):
            raise ValueError(f"Track CSV needs columns {sorted(required)}")
        points = np.column_stack((data["x_m"], data["y_m"]))
        widths_left = np.asarray(data["w_left_m"], dtype=float)
        widths_right = np.asarray(data["w_right_m"], dtype=float)
        if len(points) < 4 or not np.isfinite(points).all():
            raise ValueError("Track requires at least four finite points")
        if np.any(widths_left <= 0) or np.any(widths_right <= 0):
            raise ValueError("Track widths must be positive")

        # Data follows a closed loop, but the first point is not duplicated.
        # Append first point to enforce periodic geometry.
        step = np.linalg.norm(np.diff(points, axis=0), axis=1)
        closing_step = np.linalg.norm(points[0] - points[-1])
        if np.any(step <= 1e-10) or closing_step <= 1e-10:
            raise ValueError("Track has repeated neighboring points")
        self.s_nodes = np.concatenate(([0.0], np.cumsum(np.r_[step, closing_step])))
        self.length = float(self.s_nodes[-1])
        closed_points = np.vstack([points, points[0]])
        self._x = CubicSpline(self.s_nodes, closed_points[:, 0], bc_type="periodic")
        self._y = CubicSpline(self.s_nodes, closed_points[:, 1], bc_type="periodic")
        self._wl = CubicSpline(self.s_nodes, np.r_[widths_left, widths_left[0]], bc_type="periodic")
        self._wr = CubicSpline(self.s_nodes, np.r_[widths_right, widths_right[0]], bc_type="periodic")
        self._sample_s = np.linspace(0, self.length, max(2000, len(points)*2), endpoint=False)
        self._tree = cKDTree(self.center(self._sample_s))

    def _wrap(self, s):
        return np.mod(s, self.length)

    def center(self, s):
        t = self._wrap(s)
        return np.stack([self._x(t), self._y(t)], axis=-1)

    def tangent(self, s):
        t = self._wrap(s)
        derivative = np.stack([self._x(t, 1), self._y(t, 1)], axis=-1)
        return derivative / np.linalg.norm(derivative, axis=-1, keepdims=True)

    def normal(self, s):
        t = self.tangent(s)
        return np.stack([-t[..., 1], t[..., 0]], axis=-1)

    def heading(self, s):
        t = self.tangent(s)
        return np.arctan2(t[..., 1], t[..., 0])

    def curvature(self, s):
        t = self._wrap(s)
        dx, dy = self._x(t, 1), self._y(t, 1)
        ddx, ddy = self._x(t, 2), self._y(t, 2)
        return (dx * ddy - dy * ddx) / (dx * dx + dy * dy) ** 1.5

    def boundaries(self, s):
        c, n = self.center(s), self.normal(s)
        t = self._wrap(s)
        return c + self._wl(t)[..., None] * n, c - self._wr(t)[..., None] * n

    def project(self, x, y):
        """Return nearest arc-length coordinate s on centerline (meters)."""
        p = np.array([x, y], dtype=float)
        _, idx = self._tree.query(p)
        guess = self._sample_s[idx]
        radius = 2 * self.length / len(self._sample_s)
        result = minimize_scalar(
            lambda s: float(np.sum((self.center(s) - p) ** 2)),
            bounds=(guess - radius, guess + radius), method="bounded",
            options={"xatol": 1e-9},
        )
        return float(self._wrap(result.x))

    def errors(self, x, y, s):
        """Contouring (+ left) and lag (+ forward) errors in meters."""
        delta = np.array([x, y]) - self.center(s)
        return float(np.dot(delta, self.normal(s))), float(np.dot(delta, self.tangent(s)))

    def within_bounds(self, x, y, car_half_width=0.0):
        s = self.project(x, y)
        ec, _ = self.errors(x, y, s)
        return bool(-float(self._wr(s)) + car_half_width <= ec <= float(self._wl(s)) - car_half_width)


def plot_track(track, save_path):
    import matplotlib.pyplot as plt
    s = np.linspace(0, track.length, 1500)
    c, (left, right) = track.center(s), track.boundaries(s)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(left[:, 0], left[:, 1], "--", label="Left boundary")
    ax.plot(right[:, 0], right[:, 1], "--", label="Right boundary")
    ax.plot(c[:, 0], c[:, 1], label="Centerline")
    ax.scatter([c[0, 0]], [c[0, 1]], marker="o", label="Start", s=70)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="Path to training_track.csv")
    parser.add_argument("--plot", type=Path, default=Path("track_preview.png"))
    args = parser.parse_args()
    track = Track(args.csv)
    plot_track(track, args.plot)
    print(f"Track length: {track.length:.2f} m")
    print(f"Start center: {track.center(0)}")
    print(f"Start heading: {track.heading(0):.3f} rad")
    print(f"Start widths (L/R): {float(track._wl(0)):.2f}/{float(track._wr(0)):.2f} m")
    print(f"Saved plot: {args.plot}")
