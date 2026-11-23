"""Run the existing kinematic MPCC against the team's shared NumPy RacingEnv.

From repository root:
    python -m controllers.mpc.run_racing_env --steps 20
    python -m controllers.mpc.run_racing_env --steps 200 --plot mpcc_env.png

This is a first integration, NOT a dynamic-model MPCC: the predictor is kinematic,
while RacingEnv simulates lateral tire forces and yaw-rate dynamics.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# mpcc.py originally imports its adjacent prototype modules as top-level modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim import RacingEnv  # noqa: E402
from bicycle_model import CarParams  # noqa: E402
from mpcc import MPCC, MPCCParams  # noqa: E402


class SharedTrackAdapter:
    """Only adapt vector/scalar method shapes; use the SAME shared track geometry.

    MPCC's old prototype expects vectorized center/tangent and width callables.
    The team API supplies point_at, heading, curvature and width at scalar s.
    """

    def __init__(self, track):
        self.track = track
        self.length = track.length

    def _map(self, fn, s):
        arr = np.asarray(s, dtype=float)
        values = np.asarray([fn(float(v)) for v in arr.reshape(-1)])
        return values.reshape(arr.shape + values.shape[1:])

    def center(self, s):
        return self._map(self.track.point_at, s)

    def heading(self, s):
        return self._map(self.track.heading, s)

    def tangent(self, s):
        ang = self.heading(s)
        return np.stack((np.cos(ang), np.sin(ang)), axis=-1)

    def curvature(self, s):
        return self._map(self.track.curvature, s)

    def _wl(self, s):
        return self._map(lambda v: self.track.width(v)[0], s)

    def _wr(self, s):
        return self._map(lambda v: self.track.width(v)[1], s)

    def project(self, x, y):
        return self.track.project(x, y, 0.0)[0]

    def errors(self, x, y, s):
        cx, cy = self.track.point_at(s)
        phi = self.track.heading(s)
        dx, dy = x - cx, y - cy
        return float(-np.sin(phi) * dx + np.cos(phi) * dy), float(np.cos(phi) * dx + np.sin(phi) * dy)


def controller_from_env(env: RacingEnv, *, horizon=12, speed_cap=3.0):
    """Match the shared car's real steering, acceleration and dimension limits."""
    p = env.car_params
    car = CarParams(
        wheelbase=p["wheelbase"], dt=env.dt,
        max_steer=p["delta_max"], max_steer_rate=p["delta_rate_max"],
        max_accel=p["a_max"], min_accel=-p["b_max"],
        max_speed=min(speed_cap, p["v_max"]),
    )
    cfg = MPCCParams(
        horizon=horizon,
        safety_margin=p["width"] / 2.0 + 0.03,
        lateral_accel_limit=min(0.75 * p["a_lat_max"], 5.0),
        speed_floor=0.5,
        speed_buffer=0.5,
        speed_accel_limit=min(0.75 * p["a_max"], 3.0),
    )
    return MPCC(SharedTrackAdapter(env.track), car, cfg)


def safe_stop(env, info):
    """Brake without a stale MPCC steering command on solver failure."""
    return env.to_action(-env.car_params["b_max"], float(info["delta"]))


def drive(env, controller, *, steps=400, seed=0, verbose=True, retry=True):
    _, info = env.reset(seed=seed)
    fallbacks = 0
    recovered = 0
    failure_events = []
    solver_ms = []
    history = [(info["x"], info["y"])]
    result = None
    for i in range(steps):
        # Old MPCC state uses a single longitudinal speed; RacingEnv also exposes
        # vy/r, intentionally not used by this first kinematic controller.
        state = np.array([info["x"], info["y"], info["psi"], max(0., info["vx"])])
        t0 = time.perf_counter()
        try:
            control, _, _ = controller.solve(state, float(info["progress"]), float(info["delta"]))
            action = env.to_action(float(control[0]), float(control[1]))
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            original_status = controller.opti.stats().get("return_status", "unknown")
            # Occasionally the shifted optimization guess is poor. Retry once
            # from a track-aligned trajectory, anchored to measured progress.
            controller.guess_x = controller.guess_u = None
            try:
                if not retry:
                    raise exc
                control, _, _ = controller.solve(state, float(info["progress"]), float(info["delta"]))
                action = env.to_action(float(control[0]), float(control[1]))
                recovered += 1
                if verbose:
                    print(f"MPCC recovered step {i}: first status={original_status}")
            except (RuntimeError, ValueError, FloatingPointError):
                fallbacks += 1
                second_status = controller.opti.stats().get("return_status", "unknown") if retry else "not_retried"
                failure_events.append((i, original_status, second_status))
                action = safe_stop(env, info)
                if verbose:
                    print(f"MPCC fallback step {i}: first={original_status} retry={second_status}")
        solver_ms.append((time.perf_counter() - t0) * 1000.)
        _, _, terminated, truncated, info = env.step(action)
        history.append((info["x"], info["y"]))
        if verbose and (i % 20 == 0 or info["lap_completed"]):
            print(f"step={i:4d} t={info['t']:6.2f}s vx={info['vx']:5.2f} "
                  f"e_c={info['e_c']:+6.3f}m progress={info['progress']:6.1f}m "
                  f"solve={solver_ms[-1]:7.1f}ms off_track={info['off_track']} fallbacks={fallbacks}")
        if terminated or truncated:
            break
    result = {"steps": len(solver_ms), "time": info["t"], "lap": info["lap"],
              "progress": info["progress"], "off_track_count": info["off_track_count"],
              "solver_fallbacks": fallbacks, "solver_recoveries": recovered,
              "failure_events": failure_events, "over_50ms": int(np.sum(np.asarray(solver_ms) > 50.0)),
              "mean_solve_ms": float(np.mean(solver_ms)),
              "max_solve_ms": float(np.max(solver_ms)), "history": np.array(history),
              "info": info}
    return result


def save_plot(env, result, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ss = np.linspace(0, env.track.length, 600)
    c = np.array([env.track.point_at(float(s)) for s in ss])
    w = np.array([env.track.width(float(s)) for s in ss])
    theta = np.array([env.track.heading(float(s)) for s in ss])
    normal = np.column_stack((-np.sin(theta), np.cos(theta)))
    left, right = c + w[:, :1] * normal, c - w[:, 1:] * normal
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(*left.T, '--', alpha=0.5, label='left edge')
    ax.plot(*right.T, '--', alpha=0.5, label='right edge')
    ax.plot(*c.T, alpha=0.5, label='centerline')
    ax.plot(*result["history"].T, label='MPCC + RacingEnv')
    ax.set_aspect('equal'); ax.grid(alpha=0.25); ax.legend()
    ax.set_title(f"{env.track.name} | {result['time']:.1f}s | laps {result['lap']} | off-track {result['off_track_count']}")
    fig.tight_layout(); fig.savefig(filename, dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--track', choices=['training', 'hidden'], default='training')
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--horizon', type=int, default=12)
    p.add_argument('--speed', type=float, default=3.0)
    p.add_argument('--plot', type=Path)
    p.add_argument('--no-retry', action='store_true', help='Disable one cold-start solver retry')
    p.add_argument('--log', type=Path)
    p.add_argument('--off-track', choices=['grass', 'terminate'], default='grass')
    args = p.parse_args()
    env = RacingEnv(car='numpy', track=args.track, off_track=args.off_track, max_laps=1,
                    time_limit=args.steps * 0.05, log=True)
    try:
        controller = controller_from_env(env, horizon=args.horizon, speed_cap=args.speed)
        result = drive(env, controller, steps=args.steps, retry=not args.no_retry)
        print(f"RESULT steps={result['steps']} laps={result['lap']} "
              f"progress={result['progress']:.2f}/{env.track.length:.2f}m "
              f"off_track={result['off_track_count']} fallbacks={result['solver_fallbacks']} "
              f"recoveries={result['solver_recoveries']} over_50ms={result['over_50ms']} "
              f"mean_solver={result['mean_solve_ms']:.1f}ms max_solver={result['max_solve_ms']:.1f}ms")
        if args.plot:
            save_plot(env, result, args.plot)
            print(f"saved plot: {args.plot}")
        if args.log:
            env.save_log(str(args.log))
            print(f"saved CSV: {args.log}")
    finally:
        env.close()


if __name__ == '__main__':
    main()
