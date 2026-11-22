"""Drive laps with a simple steer-toward-the-centerline controller, shown in a
basic Pygame top-down view.

Uses only the agreed interface (env.step, env.to_action, info, track.point_at)
plus env.car_params for the wheelbase.

    python sim/examples/drive_lap.py                    # 1 lap, training track, Pygame view
    python sim/examples/drive_lap.py --laps 3 --track hidden
    python sim/examples/drive_lap.py --no-view --plot   # headless, save a matplotlib plot
    python sim/examples/drive_lap.py --video lap.mp4    # record the lap as a video
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sim import RacingEnv  # noqa: E402


def controller(env, info, a_lat=6.0, v_cap=7.0, lookahead_time=0.35):
    """Steer toward a centerline point ahead; slow down for curves ahead.

    Demo only - Nathaly's Pure Pursuit is the real baseline controller.
    """
    p = env.car_params
    track = env.track
    # speed: slow enough for the tightest curve in the next 4 m (v^2 * kappa <= a_lat)
    k_ahead = np.max(np.abs(track.lookahead(info["s"], np.linspace(0.0, 4.0, 17))))
    v_target = min(v_cap, math.sqrt(a_lat / max(k_ahead, 1e-3)))
    accel = 3.0 * (v_target - info["vx"])

    # steering: pure pursuit toward the centerline point Ld metres ahead
    ld = max(0.6, lookahead_time * info["vx"])
    x_c, y_c = track.point_at(info["s"] + ld)
    dx, dy = x_c - info["x"], y_c - info["y"]
    c, s = math.cos(info["psi"]), math.sin(info["psi"])
    px, py = c * dx + s * dy, -s * dx + c * dy          # target in the car frame
    steer = math.atan2(2.0 * p["wheelbase"] * py, px * px + py * py)
    # The car understeers (the tires slip), so it needs more steering than geometry says,
    # more so at higher speed: delta = L/R + K*v^2/R, i.e. (1 + K*v^2/L) times the geometric angle.
    steer *= 1.0 + p["understeer_gradient"] * info["vx"] ** 2 / p["wheelbase"]
    return env.to_action(accel, steer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="training")
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument("--no-view", action="store_true", help="no Pygame window (runs as fast as possible)")
    ap.add_argument("--plot", action="store_true", help="save a matplotlib plot of the driven line")
    ap.add_argument("--log", default=None, help="save the per-step log to this CSV")
    ap.add_argument("--video", default=None, help="record the run to this MP4 file (no window)")
    args = ap.parse_args()

    if args.video:
        from sim.video import record_episode
        env = RacingEnv(track=args.track, max_laps=args.laps, time_limit=120.0, render_mode="rgb_array")
        info = record_episode(env, lambda obs, info: controller(env, info), args.video, seed=0)
        print(f"saved {args.video}: {info['lap']} lap(s) in {info['t']:.2f} s, "
              f"off-track events {info['off_track_count']}")
        return

    env = RacingEnv(car="numpy", track=args.track, off_track="grass",
                    max_laps=args.laps, time_limit=120.0)
    view = None
    if not args.no_view:
        from sim.viewer import TopDownView
        p = env.car_params
        view = TopDownView(env.track, car_length=p["length"], car_width=p["width"])

    obs, info = env.reset(seed=0)
    terminated = truncated = False
    t_wall = time.perf_counter()
    while not (terminated or truncated):
        action = controller(env, info)
        obs, reward, terminated, truncated, info = env.step(action)
        if info["lap_completed"]:
            lap = env.lap_log[-1]
            print(f"lap {lap['lap']}: {lap['lap_time']:.3f} s, off-track events {lap['off_track_count']}")
        if view is not None:
            text = (f"t {info['t']:5.1f} s   v {info['vx']:4.1f} m/s   lap {info['lap']}   "
                    f"off-track {info['off_track_count']}")
            if not view.draw([{"x": info["x"], "y": info["y"], "psi": info["psi"]}], text):
                break
            # real-time pacing (game mode): wait until the wall clock catches up
            time.sleep(max(0.0, t_wall + info["t"] - time.perf_counter()))
    if view is not None:
        view.close()

    log = env.get_log()
    print(f"track {env.track.name} ({env.track.length:.2f} m): {env.lap} lap(s) in {info['t']:.2f} s, "
          f"max speed {log['vx'].max():.2f} m/s, max |e_c| {np.abs(log['e_c']).max():.3f} m, "
          f"off-track events {info['off_track_count']}")
    if truncated:
        print("time limit reached")
    if args.log:
        env.save_log(args.log)
        print("log saved to", args.log)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tab = env.track.centerline_table(s_min=0.0, s_max=env.track.length)
        nx, ny = -np.sin(tab["theta"]), np.cos(tab["theta"])
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(tab["x"], tab["y"], "k--", lw=0.6)
        ax.plot(tab["x"] + nx * tab["w_left"], tab["y"] + ny * tab["w_left"], "k", lw=1)
        ax.plot(tab["x"] - nx * tab["w_right"], tab["y"] - ny * tab["w_right"], "k", lw=1)
        sc = ax.scatter(log["x"], log["y"], c=log["vx"], s=3, cmap="viridis")
        fig.colorbar(sc, label="speed (m/s)")
        ax.set_aspect("equal")
        ax.set_title(f"{env.track.name}: lap time {env.lap_times[0]:.2f} s" if env.lap_times else env.track.name)
        out = f"drive_lap_{env.track.name}.png"
        fig.savefig(out, dpi=120)
        print("plot saved to", out)


if __name__ == "__main__":
    main()
