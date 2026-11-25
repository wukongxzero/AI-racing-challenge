"""Drive Pure Pursuit laps, optionally in the Pygame view or as a video.

From the repository root:
    python -m controllers.pure_pursuit.run                       # 2 laps, training track, live window
    python -m controllers.pure_pursuit.run --track hidden --laps 3
    python -m controllers.pure_pursuit.run --no-view             # headless, just lap times
    python -m controllers.pure_pursuit.run --video pp_lap.mp4    # record a video (no window)
    python -m controllers.pure_pursuit.run --k-v 0.2 --lat-margin 0.85   # try other settings

For the full metrics table and plots use:  python -m eval.evaluate
"""
from __future__ import annotations

import argparse
import time

from sim import RacingEnv
from controllers.pure_pursuit.pure_pursuit import PurePursuit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="training")
    ap.add_argument("--laps", type=int, default=2)
    ap.add_argument("--mu-scale", type=float, default=1.0)
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--video", default=None)
    ap.add_argument("--ld-min", type=float, default=None)
    ap.add_argument("--k-v", type=float, default=None)
    ap.add_argument("--lat-margin", type=float, default=None)
    args = ap.parse_args()
    overrides = {k: v for k, v in (("ld_min", args.ld_min), ("k_v", args.k_v),
                                   ("lat_margin", args.lat_margin)) if v is not None}

    if args.video:
        from sim.video import record_episode
        env = RacingEnv(track=args.track, max_laps=args.laps, time_limit=120.0,
                        mu_scale=args.mu_scale, render_mode="rgb_array")
        pp = PurePursuit(env, **overrides)
        info = record_episode(env, lambda obs, info: pp(info), args.video, seed=0)
        print(f"saved {args.video}: {info['lap']} lap(s), lap times {[round(t, 2) for t in info['lap_times']]}")
        return

    env = RacingEnv(track=args.track, off_track="grass", max_laps=args.laps, time_limit=120.0,
                    mu_scale=args.mu_scale)
    pp = PurePursuit(env, **overrides)
    view = None
    if not args.no_view:
        from sim.viewer import TopDownView
        p = env.car_params
        view = TopDownView(env.track, car_length=p["length"], car_width=p["width"], caption="Pure Pursuit")

    obs, info = env.reset(seed=0)
    done = False
    t_wall = time.perf_counter()
    while not done:
        obs, reward, terminated, truncated, info = env.step(pp(info))
        done = terminated or truncated
        if info["lap_completed"]:
            lap = env.lap_log[-1]
            print(f"lap {lap['lap']}: {lap['lap_time']:.3f} s, off-track events {lap['off_track_count']}")
        if view is not None:
            text = (f"Pure Pursuit   t {info['t']:5.1f} s   v {info['vx']:4.1f} / {pp.last['v_target']:4.1f} m/s   "
                    f"Ld {pp.last['ld']:.2f} m   lap {info['lap']}   off-track {info['off_track_count']}")
            if not view.draw([{"x": info["x"], "y": info["y"], "psi": info["psi"]}], text):
                break
            time.sleep(max(0.0, t_wall + info["t"] - time.perf_counter()))   # real time
    if view is not None:
        view.close()
    print(f"done: {env.lap} lap(s) in {info['t']:.2f} s, off-track events {info['off_track_count']}")


if __name__ == "__main__":
    main()
