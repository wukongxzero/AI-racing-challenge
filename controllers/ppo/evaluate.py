"""Evaluate a trained PPO policy under race rules.

Race rules = the same as every controller: start line at rest, grass (not episode end)
when off track, deterministic policy, frozen normalisation.

From the repo root:
    python -m controllers.ppo.evaluate runs/ppo/my_run                    # 3 laps, training track
    python -m controllers.ppo.evaluate runs/ppo/my_run --track hidden     # generalisation
    python -m controllers.ppo.evaluate runs/ppo/my_run --mu-scale 0.7     # 30 % less grip
    python -m controllers.ppo.evaluate runs/ppo/my_run --video lap.mp4 --log run.csv
    python -m controllers.ppo.evaluate runs/ppo/my_run --episodes 20 --random-start
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from controllers.ppo.driver import PPODriver
from sim import RacingEnv


def run_episode(env, driver, options=None, seed=None, frames=None):
    obs, info = env.reset(seed=seed, options=options)
    actions = []
    done = False
    while not done:
        a = driver.act(info)
        actions.append(a)
        obs, _, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        if frames is not None:
            frames.append(env.render())
    du = np.linalg.norm(np.diff(np.asarray(actions), axis=0), axis=1) if len(actions) > 1 else [0.0]
    return {
        "lap_times": [lap["lap_time"] for lap in env.lap_log],
        "clean": [lap["clean"] for lap in env.lap_log],
        "off_track_count": info["off_track_count"],
        "laps": info["lap"],
        "time": info["t"],
        "mean_du": float(np.mean(du)),    # control smoothness: mean |u_n - u_{n-1}|
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="run folder (uses best/ then final/) or a model .zip")
    ap.add_argument("--track", default="training")
    ap.add_argument("--car", default="numpy")
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--mu-scale", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=1,
                    help="from the start line the run is deterministic, so >1 needs --random-start")
    ap.add_argument("--random-start", action="store_true",
                    help="small random offset/heading at the start line (robustness)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", default=None, help="save the first episode as MP4")
    ap.add_argument("--log", default=None, help="save the first episode's per-step log as CSV")
    ap.add_argument("--json", default=None, help="save the summary as JSON")
    args = ap.parse_args()

    driver = PPODriver(args.model)
    env = RacingEnv(car=args.car, track=args.track, off_track="grass", max_laps=args.laps,
                    time_limit=args.time_limit, mu_scale=args.mu_scale,
                    render_mode="rgb_array" if args.video else None)
    print(f"model: {driver.model_path}   track: {args.track}   mu_scale: {args.mu_scale}")

    options = {"s0": 0.0, "v0": 0.0}
    if args.random_start:
        options.update({"e_c0": (-0.2, 0.2), "e_psi0": (-0.1, 0.1)})
    results = []
    for ep in range(args.episodes):
        frames = [] if (args.video and ep == 0) else None
        r = run_episode(env, driver, options, seed=args.seed + ep, frames=frames)
        results.append(r)
        laps = " ".join(f"{t:.2f}{'' if c else '*'}" for t, c in zip(r["lap_times"], r["clean"]))
        print(f"episode {ep}: laps {r['laps']}/{args.laps}  times [{laps}]  "
              f"off-track {r['off_track_count']}  mean|du| {r['mean_du']:.3f}")
        if ep == 0:
            if frames:
                from sim.video import save_video
                save_video(frames, args.video)
                print(f"video: {args.video}")
            if args.log:
                env.save_log(args.log)
                print(f"log: {args.log}")

    all_laps = [t for r in results for t in r["lap_times"]]
    clean = [c for r in results for c in r["clean"]]
    summary = {
        "model": driver.model_path, "track": args.track, "mu_scale": args.mu_scale,
        "episodes": args.episodes, "laps_requested": args.laps * args.episodes,
        "laps_completed": len(all_laps),
        "completion_rate": len(all_laps) / (args.laps * args.episodes),
        "clean_lap_rate": float(np.mean(clean)) if clean else 0.0,
        "mean_lap_time": float(np.mean(all_laps)) if all_laps else None,
        "best_lap_time": float(np.min(all_laps)) if all_laps else None,
        "mean_du": float(np.mean([r["mean_du"] for r in results])),
        "off_track_events": int(sum(r["off_track_count"] for r in results)),
    }
    print("(* = lap with an off-track event)")
    print(json.dumps(summary, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
