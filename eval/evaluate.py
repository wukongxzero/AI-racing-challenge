"""Run drivers on the tracks, save logs + metrics, print the lap-time table, make plots.

From the repository root:
    python -m eval.evaluate                                  # Pure Pursuit, both tracks, 3 laps
    python -m eval.evaluate --drivers pure_pursuit --tracks training --laps 5
    python -m eval.evaluate --mu-scale 0.7                   # robustness: 30 % less grip
    python -m eval.evaluate --plots-only                     # re-plot from saved results/

Outputs go to results/ (git-ignored): <driver>_<track>.csv / .json, table.md, *.png.

Adding a driver = add a function to DRIVERS that takes the env and returns
policy(obs, info) -> [throttle, steering]. MPCC and PPO slots are marked below.
"""
from __future__ import annotations

import argparse
import glob
import os

from sim import RacingEnv
from sim.track import load_track
from eval.metrics import run_episode, save_result, load_result, format_table, slug
from eval import plots


def make_pure_pursuit(env):
    from controllers.pure_pursuit.pure_pursuit import PurePursuit
    pp = PurePursuit(env)
    policy = lambda obs, info: pp(info)  # noqa: E731
    policy.controller = pp
    return policy


def make_mpcc(env, preset="RACE"):
    from controllers.mpc.driver import MPCCDriver
    from controllers.mpc import driver as d
    from controllers.mpc.dynamic_mpcc import DynamicConfig
    return MPCCDriver(env, DynamicConfig(**getattr(d, preset)))   # callable(obs, info), has reset()


# TODO (Pavan):   def make_ppo(env): model = PPO.load(...); return lambda obs, info: model.predict(obs, deterministic=True)[0]
DRIVERS = {"pure_pursuit": ("Pure Pursuit", make_pure_pursuit),
           "mpcc": ("MPCC", make_mpcc),
           "mpcc_robust": ("MPCC robust", lambda env: make_mpcc(env, "ROBUST"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drivers", nargs="+", default=list(DRIVERS), choices=list(DRIVERS))
    ap.add_argument("--tracks", nargs="+", default=["training", "hidden"])
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=180.0)
    ap.add_argument("--mu-scale", type=float, default=1.0)
    ap.add_argument("--car", default="numpy")
    ap.add_argument("--out", default="results")
    ap.add_argument("--plots-only", action="store_true")
    args = ap.parse_args()
    tag = "" if args.mu_scale == 1.0 else f"_mu{args.mu_scale:g}"
    out = args.out + tag

    results = []
    if args.plots_only:
        for f in sorted(glob.glob(os.path.join(out, "*.json"))):
            results.append(load_result(f[:-5]))
    else:
        for track in args.tracks:
            for key in args.drivers:
                name, make = DRIVERS[key]
                env = RacingEnv(car=args.car, track=track, off_track="grass", max_laps=args.laps,
                                time_limit=args.time_limit, mu_scale=args.mu_scale)
                policy = make(env)
                r = run_episode(env, policy, name=name, seed=0)
                r["controller"] = getattr(policy, "controller", None)
                save_result(r, out)
                laps = ", ".join(f"{t:.2f}" for t in r["metrics"]["lap_times"])
                print(f"{name:>14} on {track:<8}: laps [{laps}] s, off-track {r['metrics']['off_track_events']}")
                results.append(r)

    table = format_table(results)
    print("\n" + table)
    with open(os.path.join(out, "table.md"), "w") as f:
        f.write(table + "\n")

    for track in sorted({r["track"] for r in results}):
        tr = load_track(track)
        rs = [r for r in results if r["track"] == track]
        plots.plot_racing_lines(tr, rs, os.path.join(out, f"lines_{track}.png"))
        profiles = {}
        for r in rs:
            c = r.get("controller")
            if c is not None and hasattr(c, "prof_v"):
                profiles[f"{r['name']} target"] = (c.prof_s, c.prof_v)
        plots.plot_speed_vs_s(rs, os.path.join(out, f"speed_{track}.png"), track=tr, profiles=profiles)
        for r in rs:
            plots.plot_speed_map(tr, r, os.path.join(out, f"speedmap_{slug(r['name'])}_{track}.png"))
    plots.plot_lap_times(results, os.path.join(out, "lap_times.png"))
    print(f"\nsaved logs, table.md and plots to {out}/")


if __name__ == "__main__":
    main()
