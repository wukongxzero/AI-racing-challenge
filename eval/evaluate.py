"""Run drivers on the tracks, save logs + metrics, print the lap-time table, make plots.

From the repository root:
    python -m eval.evaluate                                  # all drivers, both tracks, 3 laps
    python -m eval.evaluate --drivers mpcc --tracks training --laps 2   # just MPCC
    python -m eval.evaluate --drivers mpcc --mpcc-speed 6    # MPCC with another speed cap
    python -m eval.evaluate --drivers pure_pursuit --tracks training --laps 5
    python -m eval.evaluate --mu-scale 0.7                   # robustness: 30 % less grip, controllers told
    python -m eval.evaluate --mu-scale 0.7 --controllers-assume-mu 1.0   # ... controllers NOT told
    python -m eval.evaluate --plots-only                     # re-plot from saved results/

Outputs go to results/ (git-ignored; --mu-scale runs go to results/mu<value>/,
uninformed runs to results/mu<value>/assume_mu<value>/): <driver>_<track>.csv / .json, table.md, *.png.

Adding a driver = add a function to DRIVERS that takes the env and returns
policy(obs, info) -> [throttle, steering] (adapters live in eval/drivers.py). The PPO slot is marked below.
"""
from __future__ import annotations

import argparse
import glob
import os

from sim import RacingEnv
from sim.track import load_track
from eval.metrics import run_episode, save_result, load_result, format_table, slug
from eval import plots


def make_pure_pursuit(env, args):
    from eval.drivers import PurePursuitDriver
    return PurePursuitDriver(env)


def make_mpcc(env, args):
    from eval.drivers import MPCCDriver
    return MPCCDriver(env, speed=args.mpcc_speed, horizon=args.mpcc_horizon, substeps=args.mpcc_substeps)


# TODO (Pavan): def make_ppo(env, args): model = PPO.load(...); return lambda obs, info: model.predict(obs, deterministic=True)[0]
DRIVERS = {"pure_pursuit": ("Pure Pursuit", make_pure_pursuit),
           "mpcc": ("MPCC", make_mpcc)}


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
    ap.add_argument("--controllers-assume-mu", type=float, default=None,
                    help="uninformed robustness test: controllers plan with this grip (e.g. 1.0) "
                         "while the road has --mu-scale. Default: controllers are told the real grip.")
    ap.add_argument("--mpcc-speed", type=float, default=8.0, help="MPCC speed cap (m/s)")
    ap.add_argument("--mpcc-horizon", type=int, default=6)
    ap.add_argument("--mpcc-substeps", type=int, default=2)
    args = ap.parse_args()
    out = args.out if args.mu_scale == 1.0 else os.path.join(args.out, f"mu{args.mu_scale:g}")
    if args.controllers_assume_mu is not None:
        out = os.path.join(out, f"assume_mu{args.controllers_assume_mu:g}")

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
                print(f"running {name} on {track} ...", flush=True)
                plan_env = env
                if args.controllers_assume_mu is not None:
                    from eval.drivers import AssumedGrip
                    plan_env = AssumedGrip(env, args.controllers_assume_mu, car=args.car)
                policy = make(plan_env, args)
                r = run_episode(env, policy, name=name, seed=0)
                r["controller"] = getattr(policy, "controller", None)
                r["metrics"].update(getattr(policy, "stats", {}))
                r["metrics"].update({f"setting_{k}": v for k, v in getattr(policy, "settings", {}).items()})
                save_result(r, out)
                laps = ", ".join(f"{t:.2f}" for t in r["metrics"]["lap_times"])
                extra = "".join(f", {k} {v}" for k, v in getattr(policy, "stats", {}).items())
                print(f"{name:>14} on {track:<8}: laps [{laps}] s, off-track {r['metrics']['off_track_events']}{extra}")
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
