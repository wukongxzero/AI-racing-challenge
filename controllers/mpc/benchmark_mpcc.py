"""Repeatable head-to-head MPCC benchmark on one machine.

Example:
  python3 -m controllers.mpc.benchmark_mpcc --speeds 3 4 5 --steps 950 --output benchmark.csv

Runs use identical track, seed=0, simulation step 0.05s and the same
shared NumPy RacingEnv. Both controllers retain their current horizons
(kinematic 12, dynamic 8), so this evaluates each *configured controller*,
not a matched-horizon optimization experiment.
"""
import argparse
import csv
import json
from pathlib import Path

from sim import RacingEnv
from .run_racing_env import drive as drive_kinematic, controller_from_env
from .run_dynamic_mpcc import drive_dynamic
from .dynamic_mpcc import DynamicMPCC, DynamicConfig

FIELDS = ['controller', 'speed_cap_mps', 'track', 'seed', 'steps', 'lap',
          'lap_time_s', 'progress_m', 'track_length_m', 'off_track_count',
          'solver_recoveries', 'solver_fallbacks', 'over_50ms',
          'mean_solve_ms', 'max_solve_ms', 'dynamic_steps', 'launch_steps']


def run_case(mode, speed, track='training', steps=950, seed=0):
    if mode not in ('kinematic', 'dynamic'):
        raise ValueError('mode must be kinematic or dynamic')
    if speed <= 0 or steps <= 0:
        raise ValueError('speed and steps must be positive')
    env = RacingEnv(car='numpy', track=track, off_track='grass',
                    max_laps=1, time_limit=steps*0.05, log=False)
    try:
        if mode == 'kinematic':
            controller = controller_from_env(env, speed_cap=speed)
            result = drive_kinematic(env, controller, steps=steps,
                                     seed=seed, verbose=False)
        else:
            if seed != 0:
                raise ValueError('Dynamic runner currently uses seed=0; use --seed 0')
            controller = DynamicMPCC(env, DynamicConfig(speed=speed))
            result = drive_dynamic(env, controller, steps=steps, verbose=False)
        return dict(controller=mode, speed_cap_mps=speed, track=track,
                    seed=seed, steps=result['steps'], lap=result['lap'],
                    lap_time_s=round(result['time'], 3),
                    progress_m=round(result['progress'], 3),
                    track_length_m=round(env.track.length, 3),
                    off_track_count=result['off_track_count'],
                    solver_recoveries=result['solver_recoveries'],
                    solver_fallbacks=result['solver_fallbacks'],
                    over_50ms=result['over_50ms'],
                    mean_solve_ms=round(result['mean_solve_ms'], 3),
                    max_solve_ms=round(result['max_solve_ms'], 3),
                    dynamic_steps=result.get('dynamic_steps', 0),
                    launch_steps=result.get('launch_steps', 0))
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--speeds', nargs='+', type=float, default=[3, 4, 5])
    ap.add_argument('--track', choices=['training', 'hidden'], default='training')
    ap.add_argument('--steps', type=int, default=950)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output', type=Path, default=Path('mpcc_benchmark.csv'))
    args = ap.parse_args()
    if args.seed != 0:
        ap.error('Dynamic runner supports seed=0 only')
    rows = []
    for speed in args.speeds:
        for mode in ('kinematic', 'dynamic'):
            print(f'Running {mode} at {speed:g} m/s on {args.track}...', flush=True)
            row = run_case(mode, speed, args.track, args.steps, args.seed)
            rows.append(row)
            print(json.dumps(row), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
    print(f'Saved {len(rows)} cases to {args.output}')


if __name__ == '__main__':
    main()
