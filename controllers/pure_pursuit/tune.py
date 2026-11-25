"""Grid search for Pure Pursuit's parameters on the TRAINING track only.

The hidden track is never used for tuning (same rule as PPO's training), so the
hidden-track result stays a fair generalization test.

    python -m controllers.pure_pursuit.tune              # prints the best settings + saves results/pp_tuning.csv

Score: best lap time over 3 laps; any off-track event disqualifies a setting.
"""
from __future__ import annotations

import csv
import itertools
import os

from sim import RacingEnv
from controllers.pure_pursuit.pure_pursuit import PurePursuit
from eval.metrics import run_episode

GRID = {
    "ld_min": [0.4, 0.6, 0.8],
    "k_v": [0.15, 0.2, 0.25, 0.3, 0.35],
    "lat_margin": [0.8, 0.85, 0.9, 0.95],
}


def main(out="results"):
    os.makedirs(out, exist_ok=True)
    rows = []
    keys = list(GRID)
    for values in itertools.product(*GRID.values()):
        params = dict(zip(keys, values))
        env = RacingEnv(track="training", max_laps=3, time_limit=120.0, log=True)
        pp = PurePursuit(env, **params)
        m = run_episode(env, lambda obs, info: pp(info), name="pp")["metrics"]
        rows.append({**params, "best_lap": m["best_lap"], "laps": m["laps_completed"],
                     "off_track_events": m["off_track_events"], "max_abs_e_c": m["max_abs_e_c"],
                     "steer_rate_rms": m["steer_rate_rms"]})
        print(rows[-1])
    with open(os.path.join(out, "pp_tuning.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["off_track_events"] == 0 and r["laps"] == 3]
    ok.sort(key=lambda r: r["best_lap"])
    print("\nbest clean settings:")
    for r in ok[:5]:
        print(r)


if __name__ == "__main__":
    main()
