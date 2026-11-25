"""Quick checks for Pure Pursuit and the eval pipeline (a few seconds).

    python -m controllers.pure_pursuit.test_pure_pursuit
"""
import math

import numpy as np

from sim import RacingEnv
from controllers.pure_pursuit.pure_pursuit import PurePursuit
from eval.metrics import run_episode, compute_metrics


def test_speed_profile_respects_limits():
    env = RacingEnv(track="training")
    pp = PurePursuit(env)
    car = env.car_params
    assert pp.prof_v.max() <= car["v_max"] + 1e-9
    # never faster than the cornering limit
    assert np.all(pp.prof_v <= pp.prof_v_corner + 1e-9)
    # braking between neighbours never exceeds the planned deceleration
    v2 = pp.prof_v ** 2
    decel = (v2 - np.roll(v2, -1)) / (2 * pp._ds)
    assert decel.max() <= pp.p.brake_margin * car["b_max"] + 1e-6


def test_steers_toward_goal():
    env = RacingEnv(track="training")
    pp = PurePursuit(env)
    _, info = env.reset(options={"e_c0": 0.4})          # start 0.4 m left of the centerline
    action = pp(info)
    assert action[1] < 0, "car left of the line on a straight should steer right"


def test_clean_laps_both_tracks():
    for track, max_time in (("training", 26.0), ("hidden", 21.0)):
        env = RacingEnv(track=track, max_laps=2, time_limit=80.0)
        pp = PurePursuit(env)
        r = run_episode(env, lambda obs, info: pp(info), name="Pure Pursuit")
        m = r["metrics"]
        assert m["laps_completed"] == 2, (track, m)
        assert m["off_track_events"] == 0, (track, m)
        assert m["best_lap"] < max_time, (track, m["best_lap"])


def test_metrics_counts_events():
    log = {"t": np.arange(6) * 0.05, "off_track": np.array([0, 1, 1, 0, 1, 0], bool),
           "steer": np.zeros(6), "throttle": np.zeros(6), "e_c": np.zeros(6), "vx": np.ones(6)}
    m = compute_metrics(log, [], laps_asked=1)
    assert m["off_track_events"] == 2
    assert math.isclose(m["off_track_time_frac"], 0.5)
    assert m["completion"] == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
