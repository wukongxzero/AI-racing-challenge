"""Quick checks for the PPO wrapper (no training).  python -m controllers.ppo.test_ppo"""
import numpy as np

from controllers.ppo.wrapper import PPO_OBS_NAMES, RACE_START, RewardConfig, make_env, obs_from_info


def test_obs_shape_and_order():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (15,) and len(PPO_OBS_NAMES) == 15
    assert env.observation_space.contains(obs)
    assert np.allclose(obs, obs_from_info(info))
    assert obs[PPO_OBS_NAMES.index("w_half")] > 0


def test_random_starts_cover_the_lap():
    env = make_env()
    s0 = [env.reset(seed=i)[1]["s"] for i in range(30)]
    assert max(s0) - min(s0) > 0.5 * env.unwrapped.track.length


def test_off_track_terminates_with_penalty():
    env = make_env(start=dict(RACE_START, s0=0.0))
    env.reset(seed=0)
    for _ in range(400):    # full gas, full left: leaves the road
        _, r, term, trunc, info = env.step([1.0, 1.0])
        if term:
            break
    assert term and info["end_reason"] == "off_track" and r < -5


def test_stall_terminates():
    env = make_env(start=dict(RACE_START, s0=0.0))
    env.reset(seed=0)
    for step in range(100):
        _, _, term, _, info = env.step([0.0, 0.0])   # never moves
        if term:
            break
    assert term and info["end_reason"] == "stall" and step + 1 == 40   # 40 steps = 2 s


def test_progress_reward_matches_ds():
    env = make_env(reward="progress", start=dict(RACE_START, s0=0.0))
    env.reset(seed=0)
    for _ in range(10):
        _, r, _, _, info = env.step([0.5, 0.0])
        assert abs(r - info["ds"]) < 1e-9


def test_smoothness_penalty():
    cfg = RewardConfig(w_prog=0, w_c=0, w_psi=0, w_time=0, w_du=1.0)
    env = make_env(reward=cfg, start=dict(RACE_START, s0=0.0, v0=2.0))
    env.reset(seed=0)
    _, r, *_ = env.step([0.0, 1.0])      # jump of 1 in steering from the zero start
    assert abs(r + 1.0) < 1e-6


def test_race_mode_keeps_going_on_grass():
    env = make_env(train=False, time_limit=20.0)
    env.reset(seed=0)
    for _ in range(200):
        _, _, term, trunc, info = env.step([1.0, 1.0])
        assert not term
    assert info["off_track_count"] >= 1


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
