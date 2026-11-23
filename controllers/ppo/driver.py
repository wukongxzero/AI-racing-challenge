"""Use a trained PPO policy as a driver (game, evaluation, races).

    from controllers.ppo.driver import PPODriver
    driver = PPODriver("runs/ppo/my_run")             # best/ if present, else final/
    obs, info = env.reset()                            # any RacingEnv
    action = driver.act(info)                          # [throttle, steering] in [-1, 1]
    obs, reward, terminated, truncated, info = env.step(action)

It only reads the `info` dict, so it works with the plain RacingEnv the game uses.
Observations are normalised with the frozen VecNormalize statistics saved with the
policy, and the policy is deterministic (plan, step 6).
"""
from __future__ import annotations

import os
import pickle

import numpy as np
from stable_baselines3 import PPO

from controllers.ppo.wrapper import obs_from_info


def find_model(path):
    """(model.zip, vecnormalize.pkl) for a run folder, a best/ or final/ folder, or a model file."""
    if os.path.isfile(path):
        folder = os.path.dirname(path)
        return path, os.path.join(folder, "vecnormalize.pkl")
    for sub in ("best", "final", ""):
        folder = os.path.join(path, sub)
        for name in ("best_model.zip", "model.zip"):
            if os.path.isfile(os.path.join(folder, name)):
                return os.path.join(folder, name), os.path.join(folder, "vecnormalize.pkl")
    raise FileNotFoundError(f"no model.zip / best_model.zip under {path}")


class PPODriver:
    def __init__(self, path):
        model_path, norm_path = find_model(path)
        self.model = PPO.load(model_path, device="cpu")
        with open(norm_path, "rb") as f:
            self.vecnorm = pickle.load(f)      # VecNormalize without its env: stats only
        self.vecnorm.training = False
        self.model_path = model_path

    def act(self, info) -> np.ndarray:
        obs = self.vecnorm.normalize_obs(obs_from_info(info)[None, :])
        action, _ = self.model.predict(obs, deterministic=True)
        return np.clip(action[0], -1.0, 1.0).astype(np.float32)

    __call__ = act
