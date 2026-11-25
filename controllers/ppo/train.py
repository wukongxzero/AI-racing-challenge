"""Train PPO (Stable-Baselines3) on the racing env.

From the repo root:
    python -m controllers.ppo.train                          # plan defaults: 1M steps, 8 envs
    python -m controllers.ppo.train --reward progress --run-name step3_progress
    python -m controllers.ppo.train --timesteps 20000 --n-envs 2   # quick smoke test
    tensorboard --logdir runs/ppo

Output in runs/ppo/<run-name>/ (git-ignored):
    config.json                    everything needed to reproduce the run
    best/best_model.zip + vecnormalize.pkl   best evaluation from the start line (use this)
    final/model.zip + vecnormalize.pkl  weights at the end of training
    checkpoints/                   every --checkpoint-every steps
    tb/                            TensorBoard logs
The policy and the VecNormalize statistics always belong together.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from controllers.ppo.wrapper import RewardConfig, make_env

EPISODE_KEYS = ("ep_progress", "lap", "end_reason")


class EpisodeStats(BaseCallback):
    """Logs racing-specific episode stats to TensorBoard: metres driven per episode,
    laps, and why episodes end (off_track / reverse / stall / time_limit)."""

    def __init__(self):
        super().__init__()
        self._episodes = []

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                self._episodes.append(info["episode"])
        return True

    def _on_rollout_end(self):
        if not self._episodes:
            return
        eps = self._episodes
        self.logger.record("race/ep_progress_m", float(np.mean([e["ep_progress"] for e in eps])))
        self.logger.record("race/ep_laps", float(np.mean([e["lap"] for e in eps])))
        reasons = Counter(e["end_reason"] for e in eps)
        for r in ("off_track", "reverse", "stall", "time_limit"):
            self.logger.record(f"race/end_{r}", reasons.get(r, 0) / len(eps))
        self._episodes = []


class SaveVecNormalize(BaseCallback):
    """Saves the VecNormalize statistics next to a model (called on a new best model)."""

    def __init__(self, path):
        super().__init__()
        self.path = path

    def _on_step(self):
        self.model.get_vec_normalize_env().save(self.path)
        return True


class CheckpointWithNorm(BaseCallback):
    def __init__(self, every, folder):
        super().__init__()
        self.every, self.folder = every, folder

    def _on_step(self):
        if self.num_timesteps % self.every < self.training_env.num_envs:
            os.makedirs(self.folder, exist_ok=True)
            tag = f"{self.num_timesteps // 1000}k"
            self.model.save(os.path.join(self.folder, f"model_{tag}.zip"))
            self.model.get_vec_normalize_env().save(os.path.join(self.folder, f"vecnormalize_{tag}.pkl"))
        return True


def build_vec_env(n_envs, seed, subproc=True, **env_kwargs):
    def factory(rank):
        def _init():
            env = make_env(**env_kwargs)
            env.reset(seed=seed + rank)
            return env
        return _init
    fns = [factory(i) for i in range(n_envs)]
    venv = SubprocVecEnv(fns) if subproc and n_envs > 1 else DummyVecEnv(fns)
    return VecMonitor(venv, info_keywords=EPISODE_KEYS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default=None, help="folder under runs/ppo (default: time stamp)")
    ap.add_argument("--timesteps", type=float, default=1e6)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--track", default="training")
    ap.add_argument("--car", default="numpy")
    ap.add_argument("--reward", default="full", choices=["full", "progress"])
    for k, v in RewardConfig().as_dict().items():   # override single weights, e.g. --w_du 0
        ap.add_argument(f"--{k}", type=float, default=None, help=f"reward weight (default {v})")
    ap.add_argument("--mu-scale", type=float, default=1.0)
    # PPO hyperparameters (plan section 5: SB3 defaults)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=2048, help="rollout length per env")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=50_000, help="timesteps between evaluations")
    ap.add_argument("--checkpoint-every", type=int, default=100_000)
    ap.add_argument("--resume", default=None, help="run folder to continue from (uses final/)")
    ap.add_argument("--mu-range", type=float, nargs=2, default=None,
                    help="min and max friction scale sampled per training episode (e.g. 0.6 1.0)")
    args = ap.parse_args()

    reward = RewardConfig.preset(args.reward)
    for k in reward.as_dict():
        if getattr(args, k) is not None:
            setattr(reward, k, getattr(args, k))

    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join("runs", "ppo", run_name)
    os.makedirs(out, exist_ok=True)
    config = {**vars(args), "reward_weights": reward.as_dict(), "run_name": run_name}
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    env_kwargs = dict(track=args.track, car=args.car, reward=reward, mu_scale=args.mu_scale)
    venv = build_vec_env(args.n_envs, args.seed, train=True, **env_kwargs, mu_range=args.mu_range)
    # Evaluation: start line at rest, same termination rules as training, frozen stats.
    eval_env = build_vec_env(1, args.seed + 10_000, subproc=False, train=True,
                             start={"s0": 0.0, "e_c0": 0.0, "e_psi0": 0.0, "v0": 0.0}, **env_kwargs)

    if args.resume:
        src = os.path.join(args.resume, "final")
        venv = VecNormalize.load(os.path.join(src, "vecnormalize.pkl"), venv)
        venv.training = True
        model = PPO.load(os.path.join(src, "model.zip"), env=venv, device="cpu",
                         tensorboard_log=os.path.join(out, "tb"))
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=args.gamma)
        model = PPO("MlpPolicy", venv, learning_rate=args.lr, n_steps=args.n_steps,
                    batch_size=args.batch_size, n_epochs=args.n_epochs, gamma=args.gamma,
                    gae_lambda=args.gae_lambda, clip_range=args.clip_range, ent_coef=args.ent_coef,
                    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
                    tensorboard_log=os.path.join(out, "tb"), seed=args.seed, device="cpu", verbose=1)
    eval_env = VecNormalize(eval_env, training=False, norm_reward=False, clip_obs=10.0)

    best = os.path.join(out, "best")
    os.makedirs(best, exist_ok=True)
    callbacks = CallbackList([
        EpisodeStats(),
        EvalCallback(eval_env, n_eval_episodes=1, deterministic=True,
                     eval_freq=max(args.eval_every // args.n_envs, 1), best_model_save_path=best,
                     log_path=os.path.join(out, "eval"),
                     callback_on_new_best=SaveVecNormalize(os.path.join(best, "vecnormalize.pkl"))),
        CheckpointWithNorm(args.checkpoint_every, os.path.join(out, "checkpoints")),
    ])

    print(f"run folder: {out}")
    model.learn(total_timesteps=int(args.timesteps), callback=callbacks, tb_log_name="ppo",
                reset_num_timesteps=not args.resume)

    final = os.path.join(out, "final")
    os.makedirs(final, exist_ok=True)
    model.save(os.path.join(final, "model.zip"))
    venv.save(os.path.join(final, "vecnormalize.pkl"))
    venv.close()
    print(f"done. evaluate with:  python -m controllers.ppo.evaluate {out}")


if __name__ == "__main__":
    main()
