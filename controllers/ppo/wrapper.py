"""PPO view of the racing env: 15-D observation, plan reward, plan termination.

Everything here follows docs/ppo_plan.tex, section 4 (RL problem formulation):

    obs (15) = vx, vy, yaw_rate, delta, e_c, e_psi, prev_throttle, prev_steer,
               kappa_ahead at 0.5..3.0 m (6), w_half
    reward   = w_prog*ds - w_c*e_c^2 - w_psi*e_psi^2 - w_du*|u - u_prev|^2 - w_time
               - R_crash (on the step the car leaves the road)
    terminated: off track | reverse heading (|e_psi| > pi/2) | stall (< 0.3 m in 2 s)
    truncated:  time limit (env time_limit, 50 s = 1000 steps)

No global x, y, psi in the observation, so the policy can't memorise the map.
The observation is built from `info` only (obs_from_info), so the game can drive the
policy from any RacingEnv without this wrapper.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np

from sim import RacingEnv

LOOKAHEAD = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
PPO_OBS_NAMES = (("vx", "vy", "yaw_rate", "delta", "e_c", "e_psi", "prev_throttle", "prev_steer")
                 + tuple(f"kappa_ahead_{d:g}" for d in LOOKAHEAD) + ("w_half",))

# Training starts: anywhere on the lap, slightly off the line and moving (plan 4.5).
TRAIN_START = {"s0": None, "e_c0": (-0.3, 0.3), "e_psi0": (-0.15, 0.15), "v0": (0.0, 2.5)}
# Race / evaluation start: on the start line, centred, at rest.
RACE_START = {"s0": 0.0, "e_c0": 0.0, "e_psi0": 0.0, "v0": 0.0}


@dataclass
class RewardConfig:
    """Reward weights (plan 4.4). Change one at a time and note it in the run name."""
    w_prog: float = 1.0     # per metre of centerline progress
    w_c: float = 0.05       # lateral offset^2 (m^2)
    w_psi: float = 0.05     # heading error^2 (rad^2)
    w_du: float = 0.05      # |u - u_prev|^2, both commands in [-1, 1]
    w_time: float = 0.01    # per step, so standing still is never free
    r_crash: float = 10.0   # once, on the step the car leaves the road

    @classmethod
    def preset(cls, name: str) -> "RewardConfig":
        if name == "full":
            return cls()
        if name == "progress":   # plan step 3: progress + crash only
            return cls(w_c=0.0, w_psi=0.0, w_du=0.0, w_time=0.0)
        raise ValueError(f"unknown reward preset {name!r} (use 'full' or 'progress')")

    def as_dict(self):
        return asdict(self)


def obs_from_info(info) -> np.ndarray:
    """The 15-D PPO observation from a RacingEnv info dict (order = PPO_OBS_NAMES)."""
    base = [info[k] for k in PPO_OBS_NAMES[:8]]
    w_half = 0.5 * (info["w_left"] + info["w_right"])
    return np.concatenate([base, info["kappa_ahead"], [w_half]]).astype(np.float32)


class PPORacingWrapper(gym.Wrapper):
    """Turns RacingEnv into the PPO problem of the plan.

    Args:
        env:              a RacingEnv made with lookahead=LOOKAHEAD.
        reward:           RewardConfig.
        start:            reset options used when reset() gets none (SB3 never passes any).
                          s0=None -> uniform over the whole lap.
        early_termination: stop on reverse heading / stall (training). Off track ends the
                          episode only if the env itself uses off_track="terminate".
        stall_steps, stall_progress: stall = less than stall_progress m in stall_steps steps.
    """

    def __init__(self, env: RacingEnv, reward: RewardConfig | None = None, start=None,
                 early_termination=True, stall_steps=40, stall_progress=0.3):
        super().__init__(env)
        if tuple(env.unwrapped.lookahead) != LOOKAHEAD:
            raise ValueError(f"RacingEnv must use lookahead={LOOKAHEAD}")
        self.reward_cfg = reward or RewardConfig()
        self.start = dict(TRAIN_START if start is None else start)
        self.early_termination = early_termination
        self.stall_steps = stall_steps
        self.stall_progress = stall_progress

        # Finite bounds taken from the env's own limits (VecNormalize does the scaling).
        base = env.unwrapped.observation_space
        names = env.unwrapped.obs_names
        low, high = [], []
        for k in PPO_OBS_NAMES:
            if k == "w_half":
                low.append(0.0)
                high.append(10.0)
            else:
                i = names.index(k)
                low.append(base.low[i])
                high.append(base.high[i])
        self.observation_space = gym.spaces.Box(np.array(low, np.float32),
                                                np.array(high, np.float32), dtype=np.float32)
        self.action_space = env.action_space

    def _reset_options(self, options):
        if options is not None:
            return options
        opt = dict(self.start)
        if opt.get("s0") is None:
            opt["s0"] = (0.0, self.env.unwrapped.track.length)
        return opt

    def reset(self, *, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=self._reset_options(options))
        self._progress_hist = deque([info["progress"]], maxlen=self.stall_steps + 1)
        self._start_progress = info["progress"]
        self._prev_u = np.zeros(2, dtype=np.float32)
        info = self._add_episode_info(info, end_reason="")
        return self._obs(info), info

    def step(self, action):
        u = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
        _, _, terminated, truncated, info = self.env.step(u)
        c = self.reward_cfg

        du2 = float(np.sum((u - self._prev_u) ** 2))
        self._prev_u = u
        reward = (c.w_prog * info["ds"] - c.w_c * info["e_c"] ** 2 - c.w_psi * info["e_psi"] ** 2
                  - c.w_du * du2 - c.w_time)

        reason = ""
        if info["off_track"] and terminated:
            reward -= c.r_crash
            reason = "off_track"
        elif terminated:
            reason = "laps"                       # env max_laps reached
        if self.early_termination and not terminated:
            self._progress_hist.append(info["progress"])
            if abs(info["e_psi"]) > np.pi / 2:
                terminated, reason = True, "reverse"
            elif (len(self._progress_hist) > self.stall_steps
                  and self._progress_hist[-1] - self._progress_hist[0] < self.stall_progress):
                terminated, reason = True, "stall"
        if truncated and not terminated:
            reason = "time_limit"
        truncated = truncated and not terminated

        info = self._add_episode_info(info, end_reason=reason)
        return self._obs(info), float(reward), terminated, truncated, info

    def _add_episode_info(self, info, end_reason):
        info = dict(info)
        info["ep_progress"] = info["progress"] - self._start_progress   # metres this episode
        info["end_reason"] = end_reason
        return info

    def _obs(self, info):
        return np.clip(obs_from_info(info), self.observation_space.low, self.observation_space.high)


def make_env(track="training", car="numpy", reward="full", train=True, mu_scale=1.0,
             time_limit=50.0, render_mode=None, max_laps=None, start=None):
    """One PPO env.

    train=True:  off track ends the episode, random starts, stall/reverse termination.
    train=False: race rules - grass instead of termination, start line at rest, no early
                 termination (the episode ends on max_laps or time_limit).
    """
    reward_cfg = reward if isinstance(reward, RewardConfig) else RewardConfig.preset(reward)
    env = RacingEnv(car=car, track=track, off_track="terminate" if train else "grass",
                    time_limit=time_limit, max_laps=max_laps, lookahead=LOOKAHEAD,
                    mu_scale=mu_scale, log=not train, render_mode=render_mode)
    if start is None:
        start = TRAIN_START if train else RACE_START
    return PPORacingWrapper(env, reward_cfg, start=start, early_termination=train)
