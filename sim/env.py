"""RacingEnv - Gymnasium-style racing environment (agreed team interface).

    env = RacingEnv(car="numpy")
    obs, info = env.reset(seed=None, options=None)
    obs, reward, terminated, truncated, info = env.step(action)  # [throttle, steering] in [-1, 1]
    action = env.to_action(accel, steer)                         # m/s^2, rad -> [-1, 1]

Each step holds the command for 50 ms (20 Hz). Evaluation mode is the default:
the simulation waits for the controller, so lap times do not depend on
computer speed.
"""
from __future__ import annotations

import csv
from dataclasses import replace

import numpy as np

from .car_numpy import CarParams, NumpyCar
from .track import Track, load_track

try:  # Gymnasium is optional; the env works the same without it.
    import gymnasium as gym
    from gymnasium import spaces
    _Base = gym.Env
except ImportError:  # pragma: no cover
    gym = None

    class _Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            shape = shape or np.shape(low)
            self.low = np.broadcast_to(np.asarray(low, dtype), shape).copy()
            self.high = np.broadcast_to(np.asarray(high, dtype), shape).copy()
            self.shape, self.dtype = tuple(shape), dtype

        def sample(self):
            lo = np.where(np.isfinite(self.low), self.low, -1.0)
            hi = np.where(np.isfinite(self.high), self.high, 1.0)
            return np.random.uniform(lo, hi).astype(self.dtype)

        def contains(self, x):
            x = np.asarray(x)
            return x.shape == self.shape and bool(np.all(x >= self.low) and np.all(x <= self.high))

    class spaces:  # noqa: N801 - mimic gymnasium.spaces
        Box = _Box

    class _Base:
        metadata = {}
        np_random = np.random.default_rng()

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self.np_random = np.random.default_rng(seed)

        def close(self):
            pass


OBS_NAMES = ("x", "y", "psi", "vx", "vy", "yaw_rate", "delta",
             "s", "e_c", "e_psi", "kappa", "w_left", "w_right",
             "prev_throttle", "prev_steer")


class RacingEnv(_Base):
    """Racing environment.

    Args:
        car:            "numpy" (stand-in) or "mujoco" (later, same interface).
        track:          "training", "hidden", or a path to a track CSV.
        off_track:      "grass"     - car slows on grass and keeps going (races);
                        "terminate" - episode ends when the car leaves the road.
        max_laps:       episode ends (terminated) after this many laps; None = no limit.
        time_limit:     episode ends (truncated) after this many seconds; None = no limit.
        lookahead:      distances ahead (m) for road curvature / direction ahead
                        (default 0.5, 1.0, ... 3.0 m - the 6 points PPO uses).
        mu_scale:       friction multiplier for robustness tests (e.g. 0.7 = 30 % less
                        grip). Can also be set per episode: reset(options={"mu_scale": ...}).
        reward_fn:      optional f(info) -> float replacing the default reward.
        off_track_penalty: default-reward penalty per step spent off track.
        car_params:     CarParams override (everyone should use the defaults).
        log:            record the per-step log (turn off for fast training).
    """

    metadata = {"render_modes": [], "render_fps": 20}

    def __init__(self, car="numpy", track="training", off_track="grass", max_laps=None,
                 time_limit=None, lookahead=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0), mu_scale=1.0,
                 reward_fn=None, off_track_penalty=1.0, car_params=None, log=True,
                 lap_min_fraction=0.9):
        if car == "mujoco":
            raise NotImplementedError("MuJoCo car not available yet - use car='numpy'.")
        if car != "numpy":
            raise ValueError(f"unknown car '{car}'")
        if off_track not in ("grass", "terminate"):
            raise ValueError("off_track must be 'grass' or 'terminate'")

        self.track: Track = track if isinstance(track, Track) else load_track(track)
        self._params = replace(car_params) if car_params is not None else CarParams()
        self._mu_base = self._params.mu
        self.mu_scale = float(mu_scale)
        # The car sits behind a small internal boundary (reset / step / state),
        # so the MuJoCo car can drop in here without changing anything else.
        self._car = NumpyCar(self._params)
        self.dt = self._params.dt
        self.off_track_mode = off_track
        self.max_laps = max_laps
        self.time_limit = time_limit
        self.lookahead = tuple(float(d) for d in lookahead)
        self.reward_fn = reward_fn
        self.off_track_penalty = float(off_track_penalty)
        self.log_enabled = log
        self.lap_min_fraction = lap_min_fraction

        self.obs_names = OBS_NAMES + tuple(f"kappa_ahead_{d:g}" for d in self.lookahead)
        n = len(self.obs_names)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._info = None

    # ------------------------------------------------------------- public info
    @property
    def car_params(self) -> dict:
        """Read-only copy of the shared car values (wheelbase, mu, v_max, limits...).
        `mu` already includes the current mu_scale."""
        return self._params.to_dict()

    def to_action(self, accel, steer):
        """Convert (acceleration m/s^2, steering angle rad) to [throttle, steering] in [-1, 1].

        Exact inverse of the command mapping: +a_max <-> +1, -b_max <-> -1,
        +-delta_max <-> +-1. Drag, rolling resistance and grip limits are NOT
        compensated (see car_params).
        """
        p = self._params
        thr = accel / p.a_max if accel >= 0 else accel / p.b_max
        return np.clip(np.array([thr, steer / p.delta_max], dtype=np.float32), -1.0, 1.0)

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        """Start an episode.

        options (all optional):
            s0, e_c0, e_psi0, v0   start progress (m), lateral offset (m),
                                   heading error (rad), forward speed (m/s).
            random_start: True     draw s0 ~ U[0, L), e_c0 ~ U(+-0.3),
                                   e_psi0 ~ U(+-0.2), v0 ~ U[0, 3] (explicit values win).
            mu_scale               friction multiplier from now on (robustness tests).
        Default: on the start line, centred, at rest (race start).
        """
        super().reset(seed=seed)
        opt = dict(options or {})
        if "mu_scale" in opt:
            self.mu_scale = float(opt["mu_scale"])
        self._params.mu = self._mu_base * self.mu_scale
        if opt.get("random_start"):
            rng = self.np_random
            opt.setdefault("s0", float(rng.uniform(0.0, self.track.length)))
            opt.setdefault("e_c0", float(rng.uniform(-0.3, 0.3)))
            opt.setdefault("e_psi0", float(rng.uniform(-0.2, 0.2)))
            opt.setdefault("v0", float(rng.uniform(0.0, 3.0)))
        s0 = float(opt.get("s0", 0.0)) % self.track.length
        e_c0 = float(opt.get("e_c0", 0.0))
        e_psi0 = float(opt.get("e_psi0", 0.0))
        v0 = float(opt.get("v0", 0.0))

        x, y, psi = self.track.frenet_to_cartesian(s0, e_c0, e_psi0)
        self._car.reset(x, y, psi, vx=v0)

        self.t = 0.0
        self.steps = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self._s = s0
        self._progress = s0                 # unwrapped progress (m), absolute
        # A lap counts at each start-line crossing; a first partial lap from a
        # mid-track start (less than lap_min_fraction of L) is not counted.
        L = self.track.length
        self._next_line = L
        self._lap_counts = (L - s0) >= self.lap_min_fraction * L
        self.lap = 0
        self.lap_times = []
        self.lap_log = []
        self._lap_start_t = 0.0 if self._lap_counts else None
        self._lap_off_events = 0
        self.off_track_count = 0
        self._off = False
        self._log = {k: [] for k in self._log_keys()} if self.log_enabled else None

        self._update_track_state()
        self._off = self._is_off()
        info = self._make_info(lap_completed=False)
        self._record(info)
        return self._obs(info), info

    # ------------------------------------------------------------------- step
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
        on_grass = self._off and self.off_track_mode == "grass"
        self._car.step(a[0], a[1] * self._params.delta_max, grass=on_grass)
        self.t += self.dt
        self.steps += 1
        self.prev_action = a.astype(np.float32)

        prog_before = self._progress
        self._update_track_state()
        ds = self._progress - prog_before

        was_off = self._off
        self._off = self._is_off()
        if self._off and not was_off:
            self.off_track_count += 1
            self._lap_off_events += 1

        lap_completed = self._check_lap(prog_before)
        info = self._make_info(lap_completed, ds)

        if self.reward_fn is not None:
            reward = float(self.reward_fn(info))
        else:
            reward = ds - (self.off_track_penalty if self._off else 0.0)

        terminated = bool(
            (self.off_track_mode == "terminate" and self._off)
            or (self.max_laps is not None and self.lap >= self.max_laps))
        truncated = bool(self.time_limit is not None and self.t >= self.time_limit - 1e-9
                         and not terminated)
        self._record(info)
        return self._obs(info), reward, terminated, truncated, info

    # ---------------------------------------------------------------- helpers
    def _update_track_state(self):
        st = self._car.state
        s, e_c, e_psi, kappa = self.track.project(st[0], st[1], st[2], self._s)
        L = self.track.length
        ds = s - self._s
        if ds < -0.5 * L:
            ds += L
        elif ds > 0.5 * L:
            ds -= L
        self._progress += ds
        self._s, self._e_c, self._e_psi, self._kappa = s, e_c, e_psi, kappa
        self._w_left, self._w_right = self.track.width(s)

    def _is_off(self):
        """Off track = any part of the car past the road edge:
        |e_c| > half-width - half car width."""
        half_car = 0.5 * self._params.width
        return self._e_c > self._w_left - half_car or self._e_c < -(self._w_right - half_car)

    def _check_lap(self, prog_before):
        if self._progress < self._next_line:
            return False
        # interpolate the exact crossing time inside the 50 ms step
        frac = (self._next_line - prog_before) / max(self._progress - prog_before, 1e-12)
        t_cross = self.t - self.dt + frac * self.dt
        self._next_line += self.track.length
        if not self._lap_counts:          # first partial lap: start timing now
            self._lap_counts = True
            self._lap_start_t = t_cross
            self._lap_off_events = 0
            return False
        lap_time = t_cross - self._lap_start_t
        self.lap += 1
        self.lap_times.append(lap_time)
        self.lap_log.append({"lap": self.lap, "lap_time": lap_time,
                             "off_track_count": self._lap_off_events})
        self._lap_start_t = t_cross
        self._lap_off_events = 1 if self._off else 0  # still off when crossing
        return True

    def _make_info(self, lap_completed, ds=0.0):
        st = self._car.state
        info = {
            # time and car state
            "t": self.t, "x": st[0], "y": st[1], "psi": st[2], "vx": st[3], "vy": st[4],
            "yaw_rate": st[5], "delta": st[6],
            # command that produced this state
            "prev_throttle": float(self.prev_action[0]), "prev_steer": float(self.prev_action[1]),
            # track outputs
            "s": self._s, "e_c": self._e_c, "e_psi": self._e_psi, "kappa": self._kappa,
            "w_left": self._w_left, "w_right": self._w_right,
            "kappa_ahead": self.track.lookahead(self._s, self.lookahead),
            "heading_ahead": self.track.heading_ahead(self._s, self.lookahead, st[2]),
            # progress (ds / progress_rate help detect stall or driving backwards)
            "progress": self._progress, "lap_progress": self._s / self.track.length,
            "ds": ds, "progress_rate": ds / self.dt,
            # off track and laps
            "off_track": self._off, "off_track_count": self.off_track_count,
            "lap": self.lap, "lap_times": list(self.lap_times), "lap_completed": lap_completed,
            "mu_scale": self.mu_scale,
        }
        return info

    def _obs(self, info):
        base = [info[k] for k in OBS_NAMES]
        return np.concatenate([base, info["kappa_ahead"]]).astype(np.float32)

    # ---------------------------------------------------------------- logging
    @staticmethod
    def _log_keys():
        return ("t", "x", "y", "psi", "vx", "vy", "yaw_rate", "delta",
                "throttle", "steer", "s", "progress", "e_c", "e_psi", "off_track", "lap")

    def _record(self, info):
        if self._log is None:
            return
        for k in self._log_keys():
            if k == "throttle":
                self._log[k].append(info["prev_throttle"])
            elif k == "steer":
                self._log[k].append(info["prev_steer"])
            else:
                self._log[k].append(info[k])

    def get_log(self):
        """Per-step log as a dict of numpy arrays (row i = state after command i)."""
        if self._log is None:
            return None
        return {k: np.asarray(v) for k, v in self._log.items()}

    def save_log(self, path):
        """Write the per-step log to CSV."""
        log = self.get_log()
        if log is None:
            raise RuntimeError("logging disabled (log=False)")
        keys = list(log)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for row in zip(*(log[k] for k in keys)):
                w.writerow([int(v) if isinstance(v, (bool, np.bool_)) else v for v in row])

    def render(self):
        return None  # top-down Pygame view lives in the game code
