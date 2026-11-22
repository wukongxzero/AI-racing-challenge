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
import time
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
        render_mode:    None (default, no drawing - fastest), "rgb_array" (render() returns
                        a top-down picture, e.g. for Gymnasium's RecordVideo) or "human"
                        (a live Pygame window, drawn every step in real time).
    """

    # 20 fps = one picture per 50 ms step, so recorded videos play in real time
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, car="numpy", track="training", off_track="grass", max_laps=None,
                 time_limit=None, lookahead=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0), mu_scale=1.0,
                 reward_fn=None, off_track_penalty=1.0, car_params=None, log=True,
                 render_mode=None):
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"render_mode must be None, 'human' or 'rgb_array', not {render_mode!r}")
        self.render_mode = render_mode
        self._view = None                  # Pygame viewer, created on the first render
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
        self._params.mu = self._mu_base * self.mu_scale   # also applied again at every reset
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

        self.obs_names = OBS_NAMES + tuple(f"kappa_ahead_{d:g}" for d in self.lookahead)
        # Generous physical limits for each obs value (Gymnasium wants finite bounds).
        # Values are clipped to them; only absurd states (e.g. 1 km off track) get clipped.
        p, L, tr = self._params, self.track.length, self.track
        # vx gets 50 % headroom above v_max (another car might overshoot it slightly);
        # x, y: the track's own area plus 1 km on every side.
        limits = {"psi": np.pi, "vx": 1.5 * p.v_max, "vy": 20.0, "yaw_rate": 50.0,
                  "delta": p.delta_max, "s": L, "e_c": 50.0, "e_psi": np.pi, "kappa": 10.0,
                  "w_left": 10.0, "w_right": 10.0, "prev_throttle": 1.0, "prev_steer": 1.0}
        high = np.array([limits.get(k, 10.0) for k in self.obs_names], dtype=np.float32)  # kappa_ahead: 10
        low = -high
        for k in ("vx", "s", "w_left", "w_right"):           # these are never negative
            low[self.obs_names.index(k)] = 0.0
        for k, pts in (("x", tr.x_knots), ("y", tr.y_knots)):
            i = self.obs_names.index(k)
            low[i], high[i] = pts.min() - 1000.0, pts.max() + 1000.0
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._info = None
        self._started = False             # step() before reset() is an error

    # ------------------------------------------------------------- public info
    @property
    def car_params(self) -> dict:
        """Read-only copy of the shared car values (wheelbase, mu, v_max, limits...).
        `mu` already includes the current mu_scale."""
        return self._params.to_dict()

    def to_action(self, accel, steer):
        """Convert (acceleration m/s^2, steering angle rad) to [throttle, steering] in [-1, 1].

        Exact inverse of the command mapping: +a_max <-> +1, -b_max <-> -1,
        +-delta_max <-> +-1. The car still applies its own limits (top speed, no
        reverse, tire grip, steering rate), so the result is not guaranteed.
        """
        p = self._params
        thr = accel / p.a_max if accel >= 0 else accel / p.b_max
        return np.clip(np.array([thr, steer / p.delta_max], dtype=np.float32), -1.0, 1.0)

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        """Start an episode.

        options (all optional):
            s0, e_c0, e_psi0, v0   start progress (m), lateral offset (m),
                                   heading error (rad), forward speed (m/s). Each may be
                                   a number, or a (low, high) pair = drawn uniformly
                                   at every reset, e.g. {"v0": (0.5, 2.5)}.
            random_start: True     defaults for anything not given: s0 ~ U[0, L),
                                   e_c0 ~ U(+-0.3), e_psi0 ~ U(+-0.2), v0 ~ U[0, 3].
            mu_scale               friction multiplier from now on (robustness tests).
        Default: on the start line, centred, at rest (race start). Unknown option
        names raise an error, so a typo can't silently be ignored.
        Laps: only a start ON the start line (s0 = 0) times its first lap; from any
        other start the timing begins at the first start-line crossing.
        """
        super().reset(seed=seed)
        opt = dict(options or {})
        unknown = set(opt) - self._RESET_OPTIONS
        if unknown:
            raise ValueError(f"unknown reset option(s) {sorted(unknown)}; "
                             f"allowed: {sorted(self._RESET_OPTIONS)}")
        if "mu_scale" in opt:
            self.mu_scale = float(opt["mu_scale"])
        self._params.mu = self._mu_base * self.mu_scale
        if opt.get("random_start"):
            defaults = {"s0": (0.0, self.track.length), "e_c0": (-0.3, 0.3),
                        "e_psi0": (-0.2, 0.2), "v0": (0.0, 3.0)}
            for k, rng_range in defaults.items():
                opt.setdefault(k, rng_range)
        start = {k: self._draw(opt.get(k, 0.0)) for k in ("s0", "e_c0", "e_psi0", "v0")}
        s0 = start["s0"] % self.track.length
        e_c0, e_psi0, v0 = start["e_c0"], start["e_psi0"], start["v0"]

        x, y, psi = self.track.frenet_to_cartesian(s0, e_c0, e_psi0)
        self._car.reset(x, y, psi, vx=v0)

        self.t = 0.0
        self.steps = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self._s = s0
        self._progress = s0                 # unwrapped progress (m), absolute
        # A lap counts at each start-line crossing (progress reaches the next multiple of
        # L, so it must cover the whole track). Only a start on the line times lap 1;
        # from mid-track the first crossing just starts the timing.
        L = self.track.length
        self._next_line = L
        self._lap_counts = s0 == 0.0
        self.lap = 0
        self.lap_times = []
        self.lap_log = []
        self._lap_start_t = 0.0 if self._lap_counts else None
        self._lap_off_events = 0
        self.off_track_count = 0
        self._off = False
        self._log = {k: [] for k in self._log_keys()} if self.log_enabled else None
        self._started = True

        self._update_track_state()
        self._off = self._is_off()
        info = self._make_info(lap_completed=False)
        self._record(info)
        self._info = info
        if self.render_mode == "human":
            self.render()
        return self._obs(info), info

    # ------------------------------------------------------------------- step
    def step(self, action):
        if not self._started:
            raise RuntimeError("call reset() before step()")
        a = np.asarray(action, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(a)):          # NaN / inf would silently become full throttle
            raise ValueError(f"action must be finite, got {a.tolist()}")
        a = np.clip(a, -1.0, 1.0)
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
        self._info = info
        if self.render_mode == "human":
            self.render()
        return self._obs(info), reward, terminated, truncated, info

    # ---------------------------------------------------------------- helpers
    _RESET_OPTIONS = {"s0", "e_c0", "e_psi0", "v0", "random_start", "mu_scale"}

    def _draw(self, value):
        """A number as is, or a (low, high) pair drawn uniformly with the env's seeded RNG."""
        if isinstance(value, (tuple, list)):
            low, high = value
            return float(self.np_random.uniform(low, high))
        return float(value)

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
                             "off_track_count": self._lap_off_events,
                             "clean": self._lap_off_events == 0})   # no off-track = clean lap
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
        obs = np.concatenate([base, info["kappa_ahead"]]).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

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

    # ---------------------------------------------------------------- render
    RENDER_SIZE = (960, 640)   # pixels; both multiples of 16, which video encoders prefer

    def render(self):
        """Draw the current state top-down (Gymnasium convention).

        "rgb_array": returns the picture as a (640, 960, 3) uint8 array.
        "human":     shows it in a window, paced to real time (20 fps); returns None.
        None:        does nothing.
        The game draws its own scene with sim/viewer.py; this is for recording and watching.
        """
        if self.render_mode is None or self._info is None:
            return None
        if self._view is None:
            from .viewer import TopDownView
            p = self._params
            self._view = TopDownView(self.track, window=self.RENDER_SIZE, car_length=p.length,
                                     car_width=p.width, headless=(self.render_mode == "rgb_array"))
            self._next_frame_time = time.perf_counter()
        info = self._info
        color = (240, 140, 30) if info["off_track"] else (220, 40, 40)   # orange when off track
        text = (f"t {info['t']:5.1f} s   v {info['vx']:4.1f} m/s   lap {info['lap']}   "
                f"off-track {info['off_track_count']}")
        self._view.draw([{"x": info["x"], "y": info["y"], "psi": info["psi"], "color": color}], text)
        if self.render_mode == "rgb_array":
            return self._view.frame()
        # human: wait so that one 50 ms step also takes 50 ms on screen
        self._next_frame_time += 1.0 / self.metadata["render_fps"]
        time.sleep(max(0.0, self._next_frame_time - time.perf_counter()))
        self._next_frame_time = max(self._next_frame_time, time.perf_counter())
        return None

    def close(self):
        if self._view is not None:
            self._view.close()
            self._view = None
