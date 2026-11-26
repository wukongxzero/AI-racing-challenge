"""Adapters: turn each teammate's controller into policy(obs, info) -> [throttle, steering].

eval/evaluate.py imports these. Each adapter only calls the owner's public code; it never
changes how their controller works, so the owner stays in charge of tuning it.
"""
from __future__ import annotations

import numpy as np


class AssumedGrip:
    """The env as a controller *believes* it is: same everything, but car_params as if the
    grip were mu_scale=assumed_mu. Used for the "uninformed" robustness test: the road is
    slippery (the real env has e.g. mu_scale=0.7) but Pure Pursuit / MPCC plan for normal
    grip, the same position PPO is in (PPO never reads car_params).

        planning_env = AssumedGrip(env, assumed_mu=1.0)
        driver = PurePursuitDriver(planning_env)     # actions still go to the real env
    """

    def __init__(self, env, assumed_mu=1.0, car="numpy"):
        from sim import RacingEnv
        self._env = env
        ref = RacingEnv(car=car, track=env.track.name, mu_scale=assumed_mu)
        self.car_params = dict(ref.car_params)
        ref.close()
        self.assumed_mu = assumed_mu

    def __getattr__(self, name):          # everything except car_params comes from the real env
        return getattr(self._env, name)


class PurePursuitDriver:
    name = "Pure Pursuit"

    def __init__(self, env, **params):
        from controllers.pure_pursuit.pure_pursuit import PurePursuit
        self.controller = PurePursuit(env, **params)

    def __call__(self, obs, info):
        return self.controller(info)

    def reset(self):
        self.controller.reset()


class MPCCDriver:
    """Berinay's dynamic MPCC, driven exactly like controllers/mpc/run_dynamic_mpcc.py:
    below 1.5 m/s her kinematic MPCC launches the car; above it the dynamic MPCC drives.
    A failed solve is retried once from a cold start; if that fails too, the car brakes.

    Defaults = her best high-speed benchmark settings (speed cap 8 m/s, horizon 6,
    RK4 substeps 2, the same as her --fast preset).
    """
    name = "MPCC"

    def __init__(self, env, speed=8.0, horizon=6, substeps=2):
        from controllers.mpc.dynamic_mpcc import DynamicMPCC, DynamicConfig
        self.env = env
        self.settings = {"speed": speed, "horizon": horizon, "substeps": substeps}
        self.mpc = DynamicMPCC(env, DynamicConfig(horizon=horizon, speed=speed, substeps=substeps))
        self.reset()

    def reset(self):
        self.mpc.guess_x = self.mpc.guess_u = self.mpc.guess_lam = None
        self.mpc.launch.guess_x = self.mpc.launch.guess_u = None
        self.stats = {"mpcc_recoveries": 0, "mpcc_fallbacks": 0, "mpcc_launch_steps": 0}

    def _clear_guesses(self, method):
        self.mpc.guess_x = self.mpc.guess_u = self.mpc.guess_lam = None
        method.guess_x = method.guess_u = None

    def __call__(self, obs, info):
        if info["vx"] < 1.5:
            self.stats["mpcc_launch_steps"] += 1
            state = np.array([info["x"], info["y"], info["psi"], max(0.0, info["vx"])])
            method = self.mpc.launch
        else:
            state = np.array([info["x"], info["y"], info["psi"], info["vx"], info["vy"], info["yaw_rate"]])
            method = self.mpc
        args = (state, float(info["progress"]), float(info["delta"]))
        try:
            u = method.solve(*args)[0]
        except (RuntimeError, ValueError, FloatingPointError):
            self._clear_guesses(method)
            try:
                u = method.solve(*args)[0]
                self.stats["mpcc_recoveries"] += 1
            except (RuntimeError, ValueError, FloatingPointError):
                self.stats["mpcc_fallbacks"] += 1
                return self.env.to_action(-self.env.car_params["b_max"], float(info["delta"]))
        return self.env.to_action(float(u[0]), float(u[1]))
