"""MPCC as a `policy(obs, info) -> [throttle, steering]` for eval/ and the game.

    from controllers.mpc.driver import MPCCDriver
    driver = MPCCDriver(env)                 # builds the dynamic MPCC from env.car_params
    action = driver(obs, info)

Below `launch_speed` the simulator is partly kinematic, so the kinematic MPCC launches
the car; above it the dynamic (Pacejka) MPCC drives. If a solve fails (or hits the time
cap), the car applies the next input of its last successful plan (a plan made
<= horizon-1 steps ago) and the next solve starts cold. Without a usable plan
it retries once from a fresh guess; if that fails too it brakes, holding steering.
"""
from __future__ import annotations

import numpy as np

from sim.car_casadi import state_from_info
from .dynamic_mpcc import DynamicConfig, DynamicMPCC
from .run_racing_env import safe_stop


class MPCCDriver:
    def __init__(self, env, config: DynamicConfig | None = None, launch_speed=1.5):
        self.env = env
        self.mpcc = DynamicMPCC(env, config or DynamicConfig())
        self.launch_speed = launch_speed
        self._warm_up()
        self.reset()

    def _warm_up(self):
        """Solve once with each solver before the race: the first IPOPT call is slow."""
        tr = self.env.track
        x, y = tr.point_at(0.0)
        psi = tr.heading(0.0)
        for solver, state in ((self.mpcc.launch, np.array([x, y, psi, 0.0])),
                              (self.mpcc, np.array([x, y, psi, 2.0, 0.0, 0.0]))):
            try:
                solver.solve(state, 0.0, 0.0)
            except (RuntimeError, ValueError, FloatingPointError):
                pass

    def reset(self):
        self.mpcc.guess_x = self.mpcc.guess_u = self.mpcc.guess_lam = None
        self.mpcc.launch.guess_x = self.mpcc.launch.guess_u = None
        self.recoveries = 0      # solved on the cold retry
        self.plan_reuses = 0     # used the previous plan's next input
        self.fallbacks = 0       # braked
        self._plan = None        # (inputs (3, N), steps since it was made, solver)

    def __call__(self, obs, info):
        launch = info["vx"] < self.launch_speed
        if launch:
            state = np.array([info["x"], info["y"], info["psi"], max(0.0, info["vx"])])
            solver = self.mpcc.launch
        else:
            state = state_from_info(info, "dynamic")
            solver = self.mpcc
        args = (state, float(info["progress"]), float(info["delta"]))
        try:
            u, _, up = solver.solve(*args)
            self._plan = (up, 0, solver)
        except (RuntimeError, ValueError, FloatingPointError):
            solver.guess_x = solver.guess_u = None
            self.mpcc.guess_lam = None
            if self._plan is not None and self._plan[2] is solver and self._plan[1] + 1 < self._plan[0].shape[1]:
                up, age, _ = self._plan
                self._plan = (up, age + 1, solver)
                self.plan_reuses += 1
                return self.env.to_action(float(up[0, age + 1]), float(up[1, age + 1]))
            try:
                u, _, up = solver.solve(*args)
                self._plan = (up, 0, solver)
                self.recoveries += 1
            except (RuntimeError, ValueError, FloatingPointError):
                self.fallbacks += 1
                return safe_stop(self.env, info)
        return self.env.to_action(float(u[0]), float(u[1]))


# Presets used in eval/evaluate.py (tuned on the training track only).
# RACE: fastest; plans with the grip the env reports (car_params, includes mu_scale).
RACE = dict(horizon=8, substeps=2, speed=8.0, max_cpu_time=0.035, lat_frac=0.92)
# ROBUST: plans corners with 62 % of the reported grip, so it stays on the road even
# if the real grip is ~30 % lower than the model thinks (friction not known).
ROBUST = dict(RACE, lat_frac=0.62)
