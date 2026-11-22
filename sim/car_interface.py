"""What a car must provide so RacingEnv can drive it - the boundary the MuJoCo car plugs into.

RacingEnv only ever talks to a car through this interface. A new car (e.g. the MuJoCo car)
implements it, gets one line in make_car(), and nothing else changes: not the track code,
not RacingEnv's public API, not anyone's controller.

A car must have:
    p                    car values: a CarParams (sim/car_numpy.py). RacingEnv reads dt,
                         delta_max, width, length, mu and v_max from it, and shows all of it
                         as env.car_params - so MPCC and Pure Pursuit get this car's numbers.
    reset(x, y, psi, vx=0.0)
                         put the car at a pose (centre of mass, m / rad) with forward speed vx.
    step(throttle, delta_target, grass=False)
                         advance one control period (p.dt = 50 ms) with throttle in [-1, 1]
                         (+1 = full gas, -1 = full brake) and a steering target in rad. The car
                         itself applies the limits: steering at most p.delta_rate_max toward the
                         target, |delta| <= p.delta_max, no reverse, top speed. grass=True means
                         the car is off track: less grip and extra drag.
    state                [x, y, psi, vx, vy, r, delta] after the last step (SI units, left
                         positive, vx/vy in the car frame, r = yaw rate, delta = wheel angle).
"""
from __future__ import annotations

from typing import Protocol

from .car_numpy import CarParams, NumpyCar


class Car(Protocol):
    p: CarParams
    state: list

    def reset(self, x: float, y: float, psi: float, vx: float = 0.0) -> None: ...

    def step(self, throttle: float, delta_target: float, grass: bool = False) -> object: ...


def make_car(kind: str, params: CarParams) -> Car:
    """Create the car RacingEnv drives: "numpy" now, "mujoco" once it exists."""
    if kind == "numpy":
        return NumpyCar(params)
    if kind == "mujoco":
        raise NotImplementedError("MuJoCo car not available yet - use car='numpy'.")
    raise ValueError(f"unknown car {kind!r} (use 'numpy' or 'mujoco')")
