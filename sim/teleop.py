"""Keyboard teleop: arrow keys -> the same [throttle, steering] commands every controller sends.

Fairness: the human goes through exactly the same limits as the controllers. The car
turns its wheels toward the steering command at most 3.2 rad/s and never past 0.42 rad;
throttle/brake map to the same acceleration limits.

    teleop = KeyboardTeleop(env.car_params)
    action = teleop.action(up, down, left, right, speed=info["vx"])   # once per 50 ms step
    action = teleop.action_from_pygame(speed=info["vx"])              # same, reading the keys

Keys:
    Up      throttle, ramping from 0 to full in `throttle_ramp_time` (smooth with on/off keys)
    Down    full brake (no reverse)
    none    coast (throttle 0)
    Left / Right   full steering command left / right; release = straight.

Optional steering assist (off by default, for the human car only; report it if used):
at speed, a full Left/Right asks only for the steering angle that uses the tires' grip
(a_lat_max), so holding a key can't throw the car into a slide. It only ever asks for LESS
steering than without it, so it gives no advantage over the controllers.
"""
from __future__ import annotations

import numpy as np

class KeyboardTeleop:
    def __init__(self, car_params=None, throttle_ramp_time=0.3, dt=0.05, steer_assist=False):
        self.car_params = car_params
        self.throttle_step = dt / throttle_ramp_time if throttle_ramp_time > 0 else 1.0
        self.steer_assist = steer_assist
        self.throttle = 0.0
        if steer_assist and car_params is None:
            raise ValueError("steer_assist needs car_params (pass env.car_params)")

    def reset(self):
        self.throttle = 0.0

    def steer_limit(self, speed):
        """Largest steering command (0..1) the assist allows at this speed."""
        p = self.car_params
        if speed < 1.0:
            return 1.0
        L = p["wheelbase"]
        # angle for a circle at the grip limit: delta = L/R + K*a_y with R = v^2 / a_lat_max
        delta = (L / speed ** 2 + p["understeer_gradient"]) * p["a_lat_max"]
        return min(1.0, delta / p["delta_max"])

    def action(self, up=False, down=False, left=False, right=False, speed=0.0):
        """The [throttle, steering] command for this step from which arrow keys are held."""
        if down:                                     # brake wins over gas
            self.throttle = -1.0
        elif up:                                     # ramp up gas (from 0 if braking before)
            self.throttle = min(1.0, max(self.throttle, 0.0) + self.throttle_step)
        else:
            self.throttle = 0.0                      # coast
        steer = float(left) - float(right)           # +1 left, -1 right, 0 both or none
        if self.steer_assist:
            steer *= self.steer_limit(speed)
        return np.array([self.throttle, steer], dtype=np.float32)

    def action_from_pygame(self, speed=0.0):
        """Read the arrow keys from Pygame (a Pygame window must be open and its events
        pumped, e.g. by TopDownView.draw)."""
        import pygame
        k = pygame.key.get_pressed()
        return self.action(k[pygame.K_UP], k[pygame.K_DOWN], k[pygame.K_LEFT], k[pygame.K_RIGHT],
                           speed=speed)
