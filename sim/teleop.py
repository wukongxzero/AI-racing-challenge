"""Keyboard teleop: arrow keys -> the same [throttle, steering] commands every controller sends.

Fairness: the human goes through exactly the same limits as the controllers. The car
turns its wheels toward the steering command at most 3.2 rad/s and never past 0.42 rad;
throttle/brake map to the same acceleration limits.

    teleop = KeyboardTeleop(env.car_params, track=env.track)
    action = teleop.action(up, down, left, right, speed=info["vx"], s=info["s"])   # every 50 ms
    action = teleop.action_from_pygame(speed=info["vx"], s=info["s"])             # same, from keys

Keys:
    Up      throttle, ramping from 0 to full in `throttle_ramp_time` (smooth with on/off keys)
    Down    full brake (no reverse)
    none    coast (throttle 0)
    Left / Right   full steering command left / right; release = straight.

Human-only assists (both ON by default - they make driving with on/off keys much easier). They only ever REDUCE the human's input (less steering, less gas,
more braking), never ask for more than the controllers can command. Report them in results.

Steering assist (steer_assist): at speed, a full Left/Right asks only for assist_margin x the
steering angle that uses the tires' grip (a_lat_max), so holding a key can't throw the car
into a slide. assist_margin 1.0 = exactly the steady-cornering limit (strongest); higher =
more steering allowed for turning in and corrections (default 1.4, tuned by play-testing).

Braking assist (brake_assist, needs `track`): like the braking assist in racing games. It
looks at the road up to 8 m ahead; if the car is too fast to still brake down to a corner's
grip-limited speed, it brakes (or cuts the gas) for you, so you can hold the gas and steer.
"""
from __future__ import annotations

import math

import numpy as np


class KeyboardTeleop:
    def __init__(self, car_params=None, throttle_ramp_time=0.3, dt=0.05, steer_assist=True,
                 assist_margin=1.4, track=None, brake_assist=None, brake_margin=0.5):
        """brake_assist: None = on when a track is given. brake_margin: fraction of the usable
        grip the braking assist plans corners with (lower = slower, safer corners). 0.5 let a
        simulated gas-always-held driver lap with 0 off-tracks, barely slower than 0.8."""
        self.car_params = car_params
        self.throttle_step = dt / throttle_ramp_time if throttle_ramp_time > 0 else 1.0
        self.steer_assist = steer_assist
        self.assist_margin = assist_margin
        self.track = track
        self.brake_assist = (track is not None) if brake_assist is None else brake_assist
        self.brake_margin = brake_margin
        self.throttle = 0.0
        if (steer_assist or self.brake_assist) and car_params is None:
            raise ValueError("the assists need car_params: KeyboardTeleop(env.car_params, track=env.track), "
                             "or turn them off")
        if self.brake_assist and track is None:
            raise ValueError("the braking assist needs the track: KeyboardTeleop(..., track=env.track)")

    def reset(self):
        self.throttle = 0.0

    def steer_limit(self, speed):
        """Largest steering command (0..1) the steering assist allows at this speed."""
        p = self.car_params
        if speed < 1.0:
            return 1.0
        L = p["wheelbase"]
        # angle for a circle at the grip limit: delta = L/R + K*a_y with R = v^2 / a_lat_max
        delta = (L / speed ** 2 + p["understeer_gradient"]) * p["a_lat_max"]
        return min(1.0, self.assist_margin * delta / p["delta_max"])

    def speed_limit(self, s):
        """Fastest speed (m/s) at which the car can still brake down for every corner in the
        next 8 m: for a corner at distance d with curvature k, v^2 <= a_lat/k + 2*b*d."""
        p = self.car_params
        a_lat = self.brake_margin * p["a_lat_max"]      # corner speed: v^2 * k <= a_lat
        b = 0.7 * p["b_max"]                            # plan with gentle (70 %) braking
        best = math.inf
        for d in np.arange(0.0, 8.01, 0.25):
            k = abs(self.track.curvature(s + d))
            if k > 1e-3:
                best = min(best, a_lat / k + 2.0 * b * d)
        return math.sqrt(best)

    def action(self, up=False, down=False, left=False, right=False, speed=0.0, s=None):
        """The [throttle, steering] command for this step from which arrow keys are held."""
        if down:                                     # brake wins over gas
            self.throttle = -1.0
        elif up:                                     # ramp up gas (from 0 if braking before)
            self.throttle = min(1.0, max(self.throttle, 0.0) + self.throttle_step)
        else:
            self.throttle = 0.0                      # coast
        throttle = self.throttle
        if self.brake_assist and s is not None:
            # too fast for what's ahead -> less gas or brake (never more than the human asks)
            throttle = max(-1.0, min(throttle, 1.5 * (self.speed_limit(s) - speed)))
        steer = float(left) - float(right)           # +1 left, -1 right, 0 both or none
        if self.steer_assist:
            steer *= self.steer_limit(speed)
        return np.array([throttle, steer], dtype=np.float32)

    def action_from_pygame(self, speed=0.0, s=None):
        """Read the arrow keys from Pygame (a Pygame window must be open and its events
        pumped, e.g. by TopDownView.draw)."""
        import pygame
        k = pygame.key.get_pressed()
        return self.action(k[pygame.K_UP], k[pygame.K_DOWN], k[pygame.K_LEFT], k[pygame.K_RIGHT],
                           speed=speed, s=s)
