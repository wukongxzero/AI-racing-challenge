"""Pure Pursuit baseline controller (Nathaly).

Two independent parts, each simple enough to explain on one slide:

1. STEERING - Pure Pursuit (Coulter, 1992)
   Pick a "goal" point on the centerline a distance Ld ahead of the car. Find the circle
   that starts at the rear axle, points the way the car points, and passes through the
   goal point. Steer the front wheels so the car drives that circle.

       goal point in the car frame (x forward, y left): (gx, gy),  Ld^2 = gx^2 + gy^2
       circle curvature      k = 2 * gy / Ld^2
       steering (bicycle)    delta = atan(L * k)            L = wheelbase

   The lookahead grows with speed, Ld = Ld_min + k_v * v, so the car looks further ahead
   when it is fast (smoother, more stable) and closer when slow (cuts corners less).

   Understeer correction: the geometry above assumes the tires don't slip. Our car
   understeers (env.car_params["understeer_gradient"] = K), so a circle of radius R at
   speed v really needs delta = L/R + K v^2 / R. We therefore multiply the geometric angle
   by (1 + K v^2 / L).

2. SPEED - a speed profile along the track, computed once at start-up
   a) Cornering limit at every point s:   v_corner(s) = min( sqrt(a_lat / |kappa(s)|), v_max )
      (from v^2 * kappa <= a_lat, the sideways grip limit). This is the plan in the
      preliminary report, with a_lat = margin * a_lat_max.
   b) Braking pass (backwards around the lap): the car must be able to slow down in time
      for each corner:  v(s) <= sqrt( v(s+ds)^2 + 2 * b * ds ).
   c) Acceleration pass (forwards): the car can't speed up faster than the motor allows:
      v(s+ds) <= sqrt( v(s)^2 + 2 * a * ds ).
   Both passes are repeated twice so they wrap correctly around the lap.
   A proportional speed controller with feed-forward tracks the profile:
       accel = k_p * (v_ref - v) + (feed-forward from the profile slope)

Uses only the agreed interface: info (car state + s), env.track (point_at, curvature,
length) and env.car_params. It never touches the car's internals, so it works unchanged on
the MuJoCo car.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

G = 9.81


@dataclass
class PurePursuitParams:
    # steering
    ld_min: float = 0.4         # m, lookahead at standstill (tuned, see tune.py)
    k_v: float = 0.15           # s, extra lookahead per m/s (Ld = ld_min + k_v * v); tuned, 0.10 spins out on the hidden track
    understeer_comp: bool = True  # multiply by (1 + K v^2 / L)
    # speed profile
    lat_margin: float = 0.9     # (tuned) use this fraction of the car's sideways grip a_lat_max
    brake_margin: float = 0.7   # fraction of max braking used when planning
    accel_margin: float = 1.0   # fraction of max acceleration used when planning
    v_max: float | None = None  # cap below the car's top speed (None = car's v_max)
    ds: float = 0.1             # m, spacing of the speed profile
    # speed tracking
    k_p: float = 3.0            # 1/s, proportional gain on speed error
    preview: float = 0.3        # s, read the profile this far ahead of the car (actuator lag)


class PurePursuit:
    """Pure Pursuit steering + curvature-based speed profile.

    Usage:
        pp = PurePursuit(env)                 # builds the speed profile for env.track
        obs, info = env.reset()
        action = pp(info)                     # [throttle, steering] in [-1, 1]
    """

    name = "Pure Pursuit"

    def __init__(self, env, params: PurePursuitParams | None = None, **overrides):
        self.env = env
        self.p = params if params is not None else PurePursuitParams()
        for k, v in overrides.items():          # e.g. PurePursuit(env, k_v=0.3)
            if not hasattr(self.p, k):
                raise ValueError(f"unknown Pure Pursuit parameter {k!r}")
            setattr(self.p, k, v)
        self.track = env.track
        self.car = env.car_params
        self.build_speed_profile()
        self.last = {}                           # internals of the last call, for plots/debugging

    # ------------------------------------------------------------ speed profile
    def build_speed_profile(self):
        """Fill self.prof_s, self.prof_v: target speed every ds metres around the lap."""
        p, car, tr = self.p, self.car, self.track
        L = tr.length
        n = int(round(L / p.ds))
        ds = L / n                                   # exact spacing so the lap closes
        s = np.arange(n) * ds
        kappa = np.array([tr.curvature(si) for si in s])
        v_top = min(car["v_max"], p.v_max) if p.v_max else car["v_max"]
        a_lat = p.lat_margin * car["a_lat_max"]
        b = p.brake_margin * car["b_max"]
        a = p.accel_margin * car["a_max"]

        # a) cornering limit
        v = np.minimum(np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-6)), v_top)
        self.prof_v_corner = v.copy()
        # b) braking pass, backwards; c) acceleration pass, forwards. Twice for the wrap.
        for _ in range(2):
            for i in range(n - 1, -1, -1):
                v[i] = min(v[i], math.sqrt(v[(i + 1) % n] ** 2 + 2.0 * b * ds))
        for _ in range(2):
            for i in range(n):
                v[(i + 1) % n] = min(v[(i + 1) % n], math.sqrt(v[i] ** 2 + 2.0 * a * ds))
        self.prof_s, self.prof_v, self.prof_kappa, self._ds = s, v, kappa, ds

    def v_ref(self, s):
        """Target speed at progress s (wraps around the lap, linear interpolation)."""
        L = self.track.length
        x = (s % L) / self._ds
        i = int(x)
        f = x - i
        n = len(self.prof_v)
        return (1 - f) * self.prof_v[i % n] + f * self.prof_v[(i + 1) % n]

    # ------------------------------------------------------------------ control
    def lookahead_distance(self, v):
        return self.p.ld_min + self.p.k_v * max(v, 0.0)

    def __call__(self, info):
        """One control step: info dict from env.reset/env.step -> [throttle, steering]."""
        car, p = self.car, self.p
        x, y, psi, v, s = info["x"], info["y"], info["psi"], info["vx"], info["s"]
        L = car["wheelbase"]

        # Pure Pursuit geometry is about the REAR axle (x, y is the centre of mass).
        lr = car.get("lr", 0.5 * L)
        xr = x - lr * math.cos(psi)
        yr = y - lr * math.sin(psi)

        # Goal point: walk along the centerline from the car's progress s until it is Ld
        # away from the rear axle. (s + Ld is a good first guess; a few fixed-point steps
        # correct it when the car is off the centerline or the road curves.)
        ld = self.lookahead_distance(v)
        s_goal = s + ld
        for _ in range(3):
            gx_w, gy_w = self.track.point_at(s_goal)
            dist = math.hypot(gx_w - xr, gy_w - yr)
            s_goal += ld - dist
        gx_w, gy_w = self.track.point_at(s_goal)

        # Goal point in the car frame
        dx, dy = gx_w - xr, gy_w - yr
        c, sn = math.cos(psi), math.sin(psi)
        gx, gy = c * dx + sn * dy, -sn * dx + c * dy
        ld2 = max(gx * gx + gy * gy, 1e-6)

        curv = 2.0 * gy / ld2                    # curvature of the arc to the goal point
        steer = math.atan(L * curv)              # bicycle model
        if p.understeer_comp:
            steer *= 1.0 + car["understeer_gradient"] * v * v / L

        # Speed: track the profile a little ahead (the car can't change speed instantly)
        s_prev = s + max(v, 0.0) * p.preview
        v_target = self.v_ref(s_prev)
        # feed-forward: the acceleration the profile itself asks for, a = v dv/ds
        dvds = (self.v_ref(s_prev + 0.2) - self.v_ref(s_prev - 0.2)) / 0.4
        accel = p.k_p * (v_target - v) + v_target * dvds

        self.last = {"s_goal": s_goal, "goal": (gx_w, gy_w), "ld": math.sqrt(ld2),
                     "curv": curv, "steer_rad": steer, "v_target": v_target, "accel": accel}
        return self.env.to_action(accel, steer)

    def reset(self):
        self.last = {}
