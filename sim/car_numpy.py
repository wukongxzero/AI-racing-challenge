"""NumPy stand-in car: dynamic single-track (bicycle) model with tire grip limit.

State (SI, left positive):
    x, y       position of the centre of gravity (m)
    psi        heading (rad)
    vx, vy     forward / sideways speed in the car frame (m/s)
    r          yaw rate (rad/s)
    delta      current front wheel steering angle (rad)

Model:
  * Lateral tire forces: simplified Pacejka  Fy = mu*Fz*sin(C*atan(B*alpha)).
    Peak force mu*Fz  ->  cornering is limited to about mu*g (~1 g).
  * Friction ellipse per axle: longitudinal force uses up lateral grip.
  * All-wheel drive (like the F1TENTH Traxxas 4x4); drive and brake force split
    by static axle load. No reverse: full brake stops the car.
  * Resistances: rolling + aero drag (sets the top speed) + extra drag on grass.
  * Below ~0.8 m/s the model blends into a kinematic bicycle (the dynamic one is
    singular at standstill); its accelerations are capped at the grip limit too.
  * Steering moves toward the commanded angle no faster than delta_rate_max.
  * RK4 integration, `substeps` per 50 ms control step.

Values marked PROVISIONAL are starting guesses (top speed / accel limits are to
be set by the team once the car drives).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

G = 9.81


@dataclass
class CarParams:
    # geometry / inertia (F1TENTH defaults, 1:10 scale)
    mass: float = 3.74            # kg
    Iz: float = 0.04712           # kg m^2
    lf: float = 0.15875           # CG -> front axle (m)
    lr: float = 0.17145           # CG -> rear axle (m)
    length: float = 0.58          # body length (m), for drawing
    width: float = 0.31           # body width (m), for drawing
    # steering (team values)
    delta_max: float = 0.42       # rad
    delta_rate_max: float = 3.2   # rad/s
    # tires
    mu: float = 1.0               # friction coefficient -> ~1 g lateral limit
    B: float = 8.0                # Pacejka stiffness factor
    C: float = 1.5                # Pacejka shape factor
    # longitudinal (PROVISIONAL)
    a_max: float = 5.0            # m/s^2 at throttle +1
    b_max: float = 8.0            # m/s^2 braking at throttle -1
    v_max: float = 8.0            # m/s top speed on tarmac (sets aero drag)
    a_roll: float = 0.2           # m/s^2 rolling resistance
    # grass (off track, PROVISIONAL)
    grass_mu_factor: float = 0.6  # grip multiplier on grass
    grass_drag: float = 2.0       # extra linear drag on grass (1/s)
    # integration
    dt: float = 0.05              # control period (20 Hz)
    substeps: int = 10            # physics steps per control step
    v_blend_low: float = 0.3      # below: pure kinematic model
    v_blend_high: float = 0.8     # above: pure dynamic model
    kin_tau: float = 0.05         # relaxation time of vy, r to kinematic values (s)

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr

    @property
    def c_drag(self) -> float:
        """Aero drag coefficient (1/m) so that full throttle tops out at v_max."""
        return (self.a_max - self.a_roll) / self.v_max ** 2

    def to_dict(self):
        d = asdict(self)
        d["wheelbase"] = self.wheelbase
        d["c_drag"] = self.c_drag
        return d


class NumpyCar:
    """Stand-in car. RacingEnv drives it; controllers should use RacingEnv only."""

    STATE_NAMES = ("x", "y", "psi", "vx", "vy", "r", "delta")

    def __init__(self, params: CarParams | None = None):
        self.p = params or CarParams()
        self.state = [0.0] * 7

    def reset(self, x, y, psi, vx=0.0, vy=0.0, r=0.0, delta=0.0):
        self.state = [float(x), float(y), float(psi), max(0.0, float(vx)), float(vy),
                      float(r), float(delta)]

    # -------------------------------------------------------------- dynamics
    def _deriv(self, s, a_cmd, ddelta, grass):
        p = self.p
        x, y, psi, vx, vy, r, delta = s
        mu = p.mu * (p.grass_mu_factor if grass else 1.0)
        m, L = p.mass, p.wheelbase
        Fzf = m * G * p.lr / L
        Fzr = m * G * p.lf / L

        # longitudinal tire forces from the command (AWD + brakes, split by axle load)
        if a_cmd >= 0.0 or vx > 1e-3:
            Fxf, Fxr = Fzf / (m * G) * m * a_cmd, Fzr / (m * G) * m * a_cmd
        else:
            Fxf = Fxr = 0.0
        Fxf = max(-mu * Fzf, min(mu * Fzf, Fxf))
        Fxr = max(-mu * Fzr, min(mu * Fzr, Fxr))

        # resistances (act on the body, only while moving forward)
        if vx > 1e-3:
            F_res = -m * (p.a_roll + p.c_drag * vx * vx + (p.grass_drag * vx if grass else 0.0))
        else:
            F_res = 0.0

        cpsi, spsi = math.cos(psi), math.sin(psi)
        dx = vx * cpsi - vy * spsi
        dy = vx * spsi + vy * cpsi

        # ---- dynamic model
        w = (vx - p.v_blend_low) / (p.v_blend_high - p.v_blend_low)
        w = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)
        cd, sd = math.cos(delta), math.sin(delta)
        if w > 0.0:
            vxs = max(vx, 1e-3)
            alpha_f = delta - math.atan2(vy + p.lf * r, vxs)
            alpha_r = -math.atan2(vy - p.lr * r, vxs)
            # friction ellipse: remaining lateral capacity
            fyf_cap = math.sqrt(max((mu * Fzf) ** 2 - Fxf ** 2, 0.0))
            fyr_cap = math.sqrt(max((mu * Fzr) ** 2 - Fxr ** 2, 0.0))
            Fyf = fyf_cap * math.sin(p.C * math.atan(p.B * alpha_f))
            Fyr = fyr_cap * math.sin(p.C * math.atan(p.B * alpha_r))
            dvx_d = (Fxr + Fxf * cd - Fyf * sd + F_res) / m + vy * r
            dvy_d = (Fyr + Fxf * sd + Fyf * cd) / m - vx * r
            dr_d = (p.lf * (Fyf * cd + Fxf * sd) - p.lr * Fyr) / p.Iz
        else:
            dvx_d = dvy_d = dr_d = 0.0

        # ---- kinematic model (low speed)
        if w < 1.0:
            ax = (Fxf + Fxr + F_res) / m
            td = math.tan(delta)
            r_kin = vx * td / L
            vy_kin = p.lr * r_kin
            dr_kin = (ax * td + vx * ddelta / (cd * cd)) / L
            dvx_k = ax
            dvy_k = p.lr * dr_kin + (vy_kin - vy) / p.kin_tau
            dr_k = dr_kin + (r_kin - r) / p.kin_tau
            # the kinematic model has no grip limit: cap it at what the tires can do
            ay_max = mu * G
            ay_k = max(-ay_max, min(ay_max, dvy_k + vx * r))
            dvy_k = ay_k - vx * r
            yaw_acc_max = mu * m * G * 2.0 * p.lf * p.lr / L / p.Iz
            dr_k = max(-yaw_acc_max, min(yaw_acc_max, dr_k))
        else:
            dvx_k = dvy_k = dr_k = 0.0

        return (dx, dy, r,
                w * dvx_d + (1 - w) * dvx_k,
                w * dvy_d + (1 - w) * dvy_k,
                w * dr_d + (1 - w) * dr_k,
                ddelta)

    def step(self, throttle, delta_target, grass=False):
        """Advance one control period with throttle in [-1, 1] and a steering
        target in rad (clipped to +-delta_max)."""
        p = self.p
        throttle = max(-1.0, min(1.0, float(throttle)))
        a_cmd = throttle * (p.a_max if throttle >= 0.0 else p.b_max)
        delta_target = max(-p.delta_max, min(p.delta_max, float(delta_target)))
        h = p.dt / p.substeps
        s = self.state
        for _ in range(p.substeps):
            ddelta = (delta_target - s[6]) / h
            ddelta = max(-p.delta_rate_max, min(p.delta_rate_max, ddelta))
            k1 = self._deriv(s, a_cmd, ddelta, grass)
            k2 = self._deriv([a + 0.5 * h * b for a, b in zip(s, k1)], a_cmd, ddelta, grass)
            k3 = self._deriv([a + 0.5 * h * b for a, b in zip(s, k2)], a_cmd, ddelta, grass)
            k4 = self._deriv([a + h * b for a, b in zip(s, k3)], a_cmd, ddelta, grass)
            s = [a + h / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4)
                 for a, b1, b2, b3, b4 in zip(s, k1, k2, k3, k4)]
            s[6] = max(-p.delta_max, min(p.delta_max, s[6]))
            if s[3] < 0.0:          # no reverse: brakes stop the car
                s[3] = 0.0
            if s[3] == 0.0 and a_cmd <= 0.0:
                s[4] = 0.0
                s[5] = 0.0
        s[2] = (s[2] + math.pi) % (2.0 * math.pi) - math.pi
        self.state = s
        return self.state_dict()

    def state_dict(self):
        return dict(zip(self.STATE_NAMES, self.state))
