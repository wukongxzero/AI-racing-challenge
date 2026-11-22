"""NumPy car: dynamic bicycle model with Pacejka tires (the team's planned model).

State (SI, left positive):
    x, y       position of the centre of mass (m)
    psi        heading (rad)
    vx, vy     forward / sideways speed in the car frame (m/s)
    r          yaw rate (rad/s)
    delta      current front wheel angle (rad) - part of the state because it is rate-limited

Inputs each physics step: acceleration a (m/s^2, negative = braking) and a target
steering angle. The wheels turn toward the target at most delta_rate_max.

Equations (dynamic model):
    slip angles  alpha_f = delta - atan((vy + lf*r) / vx),  alpha_r = -atan((vy - lr*r) / vx)
    tire force   Fy = D sin(C atan(B a - E (B a - atan(B a))))     (Pacejka "Magic Formula")
                 D = mu * axle load  ->  the force has a maximum = the grip limit
    x' = vx cos psi - vy sin psi        y' = vx sin psi + vy cos psi        psi' = r
    vx' = a - Fyf sin(delta)/m + vy r   (the front tire's side force slows the car in corners)
    vy' = (Fyf cos(delta) + Fyr)/m - vx r
    r'  = (lf Fyf cos(delta) - lr Fyr) / Iz

Low speed: the slip angles divide by vx, so the dynamic model breaks down near standstill.
Below 0.3 m/s total speed we use the kinematic bicycle (no tire slip); between 0.3 and 1 m/s
we blend the two results; above 1 m/s it is purely dynamic. Total speed (not just forward
speed) decides, so a car sliding sideways after a spin stays in the dynamic model and is
slowed by its tires; the kinematic model keeps only the forward speed.

Known simplifications (state them in the report): no load transfer, no longitudinal tire
slip, braking/accelerating does not reduce cornering grip, no air drag (top speed is a hard
limit). On grass the grip drops and extra drag slows the car.

Integration: RK4 with 5 ms physics steps inside each 50 ms control step.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

G = 9.81


@dataclass
class CarParams:
    # geometry / inertia: F1TENTH 1:10 car (f1tenth_gym defaults)
    mass: float = 3.74            # kg
    Iz: float = 0.04712           # kg m^2
    lf: float = 0.15875           # centre of mass -> front axle (m)
    lr: float = 0.17145           # centre of mass -> rear axle (m)
    length: float = 0.58          # body length (m), for drawing
    width: float = 0.31           # body width (m), for off-track check and drawing
    # steering (team values)
    delta_max: float = 0.42       # rad
    delta_rate_max: float = 3.2   # rad/s
    # tires: tire model (simplified Pacejka, no E term) from Liniger, Domahidi & Morari (2015).
    # The paper prints no numbers; B, C (and the D values used to get mu) are the identified
    # 1:43 car values in the authors' public code, github.com/alexliniger/MPCC
    # (C++/Params/model.json; checked Nov 22). Peak force D = mu * axle load for our car;
    # mu 0.9 = D / axle load of their car (front 0.192 N / 0.214 N, rear 0.1737 N / 0.188 N).
    mu: float = 0.9
    B_f: float = 2.579
    C_f: float = 1.2
    E_f: float = 0.0
    B_r: float = 3.3852
    C_r: float = 1.2691
    E_r: float = 0.0
    # longitudinal (PROVISIONAL team values)
    a_max: float = 5.0            # m/s^2 at throttle +1
    b_max: float = 8.0            # m/s^2 braking at throttle -1
    v_max: float = 8.0            # m/s hard top speed
    # grass (off track, PROVISIONAL)
    grass_mu_factor: float = 0.6  # grip multiplier on grass
    grass_drag: float = 2.0       # extra slowing on grass: a -= grass_drag * vx  (1/s)
    # integration
    dt: float = 0.05              # control period (20 Hz)
    substeps: int = 10            # physics steps per control step (5 ms each)
    v_blend_low: float = 0.3      # below: kinematic model only
    v_blend_high: float = 1.0     # above: dynamic model only

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr

    @property
    def a_lat_max(self) -> float:
        """Usable sideways grip (m/s^2): the most the car reaches in steady cornering.
        Below mu*g because the soft tires don't reach their peak within the steering
        limit; 0.89 was measured from steady circles (checked by a test)."""
        return 0.89 * self.mu * G

    @property
    def understeer_gradient(self) -> float:
        """How much the car understeers, K (rad per m/s^2 of sideways acceleration): to drive a
        circle of radius R at speed v the wheels need delta = L/R + K * v^2/R. Textbook formula
        from the tires' small-slip cornering stiffness C = B*C*D per axle:
            K = m/L * (lr/C_front - lf/C_rear).
        Matches the simulated car within ~2 % in normal cornering (a test checks it); a bit
        larger near the grip limit."""
        L = self.wheelbase
        c_front = self.B_f * self.C_f * self.mu * self.mass * G * self.lr / L
        c_rear = self.B_r * self.C_r * self.mu * self.mass * G * self.lf / L
        return self.mass / L * (self.lr / c_front - self.lf / c_rear)

    def to_dict(self):
        d = asdict(self)
        d["wheelbase"] = self.wheelbase
        d["a_lat_max"] = self.a_lat_max
        d["understeer_gradient"] = self.understeer_gradient
        return d


def pacejka(alpha, B, C, D, E):
    """Lateral tire force (N) for slip angle alpha (rad)."""
    Ba = B * alpha
    return D * math.sin(C * math.atan(Ba - E * (Ba - math.atan(Ba))))


class NumpyCar:
    """Stand-in car. RacingEnv drives it; controllers should use RacingEnv only.

    Interface used by RacingEnv (a MuJoCo car must provide the same):
        reset(x, y, psi, vx), step(throttle, delta_target, grass), state (list of 7 numbers)
    """

    STATE_NAMES = ("x", "y", "psi", "vx", "vy", "r", "delta")

    def __init__(self, params: CarParams | None = None):
        self.p = params or CarParams()
        self.state = [0.0] * 7

    def reset(self, x, y, psi, vx=0.0, vy=0.0, r=0.0, delta=0.0):
        self.state = [float(x), float(y), float(psi), max(0.0, float(vx)), float(vy),
                      float(r), float(delta)]

    # ---------------------------------------------------------------- models
    def _dynamic(self, s, a, ddelta, mu):
        """Time derivative of [x, y, psi, vx, vy, r, delta] (dynamic model)."""
        p = self.p
        x, y, psi, vx, vy, r, delta = s
        L = p.wheelbase
        Fzf = p.mass * G * p.lr / L                  # static axle loads
        Fzr = p.mass * G * p.lf / L
        vxs = max(vx, 1e-3)
        alpha_f = delta - math.atan2(vy + p.lf * r, vxs)
        alpha_r = -math.atan2(vy - p.lr * r, vxs)
        Fyf = pacejka(alpha_f, p.B_f, p.C_f, mu * Fzf, p.E_f)
        Fyr = pacejka(alpha_r, p.B_r, p.C_r, mu * Fzr, p.E_r)
        c, s_ = math.cos(psi), math.sin(psi)
        cd, sd = math.cos(delta), math.sin(delta)
        return (vx * c - vy * s_,
                vx * s_ + vy * c,
                r,
                a - Fyf * sd / p.mass + vy * r,
                (Fyf * cd + Fyr) / p.mass - vx * r,
                (p.lf * Fyf * cd - p.lr * Fyr) / p.Iz,
                ddelta)

    def _kinematic(self, s, a, ddelta):
        """Time derivative of [x, y, psi, v, delta] (kinematic model: no tire slip)."""
        p = self.p
        x, y, psi, v, delta = s
        beta = math.atan(p.lr / p.wheelbase * math.tan(delta))   # velocity angle at CoM
        return (v * math.cos(psi + beta),
                v * math.sin(psi + beta),
                v / p.lr * math.sin(beta),
                a,
                ddelta)

    @staticmethod
    def _rk4(f, s, h):
        k1 = f(s)
        k2 = f([a + 0.5 * h * b for a, b in zip(s, k1)])
        k3 = f([a + 0.5 * h * b for a, b in zip(s, k2)])
        k4 = f([a + h * b for a, b in zip(s, k3)])
        return [a + h / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4)
                for a, b1, b2, b3, b4 in zip(s, k1, k2, k3, k4)]

    # ------------------------------------------------------------------ step
    def step(self, throttle, delta_target, grass=False):
        """Advance one control period (50 ms) with throttle in [-1, 1] and a steering
        target in rad (clipped to +-delta_max)."""
        p = self.p
        throttle = max(-1.0, min(1.0, float(throttle)))
        a_cmd = throttle * (p.a_max if throttle >= 0.0 else p.b_max)
        delta_target = max(-p.delta_max, min(p.delta_max, float(delta_target)))
        mu = p.mu * (p.grass_mu_factor if grass else 1.0)
        h = p.dt / p.substeps
        s = self.state

        for _ in range(p.substeps):
            vx = s[3]
            # acceleration actually applied: no reverse, hard top speed, grass drag
            a = a_cmd
            if a < 0.0 and vx <= 1e-3:
                a = 0.0                               # brakes hold the car, never reverse
            if a > 0.0 and vx >= p.v_max:
                a = 0.0                               # top speed reached
            if grass:
                a -= p.grass_drag * vx
            # steering moves toward the target, at most delta_rate_max
            ddelta = max(-p.delta_rate_max, min(p.delta_rate_max, (delta_target - s[6]) / h))

            # blend factor from TOTAL speed: 0 = kinematic only, 1 = dynamic only
            lam = (math.hypot(vx, s[4]) - p.v_blend_low) / (p.v_blend_high - p.v_blend_low)
            lam = min(1.0, max(0.0, lam))

            if lam > 0.0:
                dyn = self._rk4(lambda z: self._dynamic(z, a, ddelta, mu), s, h)
            if lam < 1.0:
                v = max(s[3], 0.0)            # forward speed only: no tire slip, sideways is lost
                k = self._rk4(lambda z: self._kinematic(z, a, ddelta), [s[0], s[1], s[2], v, s[6]], h)
                beta = math.atan(p.lr / p.wheelbase * math.tan(k[4]))
                kin = [k[0], k[1], k[2], k[3] * math.cos(beta), k[3] * math.sin(beta),
                       k[3] / p.lr * math.sin(beta), k[4]]
            if lam >= 1.0:
                s = dyn
            elif lam <= 0.0:
                s = kin
            else:
                s = [lam * d + (1.0 - lam) * k for d, k in zip(dyn, kin)]

            s[6] = max(-p.delta_max, min(p.delta_max, s[6]))
            s[3] = min(max(s[3], 0.0), p.v_max)       # no reverse, hard top speed
            if s[3] == 0.0:                           # stopped: no sliding either
                s[4] = s[5] = 0.0

        s[2] = (s[2] + math.pi) % (2.0 * math.pi) - math.pi
        self.state = s
        return self.state_dict()

    def state_dict(self):
        return dict(zip(self.STATE_NAMES, self.state))
