"""The car's equations as CasADi functions, for model-based controllers (MPCC).

These are the SAME equations the simulator runs (sim/car_numpy.py, our models document
sections 3.2-3.3); a test checks that both give identical numbers. Requires `casadi`.

    from sim.car_casadi import dynamic_model, kinematic_model, discretize
    f = dynamic_model(env.car_params)        # xdot = f(x, u)
    F = discretize(f, dt=0.05)               # x_next = F(x, u): one 50 ms step, RK4

States and inputs (SI, left positive):
    dynamic:    x = [X, Y, psi, vx, vy, r]     u = [a, delta]
    kinematic:  x = [X, Y, psi, v]             u = [a, delta]
    a = longitudinal acceleration (m/s^2, negative = braking), delta = front wheel angle (rad).

How this differs from the simulator (only matters at the edges):
  * delta is an input here; in the simulator it is a state that moves toward the command at
    most 3.2 rad/s. In an MPC, add the constraint |delta_k - delta_{k-1}| <= 3.2 * dt and
    start from info["delta"].
  * The simulator blends to the kinematic model below 1 m/s, stops at 0 m/s (no reverse),
    caps speed at v_max and adds grass drag off track. Keep a, delta within car_params limits.
"""
from __future__ import annotations

import math
from dataclasses import fields

from .car_numpy import G, CarParams

try:
    import casadi as ca
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("sim.car_casadi needs `pip install casadi`") from exc


def _params(params) -> CarParams:
    """Accept CarParams, env.car_params (a dict) or None (defaults)."""
    if params is None:
        return CarParams()
    if isinstance(params, CarParams):
        return params
    names = {f.name for f in fields(CarParams)}
    return CarParams(**{k: v for k, v in dict(params).items() if k in names})


def pacejka(alpha, B, C, D, E):
    """Lateral tire force (N): D sin(C atan(B a - E (B a - atan(B a))))."""
    Ba = B * alpha
    return D * ca.sin(C * ca.atan(Ba - E * (Ba - ca.atan(Ba))))


def dynamic_model(params=None, name="car_dynamic"):
    """Dynamic bicycle model with Pacejka tires: casadi.Function xdot = f(x[6], u[2])."""
    p = _params(params)
    x = ca.SX.sym("x", 6)
    u = ca.SX.sym("u", 2)
    psi, vx, vy, r = x[2], x[3], x[4], x[5]
    a, delta = u[0], u[1]
    L = p.wheelbase
    Fzf = p.mass * G * p.lr / L                     # static axle loads
    Fzr = p.mass * G * p.lf / L
    vxs = ca.fmax(vx, 1e-3)                         # same guard as the simulator
    alpha_f = delta - ca.atan2(vy + p.lf * r, vxs)  # slip angles
    alpha_r = -ca.atan2(vy - p.lr * r, vxs)
    Fyf = pacejka(alpha_f, p.B_f, p.C_f, p.mu * Fzf, p.E_f)
    Fyr = pacejka(alpha_r, p.B_r, p.C_r, p.mu * Fzr, p.E_r)
    xdot = ca.vertcat(
        vx * ca.cos(psi) - vy * ca.sin(psi),
        vx * ca.sin(psi) + vy * ca.cos(psi),
        r,
        a - Fyf * ca.sin(delta) / p.mass + vy * r,
        (Fyf * ca.cos(delta) + Fyr) / p.mass - vx * r,
        (p.lf * Fyf * ca.cos(delta) - p.lr * Fyr) / p.Iz,
    )
    return ca.Function(name, [x, u], [xdot], ["x", "u"], ["xdot"])


def kinematic_model(params=None, name="car_kinematic"):
    """Kinematic bicycle model (no tire slip): casadi.Function xdot = f(x[4], u[2])."""
    p = _params(params)
    x = ca.SX.sym("x", 4)
    u = ca.SX.sym("u", 2)
    psi, v = x[2], x[3]
    a, delta = u[0], u[1]
    beta = ca.atan(p.lr / p.wheelbase * ca.tan(delta))   # velocity angle at the centre of mass
    xdot = ca.vertcat(
        v * ca.cos(psi + beta),
        v * ca.sin(psi + beta),
        v / p.lr * ca.sin(beta),
        a,
    )
    return ca.Function(name, [x, u], [xdot], ["x", "u"], ["xdot"])


def discretize(f, dt=0.05, substeps=10, name=None):
    """One control step of length dt as RK4 with `substeps` sub-steps, u held constant
    (the simulator uses 10 x 5 ms per 50 ms step). Returns casadi.Function x_next = F(x, u)."""
    nx = f.size1_in(0)
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", f.size1_in(1))
    h = dt / substeps
    z = x
    for _ in range(substeps):
        k1 = f(z, u)
        k2 = f(z + 0.5 * h * k1, u)
        k3 = f(z + 0.5 * h * k2, u)
        k4 = f(z + h * k3, u)
        z = z + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ca.Function(name or f"{f.name()}_rk4", [x, u], [z], ["x", "u"], ["x_next"])


def state_from_info(info, model="dynamic"):
    """The current state for these models, from a RacingEnv info dict."""
    if model == "dynamic":
        return [info["x"], info["y"], info["psi"], info["vx"], info["vy"], info["yaw_rate"]]
    return [info["x"], info["y"], info["psi"], math.hypot(info["vx"], info["vy"])]
