"""Dynamic bicycle MPCC using the team's Pacejka/CasADi model.

State: [X,Y,psi,vx,vy,yaw_rate,progress] (7)
Decision input: [acceleration, steering_target, virtual_progress_rate] (3)
The simulator imposes a steering slew limit and uses a low-speed blend; this
controller constrains steering targets and hands off to kinematic MPCC below
1.5 m/s. Local reference boundaries are linearized once per control cycle.
"""
from dataclasses import dataclass
import numpy as np
import casadi as ca
from sim.car_casadi import dynamic_model, discretize
from .run_racing_env import SharedTrackAdapter, controller_from_env


@dataclass
class DynamicConfig:
    horizon: int = 8
    speed: float = 3.0
    substeps: int = 3
    iterations: int = 120
    margin_extra: float = 0.03
    dual_warm_start: bool = False


class DynamicMPCC:
    def __init__(self, env, config=None):
        self.env = env
        self.track = SharedTrackAdapter(env.track)
        self.cfg = config or DynamicConfig()
        p = env.car_params
        self.dt = env.dt
        self.N = self.cfg.horizon
        self.speed = min(self.cfg.speed, p['v_max'])
        self.margin = p['width'] / 2 + self.cfg.margin_extra
        # Keep a known-good, physically consistent start controller for low speed.
        self.launch = controller_from_env(env, horizon=12, speed_cap=self.speed)
        f = dynamic_model(p, name='mpcc_dynamic_rhs')
        F = discretize(f, dt=self.dt, substeps=self.cfg.substeps, name='mpcc_dynamic_rk4')
        N = self.N
        opt = ca.Opti()
        X = opt.variable(7, N+1)
        U = opt.variable(3, N)
        x0 = opt.parameter(7)
        steer_now = opt.parameter()
        # center x/y, tangent x/y, unwrapped s, widths L/R, target speed
        ref = opt.parameter(8, N+1)
        opt.subject_to(X[:, 0] == x0)
        obj = 0
        for k in range(N):
            z, u = X[:,k], U[:,k]
            a, delta, vs = u[0], u[1], u[2]
            zn = F(z[:6], ca.vertcat(a, delta))
            opt.subject_to(X[:,k+1] == ca.vertcat(zn, z[6] + self.dt * vs))
            opt.subject_to(opt.bounded(-p['b_max'], a, p['a_max']))
            opt.subject_to(opt.bounded(-p['delta_max'], delta, p['delta_max']))
            opt.subject_to(opt.bounded(0, vs, self.speed))
            prev = steer_now if k == 0 else U[1,k-1]
            opt.subject_to(opt.bounded(-p['delta_rate_max']*self.dt, delta-prev,
                                        p['delta_rate_max']*self.dt))
            ec, el = self._errors(z, ref[:,k])
            opt.subject_to(ec <= ref[5,k]-self.margin)
            opt.subject_to(ec >= -ref[6,k]+self.margin)
            obj += (220*ec**2 + 40*el**2 + 12*(z[3]-ref[7,k])**2
                    + .03*a*a + .25*delta*delta + 15*(delta-prev)**2
                    + .4*z[4]**2 + .05*z[5]**2 + .08*(vs-z[3])**2
                    -4*vs*self.dt)
        ec, el = self._errors(X[:,N], ref[:,N])
        opt.subject_to(ec <= ref[5,N]-self.margin)
        opt.subject_to(ec >= -ref[6,N]+self.margin)
        obj += 400*ec**2 + 40*el**2
        opt.subject_to(opt.bounded(0.7, X[3,:], self.speed + 0.15))
        opt.subject_to(opt.bounded(-2.0, X[4,:], 2.0))
        opt.subject_to(opt.bounded(-8.0, X[5,:], 8.0))
        opt.minimize(obj)
        opt.solver('ipopt', {'print_time':False}, {'print_level':0, 'sb':'yes',
                   'max_iter':self.cfg.iterations, 'tol':1e-4, 'acceptable_tol':2e-3,
                   'warm_start_init_point':'yes' if self.cfg.dual_warm_start else 'no'})
        self.opti, self.X, self.U, self.x0, self.steer_now, self.ref = opt,X,U,x0,steer_now,ref
        self.guess_x = None
        self.guess_u = None
        self.last_status = None
        self.guess_lam = None

    @staticmethod
    def _errors(z, r):
        dx, dy = z[0]-r[0], z[1]-r[1]
        return -r[3]*dx+r[2]*dy, r[2]*dx+r[3]*dy-(z[6]-r[4])

    def _references(self, ss):
        ss = np.asarray(ss, dtype=float)
        centers = self.track.center(ss)
        ang = self.track.heading(ss)
        tan = np.stack([np.cos(ang),np.sin(ang)],axis=-1)
        # Reuse speed planning from the validated kinematic baseline.
        return np.vstack([centers.T, tan.T, ss, self.track._wl(ss),
                          self.track._wr(ss), self.launch.speed_profile(ss)])

    def solve(self, state, progress, steer_actual):
        z0 = np.r_[np.asarray(state, dtype=float),float(progress)]
        N, dt = self.N, self.dt
        if self.guess_x is None:
            ss = progress + np.arange(N+1)*dt*max(state[3],1.0)
            xy = self.track.center(ss)
            heads = np.unwrap(self.track.heading(ss))
            heads += 2*np.pi*np.round((state[2]-heads[0])/(2*np.pi))
            X0 = np.vstack([xy.T, heads, np.full(N+1,max(1.0,state[3])),
                            np.zeros(N+1), np.zeros(N+1), ss])
            X0[:,0] = z0
            U0 = np.zeros((3,N))
            U0[2] = max(state[3],1.0)
            k = self.track.curvature(ss[:-1])
            U0[1] = np.clip(np.arctan(self.env.car_params['wheelbase']*k),
                            -self.env.car_params['delta_max'],self.env.car_params['delta_max'])
        else:
            X0 = np.column_stack([self.guess_x[:,1:],self.guess_x[:,-1]+(
                                  self.guess_x[:,-1]-self.guess_x[:,-2])])
            X0[:,0] = z0
            X0[6] = np.maximum.accumulate(X0[6])
            U0 = np.column_stack([self.guess_u[:,1:],self.guess_u[:,-1]])
        opt = self.opti
        opt.set_value(self.x0,z0)
        opt.set_value(self.steer_now,float(steer_actual))
        opt.set_value(self.ref,self._references(X0[6]))
        opt.set_initial(self.X,X0)
        opt.set_initial(self.U,U0)
        if self.cfg.dual_warm_start and self.guess_lam is not None:
            opt.set_initial(opt.lam_g, self.guess_lam)
        sol = opt.solve()
        xp,up = np.asarray(sol.value(self.X)),np.asarray(sol.value(self.U))
        self.guess_x,self.guess_u=xp,up
        if self.cfg.dual_warm_start:
            self.guess_lam = np.asarray(sol.value(opt.lam_g)).reshape(-1)
        self.last_status='Solve_Succeeded'
        return up[:2,0].copy(),xp,up
