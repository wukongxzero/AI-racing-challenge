"""Step 8: track-agnostic adaptive-speed local-reference MPCC prototype.

Run: python3 mpcc.py training_track.csv --steps 120
Dependencies: numpy scipy matplotlib casadi

This is a learning prototype with a kinematic bicycle vehicle model. Track
geometry is locally linearized at each MPC update; it is NOT a globally exact
nonlinear track-boundary formulation. Vehicle parameters are placeholders.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import casadi as ca
from track import Track
from bicycle_model import BicycleModel, CarParams


@dataclass
class MPCCParams:
    horizon: int = 12
    q_contour: float = 220.0
    q_lag: float = 40.0
    q_progress: float = 4.0
    q_accel: float = 0.02
    q_steer: float = 0.20
    q_steer_change: float = 20.0
    q_vs: float = 0.04
    q_terminal_contour: float = 400.0
    q_speed: float = 12.0
    lateral_accel_limit: float = 1.6  # m/s^2; illustrative grip constraint
    speed_floor: float = 0.8  # m/s, keep progress through tight turns
    speed_buffer: float = 0.70 # m/s, below geometric lateral acceleration threshold
    safety_margin: float = 0.15  # m, CENTER clearance only (car width not modeled)
    solver_iterations: int = 160
    speed_accel_limit: float = 1.5  # smooth spatial speed changes, m/s^2


class MPCC:
    """Solve a 20-Hz receding horizon problem using IPOPT via CasADi.

    z = [x, y, heading, speed, progress]; u = [accel, steer, progress_rate].
    The applied steering angle is tracked by BicycleModel, and the last
    applied angle is used to enforce the first step's rate limit.
    """
    def __init__(self, track: Track, car_params=None, mpcc_params=None):
        self.track = track
        self.car = car_params or CarParams()
        self.cfg = mpcc_params or MPCCParams()
        self.N = self.cfg.horizon
        self.dt = self.car.dt
        N, dt, p, c = self.N, self.dt, self.car, self.cfg
        opti = ca.Opti()
        X = opti.variable(5, N + 1)
        U = opti.variable(3, N)
        x0 = opti.parameter(5)
        last_steer = opti.parameter()
        ref = opti.parameter(8, N + 1)  # cx,cy, tx,ty, sref, wl,wr, speed target
        opti.subject_to(X[:, 0] == x0)
        objective = 0
        for k in range(N):
            z, u = X[:, k], U[:, k]
            a, steer, vs = u[0], u[1], u[2]
            # Predict same kinematic model with RK4. Steering is assumed fixed
            # during a step; slew limiting is modeled by constraints below.
            def f(w):
                return ca.vertcat(w[3]*ca.cos(w[2]),
                                  w[3]*ca.sin(w[2]),
                                  w[3]/p.wheelbase*ca.tan(steer), a, vs)
            k1 = f(z)
            k2 = f(z + 0.5*dt*k1)
            k3 = f(z + 0.5*dt*k2)
            k4 = f(z + dt*k3)
            opti.subject_to(X[:, k+1] == z + dt*(k1+2*k2+2*k3+k4)/6)
            opti.subject_to(opti.bounded(p.min_accel, a, p.max_accel))
            opti.subject_to(opti.bounded(-p.max_steer, steer, p.max_steer))
            opti.subject_to(opti.bounded(0, vs, p.max_speed))
            previous = last_steer if k == 0 else U[1, k-1]
            opti.subject_to(opti.bounded(-p.max_steer_rate*dt,
                                         steer-previous, p.max_steer_rate*dt))
            # Position errors in a local orthonormal frame. The lag error
            # accounts for the optimized progress state s.
            dx = z[0]-ref[0,k]
            dy = z[1]-ref[1,k]
            tx, ty = ref[2,k], ref[3,k]
            ec = -ty*dx + tx*dy
            el = tx*dx + ty*dy - (z[4]-ref[4,k])
            opti.subject_to(ec <= ref[5,k]-c.safety_margin)
            opti.subject_to(ec >= -ref[6,k]+c.safety_margin)
            objective += (c.q_speed*(z[3]-ref[7,k])**2
                          + c.q_contour*ec**2 + c.q_lag*el**2
                          -c.q_progress*vs*dt + c.q_accel*a**2
                          + c.q_steer*steer**2 + c.q_vs*(vs-z[3])**2
                          + c.q_steer_change*(steer-previous)**2)
        # Terminal errors and boundary constraints
        dx = X[0,N]-ref[0,N]; dy = X[1,N]-ref[1,N]
        tx = ref[2,N]; ty = ref[3,N]
        ec = -ty*dx+tx*dy
        el = tx*dx+ty*dy-(X[4,N]-ref[4,N])
        opti.subject_to(ec <= ref[5,N]-c.safety_margin)
        opti.subject_to(ec >= -ref[6,N]+c.safety_margin)
        objective += c.q_terminal_contour*ec**2+c.q_lag*el**2
        opti.subject_to(opti.bounded(0, X[3,:], p.max_speed))
        opti.minimize(objective)
        opti.solver('ipopt', {'print_time': False},
                    {'print_level': 0, 'max_iter': c.solver_iterations,
                     'tol': 1e-4, 'acceptable_tol': 2e-3,
                     'sb': 'yes'})
        self.opti, self.X, self.U = opti, X, U
        self.x0, self.last_steer, self.ref = x0, last_steer, ref
        self.guess_x = None
        self.guess_u = None
        self.last_solution = None
        self._build_speed_profile()

    def _build_speed_profile(self):
        M = max(1600, int(self.track.length / 0.075))
        grid = np.linspace(0, self.track.length, M, endpoint=False)
        ds = self.track.length / M
        kappa = np.abs(self.track.curvature(grid))
        v = np.sqrt(self.cfg.lateral_accel_limit / np.maximum(kappa, 1e-5))
        v = np.clip(v-self.cfg.speed_buffer, self.cfg.speed_floor, self.car.max_speed)
        # Closed-loop forward/backward passes enforce acceleration and braking
        # reachability without raising the original curvature speed limits.
        for _ in range(4):
            for j in range(M):
                prv = (j-1) % M
                v[j] = min(v[j], np.sqrt(v[prv]**2 + 2*self.cfg.speed_accel_limit*ds))
            for j in range(M-1, -1, -1):
                nxt = (j+1) % M
                v[j] = min(v[j], np.sqrt(v[nxt]**2 + 2*(-self.car.min_accel)*ds))
        self._speed_grid = np.r_[grid, self.track.length]
        self._speed_values = np.r_[v,v[0]]

    def speed_profile(self, s_values):
        return np.interp(np.mod(s_values,self.track.length),
                         self._speed_grid, self._speed_values)

    def references(self, s_values):
        ss = np.asarray(s_values, dtype=float)
        center = self.track.center(ss)
        tangents = self.track.tangent(ss)
        return np.vstack((center.T, tangents.T, ss,
                          self.track._wl(ss), self.track._wr(ss),
                          self.speed_profile(ss)))

    def solve(self, state, progress, steer_actual):
        """Return [acceleration, commanded steering] and planned trajectories.

        Exceptions are raised if the solver fails. The caller must handle
        failure safely (e.g., brake/stop rather than apply stale commands).
        """
        z0 = np.r_[np.asarray(state,dtype=float), progress]
        N, dt = self.N, self.dt
        if self.guess_x is None:
            predicted_s = progress + np.arange(N+1)*dt*max(state[3], 1.0)
            centers = self.track.center(predicted_s)
            headings = np.unwrap(self.track.heading(predicted_s))
            headings += 2*np.pi*np.round((state[2]-headings[0])/(2*np.pi))
            X0 = np.vstack((centers.T, headings,
                            np.full(N+1, max(state[3], 1.0)), predicted_s))
            X0[:,0] = z0
            U0 = np.zeros((3,N)); U0[2,:] = max(1.0, state[3])
        else:
            X0 = np.column_stack((self.guess_x[:,1:],
                                  self.guess_x[:,-1] +
                                  (self.guess_x[:,-1]-self.guess_x[:,-2])))
            X0[:,0] = z0
            # Avoid accidentally moving progress backwards across lap boundary
            X0[4,:] = np.maximum.accumulate(X0[4,:])
            U0 = np.column_stack((self.guess_u[:,1:],self.guess_u[:,-1]))
        opti = self.opti
        opti.set_value(self.x0,z0)
        opti.set_value(self.last_steer,float(steer_actual))
        # The local reference is frozen for one optimization pass. At the next
        # 20 Hz update it is rebuilt around the new prediction.
        opti.set_value(self.ref,self.references(X0[4,:]))
        opti.set_initial(self.X,X0)
        opti.set_initial(self.U,U0)
        solution = opti.solve()
        predicted = np.asarray(solution.value(self.X))
        actions = np.asarray(solution.value(self.U))
        self.guess_x,self.guess_u = predicted,actions
        self.last_solution = (predicted,actions)
        return actions[:2,0].copy(), predicted, actions


def safe_fallback(track, car, state):
    """Emergency deceleration with bounded geometric steering feedback.

    This is a contingency controller, not a collision-avoidance guarantee.
    """
    s = track.project(state[0], state[1])
    lateral, _ = track.errors(state[0], state[1], s)
    desired_heading = float(track.heading(s))
    heading_error = (desired_heading - state[2] + np.pi) % (2*np.pi) - np.pi
    curvature = float(track.curvature(s))
    steer = np.arctan(car.wheelbase*curvature) + 1.5*heading_error - 1.0*lateral
    return np.array([car.min_accel, np.clip(steer, -car.max_steer, car.max_steer)])


def run_demo(csv_path, steps=1800, output_path=Path('mpcc_preview.png'), max_speed=3.0):
    import matplotlib.pyplot as plt
    track = Track(csv_path)
    car_params = CarParams(max_speed=max_speed)
    vehicle = BicycleModel(car_params)
    controller = MPCC(track, car_params)
    state = np.r_[track.center(0), track.heading(0), 0.0]
    states = [state.copy()]
    times, errors = [], []
    progress = 0.0
    virtual_progress = 0.0
    fallback_count = 0
    lap_completed = False
    failure = None
    prev_projection = track.project(state[0], state[1])
    for i in range(steps):
        t0 = time.perf_counter()
        try:
            action, predicted, commands = controller.solve(state, virtual_progress, vehicle.steering)
            virtual_progress = float(predicted[4,1])
        except (RuntimeError, ValueError) as e:
            # Avoid applying an old high-speed command if IPOPT fails.
            fallback_count += 1
            errors.append((i, str(e).splitlines()[0]))
            controller.guess_x = None
            controller.guess_u = None
            action = safe_fallback(track, car_params, state)
        times.append(time.perf_counter()-t0)
        state = vehicle.step(state, action)
        states.append(state.copy())
        projection = track.project(state[0],state[1])
        # Unwrap the measured track position. Reject backward leaps to avoid
        # counting the virtual progress optimization as a physical lap.
        ds = ((projection-prev_projection+track.length/2) % track.length
              -track.length/2)
        progress += ds
        prev_projection = projection
        ec, _ = track.errors(state[0],state[1],projection)
        if i%20 == 0:
            print(f'step {i:4d} | v={state[3]:.2f} m/s | '
                  f'ec={ec:+.3f} m | measured_progress={progress:.1f} m | '
                  f'solve={times[-1]*1000:.0f} ms | fallbacks={fallback_count}')
        if not track.within_bounds(state[0],state[1],
                                   car_half_width=controller.cfg.safety_margin):
            failure = f'Car center crossed track safety margin at step {i}'
            print(failure)
            break
        if progress >= track.length and state[3] > 0.1:
            lap_completed = True
            print(f'Physical lap completed at step {i+1} (~{(i+1)*car_params.dt:.2f} s simulated)')
            break
    history = np.asarray(states)
    s = np.linspace(0,track.length,1200)
    c = track.center(s); left,right = track.boundaries(s)
    fig,ax = plt.subplots(figsize=(10,7))
    ax.plot(left[:,0],left[:,1],'--',alpha=0.6,label='Left boundary')
    ax.plot(right[:,0],right[:,1],'--',alpha=0.6,label='Right boundary')
    ax.plot(c[:,0],c[:,1],alpha=0.5,label='Centerline')
    ax.plot(history[:,0],history[:,1],linewidth=2,label='MPCC driven path')
    ax.scatter(history[0,0],history[0,1],s=55,label='Start')
    ax.scatter(history[-1,0],history[-1,1],s=55,label='End')
    ax.set_aspect('equal');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    ax.set_title('Step 8 — Generalized MPCC (kinematic simulation)')
    ax.legend();ax.grid(alpha=0.2);fig.tight_layout()
    fig.savefig(output_path,dpi=160);plt.close(fig)
    print(f'Completed {len(history)-1}/{steps} steps; '
          f'measured_progress={progress:.2f}/{track.length:.2f} m; '
          f'final speed={state[3]:.2f} m/s; fallbacks={fallback_count}')
    if times:
        ts = np.array(times)*1000
        print(f'Mean/max solve time: {ts.mean():.1f}/{ts.max():.1f} ms; '
              f'over 50 ms: {np.sum(ts>50)}/{len(ts)}')
    print(f'Saved plot: {output_path}')
    sp = np.linspace(0,track.length,1500)
    f2,a2 = plt.subplots(figsize=(9,4))
    a2.plot(sp,controller.speed_profile(sp),label='Curvature-based speed target')
    actual_s = np.unwrap(np.array([track.project(x,y) for x,y in history[:,:2]])*2*np.pi/track.length)*track.length/(2*np.pi)
    actual_s -= actual_s[0]
    a2.plot(actual_s,history[:,3],label='Actual speed',alpha=0.85)
    a2.set(xlabel='Measured track progress (m)', ylabel='Speed (m/s)',title='Speed plan vs. actual speed')
    a2.legend();a2.grid(alpha=0.25);f2.tight_layout()
    speed_plot = output_path.with_name(output_path.stem + '_speed.png')
    f2.savefig(speed_plot,dpi=150);plt.close(f2)
    print(f'Saved speed plot: {speed_plot}')
    print(f'Lap verified by position: {lap_completed}; solver failures: {fallback_count}')
    if errors:
        print('First solver errors:', errors[:3])
    if failure:
        print('RUN FAILED:', failure)
    return history, failure


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv',type=Path)
    parser.add_argument('--steps',type=int,default=1800)
    parser.add_argument('--max-speed',type=float,default=3.0)
    parser.add_argument('--plot',type=Path,default=Path('mpcc_preview.png'))
    args = parser.parse_args()
    run_demo(args.csv,args.steps,args.plot,args.max_speed)
