"""MPCC-to-MuJoCo integration scaffold.

Requires a teammate-provided MuJoCo MJCF model with a named chassis body and
steering + longitudinal actuators. All actuator units and gains MUST be
validated against that model. No physics model is bundled in this scaffold.

Usage:
  python mujoco_bridge.py --model PATH.xml --track training_track.csv \
    --body car --steer-act steering --drive-act drive --seconds 10

Install: python3 -m pip install mujoco casadi numpy scipy
"""
import argparse
from dataclasses import dataclass
import csv
import math
from pathlib import Path
import time
import numpy as np
from track import Track
from bicycle_model import CarParams
from mpcc import MPCC, MPCCParams, safe_fallback


@dataclass
class BridgeConfig:
    body: str = 'car'
    steering_actuator: str = 'steering'
    drive_actuator: str = 'drive'
    control_hz: float = 20.0
    # Control mapping is provisional, NOT universal MuJoCo semantics.
    drive_accel_scale: float = 2.0  # m/s² per ctrl unit, calibration required
    steering_ctrl_scale: float = 1.0  # rad per ctrl unit, calibration required
    car_half_width: float = 0.15


def yaw_from_quat_wxyz(q):
    w, x, y, z = map(float, q)
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def actuator_id(model, mujoco, name):
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if idx < 0:
        raise ValueError(f'Missing actuator {name!r} in MJCF model')
    return int(idx)


class MujocoMPCCBridge:
    def __init__(self, model, data, track, cfg=BridgeConfig(), car=None):
        import mujoco
        self.mj = mujoco
        self.model = model
        self.data = data
        self.track = track
        self.cfg = cfg
        self.car = car or CarParams(dt=1 / cfg.control_hz, max_speed=3.0)
        if abs(self.car.dt - 1 / cfg.control_hz) > 1e-9:
            raise ValueError('CarParams.dt must equal 1/control_hz')
        if model.opt.timestep <= 0:
            raise ValueError('MuJoCo physics timestep must be positive')
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.body)
        if self.body_id < 0:
            raise ValueError(f'Missing body {cfg.body!r} in MJCF')
        self.steer_id = actuator_id(model, mujoco, cfg.steering_actuator)
        self.drive_id = actuator_id(model, mujoco, cfg.drive_actuator)
        self.controller = MPCC(track, self.car, MPCCParams(safety_margin=cfg.car_half_width))
        self.last_steer = 0.0
        self.prev_pos = None
        self.prev_time = None
        self.prev_s = None
        self.progress = 0.0
        self.fallbacks = 0
        self.records = []
        self.latencies = []
        self.missed_deadlines = 0

    def read_state(self):
        """Read chassis pose; estimate forward speed by finite differences.

        Replace finite-difference speed with a validated joint/sensor signal
        when the actual MJCF model is available.
        """
        pos = np.asarray(self.data.xpos[self.body_id, :2], dtype=float).copy()
        yaw = yaw_from_quat_wxyz(self.data.xquat[self.body_id])
        t = float(self.data.time)
        speed = 0.0
        if self.prev_pos is not None and t > self.prev_time:
            velocity = (pos - self.prev_pos) / (t - self.prev_time)
            speed = max(0.0, float(velocity @ np.array([math.cos(yaw), math.sin(yaw)])))
        self.prev_pos, self.prev_time = pos, t
        return np.array([pos[0], pos[1], yaw, speed])

    def update_progress(self, state):
        s = float(self.track.project(state[0], state[1]))
        if self.prev_s is not None:
            length = self.track.length
            delta = (s - self.prev_s + length / 2) % length - length / 2
            self.progress += delta
        self.prev_s = s
        return s

    def _limited_ctrl(self, idx, value):
        if bool(self.model.actuator_ctrllimited[idx]):
            low, high = self.model.actuator_ctrlrange[idx]
            return float(np.clip(value, low, high))
        return float(value)

    def write_action(self, action):
        accel, target_steer = map(float, action)
        max_delta = self.car.max_steer_rate * self.car.dt
        steer = float(np.clip(target_steer, self.last_steer-max_delta,
                              self.last_steer+max_delta))
        steer = float(np.clip(steer, -self.car.max_steer, self.car.max_steer))
        self.last_steer = steer
        # IMPORTANT: assumes a signed longitudinal actuator and angle-based
        # steering actuator. Wrong gear definitions require a custom adapter.
        self.data.ctrl[self.steer_id] = self._limited_ctrl(
            self.steer_id, steer / self.cfg.steering_ctrl_scale)
        self.data.ctrl[self.drive_id] = self._limited_ctrl(
            self.drive_id, accel / self.cfg.drive_accel_scale)

    def tick(self):
        state = self.read_state()
        s = self.update_progress(state)
        start = time.perf_counter()
        failed = False
        try:
            action, _, _ = self.controller.solve(state, self.progress, self.last_steer)
            if not np.all(np.isfinite(action)):
                raise ValueError('nonfinite solver output')
        except (RuntimeError, ValueError) as exc:
            failed = True
            self.fallbacks += 1
            self.controller.guess_x = None
            self.controller.guess_u = None
            action = safe_fallback(self.track, self.car, state)
        latency = time.perf_counter()-start
        self.latencies.append(latency)
        if latency > self.car.dt:
            self.missed_deadlines += 1
        self.write_action(action)
        in_bounds = bool(self.track.within_bounds(
            state[0], state[1], car_half_width=self.cfg.car_half_width))
        self.records.append((self.data.time, *state, s, self.progress,
                             float(action[0]), float(action[1]), latency*1000,
                             int(failed), int(in_bounds)))
        return in_bounds

    def run(self, seconds=10.0, realtime=False):
        """Fixed-rate physics scheduling; realtime also paces against wall clock.

        An overrun is recorded, not hidden. This is a synchronous test harness,
        not a hard-real-time controller. Controller should not be run in a
        background GUI thread without explicit synchronization.
        """
        period = self.car.dt
        next_tick = float(self.data.time)
        t_end = float(self.data.time) + seconds
        wall_start = time.perf_counter()
        sim_start = float(self.data.time)
        while float(self.data.time) < t_end:
            if float(self.data.time) + 1e-10 >= next_tick:
                if not self.tick():
                    print('STOP: safety-margin boundary check failed')
                    break
                if self.progress >= self.track.length:
                    print('Physical lap projection reached finish')
                    break
                next_tick += period
            self.mj.mj_step(self.model, self.data)
            if realtime:
                goal = wall_start + (float(self.data.time)-sim_start)
                delay = goal - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
        return self.report()

    def report(self):
        a = np.asarray(self.latencies, dtype=float)*1000
        if not len(a):
            return {'ticks':0}
        result = {'ticks':len(a), 'mean_ms':float(np.mean(a)),
                  'p95_ms':float(np.percentile(a,95)), 'max_ms':float(np.max(a)),
                  'over_50_ms':int(np.sum(a>50)), 'fallbacks':self.fallbacks,
                  'measured_progress_m':float(self.progress)}
        print('MPCC timing:', result)
        return result

    def save_csv(self, path):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sim_time','x','y','yaw','v_forward','track_s',
                             'unwrapped_s','accel_cmd','steer_cmd','solve_ms',
                             'fallback','within_bounds'])
            writer.writerows(self.records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--track', type=Path, default=Path('training_track.csv'))
    parser.add_argument('--body', default='car')
    parser.add_argument('--steer-act', default='steering')
    parser.add_argument('--drive-act', default='drive')
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--realtime', action='store_true')
    parser.add_argument('--log', type=Path, default=Path('mujoco_mpcc_log.csv'))
    args = parser.parse_args()
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cfg = BridgeConfig(body=args.body, steering_actuator=args.steer_act,
                       drive_actuator=args.drive_act)
    bridge = MujocoMPCCBridge(model,data,Track(args.track),cfg)
    # IMPORTANT: start pose in XML must be ON the track and aligned with tangent.
    bridge.run(args.seconds,realtime=args.realtime)
    bridge.save_csv(args.log)
    print('Saved:',args.log)


if __name__ == '__main__':
    main()
