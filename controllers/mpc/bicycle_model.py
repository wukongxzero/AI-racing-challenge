"""Step 2: Simple kinematic bicycle model for the AI Racing Challenge.

State: [x (m), y (m), yaw (rad), speed (m/s)].
Action: [acceleration (m/s^2), steering command (rad)].

20 Hz, steering limit and steering slew limit match the project report.
Wheelbase, acceleration and speed limits are *illustrative* until the
simulation owner confirms the actual vehicle values.
"""
from dataclasses import dataclass
import argparse
from pathlib import Path
import numpy as np

from track import Track


@dataclass
class CarParams:
    wheelbase: float = 0.33       # m, placeholder (confirm with sim owner)
    dt: float = 0.05              # s: 20 Hz controller
    max_steer: float = 0.42       # rad, from preliminary report
    max_steer_rate: float = 3.2   # rad/s, from preliminary report
    max_accel: float = 2.0        # m/s^2, placeholder
    min_accel: float = -4.0       # m/s^2, placeholder
    max_speed: float = 5.0        # m/s, placeholder


class BicycleModel:
    def __init__(self, params=None):
        self.p = params or CarParams()
        if self.p.wheelbase <= 0 or self.p.dt <= 0:
            raise ValueError('wheelbase and dt must be positive')
        self.steering = 0.0  # physical applied steering angle

    def reset(self):
        self.steering = 0.0

    def derivative(self, state, acceleration, steering):
        x, y, yaw, v = np.asarray(state, dtype=float)
        return np.array([
            v * np.cos(yaw),
            v * np.sin(yaw),
            v / self.p.wheelbase * np.tan(steering),
            acceleration,
        ], dtype=float)

    def step(self, state, action):
        """RK4 integration; saturate acceleration and limit steering rate."""
        state = np.asarray(state, dtype=float)
        action = np.asarray(action, dtype=float)
        if state.shape != (4,) or action.shape != (2,):
            raise ValueError('state must be [x,y,yaw,v] and action [accel,steer]')
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError('state and action must contain finite numbers')
        acceleration = float(np.clip(action[0], self.p.min_accel, self.p.max_accel))
        requested = float(np.clip(action[1], -self.p.max_steer, self.p.max_steer))
        max_change = self.p.max_steer_rate * self.p.dt
        self.steering += float(np.clip(requested - self.steering, -max_change, max_change))

        dt = self.p.dt
        f = lambda z: self.derivative(z, acceleration, self.steering)
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        next_state = state + dt / 6.0 * (k1 + 2*k2 + 2*k3 + k4)
        next_state[3] = np.clip(next_state[3], 0.0, self.p.max_speed)
        next_state[2] = (next_state[2] + np.pi) % (2*np.pi) - np.pi
        return next_state


def run_demo(csv_path, output_path):
    """Open-loop tests: accelerate straight vs. accelerate with fixed turn.

    These paths do NOT automatically follow the track; that is MPCC's job.
    """
    import matplotlib.pyplot as plt
    track = Track(csv_path)
    initial = np.array([*track.center(0), float(track.heading(0)), 0.0])
    histories = {}
    for name, steer in [('Straight: steering = 0', 0.0),
                        ('Turning: steering = 0.12 rad', 0.12)]:
        car = BicycleModel()
        state = initial.copy()
        states = [state.copy()]
        for _ in range(80):  # four seconds at 20 Hz
            state = car.step(state, [1.0, steer])
            states.append(state.copy())
        histories[name] = np.asarray(states)

    s = np.linspace(0.0, track.length, 1200)
    center = track.center(s)
    left, right = track.boundaries(s)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(left[:, 0], left[:, 1], '--', label='Left boundary', alpha=0.7)
    ax.plot(right[:, 0], right[:, 1], '--', label='Right boundary', alpha=0.7)
    ax.plot(center[:, 0], center[:, 1], '-', label='Centerline', alpha=0.5)
    for name, history in histories.items():
        ax.plot(history[:, 0], history[:, 1], linewidth=2, label=name)
        print(f'{name}: final speed={history[-1,3]:.2f} m/s, '
              f'final position=({history[-1,0]:.2f}, {history[-1,1]:.2f}) m')
    ax.scatter(initial[0], initial[1], s=55, label='Start', zorder=5)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Step 2 — Bicycle dynamics (open-loop, not MPCC)')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f'Saved plot: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=Path, help='Path to training_track.csv')
    parser.add_argument('--plot', type=Path, default=Path('bicycle_preview.png'))
    args = parser.parse_args()
    run_demo(args.csv, args.plot)
