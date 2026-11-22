# Racing environment (`sim/`) – stand-in NumPy car

```python
from sim import RacingEnv

env = RacingEnv(car="numpy", track="training", off_track="grass", max_laps=3, time_limit=120)
obs, info = env.reset(seed=None, options=None)
obs, reward, terminated, truncated, info = env.step(action)   # [throttle, steering] in [-1, 1]
action = env.to_action(accel, steer)                          # m/s^2, rad -> [-1, 1]

track = env.track
s, e_c, e_psi, kappa = track.project(x, y, psi, s_prev)       # s_prev=None: search whole lap
x_c, y_c = track.point_at(s)                                  # wraps around the lap
```

Run the example and tests:

```
python sim/examples/drive_lap.py                       # 1 lap with a Pygame top-down view
python sim/examples/drive_lap.py --laps 3 --track hidden
python sim/examples/drive_lap.py --no-view --plot --log run.csv
python sim/tests/test_racing.py
```

Install: `pip install -r requirements.txt` from the repo root. The team uses `pygame-ce` (community edition of pygame, same `import pygame`); it installs on every Python version. Don't install regular `pygame` in the same Python.

## Environment
- `car`: `"numpy"` now; `"mujoco"` later behind the same interface.
- `track`: `"training"`, `"hidden"` or a CSV path.
- `off_track`: `"grass"` (less grip + extra drag, keeps going, events counted) or `"terminate"`.
  Off track = any part of the car past the road edge (|e_c| > half-width − half car width).
- `max_laps` → `terminated`; `time_limit` (s) → `truncated` (PPO: `time_limit=50` = 1000 steps).
- `mu_scale`: friction multiplier for robustness tests (creation or `reset(options={"mu_scale": 0.7})`).
- `reset(options=...)`: `s0, e_c0, e_psi0, v0`, or `{"random_start": True}` (seeded via `reset(seed=...)`).
  Default: start line, centred, at rest.
- One step = one 50 ms command (20 Hz). The sim waits for the controller (evaluation mode);
  the example shows real-time pacing for game mode.
- Default reward = metres of progress this step − `off_track_penalty` while off track.
  Pass `reward_fn=lambda info: ...` to replace it (Pavan owns the reward).
- `log=False` turns off per-step logging for fast training.

## Observation and info
`env.obs_names` gives the order of the float32 obs vector (finite bounds, so `check_env`
passes with zero warnings):
`x, y, psi, vx, vy, yaw_rate, delta, s, e_c, e_psi, kappa, w_left, w_right,
prev_throttle, prev_steer, kappa_ahead_0.5 … kappa_ahead_3`.
`info` has the same values by name plus `t, heading_ahead` (road direction ahead relative to
the car), `progress` (unwrapped m), `ds, progress_rate` (m/s along the track – negative =
going backwards, ≈0 = stalled), `lap, lap_times, lap_completed, off_track, off_track_count, mu_scale`.

## Laps and logs
- A lap counts at each start-line crossing (time interpolated inside the 50 ms step).
  Progress comes from windowed projection, so shortcuts and reversing over the line don't
  count. A first partial lap from a mid-track start (< 90 % of the lap) is not counted.
- `env.get_log()` / `env.save_log(path)`: every step's time, state, command, s, progress,
  e_c, e_psi, off-track flag and lap. `env.lap_log`: lap time and off-track count per lap.

## Track helpers (`env.track`)
`heading(s)`, `curvature(s)`, `width(s)`, `lookahead(s, dists)`, `heading_ahead(s, dists, psi)`,
`frenet_to_cartesian(s, e_c, e_psi)`, `centerline_table()`, `casadi_centerline()` (CasADi
B-splines x, y, theta, kappa over s ∈ [−5, L+20]). The centerline is a periodic cubic spline
in arc length; `length` = lap length.

## Car (`env.car_params`)
The team's planned dynamic bicycle model (equations at the top of `sim/car_numpy.py`),
F1TENTH geometry (m 3.74 kg, Iz 0.0471, lf 0.159, lr 0.171, wheelbase 0.33 m, width 0.31 m).
- Tires: Pacejka formula (Liniger et al. 2015); curve shape B, C from the authors' public code
  (github.com/alexliniger/MPCC, identified on their 1:43 car); peak force = mu × axle load,
  mu = 0.9. Usable sideways grip `a_lat_max` ≈ 7.9 m/s² (0.89·mu·g; it follows `mu_scale`) –
  use this, not mu·g, for speed profiles, and keep a margin. The car **understeers**: it needs more steering than
  the geometry says, more so at speed (the example controller compensates).
- Acceleration is applied directly (`to_action` is exact), hard top speed, no reverse,
  no air drag. Braking doesn't reduce cornering grip (known simplification).
- Steering: 0.42 rad max, 3.2 rad/s rate. Kinematic model below 0.3 m/s, blended up to 1 m/s.
- **Provisional:** a_max 5 m/s², brake 8 m/s², top speed 8 m/s, grass grip ×0.6 + extra drag.

## Viewer (`sim/viewer.py`)
`TopDownView(track).draw(cars, text)` draws any number of cars (dicts with x, y, psi,
color). Drawing only – it never steps physics – so the game can step several envs and draw
them together.
