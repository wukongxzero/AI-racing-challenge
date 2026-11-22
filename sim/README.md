# Racing environment (`sim/`)

The shared environment every controller drives: the car, the track code and a Gymnasium env with
the agreed interface. Now `car="numpy"`; the MuJoCo car later plugs in behind the same interface.
Owner: Cristhian.

**Contents:** [Quick start](#quick-start) · [Start here by role](#start-here-by-role) ·
[Environment options](#environment-options) · [Observation and info](#observation-and-info) ·
[Laps and logs](#laps-and-logs) · [Track helpers](#track-helpers-envtrack) ·
[Car](#car-envcar_params) · [CasADi car](#casadi-car-simcar_casadipy-for-mpcc) ·
[Videos](#videos-simvideopy) · [Keyboard teleop](#keyboard-teleop-simteleoppy) ·
[Viewer](#viewer-simviewerpy) · [Adding a new car](#adding-a-new-car-mujoco) ·
[Known simplifications](#known-simplifications)

## Quick start

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

```bash
python sim/examples/drive_lap.py                       # a demo lap in a Pygame window
python sim/examples/drive_lap.py --laps 3 --track hidden
python sim/examples/drive_lap.py --no-view --plot --log run.csv
python sim/examples/drive_lap.py --video lap.mp4       # record a lap
python sim/examples/drive_keyboard.py                  # drive it yourself
python sim/tests/test_racing.py                        # all tests (a few seconds)
```

Setup (once, from the repo root): `pip install -r requirements.txt` and `pip install -e .`
(see the main README). The team uses `pygame-ce` (same `import pygame`, installs on every Python
version); don't install regular `pygame` in the same Python.

## Start here by role

**PPO (Pavan)**
- Make the env with `gymnasium.make("Racing-v0", off_track="terminate", time_limit=50)` (50 s = 1000
  steps). Through `gym.make`, Gymnasium's `check_env` passes every check with zero warnings.
- Your 15 inputs are all in `info` by name: `vx, vy, yaw_rate, delta, e_c, e_psi, prev_throttle,
  prev_steer, kappa_ahead` (6 values, 0.5–3.0 m ahead), `w_left` / `w_right`. The env's own `obs`
  vector is ordered as `env.obs_names` (a test checks it).
- Termination signals: off track → `terminated` in `"terminate"` mode; reverse heading → `|e_psi| > π/2`;
  stall → `progress_rate` (m/s along the track) near 0.
- Random starts as ranges, drawn every reset and reproducible with `seed`:
  `reset(options={"random_start": True, "e_psi0": (-0.15, 0.15), "v0": (0.5, 2.5)})`.
- Fast training: `log=False`, no `render_mode` (about 3,500 steps/s per env; 8 parallel envs work).
- Videos: `render_mode="rgb_array"` + `sim.video.record_episode`. Gymnasium's `RecordVideo` needs
  `moviepy`, which can't be installed on Python 3.15.

**MPCC (Berinay)**
- Track as CasADi B-splines: `env.track.casadi_centerline()` → x(s), y(s), theta(s), kappa(s).
- The car's own equations in CasADi: `sim/car_casadi.py` (tested identical to the simulator).
- The car **understeers**: a kinematic model over-predicts turning about 2× at 5 m/s, so a kinematic
  MPCC runs wide; use `dynamic_model`, or use the difference as a model-mismatch experiment.
- Steering is rate-limited in the simulator: constrain |δ_k − δ_(k−1)| ≤ 3.2·0.05 rad, starting from
  `info["delta"]`. Positions are at the centre of mass, not the rear axle.
- Sign: our `e_c` is **positive to the left**; the common MPCC contouring formula has the opposite sign.

**Pure Pursuit, game and evaluation (Nathaly)**
- Lookahead point: `env.track.point_at(s + Ld)`; wheelbase etc. in `env.car_params`.
- Speed profile: use `env.car_params["a_lat_max"]` (≈ 7.9 m/s², the grip the car really has) with a
  margin, **not μ·g** – with μ·g the car leaves the track in tight corners. Because the car
  understeers, plain Pure Pursuit drifts wide in long corners; multiplying the steering angle by
  `1 + K·v²/L` with `K = env.car_params["understeer_gradient"]` compensates (see `drive_lap.py`).
- Results: `env.lap_log` gives lap time, off-track count and `clean` per lap; every recorded lap is a
  full lap. Count clean laps (cutting a corner across the grass is slower but can gain a little).
- Per-step logs: `env.get_log()` / `env.save_log(path)`.
- Game: the viewer draws several cars at once ("ghosts"); the human uses `sim.teleop`, whose
  input shaping must be reported in results (see [Keyboard teleop](#keyboard-teleop-simteleoppy)).

## Environment options
- `car`: `"numpy"` now; `"mujoco"` later behind the same interface.
- `track`: `"training"`, `"hidden"` or a CSV path.
- `off_track`: `"grass"` (less grip + extra drag, keeps going, events counted) or `"terminate"`.
  Off track = any part of the car past the road edge (|e_c| > half-width − half car width).
- `max_laps` → `terminated`; `time_limit` (s) → `truncated`.
- `mu_scale`: friction multiplier for robustness tests (creation or `reset(options={"mu_scale": 0.7})`).
- `reset(options=...)`: default = start line, centred, at rest. `s0, e_c0, e_psi0, v0` as numbers or
  `(low, high)` ranges drawn at every reset; `random_start: True` fills in the rest; seeded via
  `reset(seed=...)`. A misspelled option name raises an error.
- Actions must be finite: a NaN or inf action raises an error (instead of silently becoming full
  throttle). `step()` before `reset()` raises an error.
- One step = one 50 ms command (20 Hz). The sim waits for the controller (evaluation mode); the
  examples show real-time pacing for the game.
- Default reward = metres of progress this step − `off_track_penalty` while off track.
  Pass `reward_fn=lambda info: ...` to replace it (Pavan owns the reward).
- `log=False` turns off per-step logging for fast training.
- Also available as `gymnasium.make("Racing-v0", **same_arguments)` (registered on `import sim`).
- `render_mode`: `None` (default, nothing drawn, fastest), `"rgb_array"` (`render()` returns a
  960×640 top-down picture) or `"human"` (live window, real time).

## Observation and info
`env.obs_names` gives the order of the float32 obs vector (finite bounds):
`x, y, psi, vx, vy, yaw_rate, delta, s, e_c, e_psi, kappa, w_left, w_right,
prev_throttle, prev_steer, kappa_ahead_0.5 … kappa_ahead_3`.
`info` has the same values by name plus `t, heading_ahead` (road direction ahead relative to
the car), `progress` (unwrapped m), `ds, progress_rate` (m/s along the track – negative =
going backwards, ≈0 = stalled), `lap, lap_times, lap_completed, off_track, off_track_count, mu_scale`.

## Laps and logs
- A lap counts at each start-line crossing (time interpolated inside the 50 ms step).
  Progress comes from windowed projection, so reversing over the line doesn't count. Only a start
  on the start line (s0 = 0) times lap 1; from a mid-track start the first crossing just starts
  the timing, so every recorded lap is a full lap.
- `env.get_log()` / `env.save_log(path)`: every step's time, state, command, s, progress,
  e_c, e_psi, off-track flag and lap. `env.lap_log`: lap time, off-track count and `clean`
  (no off-track) per lap.

## Track helpers (`env.track`)
`heading(s)`, `curvature(s)`, `width(s)`, `lookahead(s, dists)`, `heading_ahead(s, dists, psi)`,
`frenet_to_cartesian(s, e_c, e_psi)`, `centerline_table()`, `casadi_centerline()` (CasADi
B-splines x, y, theta, kappa over s ∈ [−5, L+20]). The centerline is a periodic cubic spline
in arc length; `length` = lap length.

## Car (`env.car_params`)
The team's planned dynamic bicycle model (equations at the top of `sim/car_numpy.py`),
F1TENTH geometry (m 3.74 kg, Iz 0.0471, lf 0.159, lr 0.171, wheelbase 0.33 m, width 0.31 m).
- Tires: Pacejka formula (Liniger et al. 2015); curve shape B, C from the authors' public code
  (github.com/alexliniger/MPCC, identified on their 1:43 car); peak force = μ × axle load, μ = 0.9.
  Cross-checked against ETH ForzaETH's identified F1TENTH tires (similar handling).
- Usable sideways grip `a_lat_max` ≈ 7.9 m/s² (0.89·μ·g; follows `mu_scale`). The car
  **understeers**: it needs more steering than the geometry says, more so at speed. How much:
  `understeer_gradient` K ≈ 0.010 rad per m/s² (textbook formula from the tire stiffness; a circle
  of radius R at speed v needs δ = L/R + K·v²/R).
- Acceleration is applied directly (`to_action` is exact), hard top speed, no reverse.
- Steering: 0.42 rad max, 3.2 rad/s rate. Kinematic model below 0.3 m/s total speed, blended up
  to 1 m/s (a car sliding sideways after a spin is slowed by its tires, not redirected).
- **Provisional:** a_max 5 m/s², brake 8 m/s², top speed 8 m/s, grass grip ×0.6 + extra drag.

## CasADi car (`sim/car_casadi.py`, for MPCC)
The simulator's own equations as CasADi functions (a test checks they give identical numbers):

```python
from sim.car_casadi import dynamic_model, kinematic_model, discretize, state_from_info
f = dynamic_model(env.car_params)     # xdot = f(x, u); x = [X, Y, psi, vx, vy, r], u = [a, delta]
fk = kinematic_model(env.car_params)  # x = [X, Y, psi, v], u = [a, delta]
F = discretize(f, dt=0.05)            # x_next = F(x, u): one 50 ms step (RK4, 10 sub-steps like the sim)
x0 = state_from_info(info)            # current state from the env
```
Steering is an input here but rate-limited in the simulator (see Start here by role). Below 1 m/s
the simulator blends to the kinematic model. At 5 m/s the car turns only ~53 % as sharply as the
kinematic model predicts. Measured on one laptop: IPOPT solves a simple tracking MPC with
`dynamic_model` in ~18 ms (1 s horizon) / ~33 ms (2 s) warm-started; the budget is 50 ms.

## Videos (`sim/video.py`)
`record_episode(env, act, "lap.mp4")` runs one episode with `act(obs, info) -> action` and saves
an MP4 (20 fps = real time); the env needs `render_mode="rgb_array"`. Needs `imageio[ffmpeg]`.
Gymnasium's `RecordVideo` wrapper also works with `render_mode="rgb_array"`, but it needs
`moviepy`, which can't be installed on Python 3.15 (it requires an old `pillow`).

## Keyboard teleop (`sim/teleop.py`)
Drive it yourself: `python sim/examples/drive_keyboard.py` (Up = gas, Down = brake, Left/Right = steer,
A = steering assist, R = restart, Esc = quit). For the game:

```python
from sim.teleop import KeyboardTeleop
teleop = KeyboardTeleop(env.car_params)                 # steer_assist=True for the optional assist
action = teleop.action_from_pygame(speed=info["vx"])    # same [throttle, steering] as the controllers
```
The human goes through the same car limits as every controller. **Human-only input shaping (report
it in results):** gas ramps up over 0.3 s so on/off keys don't jerk the car (`throttle_ramp_time=0`
turns it off); steering is full left / right / straight; the optional steering assist (off by
default) caps the steering request at speed to what the tires can use. All of these only make the
human's input gentler, never stronger than what controllers can command.

## Viewer (`sim/viewer.py`)
`TopDownView(track).draw(cars, text)` draws any number of cars (dicts with x, y, psi,
color). Drawing only – it never steps physics – so the game can step several envs and draw
them together.

## Adding a new car (MuJoCo)
`sim/car_interface.py` lists exactly what a car must provide (`p` car values, `reset`, `step`,
`state`). Write a class that does that, add one line to `make_car()`, and `RacingEnv(car="mujoco")`
works with no other change. Its car values (`p`) are what `env.car_params` shows, so MPCC and
Pure Pursuit automatically get the new car's numbers.

## Known simplifications
State these in the report: no load transfer between axles, no longitudinal tire slip, braking or
accelerating doesn't reduce cornering grip, no air drag (top speed is a hard limit), tire shape from
a 1:43 RC car scaled to the F1TENTH car's weight, and provisional speed/acceleration limits.
Off track uses the car's width only (|e_c| > half road width − half car width, as agreed); when
the car is at a large angle to the road (sliding), its corners can be up to ~10 cm further out, so
off-track is counted slightly late. Measured: no difference in normal driving.
