# Racing environment (`sim/`)

The shared environment every controller drives: the car, the track code and a Gymnasium env with
the agreed interface. Now `car="numpy"`; the MuJoCo car later plugs in behind the same interface.
Owner: Cristhian. Setup: see the main README.

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

Try it: `python sim/examples/drive_lap.py` (demo lap) · `python sim/examples/drive_keyboard.py`
(drive it yourself) · `python sim/tests/test_racing.py` (tests). Options: `--help`.

## Start here by role

What the environment provides for each part.

**PPO (Pavan)**
- Also available as `gymnasium.make("Racing-v0", ...)` with the same arguments; created that way,
  Gymnasium's `check_env` passes every check with zero warnings. `time_limit=50` = 1000 steps.
- All 15 inputs of the PPO plan are in `info` by name (see [Observation and info](#observation-and-info)).
  The env's own `obs` vector is ordered as `env.obs_names`.
- Off track ends the episode in `off_track="terminate"` mode; `e_psi` is the heading error and
  `progress_rate` the speed along the track (m/s, negative = going backwards).
- `reset()` options accept `(low, high)` ranges, drawn at every reset and reproducible with `seed`.
- Speed: about 3,500 steps/s per env with `log=False`; 8 envs in parallel processes run at about
  6,300 steps/s in total on one laptop.
- Videos: `render_mode="rgb_array"` + `sim.video.record_episode(env, act, "lap.mp4")`. Gymnasium's
  `RecordVideo` needs `moviepy`, which can't be installed on Python 3.15.

**MPCC (Berinay)**
- Track as CasADi B-splines: `env.track.casadi_centerline()` → x(s), y(s), theta(s), kappa(s), over
  s ∈ [−5, L+20] m.
- The car's own equations in CasADi, tested identical to the simulator (`sim/car_casadi.py`):
  ```python
  from sim.car_casadi import dynamic_model, kinematic_model, discretize, state_from_info
  f = dynamic_model(env.car_params)   # xdot = f(x, u); x = [X, Y, psi, vx, vy, r], u = [a, delta]
  F = discretize(f, dt=0.05)          # one 50 ms step, RK4 like the simulator
  ```
- The car understeers: at 5 m/s it turns about half as sharply as a kinematic model predicts.
- The simulator limits steering to 3.2 rad/s and 0.42 rad; the current wheel angle is `info["delta"]`.
  Positions are at the centre of mass, not the rear axle.
- Our `e_c` is positive to the left; the common MPCC contouring formula has the opposite sign.
- IPOPT solved a simple tracking MPC with `dynamic_model` in ~18 ms (1 s horizon) / ~33 ms (2 s)
  warm-started on one laptop; the control period is 50 ms.

**Pure Pursuit, game and evaluation (Nathaly)**
- Centerline point at any progress: `env.track.point_at(s)`; wheelbase etc. in `env.car_params`.
- `env.car_params["a_lat_max"]` is the sideways grip the car actually reaches (≈ 7.9 m/s²); μ·g is
  about 12 % higher. The car understeers (see [Car](#car-envcar_params)).
- `env.lap_log`: lap time, off-track count and `clean` per lap; every recorded lap is a full lap.
  Corner cuts across the grass are slower but can gain a little time. `info["lap_timer"]` is the
  time in the current lap. Per-step logs: `env.get_log()` / `env.save_log(path)`.
- The viewer (`sim/viewer.py`) draws several cars at once, whole track or a close-up that follows a
  car. The human driver uses `sim.teleop` (see [Keyboard teleop](#keyboard-teleop)).

## What the env offers
- `car`: `"numpy"` now, `"mujoco"` later. `track`: `"training"`, `"hidden"` or a CSV path.
- `off_track`: `"grass"` (less grip and extra drag, keeps going, events counted) or `"terminate"`.
  Off track = any part of the car past the road edge (|e_c| > half road width − half car width).
- `max_laps` → `terminated`; `time_limit` (s) → `truncated`.
- `mu_scale`: friction multiplier (at creation or `reset(options={"mu_scale": 0.7})`).
- `reset(options=...)`: default = start line, centred, at rest. `s0, e_c0, e_psi0, v0` as numbers or
  `(low, high)` ranges; `random_start: True` draws the rest. Misspelled options raise an error.
- Default reward = metres of progress this step − `off_track_penalty` while off track;
  `reward_fn=lambda info: ...` replaces it.
- `render_mode`: `None` (default, fastest), `"rgb_array"` (960×640 picture) or `"human"` (window).
- Each step is one 50 ms command; the sim waits for the controller.

## Observation and info
`info` has everything by name each step:
- car: `x, y, psi, vx, vy, yaw_rate, delta`, last command `prev_throttle, prev_steer`, time `t`
- track: `s, e_c, e_psi, kappa, w_left, w_right`, `kappa_ahead` and `heading_ahead` (6 values each,
  0.5–3.0 m ahead), `progress` (unwrapped m), `ds`, `progress_rate`
- race: `lap, lap_times, lap_completed, lap_timer, off_track, off_track_count, mu_scale`

`obs` is a float32 vector of 21 of these, in the order of `env.obs_names`.

## Laps
A lap counts when the car crosses the start line after covering the whole track (time interpolated
within the 50 ms step); reversing over the line doesn't count. Only a start on the line times lap 1.

## Car (`env.car_params`)
The team's planned dynamic bicycle model (equations at the top of `sim/car_numpy.py`), F1TENTH size
(3.74 kg, wheelbase 0.33 m, width 0.31 m), Pacejka tires with μ = 0.9.
- Grip: `a_lat_max` ≈ 7.9 m/s². Understeer: `understeer_gradient` K ≈ 0.010 rad per m/s² (a circle
  of radius R at speed v needs δ = L/R + K·v²/R).
- Steering 0.42 rad max, 3.2 rad/s. **Provisional:** accel 5 m/s², brake 8 m/s², top speed 8 m/s.
- Tire shape: Liniger et al. (2015); the numbers are in the authors' code
  (github.com/alexliniger/MPCC), scaled to this car's weight. Similar handling to ETH ForzaETH's
  measured F1TENTH tires.

## Keyboard teleop
`python sim/examples/drive_keyboard.py`: arrow keys drive; Z = close-up / whole track, A / B =
steering / braking assist, [ / ] = steering assist strength, R = restart, Esc = quit;
`--log human.csv` saves every attempt, including the assist settings used.

For the game: `KeyboardTeleop(env.car_params, track=env.track).action_from_pygame(speed=info["vx"], s=info["s"])`
gives the same `[throttle, steering]` command as the controllers. The human goes through the same car
limits. **Human-only input shaping (report it in results):** gas ramps up over 0.3 s; steering
assist (on) caps steering at speed to what the tires can grip (strength ×1.4); braking assist (on)
cuts the gas or brakes when the car is too fast for a corner within 8 m. All of these only make
the human's input gentler.

## Adding a new car (MuJoCo)
`sim/car_interface.py` lists what a car must provide (`p`, `reset`, `step`, `state`); one line in
`make_car()` adds it. Its values become `env.car_params`.

## Known simplifications
For the report: no load transfer between axles, no longitudinal tire slip, braking doesn't reduce
cornering grip, no air drag (top speed is a hard limit), tire shape from a 1:43 RC car scaled to the
F1TENTH car, provisional speed limits. Off track uses the car's width only; when the car slides at a
large angle its corners can be up to ~10 cm further out (no difference in normal driving).
