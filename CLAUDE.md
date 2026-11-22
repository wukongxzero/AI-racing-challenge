# AI Racing Project – context for Claude Code

Read this first. It explains the whole project, who does what, what has been agreed, and
what comes next. Everything under "Agreed decisions" was approved by the whole team. These
decisions are a starting point, not a cage: **if changing one gives a better solution, do it**
– just follow the "Changing an agreed decision" steps in the Code rules so nobody on the team
is caught off guard.

---

## 1. The project in one paragraph

NYU course project for *Reinforcement Learning and Optimal Control for Autonomous Systems I*
(ROB-GY 6323). We build a 2D racing game where four "drivers" race the **same simulated car
on the same track**: an optimal controller (**MPCC**, model predictive contouring control), a
learned controller (**PPO**, reinforcement learning), a simple geometric baseline (**Pure
Pursuit**), and a **human** with the keyboard. Because the car, track and limits are
identical for everyone, differences in results come from the controllers alone. The main goal
is a fair comparison of model-based control vs learning-based control; the game is the demo.

The assignment explicitly allows building the system "with the help of AI". The teacher will
see this repo.

## 2. Team and ownership

| Person | Owns | Notes |
|---|---|---|
| **Cristhian** (repo owner of this part) | Racing environment: car simulation, track code, `RacingEnv`, keyboard teleop limiter, MuJoCo car, tracks | This is the part Claude Code is helping with |
| **Berinay** | MPCC controller (CasADi/IPOPT) | Uses the shared env + track code; does not build her own simulator |
| **Pavan** | PPO controller (Stable-Baselines3) | Trains on the training track only |
| **Nathaly** | Pure Pursuit, game integration (Pygame modes, results screen), logging/metrics, comparison plots | Consumes env logs |

Rule the team agreed on: **only decisions that affect more than one person are decided
together; anything that only affects one person is that person's own call.**

## 3. Timeline (from the preliminary report – this is the official schedule)

| Dates | Work |
|---|---|
| Nov 20–26 | Stand-in NumPy car; first Pure Pursuit, MPCC and PPO laps; evaluation and logging pipeline |
| Nov 27–Dec 3 | MuJoCo car swapped in; MPCC retuned and PPO retrained on it; first lap-time table (pre-report due **Dec 3**) |
| Dec 4–10 | Multi-seed runs, ablations, robustness tests; game integration (presentation **Dec 10**, 5 min + 2 min Q&A) |
| Dec 11–17 | Failure analysis, clean reproducible scripts, final report (**Dec 17**) |

Environment milestones inside that:
- **Now (Nov 22–23): ship the stand-in NumPy car** (`RacingEnv` + track code + example) so
  nobody is blocked.
- **~Nov 25–26: MuJoCo checkpoint** – if the real-wheel car is still glitchy, switch to the
  no-contact backup (1–2 days).
- **By Nov 27: MuJoCo car swaps in** behind the same `RacingEnv`; share the car's numbers for
  MPCC; Pavan retrains PPO. If it isn't ready, everyone stays on the NumPy car.

## 4. Tech stack

Python · NumPy/SciPy · Gymnasium (env API) · Pygame (top-down 2D game and keyboard;
the team uses `pygame-ce` – community edition, same `import pygame`, installs on every Python version) ·
MuJoCo (final physics engine) · CasADi + IPOPT (MPCC) · Stable-Baselines3 (PPO).

## 5. Agreed decisions (the interface)

### 5.1 Code interface (names are proposals the team accepted)
```python
env = RacingEnv(car="numpy")   # or car="mujoco" later – one word switches the car
obs, info = env.reset(seed=None, options=None)
obs, reward, terminated, truncated, info = env.step(action)   # action = [throttle, steering], each in [-1, 1]
action = env.to_action(accel, steer)                          # converter: m/s^2 and rad -> [-1, 1]
s, e_c, e_psi, kappa = track.project(x, y, psi, s_prev)       # progress, lateral offset, heading error, curvature
x_c, y_c = track.point_at(s)                                  # centerline point at progress s (e.g. Pure Pursuit lookahead)
```
**Additions (Nov 22, all optional – nothing above changed):**
```python
from sim import RacingEnv                     # environment code lives in sim/
track = env.track                             # the track object of an env
env.car_params                                # read-only dict: wheelbase, mu, a_lat_max (usable grip), understeer_gradient, v_max, a_max, b_max, delta_max, ...
track.project(x, y, psi)                      # s_prev=None searches the whole lap (use only at start)
track.heading(s), track.curvature(s), track.width(s)          # road direction, curvature, (w_left, w_right)
track.lookahead(s, dists), track.heading_ahead(s, dists, psi) # curvature / road direction ahead
track.casadi_centerline()                     # CasADi B-splines x(s), y(s), theta(s), kappa(s) for MPCC
sim.car_casadi.dynamic_model(env.car_params)  # the car's own equations in CasADi (+ kinematic_model, discretize)
RacingEnv(mu_scale=0.7) or reset(options={"mu_scale": 0.7})    # less grip for robustness tests
gymnasium.make("Racing-v0", **kwargs)        # same env through Gymnasium's registry
RacingEnv(render_mode="rgb_array" | "human")  # pictures for videos / a live window
from sim.video import record_episode         # save an episode as MP4 (any Python version)
from sim.teleop import KeyboardTeleop        # arrow keys -> the same [throttle, steering] commands
```
**Update (Nov 22):** `track.project` takes the car's heading `psi` – heading error can't be
computed from position alone. `track.point_at(s)` was added so Pure Pursuit can find its
lookahead point (`track.point_at(s + Ld)`). Both were announced to the team.
**Sign convention:** lateral offset `e_c` is **positive when the car is left of the
centerline**. The common MPCC form `sin(phi)(x - x_c) - cos(phi)(y - y_c)` gives the
opposite sign, so MPCC must flip it or account for it in its cost.
`reset`/`step` follow the standard Gymnasium API so PPO (Stable-Baselines3) plugs in
directly. `reward` is for PPO; Pavan owns its design.

### 5.2 Commands (what every driver sends)
- Two numbers in [-1, 1]: throttle (+1 full gas, -1 full brake, 0 coast) and steering
  (+1 full left, -1 full right, 0 straight).
- Steering command = the angle the driver wants the front wheels at. The car turns toward it
  no faster than the steering-rate limit (like a real car).
- **Control rate: 20 Hz for every controller** (command held 50 ms). Berinay confirmed.
- Human keyboard teleop: arrow keys produce the same commands and go through the same
  limiter as the controllers.
  Human-only input shaping (`sim/teleop.py`; part of the experimental setup, report it):
  the throttle command ramps 0 -> full in 0.3 s while Up is held (on/off keys would
  otherwise jerk the car; `throttle_ramp_time=0` turns it off), steering is full
  left / right / straight, and an optional speed-based steering assist (off by default)
  caps the steering request at speed. All of these only make the human's input gentler,
  never stronger than what the controllers can command.

### 5.3 Observations / car state
- Car state: position, heading, forward and sideways speed, yaw rate, current steering angle.
- The env makes all car state, all track outputs, and the previous command available;
  Pavan picks which ones PPO uses.
- Units: meters, seconds, radians. **Left = positive.**
- How it's delivered: `info` holds everything by name every step (car state, previous
  command, `s, e_c, e_psi, kappa, w_left, w_right`, `kappa_ahead` and `heading_ahead` at
  0.5–3.0 m, `progress`, `ds`, `progress_rate`, `off_track`, laps). `obs` is a float32
  vector of the main ones – car state, `s, e_c, e_psi, kappa, w_left, w_right`, previous
  command and `kappa_ahead` (21 numbers); `env.obs_names` gives its order. Everything else
  is in `info` only.

### 5.4 Track
- Files (already made, in `tracks/`):
  - `training_track.csv` – 127 m lap: main straight, chicane flick, carousel (long 180°),
    back straight, hairpin, double-apex, hairpin 2. Used for all training, tuning and races.
  - `hidden_track.csv` – 95 m of flowing curves. **Never used for PPO training**; only used
    at the end to test generalization (does a controller handle a track it has never seen?).
  - Each has a `_preview.png`.
- CSV format: columns `x_m, y_m, w_left_m, w_right_m`; one centerline point every 10 cm,
  driven counter-clockwise, first point on the start line. Road 1.5 m wide (0.75 m each
  side). No corner tighter than 2.5 m radius.
- Track code returns: progress along the lap (`s`), distance from the center line (`e_c`,
  lateral offset), angle between car heading and road direction wrapped to [-pi, pi]
  (`e_psi`), curvature (`kappa`), and the road direction at several points ahead.
- Implementation requirements:
  - Fit a smooth periodic curve through the centerline, parameterized by arc length.
  - Projection searches **near the previous progress value** so it never jumps to a nearby
    section of track.
  - Handle lap wrap-around; a lap only counts after covering the whole track (no cheating
    by reversing over the line). Only a start on the start line times lap 1.
  - Off track = any part of the car past the road edge (|e_c| > half-width − half car width).
  - Must also be able to give MPCC the centerline as a smooth curve usable inside CasADi.
  - Provide `track.point_at(s)` (centerline point at progress s, wrapping around the lap).

### 5.5 Race and training rules
- Races start on the start line and end after a set number of laps or a time limit.
- Off track in races: the car slows on "grass" (lower grip / extra drag) and keeps going;
  off-track events are counted.
- For training, Pavan decides start points, whether off-track ends the episode, and when
  episodes end. The env must support starting anywhere on the track and both off-track
  behaviors.

### 5.6 Logging (for Nathaly's metrics)
- Every step: time, car state, command, progress, lateral offset, off-track flag
  (in the `info` dict).
- Per lap: lap time and off-track count. Any other metrics are Nathaly's call.

### 5.7 Car and simulation
- Car about 1:10 scale (F1TENTH-like).
- Steering: max angle 0.42 rad, max rate 3.2 rad/s (F1TENTH defaults, starting values).
- Top speed and acceleration limits: set once the car drives. NumPy car starting values
  (provisional): top speed 8 m/s (hard limit), acceleration 5 m/s², braking 8 m/s².
- NumPy car = the team's planned dynamic bicycle model (Pacejka tires, see the models
  document): acceleration is applied directly, no air drag, braking does not reduce
  cornering grip (known simplification). Tire model from Liniger et al. (2015); the paper
  prints no numbers, so B, C come from the authors' public code (github.com/alexliniger/MPCC,
  `C++/Params/model.json`, identified on their 1:43 car; checked Nov 22). Peak force =
  mu × axle load, mu = 0.9 (their D / axle load). Cite both paper and repo in the report. Max
  sideways grip ≈ 0.8 g; the car understeers (needs more steering than geometry says).
- **The NumPy car is the fallback final car**: if the real-wheel MuJoCo car isn't working
  by ~Nov 25–26, we use the NumPy car instead of building a no-contact MuJoCo backup
  (same tire equations, so the backup would add nothing). Cristhian's plan; noted in PR #1.
- Everyone uses only these car values, so all controllers race the same car.
- Run modes: **evaluation** (simulation waits for each controller, so lap times don't depend
  on computer speed) and **game** (real time).

## 6. What to build now: the stand-in NumPy car (Cristhian's next task)

Deliverables, in this order:
1. **Track code** – load a track CSV, `track.project`, curvature and look-ahead, lap logic.
   Include a test that drives along the centerline and checks progress is smooth and the
   lap counts once.
2. **NumPy car** – simple bicycle model with the steering limits above **and a grip limit**
   (e.g. cap lateral acceleration around 1 g, or a dynamic bicycle model with tire forces).
   Without a grip limit the car can take any corner at any speed and racing is meaningless.
   Integrate with a small fixed physics step (a few ms) inside each 50 ms control step.
3. **`RacingEnv`** – `reset`, `step`, `to_action`, grass behavior, lap counting, logging,
   Gymnasium spaces declared.
4. **Example script** – drives a lap of the training track (e.g. a simple steer-toward-the-
   centerline controller) and shows a basic Pygame top-down view.

Ship a minimal working version first (track + car + reset/step), push it, then add extras.

### 6.1 What teammates' plans (preliminary report) need from the environment
- **PPO (Pavan)** uses an ego-centric observation: vx, vy, yaw rate, actual steering angle,
  e_c, e_psi, previous action, **curvature sampled at 6 distances 0.5–3.0 m ahead**, and the
  **local road half-width**. Make all of these easy to get. His episodes end on off-track,
  **reverse heading** or **stall**, and truncate at 1000 steps (50 s), with random start
  points – expose what he needs to detect those (progress rate, heading error, speed).
- **Pure Pursuit (Nathaly)** needs `track.point_at(s)` for its lookahead point, plus the car's **wheelbase**, the **friction coefficient mu**
  and **top speed** (her speed profile is v = min(sqrt(mu*g/|kappa|), v_max)).
- **MPCC (Berinay)** uses a kinematic bicycle model with wheelbase L and a 0.05 s step
  (matches 20 Hz), and needs the track as a smooth curve in CasADi.
- **Robustness tests**: run with friction reduced by 20–30% → the car's friction/grip should
  be a parameter that can be scaled at `reset`/creation.
- The repository uses one folder per workstream: `sim/`, `tracks/`, `controllers/`,
  `game/`, `eval/`. The environment code belongs in `sim/`.

## 7. What comes after (the future, so design for it now)

- **MuJoCo car (by Nov 27)** replaces the NumPy car behind `RacingEnv(car="mujoco")`.
  Plan (shared with the team, also in the preliminary report):
  1. **Real wheels first** – chassis + 4 wheels touching the floor, grip from MuJoCo's contact
     model. Start from an existing MuJoCo example car. Risk: contact tuning can be glitchy
     (jitter, bouncing, flipping, unrealistic grip).
  2. **Fallback: the NumPy car** – if the real-wheel car is still glitchy at the Nov 25–26
     checkpoint, the NumPy car (same Pacejka tire equations) stays as the final car. This
     replaces the earlier no-contact MuJoCo backup, which would have used the same
     equations and added nothing (decided Nov 22, Cristhian).
  For MPCC: with real wheels, fit a bicycle model to logged test drives (1–2 days); with the
  backup, share the tire formula and its numbers directly. Keep
  the car behind a clear internal boundary so a second car implementation drops in without
  touching the track code, `RacingEnv`'s public API, or anyone else's code. Keep the NumPy
  car working as a fallback.
- **Model numbers for MPCC**: after the MuJoCo car exists, Cristhian fits a bicycle model to
  logged test drives (least squares) and shares the parameters with Berinay.
- **Game (Nathaly)**: top-down Pygame view; modes Human vs MPCC, Human vs PPO, Human vs Pure
  Pursuit, and all-AI race; opponents are "ghosts" (visible, no collisions); results screen
  with lap times. Rendering should be separate from physics so the game can draw several
  cars stepped side by side.
- **Evaluation**: lap time, completion rate, off-track events, control smoothness, MPCC
  solve time, and generalization on the hidden track.
- **Stretch goals only if the core is done early**: 3D view via MuJoCo rendering, car-to-car
  collisions, more tracks.

## 8. Course concepts this connects to

Optimization (KKT, QPs), LQR / trajectory optimization, nonlinear optimization and SQP, MPC
(including sampling-based MPC such as MPPI), dynamic programming, value/policy iteration,
Q-learning, policy gradients (PPO). The environment owner's own course link: system
identification by least squares (fitting the bicycle model) and the model-mismatch question
(MPCC plans with an approximate model; PPO learns from the real simulator).

## 9. Code rules

- Other team members use **only** the interface above; never read the car's internals.
- SI units, radians, left positive.
- Use only the agreed car values.
- **Changing an agreed decision is allowed when it gives a better outcome.** When you do:
  1. Tell Cristhian what you're changing and why, in plain words, before or as you make it.
  2. Make the change, and update this file (the relevant section) so it stays the source of truth.
  3. Add a line to the "Change log" section at the bottom: date, what changed, who is affected.
  4. Draft a short message Cristhian can send the team (what changed, what they need to do,
     if anything). Nobody should find out about a change by having their code break.
  Prefer changes that keep old calls working (e.g. add a new optional argument or a new
  function) over changes that break teammates' code.
- Keep things simple and explainable: Cristhian must be able to explain every part in the
  presentation. Prefer clear code with short comments over clever code. Don't over-engineer
  the vehicle dynamics – "drives sensibly and has a grip limit" is the bar.
- Work in small, testable pieces; add a quick test or demo for each piece.

## 9b. Environment design notes (from planning – use these)

- **Physics step** a few ms (about 5 ms) inside each 50 ms control step; RK4 or similarly
  stable integration. A dynamic bicycle model is stiff at very low speed, so if one is used,
  blend from a kinematic model below ~0.3 m/s and the dynamic model above ~1 m/s.
- **Steering angle is part of the car state** (because it's rate-limited): the command is a
  target angle; the car moves toward it at most 3.2 rad/s.
- **Grip limit is required** (cap lateral acceleration ~1 g, or tire forces with a peak).
- **Track projection:** smooth periodic spline through the centerline, re-parameterized by arc
  length; dense lookup table; local search around the previous `s`; heading error wrapped to
  [-pi, pi]; keep the track tangent angle continuous (no 2*pi jumps) so it works in CasADi.
- **Lap logic:** a lap counts only after crossing the start line having covered most of the
  track; handle wrap-around of `s`; reversing over the line must not count.
- **Off track:** |e_c| > half road width − half car width. Racing: "grass" = lower grip / extra
  drag, car keeps going, event counted. Training: configurable (end episode or not).
- **Gymnasium:** report `terminated` (e.g. off track in training) separately from `truncated`
  (time/step limit). Declare action/observation spaces. Support starting anywhere on the track
  (`reset(options=...)`).
- **Two run modes:** evaluation/lockstep (sim waits for the controller) and real-time (game).
- **Game view:** Pygame top-down, drawn from each car's position and heading (not MuJoCo's 3D
  camera). Keep rendering separate from physics. Several cars are "ghosts": each has its own
  simulation state, stepped side by side, no collisions. Design so all cars *could* later live in
  one shared MuJoCo scene (collisions would then be a setting, not a rewrite) – stretch goal only.
- **Teleop:** basic arrow keys → throttle/brake and steering commands → same limiter.
- **Car values to add once the car drives** (teammates need them): wheelbase, friction
  coefficient mu, top speed, acceleration/braking limits. Keep grip scalable for the
  "20–30% less friction" robustness test.
- **3D view** (MuJoCo rendering shown in Pygame) is possible later – stretch goal only.

## 9c. Tracks – how they were made

- Both tracks were designed in planning and are final (`tracks/`). Previews show the track,
  start line and driving direction.
- `training_track.csv` was built from straight lines joined by circular arcs (fillets) – radii
  3–7 m – then resampled every 10 cm. Features: main straight + start line, chicane flick,
  carousel (long 180°), back straight, hairpin, double-apex right, hairpin 2.
- `hidden_track.csv` was traced from Cristhian's hand drawing, smoothed, scaled 1.5× so the
  tightest corner is 2.5 m, resampled every 10 cm.
- Rules both follow: road 1.5 m, no corner under 2.5 m radius, no two road sections closer than
  ~5 m centerline-to-centerline, counter-clockwise, first point = start line.

## 10. Change log (changes to agreed decisions – keep this updated)

| Date | Change | Who is affected | Team told? |
|---|---|---|---|
| Nov 22 | `track.project` now takes the car heading: `track.project(x, y, psi, s_prev)` (heading error needs it) | Anyone calling `project` directly; `RacingEnv` passes it automatically | Yes |
| Nov 22 | Added `track.point_at(s)` – centerline point at progress s | Nathaly (Pure Pursuit lookahead) | Yes |
| Nov 22 | Clarified sign: lateral offset positive to the left; MPCC must flip the common contouring formula | Berinay | Yes |
| Nov 22 | Added `env.car_params` (read-only car values: wheelbase, mu, v_max, …), incl. `a_lat_max` = usable grip ≈ 0.89·mu·g (use this, not mu·g, for speed profiles) | Nathaly (speed profile), Berinay (model) | Yes (PR #1 notes) |
| Nov 22 | Added optional track helpers (`heading`, `curvature`, `width`, `lookahead`, `heading_ahead`, `casadi_centerline`); `project` `s_prev` may be `None` | Everyone (optional) | Yes (PR #1 notes) |
| Nov 22 | `info`/`obs` contents fixed (see 5.3); obs has finite bounds; added `progress_rate`, `mu_scale` option | Pavan, Nathaly | Yes (PR #1 notes) |
| Nov 22 | NumPy car switched to the team's planned model (Pacejka tires, Liniger tire shape, mu 0.9, hard top speed, no drag); provisional speed/accel values (see 5.7) | Everyone (car behavior), Berinay (same equations as MPCC) | Yes (PR #1 notes) |
| Nov 22 | NumPy car is the fallback final car instead of a no-contact MuJoCo backup (Cristhian's decision; see 5.7, 7) | Everyone (plan) | Yes (PR #1 notes) |
| Nov 22 | Added `render_mode` ("rgb_array" for videos, "human" window), `gym.make("Racing-v0")` (via it `check_env` passes all checks with zero warnings), `sim.video.record_episode` for MP4s; `requirements.txt` + `imageio[ffmpeg]`. Gymnasium's `RecordVideo` needs `moviepy`, which can't install on Python 3.15 | Pavan (videos, check_env), Nathaly | Yes (PR #1 notes) |
| Nov 22 | Team uses `pygame-ce` instead of `pygame` (`requirements.txt` changed): same `import pygame`, installs on every Python version incl. 3.15; don't install both | Nathaly (game), anyone running the game | Yes (PR #1 notes) |
| Nov 22 | Track CSVs and previews moved from the repo's top level into `tracks/` (the agreed layout); `RacingEnv(track="training")` finds them | Anyone loading the CSVs by path | Yes (PR #1 notes) |
| Nov 22 | Added `sim/car_casadi.py`: the simulator's dynamic and kinematic car equations as CasADi functions (+ RK4 `discretize`, `state_from_info`); tested identical to the NumPy car | Berinay (MPCC) | Yes (PR #1 notes) |
| Nov 22 | GitHub Actions runs the sim tests on every pull request and push to `main` (`.github/workflows/tests.yml`, Python 3.12 + 3.15); main README points to the environment docs | Everyone (PRs show a pass/fail check) | Yes (PR #2 notes) |
| Nov 22 | Repo installable: `pip install -e .` once (new `pyproject.toml`) so `import sim` works from `controllers/`, `game/`, `eval/` scripts | Everyone | Yes (PR #2 notes) |
| Nov 22 | Laps: only a start on the start line times lap 1 (a mid-track start no longer records a short first lap); `lap_log` has `clean`; reset options accept `(low, high)` ranges and reject misspelled names; NaN/inf actions raise an error | Pavan (training starts), Nathaly (metrics) | Yes (PR #2 notes) |
| Nov 22 | Added keyboard teleop `sim/teleop.py` (Up gas ramp, Down brake, Left/Right steer through the car's own limiter; optional speed-based steering assist, off by default) and `sim/examples/drive_keyboard.py`. Human-only input shaping documented in 5.2 (throttle ramp 0.3 s, optional assist) | Nathaly (game), anyone driving; report it in results | Yes (PR #2 notes) |
| Nov 22 | Car physics fix: a car sliding sideways after a spin is slowed by its tires (was instantly turned into forward speed); `car_params` adds `understeer_gradient` (textbook formula, matches the car within 2 %); `sim/car_interface.py` defines what a car must provide (MuJoCo hand-over) | Everyone (car behavior after spins), Nathaly (understeer) | Yes (PR #2 notes) |
