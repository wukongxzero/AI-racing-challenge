# MPCC + shared RacingEnv (first integration)

The entry point `run_racing_env.py` runs Berinay's existing **kinematic MPCC** against
Cristhian's shared `RacingEnv(car="numpy")`, which uses dynamic bicycle / Pacejka tire
physics. It does not alter the team's simulator, MuJoCo code or other controllers.

## Run from repo root

```bash
pip install -r requirements.txt
python -m controllers.mpc.run_racing_env --steps 880 --plot mpcc_env.png --log mpcc_env.csv
python -m unittest controllers.mpc.test_racing_env_integration
```

Options: `--track training|hidden`, `--speed 3.0`, `--horizon 12`, `--off-track grass|terminate`,
`--steps N`. Keep `hidden` for final evaluation rather than controller tuning.

## How it works

- `info["x"], ["y"], ["psi"], ["vx"]` form the kinematic MPCC state.
- `info["progress"]` is the real unwrapped track progress; `info["delta"]` is the
  actual wheel steering angle for the slew-rate constraint.
- `SharedTrackAdapter` maps the older vectorized MPCC track calls to
  `env.track.point_at`, `.heading`, `.curvature`, `.width`, `.project`.
  **There is only one track geometry**, supplied by RacingEnv.
- Vehicle parameters (`dt`, wheelbase, steering rate, acceleration/brake limits,
  speed) come from `env.car_params`, not placeholder values.
- The boundary safety margin is half the body width plus 3 cm. Track constraints
  remain the older locally linearized approximation; they are not hard physical
  collision constraints.
- `env.to_action(accel, steer)` converts physical commands into normalized actions.
  On solve failure the program applies full braking while holding actual steering.
- No reference to MuJoCo internals is required. The shared car interface can later
  swap the physics model when it is available.

## Measured smoke test (NumPy car, training track, speed cap 3 m/s)

The 100-step run covered 14.08 m in 5 s with zero off-track events, zero solver
fallbacks and a 14.9 ms mean solver time (294.5 ms maximum at initial solve).
The 880-step limit run finished one physical lap at step 856 (42.80 s simulated),
with zero off-track events and six solver fallbacks. Mean solve time was ~19 ms,
but some individual solves were **over 400 ms**, so a strict 20 Hz real-time
runtime is **not yet guaranteed**. Results depend on machine and packages.

## Limitations / next steps

1. **Not dynamic MPCC yet.** The predictor ignores lateral velocity and yaw
   rate, while the simulator does not. Next integrate `sim.car_casadi` dynamics,
   including steering actuation and velocity states.
2. The solver sometimes fails around corners; inspect IPOPT status, improve
   warm starts and consider soft boundaries / a safe restoration strategy.
3. Optimize worst-case solve latency, not just the mean, before a real-time demo.
4. Test robustness across friction values and initial conditions on the training
   track; reserve the hidden track for final evaluation.
5. `max_laps=1` terminates the run at a real measured completed lap.

## Solver recovery update (Nov 22, 2026)

`run_racing_env.py` now logs IPOPT return statuses and, on a failed warm-start,
tries **once** with a freshly seeded trajectory based on measured progress.
If the retry fails it still brakes safely. Use `--no-retry` for comparisons.
This is a *simulation recovery policy*, not a hard real-time or safety guarantee.

Local training-track trial (`--steps 880`, 3 m/s, NumPy car): one lap in
42.70 simulation seconds, zero off-track steps, six recovered optimization
failures, zero brake fallbacks. Mean solver time 18.9 ms; 28 out of 854 steps
exceeded the 50 ms budget and the maximum was 454.5 ms. Results vary by
machine. Root cause statuses observed: `Infeasible_Problem_Detected` and
`Maximum_Iterations_Exceeded`. The dynamic-model MPCC is not implemented yet.
