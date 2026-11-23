# Dynamic MPCC integration (experimental)

## Status

The validated kinematic MPCC remains in `mpcc.py`. The new controller is in
`dynamic_mpcc.py`; the new simulation runner is `run_dynamic_mpcc.py`.
The vehicle is the **team's NumPy simulator**, not MuJoCo.

From the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest controllers.mpc.test_dynamic_mpcc
python3 -m controllers.mpc.run_dynamic_mpcc --steps 950 --plot dynamic_mpcc.png --log dynamic_mpcc.csv
python3 -m controllers.mpc.run_dynamic_mpcc --baseline --steps 950 --plot kinematic_baseline.png
```

## Controller

- State `z=[X,Y,psi,vx,vy,r,s]`: global position, heading, vehicle-frame velocities,
  yaw rate, and virtual progress on the track.
- Input `u=[a,delta,vs]`: requested longitudinal acceleration, steering target,
  and virtual progress rate. Only acceleration and steering are sent to RacingEnv.
- Vehicle RHS: **exact `sim.car_casadi.dynamic_model`** Pacejka dynamic bicycle function.
- Prediction integration: `sim.car_casadi.discretize` RK4, 3 substeps by default.
  The simulator uses 10 substeps; this is a speed/accuracy tradeoff, **not exactly
  the same discrete dynamics**.
- Receding-horizon nonlinear optimization: IPOPT, 8 control steps (0.4 s), 20 Hz.
- Objective: path contour/lag tracking, speed tracking, forward progress, smooth
  steering, acceleration penalty, and mild lateral-speed/yaw-rate regularization.
- Constraints: local linearized track edges with car half-width + 0.03 m clearance;
  acceleration, steering angle and rate, positive forward speed, bounded lateral
  speed and yaw rate.
- Speed reference: reused from the kinematic MPCC's curvature-based speed planner.
- At less than 1.5 m/s the model has ill-conditioned slip angles and the
  simulator blends to its kinematic dynamics. We **explicitly** hand off to
  the existing kinematic controller for launch (7 steps in first full-lap test).
- On solve failure retry once from cold start; if both attempts fail, brake.

## First controlled comparison (3 m/s, training track)

| Metric | Stable kinematic run (user Mac) | Dynamic test (this environment) |
|---|---:|---:|
| Lap time | 42.70 s | 42.50 s |
| Off-track events | 0 | 0 |
| Final braking fallbacks | 0 | 0 |
| Solver recovery events | 8 | 0 |
| Mean optimization time | 23.3 ms | 18.4 ms |
| >50 ms calls | 31 | 31 |

**Interpretation:** successful first full-lap run, not proof of real-time
readiness. Hardware, startup time, and numerical conditions vary. The new
MPCC also uses a shorter horizon than the original baseline. Repeat both
algorithms on the same machine with multiple speeds, tracks, and seeds.

## Known limitations

- The simulator's rate-limited steering actuator is approximated by bounded
  steering target changes. The simulator applies its own rate limiter.
- The simulator's low-speed blend, speed clipping, grass drag and grip changes
  are not included in the dynamic predictor.
- Local track references are frozen during each optimization (sequential
  linearization); this is not a globally exact track constraint.
- A hard 50 ms real-time deadline is not guaranteed. A more robust runtime
  needs time limits, background or compiled solver performance checks and a
  verified fallback strategy.
- More extensive robustness tests are needed before high-speed claims.
