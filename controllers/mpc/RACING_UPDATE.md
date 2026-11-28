# Dynamic MPCC racing update (2026-11-28)

Goal: make the MPCC race instead of following the centerline at a cautious speed.
All tuning used the **training track only**; the hidden track was run once at the end.

## What changed

| File | Change |
|---|---|
| `racing_line.py` (new) | Offline minimum-curvature racing line (bounded least squares on the lateral offset, 0.30 m from the edges) and a speed profile on it: `v = sqrt(lat_frac * a_lat_max / |kappa|)`, backward pass with `brake_frac * b_max`, forward pass with `a_max`. Uses only the public track API, so it works on any track. |
| `dynamic_mpcc.py` | Contour error is measured to the racing line (offset `n(s)`), speed target comes from the new profile (was: kinematic MPCC profile with 75 % grip, -0.5 m/s buffer, 3 m/s^2). Road edges and top speed are **soft** constraints (slack, heavy penalty) and the state bounds apply only to predicted states, so a measured slide no longer makes the problem infeasible. `expand=True` (SX) and optional IPOPT `max_cpu_time`. `racing_line=False` gives the old centerline behaviour. |
| `driver.py` (new) | `MPCCDriver(env, config)` = `policy(obs, info)` for `eval/` and the game. Kinematic launch below 1.5 m/s, warm-up solves before the race, and on a failed / timed-out solve it uses the next input of the last good plan instead of retrying (retry, then brake, only without a plan). Presets `RACE` and `ROBUST`. |
| `eval/evaluate.py` | `make_mpcc` filled in; drivers `mpcc` and `mpcc_robust`. |

Presets: `RACE` = horizon 8, RK4 substeps 2, cap 8 m/s, IPOPT time cap 35 ms, `lat_frac` 0.92.
`ROBUST` = same with `lat_frac` 0.62 (plans corners for ~30 % less grip than reported).

## Results (NumPy car, `python -m eval.evaluate`, 3 laps from a standing start, seed 0)

| Driver | Track | μ | Best lap | Off-track | Solve mean / max | > 50 ms |
|---|---|---|---:|---:|---|---:|
| MPCC before (dynamic, h6, 8 m/s) | training | 1.0 | 26.93 s | 0 | 9.8 / 2027 ms | 35 |
| Pure Pursuit | training | 1.0 | 24.03 s | 0 | – | – |
| **MPCC RACE** | training | 1.0 | **18.08 s** | 0 | 10.3 / 40.8 ms | 0 |
| MPCC ROBUST | training | 1.0 | 20.68 s | 0 | 9.9 / 38 ms | 0 |
| Pure Pursuit | hidden | 1.0 | 19.27 s | 0 | – | – |
| **MPCC RACE** | hidden | 1.0 | **14.46 s** | 0 | 12.0 / 61.8 ms | 1 |
| MPCC ROBUST | hidden | 1.0 | 16.46 s | 0 | 10.1 / 40 ms | 0 |

For reference, PPO (`controllers/ppo/RESULTS.md`, mean lap over 3 laps, 5 seeds): training 20.05 ± 0.48 s,
hidden 16.38 ± 0.53 s. Mean laps of MPCC RACE: training 18.35 s, hidden 14.73 s.

Robustness, μ 0.7 (30 % less grip):

| Driver | Friction known? | Training | Hidden |
|---|---|---|---|
| MPCC RACE | yes (`car_params` includes `mu_scale`, like Pure Pursuit) | 20.49 s, 0 off | 16.61 s, 0 off |
| MPCC RACE | no (model built at μ 1.0) | 24.42 s, **18 off-track** | **did not finish** |
| MPCC ROBUST | no (model built at μ 1.0) | 20.75 s, 0 off | 16.76 s, 0 off |

"Friction not known" is the fair comparison with PPO, which is not told the friction.
Like PPO's friction randomization, ROBUST trades lap time at full grip (+2.6 s training) for safety at low grip.

## Caveats

- One deterministic run per condition (the sim is deterministic; seeds would only matter with noise).
- Solve times are wall-clock on a Linux cloud machine; repeat on the presentation laptop.
- The racing line uses the track map, as Pure Pursuit's speed profile does; nothing was tuned on the hidden track.
- Next ideas: online friction estimate (switch RACE/ROBUST or scale `lat_frac`), MuJoCo car if it arrives.

## Reproduce

```bash
python -m eval.evaluate --drivers pure_pursuit mpcc mpcc_robust
python -m eval.evaluate --drivers mpcc mpcc_robust --mu-scale 0.7      # friction known to the controller
python -m unittest controllers.mpc.test_dynamic_mpcc controllers.mpc.test_racing_env_integration
```
