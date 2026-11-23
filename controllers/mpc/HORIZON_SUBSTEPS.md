# Dynamic MPCC: horizon / RK4 substeps tuning (2026-11-22)

Run from repo root with the existing dependencies installed:

```bash
python3 -m controllers.mpc.run_dynamic_mpcc --speed 4 --steps 950 --fast --plot fast_4ms.png
python3 -m controllers.mpc.run_dynamic_mpcc --speed 5 --steps 950 --fast --plot fast_5ms.png
python3 -m controllers.mpc.run_dynamic_mpcc --speed 4 --steps 950 --horizon 8 --substeps 3
```

The `--fast` option selects horizon=6, substeps=2. Defaults remain horizon=8, substeps=3 so the prior controller is unchanged. Tests use `RacingEnv(car='numpy', track='training', off_track='grass')`, seed=0, one-lap termination. Measurements were produced in the current execution environment; repeat on the target MacBook before making real-time claims.

## 4 m/s, initial 250 steps

| Horizon | RK4 substeps | Mean (ms) | P99 (ms) | >50 ms | Off-track | Fallback |
|---|---|---:|---:|---:|---:|---:|
| 8 | 3 | 19.8 | 105.5 | 4 | 0 | 0 |
| 8 | 2 | 15.6 | 85.1 | 3 | 0 | 0 |
| 6 | 3 | 13.7 | 53.7 | 3 | 0 | 0 |
| 6 | 2 | 11.4 | 48.5 | 3 | 0 | 0 |

## Full-lap validation

| Speed cap | Horizon | Substeps | Lap (s) | Mean (ms) | P99 (ms) | >50 ms | Off-track | Fallback |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 4 m/s | 6 | 2 | 33.15 | 13.1 | 71.4 | 22 / 663 | 0 | 0 |
| 4 m/s | 8 | 2 | 33.15 | 18.4 | 139.2 | 24 / 663 | 0 | 0 |
| 5 m/s | 6 | 2 | 29.65 | 13.5 | 86.3 | 20 / 593 | 0 | 0 |

**Limitations.** Shorter horizon weakens lookahead; reduced RK4 substeps reduce integration accuracy. Results here are on one track and seed, not sufficient to establish robustness. Timing counts include solve attempts, retries and Python overhead; 50 ms overruns still occur. The simulation time step is 0.05 seconds but wall-clock deterministic 20 Hz operation is **not** established. Compare high-speed/high-curvature and additional seeds before deployment.
