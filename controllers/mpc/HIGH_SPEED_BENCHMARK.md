# Dynamic MPCC high-speed benchmark

## Reproduce

From the repository root:

```bash
python3 -m controllers.mpc.benchmark_high_speed --speeds 6 7 8 --steps 950 --output high_speed.csv
```

Uses the `--fast` equivalent settings: prediction horizon 6, RK4 substeps 2, training track, deterministic reset seed 0, 20 Hz simulated controller period and NumPy/Pacejka RacingEnv plant. Produces one summary CSV plus one per-step CSV for each speed.

## Observed results (single runs)

| Speed cap | Lap (s) | Off-track | Fallback | Recovered retries | Peak abs cross-track error (m) | Peak body slip (deg) | Solver >50ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6 m/s | 28.30 | 0 | 0 | 0 | 0.014 | 6.305 | 17 |
| 7 m/s | 27.90 | 0 | 0 | 2 | 0.014 | 6.314 | 16 |
| 8 m/s | 27.75 | 0 | 0 | 3 | 0.014 | 6.298 | 15 |

Cautions:

- Cap is a maximum, **not** a constant velocity. Curvature-based speed targets reduce speed in corners, so the 8 m/s cap only modestly improves the lap time relative to 6 m/s.
- Body slip is calculated as `atan2(vy, vx)` in degrees; it is **not** front or rear tire slip angle.
- Solver timings are measured on the benchmark machine and include first-solve compilation, warm-start recovery, and occasional latency spikes. 50 ms violations mean this is **not proven real-time** at 20 Hz.
- One deterministic lap per configuration is a smoke test, not a robustness study. Test other tracks, seeds, solver limits and disturbed initial conditions before claiming robustness.
- No changes to the car's tire physics or track boundaries were made in this benchmark.
