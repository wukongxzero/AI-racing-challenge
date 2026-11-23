# MPCC speed benchmark

The benchmark compares the existing kinematic MPCC and the Pacejka-based dynamic MPCC **on the same local machine**.

```bash
python3 -m controllers.mpc.benchmark_mpcc --speeds 3 4 5 --steps 950 --output mpcc_benchmark.csv
```

To test only the faster settings:

```bash
python3 -m controllers.mpc.benchmark_mpcc --speeds 4 5 --steps 950 --output mpcc_benchmark.csv
```

The output CSV contains lap completion, lap time, off-track events, solver recoveries, brake fallbacks, solution time mean/max, and how often solver wall time exceeds the 50 ms control interval.

## Interpretation

- Tests use the same **NumPy RacingEnv**, training track, seed 0, 20 Hz simulation step and car settings.
- The baseline uses its default **12-step** horizon; the dynamic MPCC uses its **8-step** horizon. This is a fair comparison of the two working controllers, **not** a controlled comparison of model choice alone.
- The dynamic controller uses the existing kinematic controller for launch below 1.5 m/s.
- Wall-clock solve time is measured on the machine executing the benchmark. Changing machine load changes results.
- `mean_solve_ms < 50` does **not** imply that every 20 Hz deadline is met. Track `over_50ms` and `max_solve_ms`.
- A **solver recovery** still incurred a failed optimization attempt and can miss the deadline; zero brake fallbacks is not the same as zero solver issues.
- Simulation advances in fixed 0.05s steps even if wall-clock solve time exceeds 0.05s. This does **not** establish physical real-time operation.
- Do not claim the car model is validated in MuJoCo from these tests: these runs use `car='numpy'`.

## Next engineering steps

1. Profile 50 ms overruns separately by initial IPOPT solve, retry and reference construction.
2. Check hard deadlines and implement an explicit timeout/failsafe for real-time deployment.
3. Repeat on the hidden track and with alternative initial conditions.
4. Check yaw-rate, lateral slip, steer-rate and predicted-vs-measured residuals at higher speeds.
