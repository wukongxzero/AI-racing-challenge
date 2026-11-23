# Dynamic MPCC solver timing experiments

The working Dynamic MPCC remains the default. This update adds an **optional**
IPOPT multiplier warm-start (`--dual-warm-start`) and prints **p95 and p99**
control-step timing as well as mean, maximum and steps exceeding 50 ms.

## Commands (run from the repository root)

```bash
python3 -m unittest controllers.mpc.test_dynamic_mpcc
python3 -m controllers.mpc.run_dynamic_mpcc --speed 4 --steps 950
python3 -m controllers.mpc.run_dynamic_mpcc --speed 4 --steps 950 --dual-warm-start
```

The optional flag warms IPOPT multipliers (`lam_g`) in addition to state and
control guesses. Failed solves clear the multiplier guess before retrying.
It is **not** enabled by default because it did not improve typical latency
in the short test.

## Measurements from this run

Same Linux environment, training track, NumPy RacingEnv, seed 0,
`horizon=8`, `substeps=3`, `dt=0.05 s`, speed cap 4 m/s.
These timing numbers are hardware-dependent and not directly comparable
to MacBook measurements.

| 250-step test | default | dual warm-start |
|---|---:|---:|
| mean step duration | 18.0 ms | 19.9 ms |
| p95 | 17.9 ms | 21.1 ms |
| p99 | 94.1 ms | 137.8 ms |
| maximum | 515.7 ms | 387.2 ms |
| steps > 50 ms | 4 | 4 |
| off-track / fallback | 0 / 0 | 0 / 0 |

Full 4 m/s default-controller validation: 663 steps, 1 lap, 33.15 seconds,
0 off-track, 0 recoveries, 0 fallbacks, 24 steps > 50 ms; mean 20.9 ms,
p95 29.1 ms, p99 150.5 ms, max 425.4 ms.

**Interpretation:** these values measure synchronous solve-and-command overhead.
A simulated 20 Hz step is *not* a strict real-time guarantee. Occasional
outliers exceed the 50 ms deadline and need additional profiling before real
hardware or deadline-critical use. The initial solve includes compilation/
initialization overhead and may produce a large outlier.

## Next engineering tasks

* Measure optimizer-internal iterations and infeasible statuses separately
  from model/reference construction and Python overhead.
* Explore horizon and RK4 substep tradeoffs using a fixed seed and identical
  track, then retain only configurations that finish full laps safely.
* Add an explicit watchdog/control-deadline policy before real-time operation.
