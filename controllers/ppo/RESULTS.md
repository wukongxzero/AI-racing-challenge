# PPO results – friction randomization (5 seeds, updated 2026-11-25)

Training PPO with random friction (`--mu-range 0.6 1.0`) keeps **4 of 5 seeds** on track at
30% less grip (baseline: 0 of 5), at a cost of about 4–5% lap time on normal grip. One
randomized seed (`dr_s3`) still goes off track at μ 0.7, so randomization makes robustness
likely, not guaranteed.

All numbers: NumPy car, 3-lap race from a standing start on the start line, deterministic
policy, frozen `VecNormalize` stats, `best/` checkpoint of each run. Every run trained on the
**training track only**; the hidden track was used only in these final evaluations (no tuning,
retraining or model selection on it). Per-run numbers: `eval_results_2026-11-25.csv`
(40 evaluations = 10 runs × 2 tracks × 2 frictions; the 2026-11-23 file has the first 3 seeds).

## Summary over 5 seeds

"Clean" = all 3 laps finished with 0 off-track events. Lap time = mean ± std over the seeds
that finished all 3 laps.

| Condition | Baseline PPO | Friction-randomized PPO (μ 0.6–1.0) |
|---|---|---|
| Training track, μ 1.0 | 5/5 clean, **20.05 ± 0.48 s** | 5/5 clean, 20.98 ± 0.51 s |
| Training track, μ 0.7 | **0/5 clean**, 3/5 finished, 32.60 ± 1.97 s, 14.8 off-track per run | **4/5 clean**, 5/5 finished, 24.36 ± 2.38 s, 3.0 off-track per run |
| Hidden track, μ 1.0 | 5/5 clean, **16.38 ± 0.53 s** | 5/5 clean, 17.06 ± 0.48 s |
| Hidden track, μ 0.7 | **0/5 clean**, 4/5 finished, 26.15 ± 1.53 s, 16.2 off-track per run | **3/5 clean**, 5/5 finished, 20.32 ± 1.68 s, 2.4 off-track per run |

## Per seed (mean lap time s / off-track events; "0 laps" = did not finish)

| Run | Group | Train μ 1.0 | Train μ 0.7 | Hidden μ 1.0 | Hidden μ 0.7 |
|---|---|---|---|---|---|
| `full_v0fix` (seed 0) | baseline | 19.91 / 0 | 33.12 / 21 | 16.09 / 0 | 26.00 / 21 |
| `full_s1` | baseline | 20.81 / 0 | 30.43 / 24 | 17.26 / 0 | 24.17 / 18 |
| `full_s2` | baseline | 20.06 / 0 | 0 laps / 5 | 16.34 / 0 | 26.56 / 21 |
| `full_s3` | baseline | 19.99 / 0 | 0 laps / 3 | 16.34 / 0 | 0 laps / 3 |
| `full_s4` | baseline | 19.48 / 0 | 34.26 / 21 | 15.88 / 0 | 27.86 / 18 |
| `dr_s0` | randomized | 21.38 / 0 | 23.61 / 0 | 17.38 / 0 | 19.90 / 3 |
| `dr_s1` | randomized | 20.90 / 0 | 22.88 / 0 | 16.97 / 0 | 19.15 / 0 |
| `dr_s2` | randomized | 21.01 / 0 | 23.42 / 0 | 17.09 / 0 | 19.64 / 0 |
| `dr_s3` | randomized | 20.16 / 0 | 28.59 / 15 | 16.31 / 0 | 23.29 / 9 |
| `dr_s4` | randomized | 21.43 / 0 | 23.31 / 0 | 17.54 / 0 | 19.64 / 0 |

Both groups: `full` reward, SB3 PPO defaults (see README), 1M steps (62 rollouts × 16,384),
8 envs. The only difference is `--mu-range 0.6 1.0`.

**What 5 seeds change vs the 3-seed result (2026-11-23):** the low-grip benefit holds but is
smaller than 3 seeds suggested (0 off-track per run → 2–3 on average) because `dr_s3`
did not learn a robust policy. The full-grip cost is now clearer: +0.9 s (≈4.6%) on the
training track and +0.7 s (≈4.2%) on the hidden track, about 1.5–2 std of the seed spread.

**Caveats:** one deterministic 3-lap run per seed and condition; 5 seeds show spread and
failure rate, not statistical significance. NumPy car only (MuJoCo retrain pending).
Main PPO for the comparison: `dr_s1` (clean in every condition; shared in the team Drive).

## Reproduce

```bash
python -m controllers.ppo.train --run-name full_s1 --seed 1                      # baseline (full_v0fix = seed 0; seeds 0–4)
python -m controllers.ppo.train --run-name dr_s1 --seed 1 --mu-range 0.6 1.0     # randomized
python -m controllers.ppo.evaluate runs/ppo/dr_s1 --track training --mu-scale 0.7
python -m controllers.ppo.evaluate runs/ppo/dr_s1 --track hidden --mu-scale 1.0 --video lap.mp4
```

Models (`runs/`) and videos (`*.mp4`) are not in git; videos are in the team Drive folder
"AI Racing – PPO videos (2026-11-23)".
