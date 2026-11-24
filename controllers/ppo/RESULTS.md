# PPO results – friction randomization (2026-11-23)

Training PPO with random friction (`--mu-range 0.6 1.0`) keeps it on track at 30% less grip,
at a cost of about 4% lap time on normal grip.

All numbers: NumPy car, 3-lap race from a standing start on the start line, deterministic
policy, frozen `VecNormalize` stats, `best/` checkpoint of each run. Every run trained on the
**training track only**; the hidden track was used only in these final evaluations (no tuning,
retraining or model selection on it). Per-run numbers: `eval_results_2026-11-23.csv`.

## Mean over 3 seeds

| Condition | Baseline PPO | Friction-randomized PPO (μ 0.6–1.0) |
|---|---|---|
| Training track, μ 1.0 | **20.3 s**, 0 off-track | 21.1 s, 0 off-track |
| Training track, μ 0.7 | 31.8 s\*, ~22 off-track per run | **23.3 s**, 0 off-track |
| Hidden track, μ 1.0 | **16.6 s**, 0 off-track | 17.2 s, 0 off-track |
| Hidden track, μ 0.7 | 25.6 s, ~20 off-track per run | **19.6 s**, 3 off-track (all seed 0) |

\* Mean of the 2 seeds that finished; `full_s2` completed 0 laps at μ 0.7 on the training track.

## Per seed (mean lap time s / off-track events)

| Run | Group | Train μ 1.0 | Train μ 0.7 | Hidden μ 1.0 | Hidden μ 0.7 |
|---|---|---|---|---|---|
| `full_v0fix` (seed 0) | baseline | 19.91 / 0 | 33.12 / 21 | 16.09 / 0 | 26.00 / 21 |
| `full_s1` | baseline | 20.81 / 0 | 30.43 / 24 | 17.26 / 0 | 24.17 / 18 |
| `full_s2` | baseline | 20.06 / 0 | 0 laps / 5 | 16.34 / 0 | 26.56 / 21 |
| `dr_s0` | randomized | 21.38 / 0 | 23.61 / 0 | 17.38 / 0 | 19.90 / 3 |
| `dr_s1` | randomized | 20.90 / 0 | 22.88 / 0 | 16.97 / 0 | 19.15 / 0 |
| `dr_s2` | randomized | 21.01 / 0 | 23.42 / 0 | 17.09 / 0 | 19.64 / 0 |

Both groups: `full` reward, SB3 PPO defaults (see README), 1M steps (62 rollouts × 16,384),
8 envs. The only difference is `--mu-range 0.6 1.0`. Caveat: one deterministic 3-lap run per
seed and condition; enough for the low-grip gap, the ~0.7 s full-grip gap needs more seeds.

## Reproduce

```bash
python -m controllers.ppo.train --run-name full_s1 --seed 1                      # baseline (full_v0fix = seed 0)
python -m controllers.ppo.train --run-name dr_s1 --seed 1 --mu-range 0.6 1.0     # randomized
python -m controllers.ppo.evaluate runs/ppo/dr_s1 --track training --mu-scale 0.7
python -m controllers.ppo.evaluate runs/ppo/dr_s1 --track hidden --mu-scale 1.0 --video lap.mp4
```

Models (`runs/`) and videos (`*.mp4`) are not in git; videos are in the team Drive folder
"AI Racing – PPO videos (2026-11-23)".
