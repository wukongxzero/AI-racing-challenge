# PPO controller (`controllers/ppo/`)

Owner: Pavan. Stable-Baselines3 PPO on the shared `RacingEnv`. It follows `docs/ppo_plan.tex`.
Run everything from the repo root after `pip install -e .` and `pip install -r requirements.txt`.

```bash
python -m controllers.ppo.test_ppo                                   # wrapper checks (seconds)
python -m controllers.ppo.train --run-name full_s0                   # 1M steps, 8 envs, ~10 min
python -m controllers.ppo.train --run-name step3 --reward progress   # progress + crash only
python -m controllers.ppo.train --run-name no_du --w_du 0            # ablation: one weight changed
python -m controllers.ppo.train --run-name dr_s0 --mu-range 0.6 1.0  # friction randomized per episode
tensorboard --logdir runs/ppo
python -m controllers.ppo.evaluate runs/ppo/full_s0                  # race rules, 3 laps
python -m controllers.ppo.evaluate runs/ppo/full_s0 --track hidden --mu-scale 0.7 --video lap.mp4
```

## Files
| File | What it does |
|---|---|
| `wrapper.py` | The PPO problem: 15-D observation, reward, termination, start distribution (`make_env`) |
| `train.py` | SB3 PPO + `SubprocVecEnv` + `VecNormalize`; best/final/checkpoint saving; TensorBoard |
| `evaluate.py` | Race-rule evaluation: lap times, clean-lap and completion rate, smoothness, video, CSV log |
| `driver.py` | `PPODriver(run_folder).act(info)` gives `[throttle, steering]` for any `RacingEnv` (game, races) |
| `test_ppo.py` | Quick checks of the wrapper |

## The problem (plan section 4)
- **Observation (15):** `vx, vy, yaw_rate, delta, e_c, e_psi, prev_throttle, prev_steer`,
  curvature at 0.5–3.0 m ahead (6), `w_half`. There's no global x, y or psi. It is built from `info` only
  (`obs_from_info`), so it doesn't depend on the env's own 21-value `obs`.
- **Reward:** `w_prog*ds − w_c*e_c² − w_psi*e_psi² − w_du*|u−u_prev|² − w_time − R_crash·[left road]`.
  The `full` defaults are 1.0 / 0.05 / 0.05 / 0.05 / 0.01 / 10. The `progress` preset zeroes the shaping terms.
  Each weight can be overridden with `--w_c 0.1` etc.
- **Terminated:** off track (env `off_track="terminate"`), reverse heading `|e_psi| > π/2`, or stall
  (< 0.3 m in 40 steps). **Truncated:** 50 s = 1000 steps.
- **Training starts:** uniform over the lap, `e_c ±0.3 m`, `e_psi ±0.15 rad`, `v0 0.5–2.5 m/s`.
- **Friction randomization (`--mu-range LOW HIGH`, off by default):** each training episode samples
  `mu_scale ~ U(LOW, HIGH)` (`_reset_options` in `wrapper.py`). The policy is not told the friction (the
  observation stays 15-D). Training envs only: the in-training evaluation that picks `best/` stays at
  `--mu-scale` (default 1.0), and `evaluate.py` always uses the `--mu-scale` you give it.
- **Evaluation** (`evaluate.py`, and the game) uses race rules: start line at rest, grass instead of
  episode end, deterministic policy and frozen `VecNormalize` stats.

## Outputs (`runs/ppo/<run>/`, git-ignored)
`config.json` (all args and reward weights) · `best/` (best start-line evaluation; this is the
one to use) · `final/` · `checkpoints/` every 100k steps · `tb/`. A `model.zip` always goes with
the `vecnormalize.pkl` saved next to it.

TensorBoard `race/` shows metres driven per episode, laps per episode and the share of episodes ending by
`off_track` / `reverse` / `stall` / `time_limit`.

## For the game (Nathaly)
```python
from controllers.ppo.driver import PPODriver
driver = PPODriver("runs/ppo/full_s0")     # folder with best/ or final/
action = driver.act(info)                  # info from the RacingEnv of the PPO car
```
