# AI Racing Challenge

ROB-GY 6323 (Reinforcement Learning & Optimal Control) team project, Fall 2026.

Four "drivers" race the **same simulated 1:10 race car on the same track**: a Pure Pursuit baseline,
MPCC (model predictive contouring control), a PPO agent (reinforcement learning) and a human with the
keyboard. Car, track and limits are identical for everyone, so differences in results come from the
controllers alone. The simulator doubles as a Pygame game.

## Team

| Member | Name | Area |
|---|---|---|
| 1 | Cristhian Moreno | Environment & dynamics: MuJoCo car, track module, Gymnasium env |
| 2 | Berinay Demirci | Optimal control: MPCC (CasADi + IPOPT) |
| 3 | Pavan Velagaleti | Reinforcement learning: PPO observations, reward, training |
| 4 | Nathaly Berrezueta | Integration & analytics: Pure Pursuit, game modes, metrics, plots |

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .                   # once: makes `import sim` work from any folder (else "No module named 'sim'")
```

Try it:

```bash
python sim/examples/drive_lap.py         # watch a demo lap
python sim/examples/drive_keyboard.py    # drive it yourself with the arrow keys
python sim/tests/test_racing.py          # run the tests (a few seconds)
```

## Layout

| Folder | Owner | Contents |
|---|---|---|
| `sim/` | Cristhian | the shared racing environment: car, track code, Gymnasium env, viewer, keyboard teleop |
| `tracks/` | Cristhian | track CSVs: `training_track.csv` (127 m) and `hidden_track.csv` (95 m, never used for training) |
| `controllers/pure_pursuit/` | Nathaly | Pure Pursuit baseline |
| `controllers/mpc/` | Berinay | MPCC (CasADi + IPOPT) |
| `controllers/ppo/` | Pavan | observation/reward wrapper, PPO training |
| `game/` | Nathaly | Pygame front end, race modes |
| `eval/` | Nathaly | metrics, comparison plots |
| `docs/` | all | proposal, reports |

## The shared environment

Every controller drives through the same `RacingEnv`
([full guide, including a "start here by role" section: `sim/README.md`](sim/README.md)):

```python
from sim import RacingEnv
env = RacingEnv(car="numpy", track="training")   # or gymnasium.make("Racing-v0", ...)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step([throttle, steering])
```

Rules everyone follows (agreed by the team, see [`CLAUDE.md`](CLAUDE.md)):

- **Commands:** two numbers in [-1, 1], `[throttle, steering]` (+1 = full gas / full left), sent at
  **20 Hz** (each held 50 ms). The car limits steering to 0.42 rad and 3.2 rad/s for everyone.
- **Units:** meters, seconds, radians; **left = positive**.
- **Interface only:** use `RacingEnv`, `env.track`, `env.to_action` and `env.car_params`; never the
  car's internals, so the MuJoCo car can replace the NumPy car without changing your code.
- **Changes to shared decisions** are proposed to the whole team first and recorded in the
  change log in `CLAUDE.md`.

## Tests

GitHub runs the environment's tests on every pull request (Python 3.12 and 3.15): a **green check**
means the shared environment still works; a **red X** means something broke (**Details** shows which
test). Locally: `python sim/tests/test_racing.py`. Workflow: [`.github/workflows/tests.yml`](.github/workflows/tests.yml).

## Where to find things

| What | Where |
|---|---|
| How to use the environment, notes for each role | [`sim/README.md`](sim/README.md) |
| Agreed decisions, timeline, change log | [`CLAUDE.md`](CLAUDE.md) (also read by AI assistants; [`AGENTS.md`](AGENTS.md) points other AI tools there) |
| What changed in each update | the pull requests ([#1](https://github.com/wukongxzero/AI-racing-challenge/pull/1), [#2](https://github.com/wukongxzero/AI-racing-challenge/pull/2)) |
| Proposal and reports | [`docs/`](docs/) |
