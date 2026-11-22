# AI Racing Challenge

ROB-GY 6323 (Reinforcement Learning & Optimal Control) team project, Fall 2026.
2D racing sim comparing Pure Pursuit, MPC, and PPO controllers, plus a human player.

## Team

| Member | Name | Area |
|---|---|---|
| 1 | Cristhian Moreno | Environment & dynamics: MuJoCo car, track module, Gymnasium env |
| 2 | Berinay Demirci | Optimal control: MPCC (CasADi + IPOPT) |
| 3 | Pavan Velagaleti | Reinforcement learning: PPO observations, reward, training |
| 4 | Nathaly Berrezueta | Integration & analytics: Pure Pursuit, game modes, metrics, plots |

## Layout

| Folder | Owner | Contents |
|---|---|---|
| `sim/` | Cristhian | vehicle model, physics step (runs without Pygame) |
| `tracks/` | Cristhian | track definitions |
| `controllers/pure_pursuit/` | Nathaly | Pure Pursuit baseline |
| `controllers/mpc/` | Berinay | MPC (CasADi + IPOPT) |
| `controllers/ppo/` | Pavan | Gymnasium env wrapper, reward, PPO training |
| `game/` | Nathaly | Pygame front end, race modes |
| `eval/` | Nathaly | metrics, comparison plots |
| `docs/` | all | proposal, reports |

## Environment (`sim/`)

The shared racing environment all controllers use: a Gymnasium env with the car, the tracks and the
agreed interface (20 Hz, actions `[throttle, steering]` in [-1, 1], SI units, left positive).

```python
from sim import RacingEnv
env = RacingEnv(car="numpy", track="training")   # or gymnasium.make("Racing-v0", ...)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step([throttle, steering])
```

- How to use it (observations, car values, CasADi car for MPCC, videos): [`sim/README.md`](sim/README.md)
- What changed from the agreed plan, plus notes for each teammate: [PR #1 description](https://github.com/wukongxzero/AI-racing-challenge/pull/1)
- Agreed interface, timeline and change log: [`CLAUDE.md`](CLAUDE.md)
- Try it: `python sim/examples/drive_lap.py` · tests: `python sim/tests/test_racing.py`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows (PowerShell): `python -m venv .venv`, then `.venv\Scripts\Activate.ps1`, then `pip install -r requirements.txt`.
