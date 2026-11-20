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

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
