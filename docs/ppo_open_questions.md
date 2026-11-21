# PPO — open questions checklist

## Cristhian (environment) — blocks coding
- [ ] Interface: `reset()` / `step()` signatures, obs + info contents — was it agreed (planned 11-17)?
- [ ] Status: does any env version run yet (even placeholder)?
- [ ] Car limits: max steering angle, steering rate, accel/brake range, top speed
- [ ] Timing: control 20 Hz? physics dt?
- [ ] Track: which one, length, width; track module gives s, e_c, e_psi, lookahead points?
- [ ] Settings at reset: can friction / mass be changed? (robustness study)
- [ ] Speed: approx. env steps per second

## Team
- [ ] Prelim report: submitted? who submits, format?
- [ ] Metrics: lap time, off-track count, smoothness — one shared definition (Nathaly's eval)
- [ ] Eval protocol: start positions, # laps, held-out track, lockstep mode
- [ ] Confirm my extras: PPO ablations + robustness study (friction change, unseen track)

## Course (Slack / Brightspace)
- [ ] Prelim report spec + late deadline
- [ ] Pre-report (12-03) and final report requirements, page limits

## Can do now (no answers needed)
- [ ] PPO concepts (L11, Schulman 2017 §3, §5)
- [ ] Toy run: SB3 PPO on Pendulum/CartPole, read TensorBoard
- [ ] Formulation: obs, action, reward, termination → report §4.3
- [ ] Placeholder env: kinematic car on an oval
