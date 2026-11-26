# Pure Pursuit (baseline) – Nathaly

A geometric path-tracking controller plus a curvature-based speed profile. Since it uses no
optimization and no learning, it's the "how far does simple geometry get you?" reference
for MPCC and PPO.

```bash
python -m controllers.pure_pursuit.run                    # watch 2 laps (Pygame window)
python -m controllers.pure_pursuit.run --track hidden --no-view
python -m controllers.pure_pursuit.test_pure_pursuit      # checks (a few seconds)
python -m controllers.pure_pursuit.tune                   # parameter sweep, training track only
python -m eval.evaluate                                   # metrics table + plots -> results/
```

## How it works

**Steering (Coulter 1992).** Take a goal point on the centerline a distance $L_d$ ahead of the
rear axle. Write it in the car frame as $(g_x, g_y)$, with x forward and y left. One circle
starts at the rear axle, is tangent to the car's heading and passes through the goal point. Its
curvature is

$$\kappa = \frac{2 g_y}{L_d^2}, \qquad \delta = \arctan(L\,\kappa)$$

where $L$ = wheelbase (0.33 m). The lookahead grows with speed, $L_d = L_{d,\min} + k_v v$.
Because the car understeers, the geometric angle is multiplied by $(1 + K v^2 / L)$, with
$K$ = `car_params["understeer_gradient"]`.

**Speed.** The speed profile is computed once from the track curvature:

1. cornering limit $v(s) = \min\big(\sqrt{a_{lat}/|\kappa(s)|},\ v_{max}\big)$, with $a_{lat} = 0.9\cdot$`a_lat_max`;
2. backward pass so the car can brake in time: $v_i \le \sqrt{v_{i+1}^2 + 2 b\,\Delta s}$;
3. forward pass for the motor limit: $v_{i+1} \le \sqrt{v_i^2 + 2 a\,\Delta s}$.

A P-controller with feed-forward tracks the profile slightly ahead of the car (0.3 s preview):
$a = k_p (v_{ref} - v) + v_{ref}\, dv_{ref}/ds$.

## Parameters (defaults tuned on the training track only)

| name | default | meaning |
|---|---|---|
| `ld_min` | 0.4 m | lookahead at standstill |
| `k_v` | 0.15 s | extra lookahead per m/s |
| `lat_margin` | 0.9 | fraction of the car's sideways grip used by the speed profile |
| `brake_margin` | 0.7 | fraction of max braking used when planning |
| `k_p` | 3.0 1/s | speed-tracking gain |

Tuning result (`tune.py`, 60 settings): a shorter lookahead is faster but less stable.
`k_v = 0.10` is 0.1 s faster on the training track but spins out on the hidden track.
`lat_margin = 0.95` gains about 0.3 s but comes within 3.5 cm of the off-track line.

## Results (NumPy car, 3 laps from a standing start, `python -m eval.evaluate`)

| track | grip | best lap | off-track events | max \|e_c\| |
|---|---|---|---|---|
| training (127 m) | 100 % | 24.03 s | 0 | 0.49 m |
| hidden (95 m) | 100 % | 19.27 s | 0 | 0.49 m |
| training | 70 % | 27.56 s | 0 | 0.45 m |
| hidden | 70 % | 22.21 s | 0 | 0.54 m |

Off track starts at |e_c| > 0.595 m. Known weaknesses, which are useful for the comparison:

- It cuts corners: it chases a point ahead and doesn't plan a racing line.
- It slows below its target in corners, because the front tire's side force drags the car.
- It knows nothing about tire slip beyond the understeer factor.

Note: `env.car_params` reflects `mu_scale`, so in the 70 % test Pure Pursuit *knows* the grip
is lower and plans slower speeds.
