"""Tests for the stand-in racing environment.

    python sim/tests/test_racing.py      (or: pytest sim/tests)
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from sim import RacingEnv, load_track  # noqa: E402


def test_track_loads_and_closes():
    for name, L in (("training", 126.6), ("hidden", 95.0)):
        tr = load_track(name)
        assert abs(tr.length - L) < 0.5
        x0, y0 = tr.point(0.0)
        x1, y1 = tr.point(tr.length)
        assert math.hypot(x1 - x0, y1 - y0) < 1e-9
        # tightest corner no tighter than ~2.5 m radius
        k = np.abs(tr.lookahead(0.0, np.arange(0, tr.length, 0.05)))
        assert 1.0 / k.max() > 2.3


def test_project_roundtrip():
    tr = load_track("training")
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = rng.uniform(0, tr.length)
        e = rng.uniform(-0.7, 0.7)
        ep = rng.uniform(-1, 1)
        x, y, psi = tr.frenet_to_cartesian(s, e, ep)
        s2, e2, ep2, k = tr.project(x, y, psi, s - 0.2)
        ds = (s2 - s + tr.length / 2) % tr.length - tr.length / 2
        assert abs(ds) < 1e-6 and abs(e2 - e) < 1e-6 and abs(ep2 - ep) < 1e-6
        assert abs(k - tr.curvature(s)) < 1e-9


def test_project_never_jumps():
    """A point near another section of track must still project near s_prev."""
    tr = load_track("training")
    # find two track points close in space but far apart along the lap
    s = np.arange(0, tr.length, 0.1)
    P = np.array([tr.point(si) for si in s])
    best = None
    for i in range(0, len(s), 5):
        d = np.hypot(*(P - P[i]).T)
        far = np.abs((s - s[i] + tr.length / 2) % tr.length - tr.length / 2) > 10
        j = np.argmin(np.where(far, d, np.inf))
        if best is None or d[j] < best[0]:
            best = (d[j], i, j)
    _, i, j = best
    # a point beside section i, displaced toward section j
    p = P[i] + 0.4 * (P[j] - P[i])
    s_loc, *_ = tr.project(p[0], p[1], 0.0, s[i])
    assert abs((s_loc - s[i] + tr.length / 2) % tr.length - tr.length / 2) < 3.5


def test_project_wraps():
    tr = load_track("training")
    x, y, psi = tr.frenet_to_cartesian(0.05)
    s, *_ = tr.project(x, y, psi, tr.length - 0.1)
    assert abs(s - 0.05) < 1e-6


def test_point_at_wraps():
    tr = load_track("training")
    assert np.allclose(tr.point_at(tr.length + 2.0), tr.point_at(2.0))
    x0, y0 = tr.point_at(0.0)
    assert abs(x0 - tr.x_knots[0]) < 1e-9 and abs(y0 - tr.y_knots[0]) < 1e-9


def test_centerline_progress_smooth_and_lap_counts_once():
    """Move the car along the centerline at 4 m/s: progress grows smoothly
    (no jumps, also across the start line) and exactly one lap counts."""
    env = RacingEnv()
    tr = env.track
    env.reset()
    v, dt = 4.0, env.dt
    prev = 0.0
    laps_seen = 0
    n = int((tr.length + 3.0) / (v * dt))
    for k in range(1, n + 1):
        s_true = k * v * dt
        x, y, psi = tr.frenet_to_cartesian(s_true)
        # place the car on the centerline, then let the env update its track state
        env._car.reset(x, y, psi, vx=v)
        prog_before = env._progress
        env._update_track_state()
        env.t += dt
        laps_seen += env._check_lap(prog_before)
        assert abs((env._progress - prev) - v * dt) < 1e-6      # smooth progress
        assert abs(env._progress - s_true) < 1e-6
        prev = env._progress
    assert laps_seen == 1 and env.lap == 1
    assert abs(env.lap_times[0] - tr.length / v) < 1e-6        # crossing time interpolated


def test_to_action_and_limits():
    env = RacingEnv()
    p = env.car_params
    assert np.allclose(env.to_action(p["a_max"], p["delta_max"]), [1, 1])
    assert np.allclose(env.to_action(-p["b_max"], -p["delta_max"]), [-1, -1])
    assert np.allclose(env.to_action(100, -5), [1, -1])
    assert np.allclose(env.to_action(0, 0), [0, 0])


def test_steering_rate_limit():
    env = RacingEnv()
    env.reset(options={"v0": 2.0})
    _, _, _, _, info = env.step([0.0, 1.0])
    assert abs(info["delta"] - 3.2 * 0.05) < 1e-9          # one step at max rate
    for _ in range(10):
        _, _, _, _, info = env.step([0.0, 1.0])
    assert abs(info["delta"] - 0.42) < 1e-9                # saturates at max angle


def test_left_positive():
    env = RacingEnv()
    env.reset(options={"v0": 2.0})
    for _ in range(5):
        _, _, _, _, info = env.step([0.0, 1.0])
    assert info["yaw_rate"] > 0 and info["e_c"] > 0         # full left -> turns left


def test_brake_stops_no_reverse():
    env = RacingEnv()
    env.reset(options={"v0": 5.0})
    for _ in range(40):
        _, _, _, _, info = env.step([-1.0, 0.0])
    assert info["vx"] == 0.0


def _max_lat_accel(env):
    """Largest sideways acceleration (car frame) in the log, from the driven path."""
    L = env.get_log()
    c, s = np.cos(L["psi"]), np.sin(L["psi"])
    V = np.stack([L["vx"] * c - L["vy"] * s, L["vx"] * s + L["vy"] * c], 1)
    A = np.gradient(V, env.dt, axis=0)
    a_side = -s * A[:, 0] + c * A[:, 1]
    return np.abs(a_side)[2:].max()


def test_grip_limit():
    """Sideways acceleration never exceeds the tire limit mu*g (the tires' peak force),
    and random inputs never blow up the integration."""
    env = RacingEnv(off_track="grass")
    mu_g = env.car_params["mu"] * 9.81
    env.reset(options={"v0": 2.0})
    for _ in range(400):
        env.step([0.6, 0.6])                       # circle -> saturates and slides
    assert _max_lat_accel(env) < 1.05 * mu_g
    rng = np.random.default_rng(1)
    for ep in range(5):
        env.reset(seed=ep, options={"random_start": True, "v0": rng.uniform(0, 7)})
        for _ in range(300):
            _, _, _, _, info = env.step(rng.uniform(-1, 1, 2))
            assert all(np.isfinite([info["vx"], info["vy"], info["yaw_rate"]]))
        assert _max_lat_accel(env) < 1.05 * mu_g


def test_grass_slows_and_counts():
    env = RacingEnv(off_track="grass")
    env.reset(options={"e_c0": 1.2, "v0": 6.0})               # start on grass
    for _ in range(40):
        _, _, term, _, info = env.step([1.0, 0.0])
    assert not term and info["off_track"] and info["vx"] < 3.0
    env.reset(options={"v0": 3.0, "e_psi0": 0.6})              # drive off the road
    for _ in range(40):
        _, _, term, _, info = env.step([0.3, 0.0])
    assert info["off_track_count"] >= 1 and not term


def test_off_track_uses_car_width():
    """Off track as soon as any part of the car is past the edge."""
    env = RacingEnv()
    half_car = 0.5 * env.car_params["width"]
    edge = 0.75 - half_car
    _, info = env.reset(options={"e_c0": edge - 0.02})
    assert not info["off_track"]
    _, info = env.reset(options={"e_c0": edge + 0.02})
    assert info["off_track"]
    _, info = env.reset(options={"e_c0": -(edge + 0.02)})
    assert info["off_track"]


def test_mu_scale():
    """Less friction -> less grip: hard cornering reaches less sideways acceleration."""
    def max_ay(mu_scale):
        env = RacingEnv(mu_scale=mu_scale)
        _, info = env.reset(options={"v0": 5.0})
        for _ in range(60):                        # hold 5 m/s, full left steering
            _, _, _, _, info = env.step([np.clip(2.0 * (5.0 - info["vx"]), -1, 1), 1.0])
        return _max_lat_accel(env)
    full, reduced = max_ay(1.0), max_ay(0.7)
    assert reduced < 0.8 * full
    env = RacingEnv()
    base = env.car_params["mu"]
    env.reset(options={"mu_scale": 0.7})
    assert abs(env.car_params["mu"] - 0.7 * base) < 1e-12
    env.reset()                                    # setting persists until changed
    assert abs(env.car_params["mu"] - 0.7 * base) < 1e-12
    env.reset(options={"mu_scale": 1.0})
    assert abs(env.car_params["mu"] - base) < 1e-12


def test_a_lat_max_matches_car():
    """car_params['a_lat_max'] (usable grip) matches what the car reaches in steady
    cornering, within 5 %, also with reduced friction. Uses the car alone (no track,
    so no grass)."""
    from sim.car_numpy import CarParams, NumpyCar
    for mu_scale in (1.0, 0.7):
        params = CarParams()
        params.mu *= mu_scale
        best = 0.0
        for v in (5.0, 7.0):
            for steer in (0.3, 0.42):
                car = NumpyCar(params)
                car.reset(0.0, 0.0, 0.0, vx=v)
                for _ in range(120):               # hold speed, constant steering
                    car.step(np.clip(2.0 * (v - car.state[3]), -1, 1), steer)
                best = max(best, car.state[3] * car.state[5])
        a = params.a_lat_max
        assert abs(best - a) < 0.05 * a, (mu_scale, best, a)
    assert abs(RacingEnv(mu_scale=0.7).car_params["a_lat_max"] - 0.7 * CarParams().a_lat_max) < 1e-9


def test_info_for_ppo():
    env = RacingEnv()
    obs, info = env.reset()
    for k in ("vx", "vy", "yaw_rate", "delta", "e_c", "e_psi", "prev_throttle", "prev_steer",
              "w_left", "w_right", "progress_rate"):
        assert k in info
    assert len(info["kappa_ahead"]) == 6 and len(info["heading_ahead"]) == 6
    assert obs.shape == (len(env.obs_names),)
    _, _, _, _, info = env.step([1.0, 0.0])
    assert info["progress_rate"] > 0


def test_obs_matches_names():
    """obs[i] is exactly the value env.obs_names[i] names in info (PPO depends on it)."""
    env = RacingEnv()
    obs, info = env.reset(options={"random_start": True}, seed=5)
    for _ in range(30):
        obs, _, _, _, info = env.step([0.6, 0.3])
    for i, name in enumerate(env.obs_names):
        if name.startswith("kappa_ahead_"):
            want = info["kappa_ahead"][env.lookahead.index(float(name.split("_")[-1]))]
        else:
            want = info[name]
        assert abs(float(obs[i]) - float(want)) < 1e-4 * (1 + abs(float(want))), name


def test_viewer_headless():
    try:
        import pygame  # noqa: F401
    except ImportError:
        print("  (pygame not installed - skipped)")
        return
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    from sim.viewer import TopDownView
    env = RacingEnv()
    _, info = env.reset()
    view = TopDownView(env.track)
    assert view.draw([{"x": info["x"], "y": info["y"], "psi": info["psi"]},
                      {"x": 0.0, "y": 0.0, "psi": 1.0, "color": (0, 0, 255)}], text="test")
    view.close()


def test_terminate_mode():
    env = RacingEnv(off_track="terminate")
    env.reset(options={"v0": 3.0, "e_psi0": 0.6})
    term = False
    for _ in range(40):
        _, _, term, _, info = env.step([0.3, 0.0])
        if term:
            break
    assert term and info["off_track"]


def _drive(env, info, n):
    from sim.examples.drive_lap import controller
    for _ in range(n):
        _, _, term, trunc, info = env.step(controller(env, info))
        if term or trunc:
            break
    return info, term, trunc


def test_lap_counting_and_log(tmp_path=None):
    env = RacingEnv(max_laps=1, time_limit=60.0)
    _, info = env.reset()
    info, term, trunc = _drive(env, info, 2000)
    assert term and env.lap == 1 and len(env.lap_times) == 1
    assert 15 < env.lap_times[0] < 40
    log = env.get_log()
    assert len(log["t"]) == env.steps + 1
    path = os.path.join(tmp_path or os.path.dirname(__file__), "_log_test.csv")
    env.save_log(path)
    assert os.path.getsize(path) > 0
    if tmp_path is None:
        os.remove(path)


def test_partial_first_lap_not_counted():
    env = RacingEnv(time_limit=40.0)
    _, info = env.reset(options={"s0": 80.0, "v0": 2.0})
    info, _, _ = _drive(env, info, 300)                      # crosses the line once
    assert env.lap == 0 and info["progress"] > env.track.length


def test_first_lap_timed_only_from_start_line():
    """Starting 8 m after the line, the first crossing must not count as a lap (it would
    be a 94 % lap with a too-short time); the next full lap counts with a real time."""
    env = RacingEnv(max_laps=1, time_limit=80, log=False)
    _, info = env.reset(options={"s0": 8.0, "v0": 3.0})
    info, term, _ = _drive(env, info, 3000)
    assert term and env.lap == 1
    assert info["progress"] > 2 * env.track.length          # it drove ~2 laps for 1 counted
    assert 23 < env.lap_times[0] < 30                        # a full lap (~25-26 s)
    assert env.lap_log[0]["clean"] is True


def test_reset_options_checked_and_ranges():
    env = RacingEnv()
    try:
        env.reset(options={"V0": 5.0})                       # typo must not be ignored
    except ValueError:
        pass
    else:
        raise AssertionError("misspelled option accepted")
    speeds = set()
    for seed in range(20):
        _, info = env.reset(seed=seed, options={"random_start": True, "v0": (0.5, 2.5),
                                                "e_psi0": (-0.15, 0.15)})
        assert 0.5 <= info["vx"] <= 2.5 and abs(info["e_psi"]) <= 0.15 + 1e-9
        speeds.add(round(info["vx"], 6))
    assert len(speeds) > 15                                  # really drawn each reset
    a = env.reset(seed=3, options={"v0": (0.5, 2.5)})[0]
    b = env.reset(seed=3, options={"v0": (0.5, 2.5)})[0]
    assert np.array_equal(a, b)                              # and reproducible with a seed


def test_bad_actions_and_order_rejected():
    env = RacingEnv()
    try:
        env.step([0.0, 0.0])                                 # before reset
    except RuntimeError:
        pass
    else:
        raise AssertionError("step before reset accepted")
    env.reset()
    for bad in ([float("nan"), 0.0], [0.0, float("inf")]):
        try:
            env.step(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"non-finite action {bad} accepted")


def test_random_start_seeded():
    env = RacingEnv()
    o1, _ = env.reset(seed=3, options={"random_start": True})
    o2, _ = env.reset(seed=3, options={"random_start": True})
    assert np.array_equal(o1, o2)
    assert o1.shape == env.observation_space.shape and o1.dtype == np.float32


def test_gymnasium_api():
    try:
        import gymnasium as gym
        from gymnasium.utils.env_checker import check_env
    except ImportError:
        print("  (gymnasium not installed - skipped)")
        return
    import warnings
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # the render check opens a window
    env = RacingEnv(off_track="terminate", time_limit=30)
    assert isinstance(env, gym.Env)
    # Made through gym.make, all checks run (render modes too) with zero warnings
    # (Pavan's plan); unrelated warnings, e.g. deprecation notices in other Gymnasium
    # versions, are ignored
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(gym.make("Racing-v0", off_track="terminate", time_limit=30).unwrapped)
    ours = [str(w.message) for w in caught if "WARN" in str(w.message)]
    assert not ours, ours
    venv = gym.vector.SyncVectorEnv([lambda: RacingEnv(off_track="terminate", time_limit=5,
                                                       log=False)] * 2)
    obs, _ = venv.reset(seed=0)
    for _ in range(120):                      # crosses an auto-reset
        obs, *_ = venv.step(venv.action_space.sample())
    assert obs.shape == (2, env.observation_space.shape[0])


def test_casadi_centerline():
    try:
        import casadi as ca
    except ImportError:
        print("  (casadi not installed - skipped)")
        return
    tr = load_track("hidden")
    f = tr.casadi_centerline()
    for s in np.linspace(-4.0, tr.length + 19.0, 200):
        x, y = tr.point(s)
        assert abs(float(f["x"](s)) - x) < 1e-4 and abs(float(f["y"](s)) - y) < 1e-4
        dth = (float(f["theta"](s)) - tr.heading(s) + np.pi) % (2 * np.pi) - np.pi
        assert abs(dth) < 1e-3
    # usable symbolically (MPCC): gradient of the squared distance to the centerline
    S = ca.MX.sym("s")
    d2 = (f["x"](S) - 1.0) ** 2 + (f["y"](S) - 2.0) ** 2
    g = ca.Function("g", [S], [ca.gradient(d2, S)])
    assert np.isfinite(float(g(10.0)))


def test_render_modes():
    """rgb_array returns a picture that changes as the car moves; human mode runs in a
    (hidden) window; no render_mode draws nothing; a wrong mode is refused."""
    try:
        import pygame  # noqa: F401
    except ImportError:
        print("  (pygame not installed - skipped)")
        return
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env = RacingEnv(render_mode="rgb_array")
    env.reset(seed=0)
    first = env.render()
    assert first.shape == (640, 960, 3) and first.dtype == np.uint8
    for _ in range(10):
        env.step([1.0, 0.0])
    assert not np.array_equal(first, env.render())
    env.close()
    env = RacingEnv(render_mode="human")
    env.reset()
    env.step([0.5, 0.0])                       # draws itself each step
    env.close()
    env = RacingEnv()
    env.reset()
    assert env.render() is None
    try:
        RacingEnv(render_mode="video")
    except ValueError:
        pass
    else:
        raise AssertionError("bad render_mode accepted")


def test_record_episode_video(tmp_path=None):
    """record_episode writes an MP4 with one frame per step (plus the start frame)."""
    try:
        import imageio.v2 as imageio
        import pygame  # noqa: F401
    except ImportError:
        print("  (imageio or pygame not installed - skipped)")
        return
    from sim.video import record_episode
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    path = os.path.join(str(tmp_path or os.path.dirname(__file__)), "_video_test.mp4")
    env = RacingEnv(render_mode="rgb_array")
    record_episode(env, lambda obs, info: [1.0, 0.0], path, max_steps=30)
    reader = imageio.get_reader(path)
    assert reader.count_frames() == 31
    reader.close()
    if tmp_path is None:
        os.remove(path)


def test_casadi_car_matches_simulator():
    """The CasADi car (for MPCC) gives the same numbers as the simulator: same derivatives
    at random states, and the same 50 ms step (above the low-speed blend zone)."""
    try:
        import casadi as ca
    except ImportError:
        print("  (casadi not installed - skipped)")
        return
    from sim.car_casadi import discretize, dynamic_model, kinematic_model, state_from_info
    from sim.car_numpy import CarParams, NumpyCar
    rng = np.random.default_rng(0)
    p = CarParams()
    car = NumpyCar(p)
    f, fk = dynamic_model(p), kinematic_model(p)
    for _ in range(300):
        x = [rng.uniform(-20, 20), rng.uniform(-20, 20), rng.uniform(-3, 3),
             rng.uniform(0.5, 8), rng.uniform(-2, 2), rng.uniform(-4, 4)]
        a, d = rng.uniform(-8, 5), rng.uniform(-0.42, 0.42)
        num = np.array(car._dynamic(x + [d], a, 0.0, p.mu)[:6])
        assert np.allclose(np.array(f(x, [a, d])).ravel(), num, rtol=1e-10, atol=1e-10)
        numk = np.array(car._kinematic(x[:4] + [d], a, 0.0)[:4])
        assert np.allclose(np.array(fk(x[:4], [a, d])).ravel(), numk, rtol=1e-10, atol=1e-10)
    F = discretize(f, dt=p.dt, substeps=p.substeps)
    for _ in range(100):
        d, thr = rng.uniform(-0.42, 0.42), rng.uniform(-0.3, 1.0)
        x0 = [0.0, 0.0, rng.uniform(-3, 3), rng.uniform(1.5, 7.0), rng.uniform(-0.5, 0.5),
              rng.uniform(-2, 2)]
        car.reset(*x0, delta=d)
        car.step(thr, d)                              # steering already at its target
        if not (1.0 < car.state[3] < p.v_max):
            continue                                  # blend zone / top-speed cap: not compared
        a = thr * (p.a_max if thr >= 0 else p.b_max)
        nxt = np.array(F(x0, [a, d])).ravel()
        nxt[2] = (nxt[2] + np.pi) % (2 * np.pi) - np.pi
        assert np.allclose(nxt, car.state[:6], atol=1e-9)
    # takes env.car_params (incl. mu_scale) and gives finite gradients for the optimizer
    env = RacingEnv(mu_scale=0.7)
    _, info = env.reset(options={"v0": 4.0})
    x, u = ca.SX.sym("x", 6), ca.SX.sym("u", 2)
    fe = dynamic_model(env.car_params)
    J = ca.Function("J", [x, u], [ca.jacobian(fe(x, u), ca.vertcat(x, u))])
    assert np.all(np.isfinite(np.array(J(state_from_info(info), [1.0, 0.2]))))


def test_keyboard_teleop():
    """Arrow keys -> standard commands: gas ramps up, brake wins, release coasts; left is
    positive; the car still applies its own steering-rate limit; the assist only ever asks
    for less steering."""
    from sim.teleop import KeyboardTeleop
    env = RacingEnv()
    tele = KeyboardTeleop(env.car_params, throttle_ramp_time=0.3)
    thr = [tele.action(up=True)[0] for _ in range(8)]
    assert thr[0] < thr[1] < thr[5] == 1.0                    # ramps to full in 0.3 s
    assert tele.action(up=True, down=True)[0] == -1.0          # brake wins
    assert tele.action()[0] == 0.0                             # release = coast
    assert tele.action(left=True)[1] == 1.0 and tele.action(right=True)[1] == -1.0
    assert tele.action(left=True, right=True)[1] == 0.0
    env.reset(options={"v0": 3.0})                             # same limiter as controllers
    _, _, _, _, info = env.step(tele.action(left=True))
    assert abs(info["delta"] - 3.2 * 0.05) < 1e-9
    assert KeyboardTeleop(env.car_params).steer_assist            # on by default (play-tested)
    assist = KeyboardTeleop(env.car_params, steer_assist=True)
    for v in (0.0, 2.0, 5.0, 8.0):
        s = assist.action(left=True, speed=v)[1]
        assert 0.0 < s <= 1.0
    assert assist.action(left=True, speed=8.0)[1] < assist.action(left=True, speed=3.0)[1]
    try:
        KeyboardTeleop(steer_assist=True)
    except ValueError:
        pass
    else:
        raise AssertionError("assist without car_params accepted")


def test_sideways_slide_slowed_by_tires():
    """A car sliding sideways after a spin (little forward speed) is slowed by its tires at
    about mu*g - its speed must not be turned into forward speed."""
    import math
    from sim.car_numpy import CarParams, NumpyCar
    p = CarParams()
    car = NumpyCar(p)
    car.reset(0.0, 0.0, 0.0, vx=0.1, vy=2.0)
    car.step(0.0, 0.0)
    speed = math.hypot(car.state[3], car.state[4])
    assert car.state[3] < 0.3                                    # not redirected forward
    decel = (2.0 - speed) / p.dt                                 # ~ mu*g, never more
    assert 0.6 * p.mu * 9.81 < decel < 1.1 * p.mu * 9.81
    for _ in range(10):
        car.step(0.0, 0.0)
    assert math.hypot(car.state[3], car.state[4]) < 0.3          # comes to (nearly) a stop


def test_understeer_gradient_matches_car():
    """car_params['understeer_gradient'] (textbook formula) matches the simulated car in
    steady cornering within 5 %, also with reduced friction."""
    from sim.car_numpy import CarParams, NumpyCar
    for mu_scale in (1.0, 0.7):
        env = RacingEnv(mu_scale=mu_scale)
        K = env.car_params["understeer_gradient"]
        p = CarParams()
        p.mu *= mu_scale
        L = p.wheelbase
        for v, d in ((3.0, 0.04), (5.0, 0.03)):
            car = NumpyCar(p)
            car.reset(0.0, 0.0, 0.0, vx=v, delta=d)
            for _ in range(200):
                car.step(np.clip(2.0 * (v - car.state[3]), -1, 1), d)
            vx, r = car.state[3], car.state[5]
            k_measured = (d - L * r / vx) / (vx * r)
            assert abs(k_measured - K) < 0.05 * K, (mu_scale, v, k_measured, K)


def test_car_interface():
    """The NumPy car provides what sim/car_interface.py says every car must provide."""
    from sim.car_interface import make_car
    from sim.car_numpy import CarParams
    car = make_car("numpy", CarParams())
    for attr in ("p", "state", "reset", "step"):
        assert hasattr(car, attr)
    car.reset(1.0, 2.0, 0.5, vx=3.0)
    car.step(0.5, 0.1, grass=False)
    assert len(car.state) == 7 and all(np.isfinite(car.state))
    try:
        make_car("lego", CarParams())
    except ValueError:
        pass
    else:
        raise AssertionError("unknown car accepted")


def test_viewer_close_up_follows_car():
    """zoom > 1 draws the car bigger and keeps it in the middle of the window; text may have
    several lines; zoom 1 still shows the whole track."""
    try:
        import pygame  # noqa: F401
    except ImportError:
        print("  (pygame not installed - skipped)")
        return
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from sim.viewer import TopDownView
    env = RacingEnv()
    _, info = env.reset(options={"s0": 40.0})
    view = TopDownView(env.track, window=(400, 300), zoom=3.0, headless=True)
    car = {"x": info["x"], "y": info["y"], "psi": info["psi"], "color": (255, 0, 0)}
    view.draw([car], text="line one" + chr(10) + "line two")
    img = view.frame()
    red = np.argwhere((img[:, :, 0] == 255) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0))
    assert len(red) > 50                                      # car drawn big (close-up)
    assert np.all(np.abs(red.mean(0) - [150, 200]) < 3)      # and in the window centre
    whole = TopDownView(env.track, window=(400, 300), zoom=1.0, headless=True)
    assert abs(view.scale - 3.0 * whole.scale) < 1e-9
    view.set_zoom(1.0)
    assert view.background.get_size() == (400, 300)


def test_lap_timer():
    """info['lap_timer'] counts the current lap and restarts at the line; NaN until a lap
    is being timed after a mid-track start."""
    env = RacingEnv(max_laps=1, time_limit=60, log=False)
    _, info = env.reset()
    assert info["lap_timer"] == 0.0
    info, _, _ = _drive(env, info, 3000)
    assert env.lap == 1 and 0.0 <= info["lap_timer"] < env.dt
    _, info = env.reset(options={"s0": 50.0})
    assert info["lap_timer"] != info["lap_timer"]            # NaN


def test_keyboard_attempt_log(tmp_path=None):
    """drive_keyboard --log saves every step and one row per lap, like the controllers."""
    from sim.examples.drive_keyboard import save_attempt
    folder = str(tmp_path or os.path.dirname(__file__))
    env = RacingEnv(max_laps=1, time_limit=60)
    _, info = env.reset()
    _drive(env, info, 3000)
    steps = save_attempt(env, os.path.join(folder, "_human.csv"), 1)
    laps = os.path.join(folder, "_human_1_laps.csv")
    with open(laps) as f:
        rows = f.read().strip().splitlines()
    assert rows[0] == "lap,lap_time,off_track_count,clean" and len(rows) == 2
    with open(steps) as f:
        assert f.readline().startswith("t,x,y,psi")
    if tmp_path is None:
        os.remove(steps)
        os.remove(laps)


def test_mujoco_placeholder():
    try:
        RacingEnv(car="mujoco")
    except NotImplementedError:
        return
    raise AssertionError


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("FAIL", t.__name__, repr(exc))
    sys.exit(1 if failed else 0)
