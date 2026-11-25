"""Evaluation pipeline shared by every driver (Nathaly).

Every controller is run the same way and measured with the same numbers, so the
lap-time table compares controllers, not measurement code.

    from eval.metrics import run_episode, compute_metrics
    result = run_episode(env, policy, name="Pure Pursuit")   # policy(obs, info) -> action
    print(result["metrics"])

A "policy" is any function policy(obs, info) -> [throttle, steering]:
    Pure Pursuit:  lambda obs, info: pp(info)
    PPO (SB3):     lambda obs, info: model.predict(obs, deterministic=True)[0]
    MPCC:          lambda obs, info: mpcc_step(info)
The time each call takes is measured too (MPCC solve time, PPO inference time).

Metrics (all from the env's per-step log and lap log, see sim/README.md "Logging"):
    laps_completed, completion   laps finished / laps asked for
    lap_times, best_lap, mean_lap, lap_std
    clean_laps                   laps with no off-track event
    off_track_events             times the car left the road (entering the grass = 1 event)
    off_track_time_frac          share of time spent off the road
    mean_abs_e_c, max_abs_e_c    distance from the centerline (m)
    mean_speed, max_speed        m/s
    steer_rate_rms               RMS change of the steering command per second (1/s, on the
                                 [-1, 1] scale) - lower = smoother
    throttle_rate_rms            same for throttle
    steer_reversals_per_s        sign changes of the steering rate (wiggle / chatter)
    compute_ms_mean, compute_ms_max   time per controller call (ms)
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np


def run_episode(env, policy, name="controller", seed=0, reset_options=None, on_step=None):
    """Run one episode to the end (env's max_laps or time_limit) and measure it.

    Returns {"name", "track", "log", "lap_log", "compute_s", "metrics"}.
    on_step(info) is called after every step (for live viewers); return False to stop.
    """
    if env.max_laps is None and env.time_limit is None:
        raise ValueError("give the env max_laps or time_limit, or the episode never ends")
    obs, info = env.reset(seed=seed, options=reset_options)
    if hasattr(policy, "reset"):
        policy.reset()
    compute = []
    terminated = truncated = False
    while not (terminated or truncated):
        t0 = time.perf_counter()
        action = policy(obs, info)
        compute.append(time.perf_counter() - t0)
        obs, reward, terminated, truncated, info = env.step(action)
        if on_step is not None and on_step(info) is False:
            break
    log = env.get_log()
    lap_log = [dict(l) for l in env.lap_log]
    metrics = compute_metrics(log, lap_log, dt=env.dt, laps_asked=env.max_laps,
                              compute_s=np.asarray(compute))
    metrics.update({"truncated": bool(truncated), "terminated": bool(terminated),
                    "mu_scale": float(env.mu_scale)})
    return {"name": name, "track": env.track.name, "log": log, "lap_log": lap_log,
            "compute_s": np.asarray(compute), "metrics": metrics}


def compute_metrics(log, lap_log, dt=0.05, laps_asked=None, compute_s=None):
    """Numbers for one run, from env.get_log() and env.lap_log (dicts of arrays / list of dicts)."""
    lap_times = [float(l["lap_time"]) for l in lap_log]
    off = np.asarray(log["off_track"], dtype=bool)
    # an event = a step where the car goes from on-road to off-road (or starts off-road)
    events = int(off[0]) + int(np.sum(off[1:] & ~off[:-1])) if len(off) else 0
    steer = np.asarray(log["steer"], dtype=float)
    thr = np.asarray(log["throttle"], dtype=float)
    d_steer = np.diff(steer) / dt
    d_thr = np.diff(thr) / dt
    sgn = np.sign(d_steer[np.abs(d_steer) > 1e-3])
    duration = float(log["t"][-1]) if len(log["t"]) else 0.0
    m = {
        "laps_completed": len(lap_times),
        "completion": (len(lap_times) / laps_asked) if laps_asked else None,
        "lap_times": lap_times,
        "best_lap": min(lap_times) if lap_times else None,
        "mean_lap": float(np.mean(lap_times)) if lap_times else None,
        "lap_std": float(np.std(lap_times)) if len(lap_times) > 1 else None,
        "clean_laps": int(sum(bool(l.get("clean", l["off_track_count"] == 0)) for l in lap_log)),
        "off_track_events": events,
        "off_track_time_frac": float(off.mean()) if len(off) else 0.0,
        "mean_abs_e_c": float(np.mean(np.abs(log["e_c"]))),
        "max_abs_e_c": float(np.max(np.abs(log["e_c"]))),
        "mean_speed": float(np.mean(log["vx"])),
        "max_speed": float(np.max(log["vx"])),
        "steer_rate_rms": float(np.sqrt(np.mean(d_steer ** 2))) if len(d_steer) else 0.0,
        "throttle_rate_rms": float(np.sqrt(np.mean(d_thr ** 2))) if len(d_thr) else 0.0,
        "steer_reversals_per_s": (float(np.sum(sgn[1:] != sgn[:-1])) / duration) if duration > 0 and len(sgn) > 1 else 0.0,
        "duration_s": duration,
    }
    if compute_s is not None and len(compute_s):
        m["compute_ms_mean"] = float(np.mean(compute_s) * 1e3)
        m["compute_ms_max"] = float(np.max(compute_s) * 1e3)
    return m


# ------------------------------------------------------------------ saving / loading
def save_result(result, out_dir="results"):
    """Save <name>_<track>.csv (per-step log + compute time) and <name>_<track>.json (metrics, laps)."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{slug(result['name'])}_{result['track']}")
    log = dict(result["log"])
    if len(result["compute_s"]) == len(log["t"]):
        log["compute_ms"] = result["compute_s"] * 1e3
    keys = list(log)
    with open(stem + ".csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for row in zip(*(log[k] for k in keys)):
            f.write(",".join(str(int(v)) if isinstance(v, (bool, np.bool_)) else repr(float(v)) for v in row) + "\n")
    with open(stem + ".json", "w") as f:
        json.dump({"name": result["name"], "track": result["track"],
                   "metrics": result["metrics"], "lap_log": result["lap_log"]}, f, indent=2, default=float)
    return stem


def load_result(stem):
    """Inverse of save_result (stem = path without extension)."""
    with open(stem + ".json") as f:
        meta = json.load(f)
    data = np.genfromtxt(stem + ".csv", delimiter=",", names=True)
    meta["log"] = {k: data[k] for k in data.dtype.names}
    return meta


def slug(name):
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


def format_table(results, keys=("best_lap", "mean_lap", "completion", "clean_laps", "off_track_events",
                                "max_abs_e_c", "mean_speed", "steer_rate_rms", "compute_ms_mean")):
    """Plain-text (Markdown) table of metrics, one row per result."""
    head = "| controller | track | " + " | ".join(keys) + " |"
    lines = [head, "|" + "---|" * (len(keys) + 2)]
    for r in results:
        m = r["metrics"]
        cells = []
        for k in keys:
            v = m.get(k)
            cells.append("-" if v is None else (f"{v:.3f}" if isinstance(v, float) and not math.isnan(v) else str(v)))
        lines.append(f"| {r['name']} | {r['track']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
