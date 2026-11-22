"""Drive the car yourself with the arrow keys (real time, Pygame window).

    python sim/examples/drive_keyboard.py                  # training track, view follows the car
    python sim/examples/drive_keyboard.py --track hidden
    python sim/examples/drive_keyboard.py --no-assist      # start with the steering assist off
    python sim/examples/drive_keyboard.py --log human.csv  # save each attempt for the results

Keys: Up = gas, Down = brake, Left/Right = steer, Z = close-up / whole track,
      A = steering assist on/off, [ / ] = steering assist weaker / stronger,
      B = braking assist on/off (brakes for you before corners), R = restart, Esc = quit.
Same car, same limits and the same 20 Hz commands as every controller; off track the car
slows on the grass and the event is counted. With --log, every attempt (ended by R or Esc)
is saved as <name>_<n>.csv (every step, same columns as env.save_log plus the assist settings)
and <name>_<n>_laps.csv (lap time, off-track count, clean, assists used) - the same data the
controllers produce, plus what the results must report about the human's assists.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sim import RacingEnv  # noqa: E402
from sim.teleop import KeyboardTeleop  # noqa: E402
from sim.viewer import TopDownView  # noqa: E402

CLOSE_UP = 3.0          # zoom of the follow view


def assist_state(teleop):
    """The human's assist settings right now (recorded every step for honest reporting)."""
    return (bool(teleop.steer_assist), float(teleop.assist_margin), bool(teleop.brake_assist))


def describe_assists(states):
    """Readable summary of the assist settings used, e.g. 'steer x1.4 + brake'."""
    def one(st):
        steer, margin, brake = st
        parts = [f"steer x{margin:.1f}"] if steer else []
        parts += ["brake"] if brake else []
        return " + ".join(parts) or "none"
    names = list(dict.fromkeys(one(st) for st in states))         # unique, in order of use
    return names[0] if len(names) == 1 else "changed: " + " / ".join(names)


def save_attempt(env, log_path, n, assists=None):
    """Save one attempt: every step + one row per lap. `assists` = assist_state() for every
    logged step (row 0 = after reset); it adds the assist settings to both files so results
    can say exactly which assists the human used. Returns the step-log path."""
    stem, _ = os.path.splitext(log_path)
    steps_path, laps_path = f"{stem}_{n}.csv", f"{stem}_{n}_laps.csv"
    log = env.get_log()
    rows = len(log["t"])
    if assists is not None and len(assists) != rows:
        raise ValueError(f"{len(assists)} assist records for {rows} logged steps")
    with open(steps_path, "w", newline="") as f:
        w = csv.writer(f)
        keys = list(log)
        w.writerow(keys + (["steer_assist", "assist_margin", "brake_assist"] if assists else []))
        for i in range(rows):
            row = [int(log[k][i]) if isinstance(log[k][i], (bool, np.bool_)) else log[k][i] for k in keys]
            if assists:
                steer, margin, brake = assists[i]
                row += [int(steer), margin, int(brake)]
            w.writerow(row)
    with open(laps_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lap", "lap_time", "off_track_count", "clean"] + (["assists"] if assists else []))
        for lap in env.lap_log:
            row = [lap["lap"], f"{lap['lap_time']:.3f}", lap["off_track_count"], int(lap["clean"])]
            if assists:                      # steps driven during this lap: log "lap" == lap - 1
                during = [assists[i] for i in range(rows) if log["lap"][i] == lap["lap"] - 1]
                row.append(describe_assists(during or [assists[-1]]))
            w.writerow(row)
    return steps_path


def hud(info, env, teleop, action, zoom):
    """Driver info: timing, then what the car is being told and the assists, then the keys."""
    fmt = lambda v: f"{v:5.2f} s" if v == v else "  -  "            # NaN -> "-"
    last = env.lap_times[-1] if env.lap_times else float("nan")
    best = min(env.lap_times) if env.lap_times else float("nan")
    line1 = (f"lap {info['lap'] + 1}   time {fmt(info['lap_timer'])}   last {fmt(last)}   "
             f"best {fmt(best)}   v {info['vx']:4.1f} m/s")
    line2 = (f"gas {action[0]:+.2f}  steer {action[1]:+.2f}   off-track {info['off_track_count']}   "
             f"steer assist {'ON x' + format(teleop.assist_margin, '.1f') if teleop.steer_assist else 'off'}   "
             f"brake assist {'ON' if teleop.brake_assist else 'off'}")
    line3 = "[Z] view  [A] steer assist  [ ] strength  [B] brake assist  [R] restart  [Esc] quit"
    return line1 + "\n" + line2 + "\n" + line3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="training")
    ap.add_argument("--no-assist", action="store_true", help="start with the steering assist off")
    ap.add_argument("--log", default=None, help="save every attempt, e.g. --log human.csv")
    ap.add_argument("--seconds", type=float, default=None, help="quit after this many seconds")
    args = ap.parse_args()

    import pygame
    env = RacingEnv(track=args.track, off_track="grass")
    p = env.car_params
    view = TopDownView(env.track, car_length=p["length"], car_width=p["width"], zoom=CLOSE_UP,
                       caption="RacingEnv - arrow keys to drive")
    teleop = KeyboardTeleop(p, track=env.track, steer_assist=not args.no_assist)   # assists on
    obs, info = env.reset()
    assists = [assist_state(teleop)]                 # one record per logged step
    attempt = 1
    action = teleop.action()
    prev_keys = pygame.key.get_pressed()
    start = next_frame = time.perf_counter()
    running = True
    while running:
        # Read the keys as they are NOW: refresh Pygame's key state first, otherwise it is
        # from the previous frame (up to 50 ms old) and the car reacts late.
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        pressed = lambda k: keys[k] and not prev_keys[k]           # key just went down
        if pressed(pygame.K_r):
            if args.log:
                print("saved", save_attempt(env, args.log, attempt, assists))
                attempt += 1
            obs, info = env.reset()
            teleop.reset()
            assists = [assist_state(teleop)]
        if pressed(pygame.K_a):
            teleop.steer_assist = not teleop.steer_assist
        if pressed(pygame.K_b):
            teleop.brake_assist = not teleop.brake_assist
        if pressed(pygame.K_LEFTBRACKET):                          # weaker assist = more steering
            teleop.assist_margin = round(teleop.assist_margin + 0.1, 1)
        if pressed(pygame.K_RIGHTBRACKET):                         # stronger, but never below x1.0
            teleop.assist_margin = round(max(1.0, teleop.assist_margin - 0.1), 1)
        if pressed(pygame.K_z):
            view.set_zoom(1.0 if view.zoom > 1.0 else CLOSE_UP)
        prev_keys = keys

        # one 50 ms step: keys -> same command format as the controllers -> env.step
        action = teleop.action_from_pygame(info["vx"], s=info["s"])
        obs, reward, terminated, truncated, info = env.step(action)
        assists.append(assist_state(teleop))
        if info["lap_completed"]:
            lap = env.lap_log[-1]
            print(f"lap {lap['lap']}: {lap['lap_time']:.2f} s"
                  f"{'' if lap['clean'] else '  (not clean: ' + str(lap['off_track_count']) + ' off-track)'}")

        color = (240, 140, 30) if info["off_track"] else (220, 40, 40)
        running = view.draw([{"x": info["x"], "y": info["y"], "psi": info["psi"], "color": color}],
                            hud(info, env, teleop, action, view.zoom))
        if args.seconds is not None and time.perf_counter() - start > args.seconds:
            running = False

        next_frame += env.dt                                         # real time: 20 steps per s
        time.sleep(max(0.0, next_frame - time.perf_counter()))
        next_frame = max(next_frame, time.perf_counter())
    if args.log:
        print("saved", save_attempt(env, args.log, attempt, assists))
    view.close()


if __name__ == "__main__":
    main()
