"""Drive the car yourself with the arrow keys (real time, Pygame window).

    python sim/examples/drive_keyboard.py                  # training track
    python sim/examples/drive_keyboard.py --track hidden
    python sim/examples/drive_keyboard.py --assist         # start with the steering assist on

Keys: Up = gas, Down = brake, Left/Right = steer, A = steering assist on/off,
      R = restart, Esc = quit.
Same car, same limits and the same 20 Hz commands as every controller; off track the car
slows on the grass and the event is counted. Lap times are printed when you finish a lap.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sim import RacingEnv  # noqa: E402
from sim.teleop import KeyboardTeleop  # noqa: E402
from sim.viewer import TopDownView  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="training")
    ap.add_argument("--assist", action="store_true", help="start with the steering assist on")
    args = ap.parse_args()

    import pygame
    env = RacingEnv(track=args.track, off_track="grass")
    p = env.car_params
    view = TopDownView(env.track, car_length=p["length"], car_width=p["width"],
                       caption="RacingEnv - arrows: drive | A: steering assist | R: restart | Esc: quit")
    teleop = KeyboardTeleop(p, steer_assist=args.assist)
    obs, info = env.reset()
    prev_keys = pygame.key.get_pressed()
    next_frame = time.perf_counter()
    running = True
    while running:
        # one 50 ms step: read keys -> same command format as the controllers -> env.step
        keys = pygame.key.get_pressed()
        pressed = lambda k: keys[k] and not prev_keys[k]           # key just went down
        if pressed(pygame.K_r):
            obs, info = env.reset()
            teleop.reset()
        if pressed(pygame.K_a):
            teleop.steer_assist = not teleop.steer_assist
        prev_keys = keys

        obs, reward, terminated, truncated, info = env.step(teleop.action_from_pygame(info["vx"]))
        if info["lap_completed"]:
            lap = env.lap_log[-1]
            print(f"lap {lap['lap']}: {lap['lap_time']:.2f} s"
                  f"{'' if lap['clean'] else '  (not clean: ' + str(lap['off_track_count']) + ' off-track)'}")

        best = f"{min(env.lap_times):.2f}" if env.lap_times else "-"
        text = (f"v {info['vx']:4.1f} m/s   lap {info['lap'] + 1}   best {best} s   "
                f"off-track {info['off_track_count']}   assist {'ON' if teleop.steer_assist else 'off'}")
        color = (240, 140, 30) if info["off_track"] else (220, 40, 40)
        running = view.draw([{"x": info["x"], "y": info["y"], "psi": info["psi"], "color": color}], text)

        next_frame += env.dt                                         # real time: 20 steps per s
        time.sleep(max(0.0, next_frame - time.perf_counter()))
        next_frame = max(next_frame, time.perf_counter())
    view.close()


if __name__ == "__main__":
    main()
