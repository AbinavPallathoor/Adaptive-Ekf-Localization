"""MuJoCo simulation window launched by app.py. On macOS run with mjpython.

Drives the chosen route and streams one JSON sample per line on stdout at
20 Hz of simulation time (see robot.FIELDS): ground-truth pose, wheel encoder
angles, noisy gyro yaw rate and, at the fix rate, a noisy position fix.
Prints 'ready' once the window is open. Reads 'speed N' lines on stdin and
exits when its window is closed or stdin closes (the app quit or pressed Stop).
"""

import argparse
import sys
import threading
import time

import robot

FRAME_STEPS = 5  # physics steps per displayed frame at 1× speed (10 ms)


def main():
    import mujoco.viewer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', choices=robot.ROUTES, required=True)
    parser.add_argument('--speed', type=int, choices=robot.SPEEDS, default=1, help='times faster than real time')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gyro-noise', type=float, default=.01, help='σ, rad/s')
    parser.add_argument('--fix-noise', type=float, default=.1, help='σ, m')
    parser.add_argument('--fix-rate', type=float, choices=robot.FIX_RATES, default=1, help='Hz')
    args = parser.parse_args()
    model, data = robot.load()
    follower = robot.RouteFollower(args.route, robot.true_pose(data))
    sensors = robot.Sensors(args.seed, args.gyro_noise, args.fix_noise, args.fix_rate)
    settings = {'speed': args.speed}
    app_open = threading.Event()
    app_open.set()

    def watch_app():
        for line in sys.stdin:
            speed = robot.parse_command(line)
            if speed is not None:
                settings['speed'] = speed
        app_open.clear()

    threading.Thread(target=watch_app, daemon=True).start()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [.4, .4, .05]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.8, 135, -45
        print('ready', flush=True)
        steps = 0
        while viewer.is_running() and app_open.is_set():
            started = time.monotonic()
            with viewer.lock():
                # Faster speeds take more physics steps per displayed frame.
                for _ in range(settings['speed']):
                    if steps % robot.STEPS_PER_SAMPLE == 0:
                        print(robot.encode_sample(sensors.sample(robot.readings(data))), flush=True)
                        target = follower.target(data)
                    robot.apply_target(model, data, target, FRAME_STEPS)
                    steps += FRAME_STEPS
            viewer.sync()
            time.sleep(max(0, FRAME_STEPS*model.opt.timestep - (time.monotonic()-started)))


if __name__ == '__main__':
    main()
