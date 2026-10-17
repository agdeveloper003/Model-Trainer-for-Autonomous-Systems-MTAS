"""
Hand-written "rule-based" driver: it does NOT learn anything.

It is useful for two things:
  1. checking that the environment works (it must be able to drive a lap);
  2. giving us a reference lap time to beat with machine learning.

It only uses the sensors, just like the neural network will:
  - it steers towards the direction where the rays see the most free space;
  - it adjusts its speed to how much room there is ahead.

    python bot.py                            # prints the lap time on the classic circuit
    python bot.py --track random --seed 42   # same on a random circuit
"""
import argparse

import numpy as np

from env import CarCircuitEnv
from track import add_track_args, spec_from_args


def rule_based_policy(env):
    dist, ang = env.ray_dist, env.ray_angles
    car = env.car

    # steering: average of the ray angles weighted by free space
    w = dist ** 4
    target = float((w * ang).sum() / w.sum())
    steer = np.clip(target / car.p.max_steer * 1.6, -1, 1)

    # speed: how fast can I go and still stop before the wall ahead
    mid = len(dist) // 2
    front = float(dist[mid - 1: mid + 2].min())
    target_speed = np.sqrt(2 * car.p.brake_decel * 0.55 * max(front - 12.0, 0.0)) + 14.0
    throttle = np.clip((target_speed - car.speed) * 0.4, -1, 1)
    return np.array([steer, throttle])


def main():
    ap = argparse.ArgumentParser(description="Rule-based driver (reference lap time).")
    ap.add_argument("--laps", type=int, default=1)
    add_track_args(ap)
    args = ap.parse_args()
    track = spec_from_args(args).build()

    env = CarCircuitEnv(track=track, laps_to_finish=args.laps)
    env.reset()
    total_reward, info = 0.0, {}
    while True:
        _, r, term, trunc, info = env.step(rule_based_policy(env))
        total_reward += r
        if term or trunc:
            break

    outcome = "CRASH" if info.get("crashed") else ("lap completed" if info.get("laps") else "stalled / time up")
    print(f"Outcome: {outcome}")
    print(f"Laps: {info.get('laps')}  times: {[round(t, 2) for t in env.lap_times]}")
    print(f"Progress: {info.get('progress', 0):.1%} of a lap  total reward: {total_reward:.1f}")


if __name__ == "__main__":
    main()
