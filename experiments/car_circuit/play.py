"""
Drive the car yourself with the keyboard.

    python play.py                                  # classic circuit
    python play.py --track random --seed 42         # a random circuit

Arrows or WASD: up = throttle, down = brake, left/right = steer.
R restart, C switch view, V sensors on/off, ESC quit.

Try to drive a clean lap: you will notice that entering a corner too fast
makes the car "slide" (TYRES AT THE LIMIT) towards the wall.
"""
import argparse

import pygame

from env import CarCircuitEnv
from render import Renderer
from track import add_track_args, spec_from_args


def keyboard_action():
    k = pygame.key.get_pressed()
    left = k[pygame.K_LEFT] or k[pygame.K_a]
    right = k[pygame.K_RIGHT] or k[pygame.K_d]
    up = k[pygame.K_UP] or k[pygame.K_w]
    down = k[pygame.K_DOWN] or k[pygame.K_s]
    steer = float(left) - float(right)
    throttle = float(up) - float(down)
    return [steer, throttle]


def main():
    ap = argparse.ArgumentParser(description="Drive the car with the keyboard.")
    add_track_args(ap)
    track = spec_from_args(ap.parse_args()).build()

    # 60 steps per second in real time; endless laps until you crash
    env = CarCircuitEnv(track=track, dt=1 / 60, substeps=1, laps_to_finish=10_000,
                        max_time=1e9, stall_time=1e9)
    renderer = Renderer(env)
    env.reset()
    action = [0.0, 0.0]

    running = True
    while running:
        running = renderer.handle_events()
        if renderer.reset_requested:
            renderer.reset_requested = False
            env.reset()

        if not env.done:
            action = keyboard_action()
            env.step(action)

        renderer.draw(action)
        renderer.tick(60)

    renderer.close()


if __name__ == "__main__":
    main()
