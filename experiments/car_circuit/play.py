"""
Guida tu la macchina con la tastiera.

    python play.py

Frecce o WASD: su = accelera, giù = frena, sinistra/destra = sterza.
R riparti, C cambia vista, V sensori on/off, ESC esci.

Prova a fare un giro pulito: ti accorgerai che entrare in curva troppo
veloce fa "scivolare" la macchina (scritta GOMME AL LIMITE) verso il muro.
"""
import pygame

from env import CarCircuitEnv
from render import Renderer


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
    # 60 passi al secondo in tempo reale; giri infiniti finché non sbatti
    env = CarCircuitEnv(dt=1 / 60, substeps=1, laps_to_finish=10_000,
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
