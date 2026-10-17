"""
Pilota "a regole" scritto a mano: NON impara niente.

Serve a due cose:
  1. verificare che l'ambiente funzioni (deve riuscire a fare un giro);
  2. darci un tempo di riferimento da battere con il machine learning.

Usa solo i sensori, come farà la rete neurale:
  - sterza verso la direzione in cui i raggi vedono più spazio libero;
  - regola la velocità in base a quanto spazio c'è davanti.

    python bot.py            # senza grafica, stampa il tempo sul giro
    python bot.py --render   # guardalo guidare
"""
import argparse

import numpy as np

from env import CarCircuitEnv


def rule_based_policy(env):
    dist, ang = env.ray_dist, env.ray_angles
    car = env.car

    # sterzo: media degli angoli pesata sullo spazio libero
    w = dist ** 4
    target = float((w * ang).sum() / w.sum())
    steer = np.clip(target / car.p.max_steer * 1.6, -1, 1)

    # velocità: quanto posso andare veloce per fermarmi prima del muro davanti
    mid = len(dist) // 2
    front = float(dist[mid - 1: mid + 2].min())
    target_speed = np.sqrt(2 * car.p.brake_decel * 0.55 * max(front - 12.0, 0.0)) + 14.0
    throttle = np.clip((target_speed - car.speed) * 0.4, -1, 1)
    return np.array([steer, throttle])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="mostra la finestra grafica")
    ap.add_argument("--laps", type=int, default=1)
    args = ap.parse_args()

    if args.render:
        import pygame
        from render import Renderer
        env = CarCircuitEnv(dt=1 / 60, substeps=1, laps_to_finish=args.laps)
        renderer = Renderer(env, title="MTAS - pilota a regole")
    else:
        env = CarCircuitEnv(laps_to_finish=args.laps)

    env.reset()
    total_reward, info = 0.0, {}
    while True:
        _, r, term, trunc, info = env.step(rule_based_policy(env))
        total_reward += r
        if args.render:
            if not renderer.handle_events():
                break
            renderer.draw()
            renderer.tick(60)
        if term or trunc:
            break

    esito = "INCIDENTE" if info.get("crashed") else ("giro completato" if info.get("laps") else "fermo/tempo scaduto")
    print(f"Esito: {esito}")
    print(f"Giri: {info.get('laps')}  tempi: {[round(t, 2) for t in env.lap_times]}")
    print(f"Progresso: {info.get('progress', 0):.1%} di giro  ricompensa totale: {total_reward:.1f}")
    if args.render:
        pygame.time.wait(1500)
        renderer.close()


if __name__ == "__main__":
    main()
