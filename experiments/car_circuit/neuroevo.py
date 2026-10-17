"""
Neuroevoluzione: la macchina impara a guidare senza che nessuno le spieghi come.

Idea (algoritmo genetico):
  1. si crea una popolazione di piccole reti neurali con pesi casuali;
  2. ognuna guida un giro: il punteggio (fitness) dice quanto è andata bene;
  3. le migliori sopravvivono, le altre vengono sostituite da copie
     "mutate" (pesi + un po' di rumore) delle migliori;
  4. si ripete: generazione dopo generazione la guida migliora.

La rete vede le stesse 11 osservazioni dell'ambiente (raggi, velocità,
sterzo) e risponde con 2 numeri: sterzo e gas/freno.

    python neuroevo.py                     # allena (salva in checkpoints/)
    python neuroevo.py --generations 300   # allena più a lungo
    python neuroevo.py --play              # guarda guidare la rete migliore
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

from env import CarCircuitEnv

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(HERE, "checkpoints", "neuroevo_best.npz")
LOG = os.path.join(HERE, "runs", "neuroevo_log.csv")

HIDDEN = 16
MAX_TIME = 150.0                      # secondi: un giro più lento non ci interessa


# ================================================================ rete neurale
class Net:
    """Rete minuscola: 11 ingressi -> 16 neuroni nascosti -> 2 uscite (tanh)."""

    def __init__(self, n_in, n_hidden, n_out):
        self.shapes = [(n_in, n_hidden), (n_hidden,), (n_hidden, n_out), (n_out,)]
        self.size = sum(int(np.prod(s)) for s in self.shapes)

    def unpack(self, genome):
        parts, i = [], 0
        for s in self.shapes:
            n = int(np.prod(s))
            parts.append(genome[i:i + n].reshape(s))
            i += n
        return parts

    def act(self, params, obs):
        w1, b1, w2, b2 = params
        h = np.tanh(obs @ w1 + b1)
        return np.tanh(h @ w2 + b2)


def make_env():
    return CarCircuitEnv(max_time=MAX_TIME)


# ==================================================================== fitness
def evaluate(genome):
    """Fa guidare un giro alla rete. Ritorna (fitness, tempo sul giro o None)."""
    env = make_env()
    net = Net(env.obs_size, HIDDEN, env.action_size)
    params = net.unpack(genome)
    obs = env.reset()
    while True:
        obs, _, term, trunc, _ = env.step(net.act(params, obs))
        if term or trunc:
            break
    if env.laps >= 1:
        # giro finito: il punteggio ora dipende solo da quanto è stato veloce
        return env.track.length + (MAX_TIME - env.lap_times[0]) * 100.0, env.lap_times[0]
    # giro non finito: conta quanti metri ha percorso
    return max(env.progress, 0.0), None


# ============================================================= allenamento
def train(args):
    rng = np.random.default_rng(args.seed)
    env = make_env()
    net = Net(env.obs_size, HIDDEN, env.action_size)
    print(f"Rete: {env.obs_size} -> {HIDDEN} -> {env.action_size}  ({net.size} pesi)")
    print(f"Popolazione {args.pop}, elite {args.elite}, mutazione {args.sigma}, {args.workers} processi\n")

    pop = rng.normal(0.0, 0.5, size=(args.pop, net.size))
    best_fit, best_genome, best_lap = -np.inf, None, None

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "w") as f:
        f.write("generation,best_fitness,mean_fitness,best_lap_time,laps_completed\n")

    with Pool(args.workers) as pool:
        for gen in range(1, args.generations + 1):
            t0 = time.perf_counter()
            results = pool.map(evaluate, list(pop))
            fit = np.array([r[0] for r in results])
            laps = [r[1] for r in results]
            order = np.argsort(-fit)

            if fit[order[0]] > best_fit:
                best_fit, best_genome, best_lap = fit[order[0]], pop[order[0]].copy(), laps[order[0]]
                np.savez(CHECKPOINT, genome=best_genome, hidden=HIDDEN, lap_time=best_lap or -1.0)

            n_done = sum(l is not None for l in laps)
            gen_best_lap = laps[order[0]]
            lap_txt = f"{gen_best_lap:6.2f} s" if gen_best_lap else "   ---  "
            rec_txt = f"{best_lap:6.2f} s" if best_lap else "---"
            print(f"gen {gen:4d}  miglior giro {lap_txt}  giri completati {n_done:3d}/{args.pop}"
                  f"  record {rec_txt}  ({time.perf_counter() - t0:.1f} s)")
            with open(LOG, "a") as f:
                f.write(f"{gen},{fit[order[0]]:.2f},{fit.mean():.2f},{gen_best_lap or ''},{n_done}\n")

            # ---------------------------------------- nuova generazione
            elite = pop[order[:args.elite]]
            parents = pop[order[:max(args.elite, args.pop // 4)]]   # il migliore 25% può riprodursi
            children = []
            while len(children) < args.pop - args.elite:
                a, b = parents[rng.integers(len(parents), size=2)]
                mask = rng.random(net.size) < 0.5                  # incrocio: metà pesi da a, metà da b
                child = np.where(mask, a, b) + rng.normal(0.0, args.sigma, net.size)
                children.append(child)
            pop = np.vstack([elite, np.array(children)])

    print(f"\nFinito. Miglior giro: {best_lap:.2f} s" if best_lap else "\nFinito. Nessun giro completato.")
    print(f"Rete salvata in {os.path.relpath(CHECKPOINT, HERE)}  -  guardala con: python neuroevo.py --play")


# ================================================================== replay
def play():
    import pygame
    from render import Renderer

    if not os.path.exists(CHECKPOINT):
        print("Nessuna rete salvata: prima allena con  python neuroevo.py")
        return
    data = np.load(CHECKPOINT)
    # stessi passi di tempo dell'allenamento, così la guida è identica
    env = CarCircuitEnv(max_time=1e9, stall_time=1e9, laps_to_finish=10_000)
    net = Net(env.obs_size, int(data["hidden"]), env.action_size)
    params = net.unpack(data["genome"])
    renderer = Renderer(env, title="MTAS - rete neurale (neuroevoluzione)")
    fps = round(1 / env.dt)

    obs = env.reset()
    action = np.zeros(2)
    running = True
    while running:
        running = renderer.handle_events()
        if renderer.reset_requested:
            renderer.reset_requested = False
            obs = env.reset()
        if not env.done:
            action = net.act(params, obs)
            obs, *_ = env.step(action)
        renderer.draw(action, extra_lines=["Pilota: rete neurale", f"Allenamento: {float(data['lap_time']):.2f} s"])
        renderer.tick(fps)
    renderer.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play", action="store_true", help="guarda la rete migliore salvata")
    ap.add_argument("--generations", type=int, default=150)
    ap.add_argument("--pop", type=int, default=96, help="reti per generazione")
    ap.add_argument("--elite", type=int, default=6, help="migliori copiate intatte")
    ap.add_argument("--sigma", type=float, default=0.08, help="forza della mutazione")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="processi in parallelo")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.play:
        play()
    else:
        train(args)


if __name__ == "__main__":
    main()
