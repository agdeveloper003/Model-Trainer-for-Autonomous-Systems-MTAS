"""
Neuroevolution: the car learns to drive without anyone explaining how.

Idea (genetic algorithm):
  1. create a population of small neural networks with random weights;
  2. each one drives a lap: the score (fitness) says how well it did;
  3. the best survive, the others are replaced by "mutated" copies
     (weights + a bit of noise) of the best;
  4. repeat: generation after generation the driving improves.

The network sees the same 11 observations as the environment (rays, speed,
steering) and answers with 2 numbers: steering and throttle/brake.

Every track has its own saved network (checkpoints/neuroevo_<track id>.npz).

    python neuroevo.py                                  # train on the classic circuit
    python neuroevo.py --track random --seed 42         # train on a random circuit
    python neuroevo.py --generations 300                # train for longer
    python neuroevo.py --play                           # watch the best saved network
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

from env import CarCircuitEnv
from track import TrackSpec, add_track_args, spec_from_args

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(HERE, "checkpoints")
RUNS_DIR = os.path.join(HERE, "runs")

HIDDEN = 16


def checkpoint_path(spec):
    return os.path.join(CHECKPOINT_DIR, f"neuroevo_{spec.track_id()}.npz")


def lap_time_limit(track):
    """Seconds: a lap slower than this is not interesting (~13 m/s average)."""
    return max(60.0, track.length / 13.0)


def load_checkpoint(spec):
    """Return the saved data for this track, or None."""
    path = checkpoint_path(spec)
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return {"genome": data["genome"], "hidden": int(data["hidden"]),
            "lap_time": float(data["lap_time"]), "generations": int(data["generations"])}


# ============================================================ neural network
class Net:
    """Tiny network: 11 inputs -> 16 hidden neurons -> 2 outputs (tanh)."""

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


# =================================================================== fitness
_worker_env = None


def _init_worker(spec):
    """Each worker process builds the track once, not once per network."""
    global _worker_env
    track = spec.build()
    _worker_env = CarCircuitEnv(track=track, max_time=lap_time_limit(track))


def evaluate(genome):
    """Let the network drive one lap. Return (fitness, lap time or None)."""
    env = _worker_env
    net = Net(env.obs_size, HIDDEN, env.action_size)
    params = net.unpack(genome)
    obs = env.reset()
    while True:
        obs, _, term, trunc, _ = env.step(net.act(params, obs))
        if term or trunc:
            break
    if env.laps >= 1:
        # lap finished: from now on the score only depends on how fast it was
        return env.track.length + (env.max_time - env.lap_times[0]) * 100.0, env.lap_times[0]
    # lap not finished: count how many metres it covered
    return max(env.progress, 0.0), None


# ================================================================== training
def train(spec, generations=150, pop_size=96, elite=6, sigma=0.08,
          workers=None, seed=0, stop_file=None):
    workers = workers or os.cpu_count()
    rng = np.random.default_rng(seed)
    track = spec.build()
    env = CarCircuitEnv(track=track)
    net = Net(env.obs_size, HIDDEN, env.action_size)
    ckpt = checkpoint_path(spec)
    log_path = os.path.join(RUNS_DIR, f"neuroevo_{spec.track_id()}.csv")

    print(f"Track: {track.name}  {track.length:.0f} m  (id {spec.track_id()})")
    print(f"Network: {env.obs_size} -> {HIDDEN} -> {env.action_size}  ({net.size} weights)")
    print(f"Population {pop_size}, elite {elite}, mutation {sigma}, {workers} processes\n", flush=True)

    pop = rng.normal(0.0, 0.5, size=(pop_size, net.size))
    best_fit, best_lap = -np.inf, None

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(log_path, "w") as f:
        f.write("generation,best_fitness,mean_fitness,best_lap_time,laps_completed\n")

    gen = 0
    with Pool(workers, initializer=_init_worker, initargs=(spec,)) as pool:
        for gen in range(1, generations + 1):
            t0 = time.perf_counter()
            results = pool.map(evaluate, list(pop))
            fit = np.array([r[0] for r in results])
            laps = [r[1] for r in results]
            order = np.argsort(-fit)

            if fit[order[0]] > best_fit:
                best_fit, best_lap = fit[order[0]], laps[order[0]]
                np.savez(ckpt, genome=pop[order[0]], hidden=HIDDEN,
                         lap_time=best_lap or -1.0, generations=gen)

            n_done = sum(l is not None for l in laps)
            gen_best_lap = laps[order[0]]
            lap_txt = f"{gen_best_lap:6.2f} s" if gen_best_lap else "   ---  "
            rec_txt = f"{best_lap:6.2f} s" if best_lap else "---"
            print(f"gen {gen:4d}/{generations}  best lap {lap_txt}  laps completed {n_done:3d}/{pop_size}"
                  f"  record {rec_txt}  ({time.perf_counter() - t0:.1f} s)", flush=True)
            with open(log_path, "a") as f:
                f.write(f"{gen},{fit[order[0]]:.2f},{fit.mean():.2f},{gen_best_lap or ''},{n_done}\n")

            if stop_file and os.path.exists(stop_file):
                print("\nStopped by the user.")
                break

            # ---------------------------------------- next generation
            elites = pop[order[:elite]]
            parents = pop[order[:max(elite, pop_size // 4)]]   # the best 25% may reproduce
            children = []
            while len(children) < pop_size - elite:
                a, b = parents[rng.integers(len(parents), size=2)]
                mask = rng.random(net.size) < 0.5               # crossover: half weights from a, half from b
                children.append(np.where(mask, a, b) + rng.normal(0.0, sigma, net.size))
            pop = np.vstack([elites, np.array(children)])

    print(f"\nDone after {gen} generations. Best lap: {best_lap:.2f} s" if best_lap
          else f"\nDone after {gen} generations. No lap completed yet: try more generations.")
    print(f"Network saved to {os.path.relpath(ckpt, HERE)}", flush=True)
    return best_lap


# ==================================================================== replay
def play(spec):
    import pygame
    from render import Renderer

    data = load_checkpoint(spec)
    if data is None:
        print(f"No saved network for track '{spec.track_id()}': train it first.")
        return
    # same time step as training, so the driving is identical
    env = CarCircuitEnv(track=spec.build(), max_time=1e9, stall_time=1e9, laps_to_finish=10_000)
    net = Net(env.obs_size, data["hidden"], env.action_size)
    params = net.unpack(data["genome"])
    renderer = Renderer(env, title="MTAS - neural network (neuroevolution)")
    fps = round(1 / env.dt)
    trained = f"{data['lap_time']:.2f} s" if data["lap_time"] > 0 else "no full lap"
    info = ["Driver: neural network", f"Training: {trained}", f"after {data['generations']} generations"]

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
        renderer.draw(action, extra_lines=info)
        renderer.tick(fps)
    renderer.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play", action="store_true", help="watch the best saved network")
    ap.add_argument("--generations", type=int, default=150)
    ap.add_argument("--pop", type=int, default=96, help="networks per generation")
    ap.add_argument("--elite", type=int, default=6, help="best ones copied unchanged")
    ap.add_argument("--sigma", type=float, default=0.08, help="mutation strength")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel processes")
    ap.add_argument("--evo-seed", type=int, default=0, help="seed of the evolution itself")
    ap.add_argument("--stop-file", help="training stops when this file appears (used by the launcher)")
    add_track_args(ap)
    args = ap.parse_args()
    spec = spec_from_args(args)
    if args.play:
        play(spec)
    else:
        train(spec, args.generations, args.pop, args.elite, args.sigma,
              args.workers, args.evo_seed, args.stop_file)


if __name__ == "__main__":
    main()
