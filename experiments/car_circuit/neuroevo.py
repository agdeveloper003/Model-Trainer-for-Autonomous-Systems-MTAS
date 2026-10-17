"""
Neuroevolution: the car learns to drive without anyone explaining how.

Idea (genetic algorithm):
  1. create a population of small neural networks with random weights;
  2. each one drives: the score (fitness) says how well it did;
  3. the best survive, the others are replaced by "mutated" copies
     (weights + a bit of noise) of the best;
  4. repeat: generation after generation the driving improves.

The network sees the same 11 observations as the environment (rays, speed,
steering) and answers with 2 numbers: steering and throttle/brake.

Two ways to train:
  - one track:   every network drives the same circuit; the model learns
                 that circuit very well (and may only know that one);
  - generalist:  every generation brings K new random circuits (both
                 directions) and the fitness is the average over them, so a
                 network cannot memorise a layout: it has to learn to drive.
                 A fixed validation set of circuits never used in training
                 measures how well it drives on tracks it has never seen;
                 the saved model is the best one on that set.

This file holds the pieces shared with the app (network, scores, one
generation step, validation set, saving/loading models) and the command
line. To watch training live, use the app:  python app.py

    python neuroevo.py                                  # train on the classic circuit
    python neuroevo.py --track random --seed 42         # train on a random circuit
    python neuroevo.py --general --tracks 8             # train a generalist driver
    python neuroevo.py --general --device gpu           # ... simulating on the graphics card
"""
import argparse
import glob
import os
import time

import numpy as np

from env import CarCircuitEnv
from track import GENERAL_RANGES, Track, add_track_args, sample_track, spec_from_args

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(HERE, "checkpoints")
RUNS_DIR = os.path.join(HERE, "runs")

HIDDEN = 16
ELITE = 6
SIGMA = 0.08
SPEED_REF = 70.0                  # m/s, the car's top speed: scales the speed bonus
VALIDATION_SEED = 20251017        # fixed: the validation circuits never change
VALIDATION_SIZE = 12              # random circuits, plus the classic one


# ============================================================ neural network
class Net:
    """Tiny network: 11 inputs -> 16 hidden neurons -> 2 outputs (tanh)."""

    def __init__(self, n_in=11, n_hidden=HIDDEN, n_out=2):
        self.shapes = [(n_in, n_hidden), (n_hidden,), (n_hidden, n_out), (n_out,)]
        self.size = sum(int(np.prod(s)) for s in self.shapes)

    def unpack(self, genome):
        parts, i = [], 0
        for s in self.shapes:
            n = int(np.prod(s))
            parts.append(genome[i:i + n].reshape(s))
            i += n
        return parts

    @staticmethod
    def act(params, obs):
        w1, b1, w2, b2 = params
        h = np.tanh(obs @ w1 + b1)
        return np.tanh(h @ w2 + b2)

    @staticmethod
    def activations(params, obs):
        """Hidden and output values, for drawing the network."""
        w1, b1, w2, b2 = params
        h = np.tanh(obs @ w1 + b1)
        return h, np.tanh(h @ w2 + b2)


# ======================================================= scores & evolution
def lap_time_limit(track):
    """Seconds: a lap slower than this is not interesting (~13 m/s average)."""
    return max(60.0, track.length / 13.0)


def make_training_env(track):
    return CarCircuitEnv(track=track, max_time=lap_time_limit(track))


def track_score(env):
    """
    Score of a finished episode, comparable between circuits of any length.
    Returns (score, lap time or None):
      0 .. 1  share of the lap covered before crashing / stalling;
      > 1     lap completed: 1 + average speed / top speed (faster = higher).
    Any completed lap beats any incomplete one.
    """
    L = env.track.length
    if env.laps >= 1:
        lap = env.lap_times[0]
        return 1.0 + (L / lap) / SPEED_REF, lap
    return max(env.progress, 0.0) / L, None


def random_population(rng, pop_size, size):
    return rng.normal(0.0, 0.5, size=(pop_size, size))


def next_generation(pop, fit, rng, elite=ELITE, sigma=SIGMA):
    """The best `elite` survive unchanged; the rest are mutated children of the best 25%."""
    order = np.argsort(-fit)
    pop_size, size = pop.shape
    elites = pop[order[:elite]]
    parents = pop[order[:max(elite, pop_size // 4)]]
    children = []
    while len(children) < pop_size - elite:
        a, b = parents[rng.integers(len(parents), size=2)]
        mask = rng.random(size) < 0.5                       # crossover: half weights from a, half from b
        children.append(np.where(mask, a, b) + rng.normal(0.0, sigma, size))
    return np.vstack([elites, np.array(children)])


# ============================================================ validation
_validation = None


def validation_tracks():
    """Fixed circuits never used for training (same settings ranges, both directions)."""
    global _validation
    if _validation is None:
        rng = np.random.default_rng(VALIDATION_SEED)
        _validation = [sample_track(rng, GENERAL_RANGES)[1] for _ in range(VALIDATION_SIZE)] + [Track()]
    return _validation


def run_episode(genome, track):
    """One full lap attempt. Returns (score, lap time or None, crashed)."""
    env = make_training_env(track)
    params = Net().unpack(genome)
    obs = env.reset()
    while not env.done:
        obs, *_ = env.step(Net.act(params, obs))
    score, lap = track_score(env)
    return score, lap, env.crashed


def summarize(results, tracks):
    """results: list of (score, lap, crashed) on `tracks` -> averages."""
    done = [(t.length / lap) for (_, lap, _), t in zip(results, tracks) if lap]
    return {
        "score": float(np.mean([r[0] for r in results])),
        "completed": sum(1 for r in results if r[1]) / len(results),
        "crashed": sum(1 for r in results if r[2]) / len(results),
        "speed_kmh": float(np.mean(done) * 3.6) if done else None,
    }


# ==================================================================== models
def model_path(key):
    return os.path.join(CHECKPOINT_DIR, f"neuroevo_{key}.npz")


def checkpoint_path(spec):
    return model_path(spec.track_id())


def save_model(key, track_name, track_length, genome, lap_time, generations, val=None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    extra = {}
    if val:
        extra = {"val_completed": val["completed"], "val_speed": val["speed_kmh"] or 0.0}
    np.savez(model_path(key), genome=genome, hidden=HIDDEN, lap_time=lap_time or -1.0,
             generations=generations, track_name=track_name, track_length=track_length, **extra)


def _read(path):
    data = np.load(path)
    key = os.path.basename(path)[len("neuroevo_"):-len(".npz")]
    return {
        "path": path, "track_id": key, "general": key.startswith("general"),
        "genome": data["genome"], "hidden": int(data["hidden"]),
        "lap_time": float(data["lap_time"]),
        "generations": int(data["generations"]) if "generations" in data else 0,
        "track_name": str(data["track_name"]) if "track_name" in data else key,
        "track_length": float(data["track_length"]) if "track_length" in data else 0.0,
        "val_completed": float(data["val_completed"]) if "val_completed" in data else None,
        "val_speed": float(data["val_speed"]) if "val_speed" in data else None,
    }


def load_model(path):
    return _read(path) if os.path.exists(path) else None


def load_checkpoint(spec):
    """Return the saved model for this track, or None."""
    return load_model(checkpoint_path(spec))


def list_models():
    """Every saved model, most recent first."""
    paths = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "neuroevo_*.npz")),
                   key=os.path.getmtime, reverse=True)
    return [_read(p) for p in paths]


# ============================================================ command line
def main():
    from trainer import LiveTrainer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--general", action="store_true", help="train a generalist driver on random circuits")
    ap.add_argument("--tracks", type=int, default=8, help="circuits per generation (generalist)")
    ap.add_argument("--generations", type=int, default=150)
    ap.add_argument("--pop", type=int, default=96, help="networks per generation")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                    help="cpu = one worker process per core, gpu = every car at once on the graphics card")
    ap.add_argument("--workers", type=int, default=None, help="parallel processes (cpu)")
    ap.add_argument("--evo-seed", type=int, default=0, help="seed of the evolution itself")
    add_track_args(ap)
    args = ap.parse_args()

    if args.general:
        tr = LiveTrainer(args.generations, args.pop, args.workers, args.evo_seed, n_tracks=args.tracks,
                         device=args.device)
    else:
        spec = spec_from_args(args)
        tr = LiveTrainer(args.generations, args.pop, args.workers, args.evo_seed, spec=spec,
                         track=spec.build(), device=args.device)
    print(f"Training: {tr.name}   key {tr.key}")
    print(f"Network: 11 -> {HIDDEN} -> 2  ({Net().size} weights), population {args.pop}, "
          f"{tr.device_label}\n", flush=True)

    os.makedirs(RUNS_DIR, exist_ok=True)
    log_path = os.path.join(RUNS_DIR, f"neuroevo_{tr.key}.csv")
    with open(log_path, "w") as f:
        f.write("generation,survival_rate,best_lap,val_completed,val_speed_kmh,seconds\n")
    shown = 0
    try:
        while not tr.finished:
            tr.advance(100_000)                       # run every car of the generation to the end
            for h in tr.history[shown:]:
                surv = h["completed"] / h["pop"]
                line = f"gen {h['gen']:4d}/{tr.generations}  survival {surv:6.1%}"
                if h.get("best_lap"):
                    line += f"  best lap {h['best_lap']:6.2f} s"
                if h.get("val_completed") is not None:
                    spd = f"{h['val_speed']:.0f} km/h" if h["val_speed"] else "-"
                    line += f"  validation: {h['val_completed']:5.1%} completed, {spd}"
                print(line + f"  ({h['seconds']:.1f} s)", flush=True)
                with open(log_path, "a") as f:
                    f.write(f"{h['gen']},{surv:.4f},{h.get('best_lap') or ''},"
                            f"{h.get('val_completed') if h.get('val_completed') is not None else ''},"
                            f"{h.get('val_speed') or ''},{h['seconds']:.2f}\n")
            shown = len(tr.history)
    finally:
        tr.close()
    print(f"\nDone in {time.perf_counter() - tr.started:.0f} s. "
          f"Model saved to {os.path.relpath(model_path(tr.key), HERE)}")


if __name__ == "__main__":
    main()
