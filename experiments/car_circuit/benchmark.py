"""
Benchmark: how well does a model drive circuits it has never seen?

Runs the model and the rule-based bot on the validation circuits (12
random ones, both directions, plus the classic circuit; never used for
training) and prints, for each circuit, the lap time or what went wrong.

Summary:
  completed   share of circuits where the model finished the lap
  crashed     share where it hit a wall
  vs bot      model lap time / bot lap time, on circuits both finished
              (0.70 = 30% faster than the bot)

    python benchmark.py                                         # every saved model
    python benchmark.py checkpoints/neuroevo_general-k8.npz     # one model
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np

from bot import rule_based_policy
from neuroevo import list_models, load_model, run_episode, summarize, validation_tracks
from env import CarCircuitEnv


def fmt_time(t):
    if t is None:
        return "--:--.--"
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:05.2f}"


def bot_lap(track):
    env = CarCircuitEnv(track=track)
    env.reset()
    while not env.done:
        env.step(rule_based_policy(env))
    return env.lap_times[0] if env.laps else None


def _job(args):
    kind, genome, i = args
    track = validation_tracks()[i]
    return bot_lap(track) if kind == "bot" else run_episode(genome, track)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="model files (default: every saved model)")
    args = ap.parse_args()
    models = [load_model(p) for p in args.models] if args.models else list_models()
    models = [m for m in models if m]
    tracks = validation_tracks()
    n = len(tracks)

    with Pool(min(os.cpu_count() or 2, 32)) as pool:
        bots = pool.map(_job, [("bot", None, i) for i in range(n)])
        results = {m["path"]: pool.map(_job, [("model", m["genome"], i) for i in range(n)]) for m in models}

    print(f"Validation circuits: {n}\n")
    print(f"{'circuit':34s} {'km':>5s} {'bot':>9s}  " + "  ".join(f"{m['track_id'][:16]:>16s}" for m in models))
    for i, t in enumerate(tracks):
        cells = []
        for m in models:
            score, lap, crashed = results[m["path"]][i]
            cells.append(fmt_time(lap) if lap else ("crash " if crashed else "stall ") + f"{min(score, 1):4.0%}")
        print(f"{t.name[:34]:34s} {t.length / 1000:5.2f} {fmt_time(bots[i]):>9s}  "
              + "  ".join(f"{c:>16s}" for c in cells))

    print("\nSummary")
    for m in models:
        res = results[m["path"]]
        s = summarize(res, tracks)
        ratios = [r[1] / b for r, b in zip(res, bots) if r[1] and b]
        vs = f"{np.mean(ratios):.2f}x bot time" if ratios else "-"
        print(f"  {m['track_id']:40s} completed {s['completed']:6.1%}   crashed {s['crashed']:6.1%}   "
              f"vs bot {vs}  (on {len(ratios)} circuits)")
    print(f"  {'rule-based bot':40s} completed {sum(b is not None for b in bots) / n:6.1%}")


if __name__ == "__main__":
    main()
