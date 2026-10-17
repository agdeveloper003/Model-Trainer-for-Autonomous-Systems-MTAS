"""
Training engine: the genetic algorithm of neuroevo.py, run so that every
car can be watched while it drives (the app) or as fast as possible (the
command line).

The caller asks "advance k simulation steps" and receives position, speed,
sensors and score of every car on every circuit. When all cars have
crashed, stalled or finished, the generation is scored and the next one is
bred.

Two modes:
  - one track:   K = 1 fixed circuit; fitness = score on it. The best network
                 always survives unchanged and the circuit never changes, so
                 the best score can only stay equal or improve.
  - generalist:  K new random circuits every generation; fitness = average
                 score over them. Every few generations the best networks are
                 measured on the fixed validation circuits, and the model is
                 saved only when that measure improves.

Two simulation back-ends, same rules and same results (up to the last
digits of the floating point maths):
  - CPU: worker processes, one per core, each driving a slice of the
         population one car at a time (env.py);
  - GPU: every car at once as arrays on the graphics card (gpusim.py).
"""
import multiprocessing as mp
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from neuroevo import (ELITE, SIGMA, Net, make_training_env, next_generation, random_population,
                      run_episode, save_model, summarize, track_score, validation_tracks)
from track import GENERAL_RANGES, sample_track

# columns of the per-car, per-circuit state array
X, Y, HEADING, SPEED, DONE, CRASHED, PROGRESS, LAPS, LAP_TIME, SCORE = range(10)
STATE_COLS = 10
VALIDATE_EVERY = 5       # generations between two validation runs (generalist)
VALIDATE_TOP = 3         # how many of the best networks are validated
RUN_TO_END = 100_000     # advance(RUN_TO_END) = finish the generation without stopping


def gpu_available():
    """True if PyTorch with CUDA is installed (imports torch: call only when needed)."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ============================================================ CPU back-end
def _worker(conn):
    tracks, val_tracks = [], []
    envs, params, obs = [], [], []           # envs[car][circuit]

    def state():
        n, k = len(envs), len(tracks)
        s = np.zeros((n, k, STATE_COLS))
        o = np.zeros((n, k, 11), dtype=np.float32)
        for i, row in enumerate(envs):
            for j, e in enumerate(row):
                c = e.car
                score, lap = track_score(e) if e.done else (0.0, None)
                s[i, j] = (c.x, c.y, c.heading, c.speed, e.done, e.crashed, e.progress, e.laps,
                           lap if lap is not None else -1.0, score)
                o[i, j] = obs[i][j]
        return s, o

    while True:
        msg = conn.recv()
        kind = msg[0]
        if kind == "tracks":
            tracks = msg[1]
        elif kind == "val":
            val_tracks = msg[1]
        elif kind == "gen":
            genomes = msg[1]
            envs = [[make_training_env(t) for t in tracks] for _ in genomes]
            params = [Net().unpack(g) for g in genomes]
            obs = [[e.reset() for e in row] for row in envs]
            conn.send(state())
        elif kind == "step":
            k = msg[1]
            for i, row in enumerate(envs):
                p = params[i]
                for j, e in enumerate(row):
                    o = obs[i][j]
                    for _ in range(k):
                        if e.done:
                            break
                        o, *_ = e.step(Net.act(p, o))
                    obs[i][j] = o
            conn.send(state())
        elif kind == "eval":
            conn.send([(gi, vi, run_episode(g, val_tracks[vi])) for gi, g, vi in msg[1]])
        else:
            break
    conn.close()


class CpuBackend:
    """One worker process per CPU core; each drives a slice of the population."""

    def __init__(self, pop_size, workers=None):
        n = max(1, min(workers or max(1, (os.cpu_count() or 2) - 2), pop_size))
        self.slices = np.array_split(np.arange(pop_size), n)
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(n):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child,), daemon=True)
            p.start()
            self.conns.append(parent)
            self.procs.append(p)
        self.label = f"CPU · {n} processes"

    def _send_all(self, msg):
        for c in self.conns:
            c.send(msg)

    def _gather(self):
        parts = [c.recv() for c in self.conns]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    def prepare(self, tracks):
        return tracks                         # nothing to pre-compute

    def set_validation(self, tracks):
        self.val_tracks = tracks
        self._send_all(("val", tracks))

    def set_tracks(self, prepared):
        self._send_all(("tracks", prepared))

    def start(self, pop):
        for c, idx in zip(self.conns, self.slices):
            c.send(("gen", pop[idx]))
        return self._gather()

    def step(self, k):
        self._send_all(("step", k))
        return self._gather()

    def validate(self, genomes):
        """Raw results [genome][circuit] on the validation circuits."""
        jobs = [(gi, g, vi) for gi, g in enumerate(genomes) for vi in range(len(self.val_tracks))]
        chunks = [jobs[i::len(self.conns)] for i in range(len(self.conns))]
        busy = [(c, ch) for c, ch in zip(self.conns, chunks) if ch]
        for c, ch in busy:
            c.send(("eval", ch))
        res = [[None] * len(self.val_tracks) for _ in genomes]
        for c, _ in busy:
            for gi, vi, r in c.recv():
                res[gi][vi] = r
        return res

    def close(self):
        for c in self.conns:
            try:
                c.send(("quit",))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()


# ============================================================ GPU back-end
class GpuBackend:
    """Every car of the generation at once on the graphics card (gpusim.py)."""

    def __init__(self):
        import torch
        from gpusim import default_device, device_name
        self.torch = torch
        self.device = default_device()
        self.label = f"GPU · {device_name(self.device)}"
        self.sim = None

    def prepare(self, tracks):
        from gpusim import TrackBatch
        return TrackBatch(tracks, self.device)           # tensors + nearby-wall tables (in the background)

    def set_validation(self, tracks):
        self.val_tracks = tracks
        self.val_tb = self.prepare(tracks)

    def set_tracks(self, prepared):
        self.tb = prepared

    def start(self, pop):
        from gpusim import BatchSim
        self.sim = BatchSim(None, pop, self.device, track_batch=self.tb)
        return self.sim.snapshot()

    def step(self, k):
        if k >= RUN_TO_END:
            self.sim.run_to_end()
        else:
            self.sim.step(k)
        return self.sim.snapshot()

    def validate(self, genomes):
        from gpusim import BatchSim
        sim = BatchSim(None, genomes, self.device, track_batch=self.val_tb)
        sim.run_to_end()
        return sim.results()

    def close(self):
        self.sim = None


# =============================================================== trainer
class LiveTrainer:
    def __init__(self, generations=150, pop_size=96, workers=None, seed=0,
                 spec=None, track=None, n_tracks=None, device="cpu"):
        self.general = n_tracks is not None
        self.n_tracks = n_tracks if self.general else 1
        self.generations, self.pop_size, self.seed = generations, pop_size, seed
        self.rng = np.random.default_rng(seed)
        self.net = Net()
        self.pop = random_population(self.rng, pop_size, self.net.size)
        self.backend = GpuBackend() if device == "gpu" else CpuBackend(pop_size, workers)
        self.device_label = self.backend.label

        if self.general:
            self.key = f"general-k{n_tracks}"
            self.name = f"Generalist ({n_tracks} tracks/gen)"
            self.val_tracks = validation_tracks()
            self.backend.set_validation(self.val_tracks)
            # circuits of the next generation are built in the background while this one runs
            self._pool = ThreadPoolExecutor(max_workers=1)
            self._next = self._pool.submit(self._make_tracks, 1)
            self.track_specs, self.tracks = [], []
        else:
            self.key, self.name = spec.track_id(), track.name
            self.track_specs, self.tracks = [spec], [track]
            self.backend.set_tracks(self.backend.prepare(self.tracks))
        self.env_dt = make_training_env(track if track is not None else self.val_tracks[0]).dt

        self.gen = 0
        self.finished = False
        self.history = []            # one dict per finished generation
        self.record_fit = -np.inf    # one track: best fitness so far
        self.record_lap = None
        self.record_gen = None
        self.record_val = None       # generalist: best validation summary so far
        self.best_genome = None
        self.sim_time = 0.0          # simulated seconds in the current generation
        self.total_steps = 0
        self.started = time.perf_counter()
        self._start_generation()

    # ------------------------------------------------------------------
    def _make_tracks(self, gen):
        rng = np.random.default_rng([self.seed, gen])      # same seed -> same circuits
        pairs = [sample_track(rng, GENERAL_RANGES) for _ in range(self.n_tracks)]
        return pairs, self.backend.prepare([p[1] for p in pairs])

    def _start_generation(self):
        self.gen += 1
        self.sim_time = 0.0
        self.gen_started = time.perf_counter()
        if self.general:
            pairs, prepared = self._next.result()
            self.track_specs = [p[0] for p in pairs]
            self.tracks = [p[1] for p in pairs]
            self.backend.set_tracks(prepared)
            if self.gen < self.generations:
                self._next = self._pool.submit(self._make_tracks, self.gen + 1)
        self.state, self.obs = self.backend.start(self.pop)

    def advance(self, steps):
        """Simulate `steps` more steps for every car still running."""
        if self.finished or steps <= 0:
            return
        self.state, self.obs = self.backend.step(steps)
        self.sim_time += min(steps, RUN_TO_END) * self.env_dt
        self.total_steps += steps
        if self.state[:, :, DONE].all():
            self._finish_generation()

    # ------------------------------------------------------------------
    def _finish_generation(self):
        score = self.state[:, :, SCORE]                 # (pop, K)
        fit = score.mean(axis=1)
        laps = self.state[:, :, LAP_TIME]
        top = int(np.argmax(fit))
        entry = {
            "gen": self.gen,
            "completed": int((laps > 0).sum()),
            "pop": laps.size,                           # episodes: networks x circuits
            "best_progress": float(min(1.0, score[top].mean())),
            "seconds": time.perf_counter() - self.gen_started,
        }
        if not self.general:
            lap_top = laps[top, 0] if laps[top, 0] > 0 else None
            if fit[top] > self.record_fit:
                self.record_fit, self.record_lap, self.record_gen = fit[top], lap_top, self.gen
                self.best_genome = self.pop[top].copy()
                save_model(self.key, self.name, self.tracks[0].length, self.best_genome, lap_top, self.gen)
            entry.update(best_lap=lap_top, record_lap=self.record_lap)
        elif self.gen % VALIDATE_EVERY == 0 or self.gen == self.generations:
            cand = np.argsort(-fit)[:VALIDATE_TOP]
            vals = [summarize(r, self.val_tracks) for r in self.backend.validate(self.pop[cand])]
            # best = most unseen circuits completed; average score only breaks ties
            rank = [(v["completed"], v["score"]) for v in vals]
            b = int(max(range(len(vals)), key=lambda i: rank[i]))
            entry.update(val_completed=vals[b]["completed"], val_speed=vals[b]["speed_kmh"])
            if self.record_val is None or rank[b] > (self.record_val["completed"], self.record_val["score"]):
                self.record_val, self.record_gen = vals[b], self.gen
                self.best_genome = self.pop[cand[b]].copy()
                save_model(self.key, self.name, 0.0, self.best_genome, None, self.gen, vals[b])
        self.history.append(entry)

        if self.gen >= self.generations:
            self.finished = True
            return
        self.pop = next_generation(self.pop, fit, self.rng, ELITE, SIGMA)
        self._start_generation()

    def stop(self):
        """End training now (the best model so far is already saved)."""
        self.finished = True

    # ------------------------------------------------------------------
    def live_scores(self):
        """Score of every network so far, averaged over the circuits (cars still running included)."""
        s = self.state
        part = np.where(s[:, :, DONE] == 1, s[:, :, SCORE],
                        np.clip(s[:, :, PROGRESS], 0, None) / np.array([t.length for t in self.tracks]))
        return part.mean(axis=1)

    def leader(self):
        """Index of the network to show."""
        s = self.state[:, 0]
        if not self.general:
            running = np.nonzero(s[:, DONE] == 0)[0]        # the furthest car still running
            if len(running):
                return int(running[np.argmax(s[running, PROGRESS])])
            return int(np.argmax(s[:, SCORE]))
        return int(np.argmax(self.live_scores()))

    def close(self):
        if self.general:
            self._pool.shutdown(wait=False, cancel_futures=True)
        self.backend.close()
