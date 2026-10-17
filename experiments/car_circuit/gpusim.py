"""
Batched simulation in PyTorch: every car of a generation at once, on the
GPU (or the CPU).

It reproduces env.py + car.py + track.py exactly (same physics, sensors,
collisions, lap and stall rules, same scores; float64 like the original),
but instead of one Python object per car every quantity is an array with
one entry per car, shaped (networks P, circuits K). One simulation step is
a few dozen array operations for all P x K cars together:

    sensors   9 rays x every car x the walls near it
    network   P small matrix products (torch.bmm), one per network
    physics   bicycle model with grip limit, 2 sub-steps
    track     closest centre-line point (progress), corner collisions,
              lap completed, stalled, out of time

Two tricks keep the GPU busy instead of waiting:
  - nearby walls: for every centre-line point the walls within sensor
    range are listed in advance, so each car tests ~300 walls, not ~2,500;
  - CUDA graph: one step is recorded once and then replayed as a single
    launch, instead of ~150 small launches from Python.

Circuits of different sizes are padded to the same length; padding walls
have zero length and can never be hit.
"""
import math

import numpy as np
import torch

from car import CarParams
from neuroevo import SPEED_REF, Net, lap_time_limit

WINDOW = 40            # closest-point search: +- this many segments around the previous one
STALL_TIME = 6.0


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "CPU (PyTorch)"


# One GPU thread per (car, ray). Stage 1 (float32) finds the 2 closest walls among the
# nearby ones; stage 2 recomputes those 2 exactly in float64. Compiled at run time with
# NVRTC (shipped with PyTorch): no CUDA toolkit needed. --fmad=false keeps the float64
# roundings identical to NumPy on the CPU.
RAYCAST_SRC = r"""
extern "C" __global__ void raycast(
    const double* __restrict__ x, const double* __restrict__ y, const double* __restrict__ heading,
    const long long* __restrict__ seg, const long long* __restrict__ cand,
    const double* __restrict__ wa, const double* __restrict__ we,
    const float* __restrict__ wa32, const float* __restrict__ we32,
    const double* __restrict__ ang, double* __restrict__ out,
    int B, int K, int R, int Nmax, int C, int Wp, double max_range)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * R) return;
    int b = i / R, r = i - b * R;
    int k = b % K;
    double px = x[b], py = y[b];
    double h = heading[b] + ang[r];
    double dx = cos(h), dy = sin(h);
    float fx = (float)dx, fy = (float)dy, fpx = (float)px, fpy = (float)py;
    const long long* cl = cand + ((long long)k * Nmax + seg[b]) * C;
    const float* A32 = wa32 + (long long)k * Wp * 2;
    const float* E32 = we32 + (long long)k * Wp * 2;
    float inf32 = __int_as_float(0x7f800000);
    float t1 = inf32, t2 = inf32;
    long long w1 = -1, w2 = -1;
    for (int c = 0; c < C; ++c) {
        long long w = cl[c];
        float ax = A32[2 * w] - fpx, ay = A32[2 * w + 1] - fpy;
        float ex = E32[2 * w], ey = E32[2 * w + 1];
        float den = fx * ey - fy * ex;
        if (fabsf(den) <= 1e-9f) continue;
        float t = (ax * ey - ay * ex) / den;
        float u = (ax * fy - ay * fx) / den;
        if (t < -1e-4f || u < -1e-4f || u > 1.0001f) continue;
        if (t < t1) { t2 = t1; w2 = w1; t1 = t; w1 = w; }
        else if (t < t2) { t2 = t; w2 = w; }
    }
    double best = __longlong_as_double(0x7ff0000000000000LL);
    const double* A = wa + (long long)k * Wp * 2;
    const double* E = we + (long long)k * Wp * 2;
    long long ws[2] = {w1, w2};
    for (int j = 0; j < 2; ++j) {
        long long w = ws[j];
        if (w < 0) continue;
        double aox = A[2 * w] - px, aoy = A[2 * w + 1] - py;
        double ex = E[2 * w], ey = E[2 * w + 1];
        double den = dx * ey - dy * ex;
        double t = (aox * ey - aoy * ex) / den;
        double u = (aox * dy - aoy * dx) / den;
        if (fabs(den) > 1e-9 && t >= 0.0 && u >= 0.0 && u <= 1.0 && t < best) best = t;
    }
    out[i] = best < max_range ? best : max_range;
}
"""
_raycast_kernel = None


def raycast_kernel():
    """The compiled kernel, or None if it cannot be compiled here (then PyTorch ops are used)."""
    global _raycast_kernel
    if _raycast_kernel is None:
        try:
            import os
            import torch.utils.cpp_extension as ce
            if ce.CUDA_HOME is None:                 # the kernel needs no CUDA headers
                ce.CUDA_HOME = os.path.dirname(torch.__file__)
            _raycast_kernel = torch.cuda._compile_kernel(RAYCAST_SRC, "raycast", nvcc_options=["--fmad=false"])
        except Exception:
            _raycast_kernel = False
    return _raycast_kernel or None


def candidate_walls(track, max_range):
    """
    For every centre-line point: the walls a car near it could see within `max_range`.
    A car is at most half a width (+ one segment) from its closest centre-line point,
    so this is a superset of the walls the full ray cast could hit: the result is exact.
    """
    reach = 0.5 * np.linalg.norm(track.wall_b - track.wall_a, axis=1).max()
    radius = max_range + reach + track.half_width + 3.0
    d2 = ((track.center[:, None, :] - track.wall_mid[None, :, :]) ** 2).sum(-1)
    return [np.nonzero(row < radius * radius)[0] for row in d2]


class TrackBatch:
    """K circuits as padded tensors."""

    def __init__(self, tracks, device, dtype=torch.float64, max_range=150.0):
        K = len(tracks)
        n_max = max(t.N for t in tracks)
        f = dict(device=device, dtype=dtype)
        center = np.zeros((K, n_max, 2))
        tangent = np.zeros((K, n_max, 2))
        seg_len = np.ones((K, n_max))
        cum = np.zeros((K, n_max))
        wall_a = np.zeros((K, 2 * n_max + 1, 2))
        wall_e = np.zeros((K, 2 * n_max + 1, 2))             # zero length = never hit (last one: padding)
        cands = [candidate_walls(t, max_range) for t in tracks]
        c_max = max(len(c) for cs in cands for c in cs)
        cand = np.full((K, n_max, c_max), 2 * n_max)          # padding -> the zero-length wall
        start = np.zeros((K, 5))                             # x, y, heading, segment, s
        for k, t in enumerate(tracks):
            n = t.N
            center[k, :n], tangent[k, :n], seg_len[k, :n], cum[k, :n] = t.center, t.tangent, t.seg_len, t.cum
            wall_a[k, :2 * n] = t.wall_a
            wall_e[k, :2 * n] = t.wall_b - t.wall_a
            for i, c in enumerate(cands[k]):
                cand[k, i, :len(c)] = c
            seg, s, _ = t.closest(t.start_pos)
            start[k] = (t.start_pos[0], t.start_pos[1], t.start_heading, seg, s)
        self.K = K
        self.center = torch.tensor(center, **f)
        self.tangent = torch.tensor(tangent, **f)
        self.seg_len = torch.tensor(seg_len, **f)
        self.cum = torch.tensor(cum, **f)
        self.N = torch.tensor([t.N for t in tracks], device=device)
        self.length = torch.tensor([t.length for t in tracks], **f)
        self.half_w = torch.tensor([t.half_width for t in tracks], **f)
        self.max_time = torch.tensor([lap_time_limit(t) for t in tracks], **f)
        self.wall_a = torch.tensor(wall_a, **f)
        self.wall_e = torch.tensor(wall_e, **f)
        self.wall_a32, self.wall_e32 = self.wall_a.float(), self.wall_e.float()
        self.cand = torch.tensor(cand, device=device)        # (K, N, C) wall indices near each point
        self.start = torch.tensor(start, **f)


class BatchSim:
    """
    P networks (genomes) each driving all K circuits: P x K cars.
    step(n) advances, running() tells if any car is still going,
    snapshot() gives positions and sensors, results() the scores.
    """

    def __init__(self, tracks, genomes, device=None, dtype=torch.float64,
                 dt=1 / 30, substeps=2, n_rays=9, ray_fov_deg=180.0, max_range=150.0,
                 track_batch=None, use_graph=True):
        self.device = device or default_device()
        self.dtype = dtype
        f = self.f = dict(device=self.device, dtype=dtype)
        self.tb = track_batch or TrackBatch(tracks, self.device, dtype, max_range)
        self.p = p = CarParams()
        self.dt, self.substeps, self.max_range = dt, substeps, max_range
        half = math.radians(ray_fov_deg) / 2
        self.ray_angles = torch.tensor(np.linspace(-half, half, n_rays), **f)

        genomes = np.asarray(genomes, dtype=float)
        P, K = len(genomes), self.tb.K
        self.P, self.K = P, K
        parts = [Net(n_in=n_rays + 2).unpack(g) for g in genomes]
        self.w1 = torch.tensor(np.stack([q[0] for q in parts]), **f)              # (P, 11, 16)
        self.b1 = torch.tensor(np.stack([q[1] for q in parts]), **f)[:, None]     # (P, 1, 16)
        self.w2 = torch.tensor(np.stack([q[2] for q in parts]), **f)              # (P, 16, 2)
        self.b2 = torch.tensor(np.stack([q[3] for q in parts]), **f)[:, None]     # (P, 1, 2)

        # constants used inside the step
        self._offsets = torch.arange(-WINDOW, WINDOW + 1, device=self.device)
        self._kk = torch.arange(K, device=self.device)[None, :, None]
        self._kpk = self._kk[..., 0].expand(P, K)
        hl, hw = p.length / 2, p.width / 2
        self._lx = torch.tensor([hl, hl, -hl, -hl], **f)
        self._ly = torch.tensor([hw, -hw, -hw, hw], **f)
        self._inf = torch.tensor(math.inf, **f)
        self._inf32 = torch.tensor(math.inf, device=self.device, dtype=torch.float32)

        # state: fixed buffers, updated in place (needed by the CUDA graph)
        st = self.tb.start
        ones = torch.ones((P, K), **f)
        self.x, self.y, self.heading = ones * st[:, 0], ones * st[:, 1], ones * st[:, 2]
        self.speed = torch.zeros((P, K), **f)
        self.steer = torch.zeros((P, K), **f)
        self.seg = (ones * st[:, 3]).long()
        self.prev_s = ones * st[:, 4]
        self.progress = torch.zeros((P, K), **f)
        self.best_progress = torch.zeros((P, K), **f)
        self.last_improve = torch.zeros((P, K), **f)
        self.lap_time = torch.full((P, K), -1.0, **f)
        self.crashed = torch.zeros((P, K), dtype=torch.bool, device=self.device)
        self.done = torch.zeros((P, K), dtype=torch.bool, device=self.device)
        self.time_t = torch.zeros((), **f)                   # simulated seconds
        self.ray = torch.zeros((P, K, n_rays), **f)
        self.steps = 0
        self._kernel = raycast_kernel() if self.device.type == "cuda" else None
        self._sense()

        self.use_graph = use_graph and self.device.type == "cuda"
        self._graph = None

    @property
    def time(self):
        return self.steps * self.dt

    # ================================================================ track
    def _window(self, seg):
        """Indices of the segments around `seg` (P, K, 2W+1) and their geometry."""
        idx = torch.remainder(seg[..., None] + self._offsets, self.tb.N[None, :, None])
        kk = self._kk.expand_as(idx)
        return idx, self.tb.center[kk, idx], self.tb.tangent[kk, idx], self.tb.seg_len[kk, idx]

    @staticmethod
    def _project(px, py, a, d, L):
        """Point(s) -> closest point on each window segment: (t along it, squared distance)."""
        rx, ry = px - a[..., 0], py - a[..., 1]
        t = torch.minimum(torch.clamp(rx * d[..., 0] + ry * d[..., 1], min=0.0), L)
        qx, qy = px - (a[..., 0] + d[..., 0] * t), py - (a[..., 1] + d[..., 1] * t)
        return t, qx * qx + qy * qy

    def _sense(self):
        """
        Distance to the first wall along each ray (P, K, R), against the walls near each car.

        Two stages, so the GPU does not pay for float64 on every wall:
          1. float32 over all nearby walls, only to pick the 2 closest candidates per ray;
          2. float64 (exact, as env.py) for those 2 candidates only.
        """
        if self._kernel is not None:
            tb, n = self.tb, self.P * self.K * len(self.ray_angles)
            self._kernel(grid=((n + 127) // 128, 1, 1), block=(128, 1, 1),
                         args=[self.x, self.y, self.heading, self.seg, tb.cand, tb.wall_a, tb.wall_e,
                               tb.wall_a32, tb.wall_e32, self.ray_angles, self.ray,
                               self.P * self.K, self.K, len(self.ray_angles), tb.cand.shape[1],
                               tb.cand.shape[2], tb.wall_a.shape[1], float(self.max_range)])
            return
        h = self.heading[..., None] + self.ray_angles                     # (P, K, R)
        dx, dy = torch.cos(h), torch.sin(h)
        cand = self.tb.cand[self._kpk, self.seg]                           # (P, K, C)
        kc = self._kk.expand_as(cand)

        # stage 1: float32, every nearby wall
        wa, we = self.tb.wall_a32[kc, cand], self.tb.wall_e32[kc, cand]    # (P, K, C, 2)
        ex, ey = we[:, :, None, :, 0], we[:, :, None, :, 1]                # (P, K, 1, C)
        aox = wa[:, :, None, :, 0] - self.x.float()[..., None, None]
        aoy = wa[:, :, None, :, 1] - self.y.float()[..., None, None]
        fx, fy = dx.float()[..., None], dy.float()[..., None]               # (P, K, R, 1)
        denom = fx * ey - fy * ex                                          # (P, K, R, C)
        t = (aox * ey - aoy * ex) / denom
        u = (aox * fy - aoy * fx) / denom
        tol = 1e-4                                                         # generous: the exact test is stage 2
        ok = (denom.abs() > 1e-9) & (t >= -tol) & (u >= -tol) & (u <= 1 + tol)
        t = torch.where(ok, t, self._inf32)
        best = torch.topk(t, k=min(2, t.shape[-1]), dim=-1, largest=False).indices    # (P, K, R, 2)

        # stage 2: float64, exact, on the 2 candidates
        w = cand[:, :, None, :].expand(-1, -1, best.shape[2], -1).gather(-1, best)     # wall indices
        kw = self._kk[..., None].expand_as(w)
        a, e = self.tb.wall_a[kw, w], self.tb.wall_e[kw, w]                # (P, K, R, 2, 2)
        ex, ey = e[..., 0], e[..., 1]
        aox, aoy = a[..., 0] - self.x[..., None, None], a[..., 1] - self.y[..., None, None]
        dx, dy = dx[..., None], dy[..., None]
        denom = dx * ey - dy * ex
        t = (aox * ey - aoy * ex) / denom
        u = (aox * dy - aoy * dx) / denom
        hit = (denom.abs() > 1e-9) & (t >= 0) & (u >= 0) & (u <= 1)
        t = torch.where(hit, t, self._inf)
        self.ray.copy_(torch.clamp(t.amin(dim=-1), max=self.max_range))

    def _obs(self):
        p = self.p
        obs = torch.cat([self.ray / self.max_range, (self.speed / p.max_speed)[..., None],
                         (self.steer / p.max_steer)[..., None]], dim=-1)
        return obs.to(torch.float32).to(self.dtype)                        # same rounding as env.py

    # ================================================================ step
    def step(self, n=1):
        if n <= 0:
            return
        if self.use_graph:
            if self._graph is None:
                n -= self._build_graph()
            for _ in range(max(0, n)):
                self._graph.replay()
                self.steps += 1
        else:
            for _ in range(n):
                self._step()
                self.steps += 1

    def _build_graph(self):
        """Run 3 real steps on a side stream (warm-up), then record one step as a CUDA graph."""
        s = torch.cuda.Stream(self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step()
                self.steps += 1
        torch.cuda.current_stream(self.device).wait_stream(s)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._step()                                   # recorded, not executed
        return 3

    def _step(self):
        p, tb = self.p, self.tb
        run = ~self.done

        # network: one matrix product per network, all its circuits together
        hid = torch.tanh(torch.bmm(self._obs(), self.w1) + self.b1)
        act = torch.tanh(torch.bmm(hid, self.w2) + self.b2)
        steer_cmd, throttle = act[..., 0].clamp(-1, 1), act[..., 1].clamp(-1, 1)

        # physics (car.py), sub-steps
        x, y, heading, speed, steer = self.x, self.y, self.heading, self.speed, self.steer
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            target = steer_cmd * p.max_steer
            steer = steer + torch.clamp(target - steer, -p.steer_rate * h, p.steer_rate * h)
            accel = torch.where(throttle >= 0, throttle * p.engine_accel * (1.0 - speed / p.max_speed),
                                throttle * p.brake_decel) - p.rolling_decel
            yaw = speed / p.wheelbase * torch.tan(steer)
            max_yaw = p.max_lat_accel / torch.clamp(speed, min=0.1)
            sliding = yaw.abs() > max_yaw
            yaw = torch.where(sliding, torch.copysign(max_yaw, yaw), yaw)
            accel = torch.where(sliding, accel - p.scrub_decel, accel)
            speed = torch.clamp(speed + accel * h, 0.0, p.max_speed)
            heading = torch.remainder(heading + yaw * h + math.pi, 2 * math.pi) - math.pi
            x = x + speed * torch.cos(heading) * h
            y = y + speed * torch.sin(heading) * h
        self.time_t.add_(self.dt)
        t_now = self.time_t

        # progress along the track (env.py)
        idx, a, d, L = self._window(self.seg)
        tt, d2 = self._project(x[..., None], y[..., None], a, d, L)
        k = d2.argmin(dim=-1, keepdim=True)
        seg = idx.gather(-1, k)[..., 0]
        s = tb.cum[self._kpk, seg] + tt.gather(-1, k)[..., 0]
        length = tb.length[None, :]
        ds = s - self.prev_s
        ds = torch.where(ds < -length / 2, ds + length, torch.where(ds > length / 2, ds - length, ds))
        progress = self.progress + ds
        lap = progress >= length

        # collision: a corner of the car outside the edges (searched around the new segment)
        c, sn = torch.cos(heading)[..., None], torch.sin(heading)[..., None]
        cx = x[..., None] + self._lx * c - self._ly * sn                  # (P, K, 4)
        cy = y[..., None] + self._lx * sn + self._ly * c
        _, a2, d2_, L2 = self._window(seg)
        _, cd2 = self._project(cx[..., None], cy[..., None], a2[:, :, None], d2_[:, :, None], L2[:, :, None])
        crash = (cd2.amin(dim=-1).sqrt() > tb.half_w[None, :, None]).any(dim=-1)

        improve = progress > self.best_progress + 0.5
        best = torch.where(improve, progress, self.best_progress)
        last_imp = torch.where(improve, t_now, self.last_improve)
        terminated = lap | crash
        truncated = ~terminated & (((t_now - last_imp) > STALL_TIME) | (t_now >= tb.max_time[None, :]))

        # only cars still running change (in place: the CUDA graph reuses these buffers)
        for buf, new in ((self.x, x), (self.y, y), (self.heading, heading), (self.speed, speed),
                         (self.steer, steer), (self.seg, seg), (self.prev_s, s), (self.progress, progress),
                         (self.best_progress, best), (self.last_improve, last_imp)):
            buf.copy_(torch.where(run, new, buf))
        self.lap_time.copy_(torch.where(run & lap, t_now, self.lap_time))
        self.crashed.copy_(self.crashed | (run & crash))
        self.done.copy_(self.done | (run & (terminated | truncated)))
        self._sense()

    # ============================================================== results
    def running(self):
        return bool((~self.done).any())

    def scores(self):
        """track_score() of every car (P, K): share of lap, or 1 + speed bonus when completed."""
        length = self.tb.length[None, :]
        lapped = self.lap_time > 0
        return torch.where(lapped, 1.0 + (length / self.lap_time.clamp(min=1e-9)) / SPEED_REF,
                           self.progress.clamp(min=0.0) / length)

    def snapshot(self):
        """(state (P, K, 10), obs (P, K, 11)) as numpy, columns as in trainer.py."""
        score = torch.where(self.done, self.scores(), torch.zeros_like(self.progress))
        laps = (self.lap_time > 0).to(self.dtype)
        state = torch.stack([self.x, self.y, self.heading, self.speed, self.done.to(self.dtype),
                             self.crashed.to(self.dtype), self.progress, laps, self.lap_time, score], dim=-1)
        return state.cpu().numpy(), self._obs().to(torch.float32).cpu().numpy()

    def results(self):
        """Per car (score, lap time or None, crashed), as a P x K nested list."""
        sc = self.scores().cpu().numpy()
        lap = self.lap_time.cpu().numpy()
        cr = self.crashed.cpu().numpy()
        return [[(float(sc[i, k]), float(lap[i, k]) if lap[i, k] > 0 else None, bool(cr[i, k]))
                 for k in range(self.K)] for i in range(self.P)]

    def run_to_end(self, check_every=64, max_steps=100_000):
        """Advance until every car has finished (headless)."""
        while self.steps < max_steps and self.running():
            self.step(check_every)
