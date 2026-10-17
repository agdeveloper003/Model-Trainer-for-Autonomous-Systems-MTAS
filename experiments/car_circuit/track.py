"""
Circuit: geometry, progress along the track, ray sensors and the
random track generator.

Units: metres. The y axis points up (as in maths); the renderer flips it
for the screen.

A circuit is described by a few "control points". A closed Catmull-Rom
spline joins them into a smooth curve (the centre line), which is then
resampled at a constant spacing. The two edges (the "walls") are derived
from the centre line and the track width.

Random tracks are generated from a TrackSpec: the same spec (seed
included) always gives exactly the same circuit.
"""
import argparse
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# Hand-drawn circuit (metres). Driving direction: counter-clockwise.
# The first point is on the start straight.
# --------------------------------------------------------------------------
CLASSIC_LAYOUT = [
    (0, 0), (120, 0), (240, 0), (330, 10),          # main straight
    (390, 50), (405, 115),                          # long left-hander
    (375, 175), (310, 195),                         # towards the chicane
    (265, 175), (225, 195), (215, 245),             # right-left chicane
    (250, 300), (320, 320), (375, 350),             # fast esses
    (385, 410), (340, 450),                         # hairpin
    (250, 460), (150, 450), (70, 420),              # back straight
    (20, 370), (5, 300), (35, 240),                 # descent
    (60, 185), (40, 125), (-15, 90),                # slow S-bend
    (-60, 50), (-55, 10),                           # final corner
]


def catmull_rom_closed(points, samples_per_segment=40):
    """Smooth closed curve passing through every control point."""
    P = np.asarray(points, dtype=float)
    n = len(P)
    t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)[:, None]
    t2, t3 = t * t, t * t * t
    out = []
    for i in range(n):
        p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        out.append(0.5 * (2 * p1
                          + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.vstack(out)


def closed_length(poly):
    closed = np.vstack([poly, poly[:1]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def resample_closed(poly, spacing):
    """Resample a closed polyline at (roughly) constant spacing."""
    closed = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    n = max(8, int(round(total / spacing)))
    s = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(s, cum, closed[:, 0])
    y = np.interp(s, cum, closed[:, 1])
    return np.stack([x, y], axis=1)


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class Track:
    def __init__(self, control_points=None, width=14.0, spacing=2.0, name="Classic circuit"):
        self.name = name
        self.width = float(width)
        self.half_width = self.width / 2.0
        points = CLASSIC_LAYOUT if control_points is None else control_points

        c = resample_closed(catmull_rom_closed(points), spacing)
        self.center = c                                   # (N, 2) centre-line points
        self.N = len(c)

        seg = np.roll(c, -1, axis=0) - c                  # segment i: center[i] -> center[i+1]
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.tangent = seg / self.seg_len[:, None]
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)[:-1]])  # arc length
        self.length = float(self.seg_len.sum())

        # normal (pointing left) at the vertices: average of the adjacent tangents
        tv = self.tangent + np.roll(self.tangent, 1, axis=0)
        tv /= np.linalg.norm(tv, axis=1)[:, None]
        self.normal = np.stack([-tv[:, 1], tv[:, 0]], axis=1)
        self.left = c + self.normal * self.half_width
        self.right = c - self.normal * self.half_width

        # walls as segments A->B (left edge + right edge)
        self.wall_a = np.vstack([self.left, self.right])
        self.wall_b = np.vstack([np.roll(self.left, -1, 0), np.roll(self.right, -1, 0)])
        self.wall_mid = 0.5 * (self.wall_a + self.wall_b)
        self.wall_reach = 0.5 * np.linalg.norm(self.wall_b - self.wall_a, axis=1).max()

        self.start_pos = c[0].copy()
        self.start_heading = float(np.arctan2(self.tangent[0, 1], self.tangent[0, 0]))

    # ------------------------------------------------------------------
    def closest(self, pos, hint=None, window=40):
        """
        Closest point of the centre line to `pos`.

        Returns (segment_index, s, lateral):
          s       = metres travelled along the circuit from the finish line
          lateral = signed distance from the centre line (+ = left)
        `hint` is the index found at the previous step: the search only looks
        around it, which keeps it fast. Without a hint the whole track is searched.
        """
        pos = np.asarray(pos, dtype=float)
        if hint is None:
            idx = np.arange(self.N)
        else:
            idx = (hint + np.arange(-window, window + 1)) % self.N
        a, d, L = self.center[idx], self.tangent[idx], self.seg_len[idx]
        rel = pos - a
        t = np.clip((rel * d).sum(axis=1), 0.0, L)
        proj = a + d * t[:, None]
        dist2 = ((pos - proj) ** 2).sum(axis=1)
        k = int(np.argmin(dist2))
        i = int(idx[k])
        side = np.sign(_cross(d[k], rel[k])) or 1.0
        return i, float(self.cum[i] + t[k]), float(side * np.sqrt(dist2[k]))

    # ------------------------------------------------------------------
    def raycast(self, origin, directions, max_range):
        """
        Cast rays from `origin` along `directions` (R, 2 unit vectors) and
        return the distance to the first wall hit (max_range if none).
        """
        o = np.asarray(origin, dtype=float)
        near = ((self.wall_mid - o) ** 2).sum(axis=1) < (max_range + self.wall_reach) ** 2
        A = self.wall_a[near]
        E = self.wall_b[near] - A
        AO = A - o                                          # (M, 2)
        d = np.asarray(directions, dtype=float)             # (R, 2)
        dx, dy = d[:, 0:1], d[:, 1:2]                       # (R, 1)
        denom = dx * E[None, :, 1] - dy * E[None, :, 0]     # cross(d, E)   (R, M)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (AO[None, :, 0] * E[None, :, 1] - AO[None, :, 1] * E[None, :, 0]) / denom
            u = (AO[None, :, 0] * dy - AO[None, :, 1] * dx) / denom
        hit = (np.abs(denom) > 1e-9) & (t >= 0) & (u >= 0) & (u <= 1)
        t = np.where(hit, t, np.inf)
        if t.shape[1] == 0:
            return np.full(len(d), float(max_range))
        return np.minimum(t.min(axis=1), max_range)

    # ------------------------------------------------------------------
    def min_curve_radius(self):
        """Tightest corner radius (useful to check a layout)."""
        dth = np.arctan2(_cross(np.roll(self.tangent, 1, 0), self.tangent),
                         (np.roll(self.tangent, 1, 0) * self.tangent).sum(1))
        k = np.abs(dth) / self.seg_len
        return float(1.0 / k.max())

    def min_clearance(self, step=2):
        """
        Smallest distance between two parts of the centre line that are far
        apart along the track (e.g. two parallel straights). If it is smaller
        than the track width, the asphalt overlaps itself.
        """
        idx = np.arange(0, self.N, step)
        P, s = self.center[idx], self.cum[idx]
        d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        ds = np.abs(s[:, None] - s[None, :])
        ds = np.minimum(ds, self.length - ds)               # distance along the loop
        far = ds > 3 * self.width + 20
        return float(d[far].min()) if far.any() else np.inf

    def bounds(self):
        pts = np.vstack([self.left, self.right])
        return pts.min(axis=0), pts.max(axis=0)


# ==========================================================================
# Random track generator
# ==========================================================================
@dataclass
class TrackSpec:
    """Everything needed to rebuild a circuit exactly."""
    kind: str = "classic"          # "classic" (hand drawn) or "random"
    seed: int = 1
    corners: int = 12              # control points of the random layout
    length: float = 2000.0         # target length in metres
    width: float = 14.0            # track width in metres
    twistiness: float = 0.5        # 0 = almost an oval, 1 = very twisty
    clockwise: bool = False        # driving direction

    LIMITS = {"corners": (5, 24), "length": (800.0, 4000.0),
              "width": (10.0, 20.0), "twistiness": (0.0, 1.0)}

    def clamped(self):
        lim = self.LIMITS
        return TrackSpec(
            kind=self.kind, seed=int(self.seed),
            corners=int(np.clip(self.corners, *lim["corners"])),
            length=float(np.clip(self.length, *lim["length"])),
            width=float(np.clip(self.width, *lim["width"])),
            twistiness=float(np.clip(self.twistiness, *lim["twistiness"])),
            clockwise=bool(self.clockwise),
        )

    def track_id(self):
        """Short unique name, used e.g. for the saved network's file name."""
        cw = "-cw" if self.clockwise else ""
        if self.kind == "classic":
            return "classic" + cw
        s = self.clamped()
        return f"random-s{s.seed}-c{s.corners}-l{s.length:g}-w{s.width:g}-t{s.twistiness:g}{cw}"

    def build(self):
        if self.kind == "classic":
            if not self.clockwise:
                return Track()
            # same layout driven the other way: keep the first point as the start
            return Track(np.roll(np.asarray(CLASSIC_LAYOUT[::-1], float), 1, axis=0),
                         name="Classic circuit (clockwise)")
        return generate_track(self.clamped())

    def to_args(self):
        """Command-line arguments that rebuild this spec (see add_track_args)."""
        cw = ["--clockwise"] if self.clockwise else []
        if self.kind == "classic":
            return ["--track", "classic"] + cw
        s = self.clamped()
        return ["--track", "random", "--seed", str(s.seed), "--corners", str(s.corners),
                "--length", f"{s.length:g}", "--width", f"{s.width:g}",
                "--twistiness", f"{s.twistiness:g}"] + cw


def generate_track(spec, max_attempts=300):
    """
    Build a random circuit from `spec`. Deterministic: the random generator
    is seeded with spec.seed, so the same spec gives the same track.

    Recipe: control points around a circle (counter-clockwise, or reversed
    for a clockwise track), each with a
    random angle and radius; more twistiness = radii vary more, so the
    layout gets more bends and hairpins. The result is stretched and
    rotated, scaled to the requested length, and rejected if a corner is
    too tight or two parts of the track overlap. In that case the next
    candidate from the same seeded generator is tried.
    """
    rng = np.random.default_rng(spec.seed)
    n, tw = spec.corners, spec.twistiness
    half_w = spec.width / 2
    for _ in range(max_attempts):
        base = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        ang = np.sort(base + rng.uniform(-0.4, 0.4, n) * (2 * np.pi / n))
        r = 1.0 - (0.15 + 0.45 * tw) * rng.random(n)
        pts = np.stack([np.cos(ang) * r, np.sin(ang) * r], axis=1)

        stretch = rng.uniform(1.0, 1.6)                     # make it less round
        rot = rng.uniform(0.0, 2 * np.pi)
        pts[:, 0] *= stretch
        c, s = np.cos(rot), np.sin(rot)
        pts = pts @ np.array([[c, -s], [s, c]]).T

        pts *= spec.length / closed_length(catmull_rom_closed(pts))
        if spec.clockwise:
            pts = pts[::-1].copy()

        # start on the longest straight-ish stretch between two control points
        gaps = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        pts = np.roll(pts, -int(np.argmax(gaps)), axis=0)

        name = f"Random #{spec.seed}" + (" (clockwise)" if spec.clockwise else "")
        track = Track(pts, width=spec.width, name=name)
        if track.min_curve_radius() < half_w * 1.2 + 3:
            continue
        if track.min_clearance() < spec.width + 8:
            continue
        return track
    raise ValueError("Could not build a valid track with these settings. "
                     "Try another seed, fewer corners, a longer length or a narrower track.")


# Settings the generalist driver is trained and validated on.
GENERAL_RANGES = {"corners": (6, 16), "length": (1200.0, 3000.0),
                  "width": (12.0, 18.0), "twistiness": (0.0, 0.8)}


def sample_track(rng, ranges=GENERAL_RANGES):
    """A random valid circuit with random settings (and direction). Returns (spec, track)."""
    while True:
        spec = TrackSpec(
            kind="random", seed=int(rng.integers(0, 1_000_000_000)),
            corners=int(rng.integers(ranges["corners"][0], ranges["corners"][1] + 1)),
            length=float(round(rng.uniform(*ranges["length"]) / 100) * 100),
            width=float(round(rng.uniform(*ranges["width"]))),
            twistiness=float(round(rng.uniform(*ranges["twistiness"]) / 0.05) * 0.05),
            clockwise=bool(rng.random() < 0.5),
        ).clamped()
        try:
            return spec, spec.build()
        except ValueError:
            continue                                        # impossible settings: draw again


# ---------------------------------------------------------- command line
def add_track_args(parser):
    d = TrackSpec()
    g = parser.add_argument_group("track")
    g.add_argument("--track", choices=["classic", "random"], default=d.kind)
    g.add_argument("--seed", type=int, default=d.seed, help="random track seed")
    g.add_argument("--corners", type=int, default=d.corners)
    g.add_argument("--length", type=float, default=d.length, help="metres")
    g.add_argument("--width", type=float, default=d.width, help="metres")
    g.add_argument("--twistiness", type=float, default=d.twistiness, help="0..1")
    g.add_argument("--clockwise", action="store_true", help="drive the track the other way")


def spec_from_args(args):
    return TrackSpec(kind=args.track, seed=args.seed, corners=args.corners,
                     length=args.length, width=args.width, twistiness=args.twistiness,
                     clockwise=args.clockwise)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build a track and print its stats.")
    add_track_args(ap)
    spec = spec_from_args(ap.parse_args())
    t = spec.build()
    print(f"{t.name}: {t.length:.0f} m, width {t.width:g} m, tightest corner radius "
          f"{t.min_curve_radius():.1f} m, id {spec.track_id()}")
