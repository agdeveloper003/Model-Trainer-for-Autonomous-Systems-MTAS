"""
Track view: draws a circuit and the cars on it inside a rectangle of the window.

It is only an observer: it reads positions and draws them, it never
changes the simulation.

Two cameras:
  - overview: the whole circuit fits the rectangle (pre-drawn once, fast);
  - close-up: follows a car and rotates with it, so its nose always points
    up. Only the nearby pieces of track are drawn, as vector shapes.
"""
import math

import numpy as np
import pygame

from ui import BAD, GOOD, WARN, lerp_color

# ---------------------------------------------------------------- colours
GRASS_A = (54, 92, 52)
GRASS_B = (58, 98, 55)
ASPHALT = (58, 60, 66)
ASPHALT_LINE = (78, 80, 88)
KERB_RED = (196, 52, 46)
KERB_WHITE = (232, 232, 232)
CAR_BODY = (40, 140, 255)
CAR_GLASS = (20, 30, 50)
CAR_CRASH = (255, 70, 60)

OVERVIEW_CAR_SCALE = 2.5     # in the overview the car is enlarged so you can see it
FOLLOW_SCALE = 6.0           # pixels per metre in the close-up view
GRASS_STRIPE = 20.0          # metres, width of the grass stripes


def fmt_time(t):
    if t is None:
        return "--:--.--"
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:05.2f}"


def speed_color(kmh, slow=40.0, fast=200.0):
    t = (kmh - slow) / (fast - slow)
    if t < 0.5:
        return lerp_color(BAD, WARN, t * 2)
    return lerp_color(WARN, GOOD, (t - 0.5) * 2)


def car_corners(x, y, heading, length=4.5, width=2.0):
    """The 4 corners of a car body (front-left, front-right, rear-right, rear-left)."""
    c, s = math.cos(heading), math.sin(heading)
    hl, hw = length / 2, width / 2
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    return local @ np.array([[c, -s], [s, c]]).T + (x, y)


class TrackView:
    def __init__(self, track, lite=False):
        """lite = simplified drawing (asphalt and edges only) for small thumbnails."""
        self.track = track
        self.lite = lite
        self.rect = None
        self.follow = False
        self._build_geometry()

    # ======================================================== track shapes
    def _build_geometry(self):
        """Every piece of the track as a small polygon, computed once."""
        t = self.track
        L, R, N, n = t.left, t.right, t.N, t.normal
        j = np.roll(np.arange(N), -1)
        self.q_asphalt = np.stack([L, L[j], R[j], R], axis=1)            # (N, 4, 2)

        kerb = 1.0                                                       # 1 m wide, inside the edges
        Li, Ri = L - n * kerb, R + n * kerb
        self.q_kerb = np.concatenate([np.stack([L, L[j], Li[j], Li], axis=1),
                                      np.stack([R, R[j], Ri[j], Ri], axis=1)])
        self.kerb_seg = np.concatenate([np.arange(N), np.arange(N)])     # which segment each kerb is on
        self.kerb_col = [KERB_RED if (i // 3) % 2 == 0 else KERB_WHITE for i in self.kerb_seg]

        dash = np.arange(0, N, 6)                                        # dashed centre line
        self.dash = np.stack([t.center[dash], t.center[(dash + 3) % N]], axis=1)
        self.dash_seg = dash

        # chequered finish line
        c0, n0, d0 = t.center[0], t.normal[0], t.tangent[0]
        cells = 8
        size = t.width / cells
        quads, cols = [], []
        for row in range(2):
            for k in range(cells):
                base = c0 - n0 * t.half_width + n0 * (k * size) + d0 * (row * size)
                quads.append([base, base + n0 * size, base + n0 * size + d0 * size, base + d0 * size])
                cols.append((240, 240, 240) if (k + row) % 2 == 0 else (25, 25, 25))
        self.q_finish, self.finish_col = np.array(quads), cols

    def _draw_shapes(self, surf, to_screen, seg_mask, line_w):
        """Draw the track pieces whose segment is in `seg_mask` (bool array, len N)."""
        def polys(q, idx):
            if len(idx) == 0:
                return []
            pts = to_screen(q[idx].reshape(-1, 2))
            return np.asarray(pts).reshape(len(idx), -1, 2).tolist()

        idx = np.nonzero(seg_mask)[0]
        for p in polys(self.q_asphalt, idx):
            pygame.draw.polygon(surf, ASPHALT, p)
        didx = np.nonzero(seg_mask[self.dash_seg])[0]
        for a, b in polys(self.dash, didx):
            pygame.draw.line(surf, ASPHALT_LINE, a, b, line_w)
        kidx = np.nonzero(seg_mask[self.kerb_seg])[0]
        for k, p in zip(kidx, polys(self.q_kerb, kidx)):
            pygame.draw.polygon(surf, self.kerb_col[k], p)
        if seg_mask[0] or seg_mask[-1]:
            for col, p in zip(self.finish_col, polys(self.q_finish, np.arange(len(self.q_finish)))):
                pygame.draw.polygon(surf, col, p)

    # ================================================================ camera
    def set_rect(self, rect):
        """Place the view in the window; the overview is re-drawn only if the size changed."""
        rect = pygame.Rect(rect)
        if self.rect is not None and rect.size == self.rect.size:
            self.rect = rect
            return
        self.rect = rect
        lo, hi = self.track.bounds()
        pad = 20.0                                         # metres of grass around the track
        span = (hi - lo) + 2 * pad
        self.overview_scale = min(rect.w / span[0], rect.h / span[1])
        self.overview_center = (lo + hi) / 2
        self.bg_origin = self.overview_center + np.array([-rect.w, rect.h]) / 2 / self.overview_scale
        self.bg_overview = self._render_overview()

    def begin(self, surf, follow=None):
        """
        Start drawing a frame: clip to the view and draw the ground and track.
        follow = (x, y, heading) of the car to follow, or None for the overview.
        """
        r = self.rect
        if follow is not None and self.follow:
            self._cam = (FOLLOW_SCALE, np.array(follow[:2], float), r.center, math.pi / 2 - follow[2])
        else:
            self._cam = (self.overview_scale, self.overview_center, r.center, 0.0)
        surf.set_clip(r)
        if self._cam[3] or (follow is not None and self.follow):
            surf.fill(GRASS_A, r)
            self._draw_follow_track(surf)
        else:
            surf.blit(self.bg_overview, r.topleft)

    def end(self, surf):
        surf.set_clip(None)

    @property
    def car_scale(self):
        """How much cars are enlarged with the current camera."""
        return 1.0 if self._cam[0] == FOLLOW_SCALE else OVERVIEW_CAR_SCALE

    def w2s(self, p):
        return self.w2s_many([p])[0]

    def w2s_many(self, pts):
        s, c, sc, a = self._cam
        pts = np.asarray(pts, dtype=float)
        dx, dy = pts[:, 0] - c[0], pts[:, 1] - c[1]
        if a:
            ca, sa = math.cos(a), math.sin(a)
            dx, dy = ca * dx - sa * dy, sa * dx + ca * dy
        x = sc[0] + dx * s
        y = sc[1] - dy * s
        return list(zip(x.tolist(), y.tolist()))

    # ====================================================== track drawing
    def _render_overview(self):
        """Draw the whole track once onto a surface the size of the view."""
        scale = self.overview_scale
        w, h = self.rect.size
        surf = pygame.Surface((w, h))
        surf.fill(GRASS_A)
        if self.lite:
            o, t = self.bg_origin, self.track
            pts = np.stack([(t.center[:, 0] - o[0]) * scale, (o[1] - t.center[:, 1]) * scale], axis=1).tolist()
            pygame.draw.lines(surf, KERB_RED, True, pts, max(3, int(t.width * scale) + 3))
            pygame.draw.lines(surf, ASPHALT, True, pts, max(1, int(t.width * scale)))
            a = t.center[0] + t.normal[0] * t.half_width * 1.8
            b = t.center[0] - t.normal[0] * t.half_width * 1.8
            pygame.draw.line(surf, KERB_WHITE, ((a[0] - o[0]) * scale, (o[1] - a[1]) * scale),
                             ((b[0] - o[0]) * scale, (o[1] - b[1]) * scale), 2)
            return surf
        stripe = max(8, int(GRASS_STRIPE * scale))
        for i in range(0, w, stripe * 2):
            pygame.draw.rect(surf, GRASS_B, (i, 0, stripe, h))

        o = self.bg_origin

        def to_bitmap(p):
            return np.stack([(p[:, 0] - o[0]) * scale, (o[1] - p[:, 1]) * scale], axis=1)

        self._draw_shapes(surf, to_bitmap, np.ones(self.track.N, bool), max(1, int(0.25 * scale)))
        return surf

    def _draw_follow_track(self, surf):
        """Close-up view: draw only the nearby track pieces, rotated with the car."""
        s, c, _, _ = self._cam
        radius = math.hypot(*self.rect.size) / 2 / s + 2      # metres visible from the centre

        # grass stripes fixed to the ground: they make speed visible
        k0 = math.floor((c[0] - radius) / GRASS_STRIPE)
        k1 = math.ceil((c[0] + radius) / GRASS_STRIPE)
        y0, y1 = c[1] - radius, c[1] + radius
        for k in range(k0, k1):
            if k % 2:
                x0, x1 = k * GRASS_STRIPE, (k + 1) * GRASS_STRIPE
                pygame.draw.polygon(surf, GRASS_B, self.w2s_many([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))

        t = self.track
        near = ((t.center - c) ** 2).sum(axis=1) < (radius + t.width + 10) ** 2
        self._draw_shapes(surf, self.w2s_many, near, max(1, int(round(0.25 * s))))

    # ======================================================= car drawing
    def draw_trail(self, surf, trail, width):
        """trail: list of (position, km/h); drawn coloured by speed."""
        if len(trail) > 1:
            pts = self.w2s_many([p for p, _ in trail])
            for k in range(1, len(pts)):
                pygame.draw.line(surf, speed_color(trail[k][1]), pts[k - 1], pts[k], width)

    def draw_rays(self, surf, pos, dirs, dists, max_range):
        origin = self.w2s(pos)
        ends = self.w2s_many(pos + dirs * dists[:, None])
        for end, dist in zip(ends, dists):
            col = lerp_color(BAD, GOOD, dist / max_range * 2.5)
            pygame.draw.line(surf, col, origin, end, 1)
            pygame.draw.circle(surf, col, end, 3)

    def draw_car(self, surf, x, y, heading, color=CAR_BODY, scale=None, glass=True):
        scale = self.car_scale if scale is None else scale
        corners = car_corners(x, y, heading, 4.5 * scale, 2.0 * scale)
        pygame.draw.polygon(surf, color, self.w2s_many(corners))
        if glass:
            # windscreen: shows where the nose is
            fl, fr, rr, rl = corners
            g = [fl + (rl - fl) * 0.18, fr + (rr - fr) * 0.18, fr + (rr - fr) * 0.38, fl + (rl - fl) * 0.38]
            g = [p + (fr - fl) * (0.12 if i in (0, 3) else -0.12) for i, p in enumerate(g)]
            pygame.draw.polygon(surf, CAR_GLASS, self.w2s_many(g))
