"""
Pygame renderer: top-down view of the circuit.

It is only an observer: it reads the environment state and draws it.
It never changes the simulation.

Shows:
  - the track with kerbs and finish line
  - the car and its sensor rays (green = clear, red = wall close)
  - the racing line of the current and previous lap,
    coloured by speed (red = slow, green = fast)
  - a panel with speed, lap times, progress and controls

Keys handled here: C = switch view (the close-up view rotates with the car),
V = sensors on/off, R = restart, ESC = quit.
"""
import math

import numpy as np
import pygame

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
PANEL_BG = (18, 20, 26)
TEXT = (230, 232, 238)
TEXT_DIM = (140, 146, 160)
ACCENT = (90, 180, 255)
WARN = (255, 170, 40)
BAD = (255, 80, 70)
GOOD = (90, 220, 120)

PANEL_W = 300
OVERVIEW_CAR_SCALE = 2.5     # in the overview the car is enlarged so you can see it
FOLLOW_SCALE = 6.0           # pixels per metre in the view that follows the car
GRASS_STRIPE = 20.0          # metres, width of the grass stripes


def fmt_time(t):
    if t is None:
        return "--:--.--"
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:05.2f}"


def lerp_color(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def speed_color(kmh, slow=40.0, fast=200.0):
    t = (kmh - slow) / (fast - slow)
    if t < 0.5:
        return lerp_color(BAD, WARN, t * 2)
    return lerp_color(WARN, GOOD, (t - 0.5) * 2)


class Renderer:
    def __init__(self, env, size=(1280, 820), title="MTAS - Car on a circuit"):
        pygame.init()
        self.env = env
        self.track = env.track
        self.W, self.H = size
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        mono = "consolas,menlo,dejavusansmono,couriernew,monospace"
        self.font = pygame.font.SysFont(mono, 17)
        self.font_small = pygame.font.SysFont(mono, 14)
        self.font_mid = pygame.font.SysFont(mono, 24, bold=True)
        self.font_big = pygame.font.SysFont(mono, 54, bold=True)

        self.show_rays = True
        self.follow = False
        self.reset_requested = False

        self.trail, self.prev_trail = [], []
        self.best_lap = None
        self._last_time = 0.0
        self._last_laps = 0
        self._flash_text, self._flash_until = None, 0

        self._build_geometry()

        # overview: the whole circuit in the space right of the panel
        lo, hi = self.track.bounds()
        self.pad = 20.0                                    # metres of grass around it
        span = (hi - lo) + 2 * self.pad
        avail_w, avail_h = self.W - PANEL_W, self.H
        self.overview_scale = min(avail_w / span[0], avail_h / span[1])
        self.overview_center = (lo + hi) / 2
        self.bg_origin = np.array([lo[0] - self.pad, hi[1] + self.pad])  # top-left corner
        self.bg_overview = self._render_overview(span)

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

    # ================================================================ view
    def _view(self):
        """Return (scale, world centre, screen centre, rotation angle).

        In the view that follows the car the world rotates so that the nose
        always points to the top of the screen.
        """
        sc = (PANEL_W + (self.W - PANEL_W) / 2, self.H / 2)
        if self.follow:
            return FOLLOW_SCALE, self.env.car.pos.copy(), sc, math.pi / 2 - self.env.car.heading
        return self.overview_scale, self.overview_center, sc, 0.0

    def w2s(self, p):
        return self.w2s_many([p])[0]

    def w2s_many(self, pts):
        s, c, sc, a = self._view_cache
        pts = np.asarray(pts, dtype=float)
        dx, dy = pts[:, 0] - c[0], pts[:, 1] - c[1]
        if a:
            ca, sa = math.cos(a), math.sin(a)
            dx, dy = ca * dx - sa * dy, sa * dx + ca * dy
        x = sc[0] + dx * s
        y = sc[1] - dy * s
        return list(zip(x.tolist(), y.tolist()))

    # ====================================================== track drawing
    def _render_overview(self, span):
        """Draw the whole track once onto a surface (fast to copy every frame)."""
        scale = self.overview_scale
        w, h = int(span[0] * scale) + 2, int(span[1] * scale) + 2
        surf = pygame.Surface((w, h))
        surf.fill(GRASS_A)
        stripe = max(8, int(GRASS_STRIPE * scale))
        for i in range(0, w, stripe * 2):
            pygame.draw.rect(surf, GRASS_B, (i, 0, stripe, h))

        o = self.bg_origin

        def to_bitmap(p):
            return np.stack([(p[:, 0] - o[0]) * scale, (o[1] - p[:, 1]) * scale], axis=1)

        self._draw_shapes(surf, to_bitmap, np.ones(self.track.N, bool), max(1, int(0.25 * scale)))
        return surf

    def _draw_follow_track(self):
        """Close-up view: draw only the nearby track pieces, rotated with the car."""
        s, c, _, _ = self._view_cache
        radius = math.hypot(self.W - PANEL_W, self.H) / 2 / s + 2      # metres visible from the centre

        # grass stripes fixed to the ground: they make speed visible
        k0 = math.floor((c[0] - radius) / GRASS_STRIPE)
        k1 = math.ceil((c[0] + radius) / GRASS_STRIPE)
        y0, y1 = c[1] - radius, c[1] + radius
        for k in range(k0, k1):
            if k % 2:
                x0, x1 = k * GRASS_STRIPE, (k + 1) * GRASS_STRIPE
                pygame.draw.polygon(self.screen, GRASS_B, self.w2s_many([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))

        t = self.track
        near = ((t.center - c) ** 2).sum(axis=1) < (radius + t.width + 10) ** 2
        self._draw_shapes(self.screen, self.w2s_many, near, max(1, int(round(0.25 * s))))

    # ============================================================== events
    def handle_events(self):
        """Return False if the user wants to close the window."""
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    return False
                if e.key == pygame.K_c:
                    self.follow = not self.follow
                elif e.key == pygame.K_v:
                    self.show_rays = not self.show_rays
                elif e.key == pygame.K_r:
                    self.reset_requested = True
        return True

    def tick(self, fps=60):
        return self.clock.tick(fps)

    def close(self):
        pygame.quit()

    # ============================================================ drawing
    def _update_history(self):
        env = self.env
        if env.time < self._last_time:            # the environment was reset
            self.trail = []
            self._last_laps = 0
        self._last_time = env.time

        if env.laps > self._last_laps:            # lap completed
            self._last_laps = env.laps
            lap = env.last_lap_time
            record = self.best_lap is None or lap < self.best_lap
            if record:
                self.best_lap = lap
            self.prev_trail = self.trail
            self.trail = []
            self._flash_text = ("RECORD! " if record else "LAP ") + fmt_time(lap)
            self._flash_until = pygame.time.get_ticks() + 2500

        pos = env.car.pos
        if not self.trail or np.hypot(*(pos - self.trail[-1][0])) > 1.5:
            self.trail.append((pos.copy(), env.car.speed * 3.6))

    def draw(self, action=None, extra_lines=None):
        self._update_history()
        self._view_cache = self._view()
        env, car = self.env, self.env.car

        self.screen.fill(GRASS_A)
        if self.follow:
            self._draw_follow_track()
        else:
            self.screen.blit(self.bg_overview, self.w2s(self.bg_origin))

        s = self._view_cache[0]
        width = max(1, int(round(0.8 * s))) if self.follow else 2
        for trail, w in ((self.prev_trail, max(1, width - 1)), (self.trail, width)):
            if len(trail) > 1:
                pts = self.w2s_many([p for p, _ in trail])
                for k in range(1, len(pts)):
                    pygame.draw.line(self.screen, speed_color(trail[k][1]), pts[k - 1], pts[k], w)

        if self.show_rays:
            origin = self.w2s(car.pos)
            for d, dist in zip(env.ray_dirs, env.ray_dist):
                end = self.w2s(car.pos + d * dist)
                col = lerp_color(BAD, GOOD, dist / env.max_range * 2.5)
                pygame.draw.line(self.screen, col, origin, end, 1)
                pygame.draw.circle(self.screen, col, end, 3)

        self._draw_car()
        pygame.draw.rect(self.screen, PANEL_BG, (0, 0, PANEL_W, self.H))
        self._draw_panel(action, extra_lines)
        self._draw_overlays()
        pygame.display.flip()

    def _draw_car(self):
        car = self.env.car
        k = 1.0 if self.follow else OVERVIEW_CAR_SCALE
        corners = car.pos + (car.corners() - car.pos) * k
        body = CAR_CRASH if self.env.crashed else CAR_BODY
        pygame.draw.polygon(self.screen, body, self.w2s_many(corners))
        # windscreen: shows where the nose is
        fl, fr, rr, rl = corners
        glass = [fl + (rl - fl) * 0.18, fr + (rr - fr) * 0.18, fr + (rr - fr) * 0.38, fl + (rl - fl) * 0.38]
        glass = [g + (fr - fl) * (0.12 if i in (0, 3) else -0.12) for i, g in enumerate(glass)]
        pygame.draw.polygon(self.screen, CAR_GLASS, self.w2s_many(glass))

    # ---------------------------------------------------------------- panel
    def _text(self, txt, x, y, font=None, color=TEXT):
        surf = (font or self.font).render(txt, True, color)
        self.screen.blit(surf, (x, y))
        return surf.get_height()

    def _bar(self, x, y, w, h, frac, color, centered=False):
        pygame.draw.rect(self.screen, (45, 48, 58), (x, y, w, h))
        frac = max(-1.0, min(1.0, frac))
        if centered:
            mid = x + w / 2
            fw = abs(frac) * w / 2
            pygame.draw.rect(self.screen, color, (mid if frac >= 0 else mid - fw, y, fw, h))
            pygame.draw.line(self.screen, TEXT_DIM, (mid, y - 2), (mid, y + h + 2))
        else:
            pygame.draw.rect(self.screen, color, (x, y, max(0.0, frac) * w, h))

    def _draw_panel(self, action, extra_lines):
        env, car = self.env, self.env.car
        x, y = 20, 18
        y += self._text("MTAS", x, y, self.font_mid, ACCENT) + 2
        y += self._text(f"{self.track.name}  {self.track.length / 1000:.2f} km", x, y, self.font_small, TEXT_DIM) + 18

        kmh = car.speed * 3.6
        self._text(f"{kmh:5.0f}", x - 6, y, self.font_big, speed_color(kmh))
        self._text("km/h", x + 175, y + 32, self.font, TEXT_DIM)
        y += 70
        self._bar(x, y, PANEL_W - 40, 8, car.speed / car.p.max_speed, speed_color(kmh))
        y += 26

        rows = [
            ("Lap", f"{env.laps + 1}"),
            ("Lap time", fmt_time(env.time - env.lap_start_time)),
            ("Last lap", fmt_time(env.last_lap_time)),
            ("Best lap", fmt_time(self.best_lap)),
        ]
        for label, val in rows:
            self._text(label, x, y, self.font, TEXT_DIM)
            self._text(val, x + 150, y, self.font, TEXT)
            y += 24

        y += 8
        lap_frac = (env.progress / self.track.length) - env.laps
        self._text(f"Lap progress    {max(0.0, lap_frac):5.1%}", x, y, self.font, TEXT_DIM)
        y += 22
        self._bar(x, y, PANEL_W - 40, 8, lap_frac, ACCENT)
        y += 24

        # current controls
        if action is not None:
            steer, thr = float(action[0]), float(action[1])
            self._text("Throttle / brake", x, y, self.font_small, TEXT_DIM)
            y += 18
            self._bar(x, y, PANEL_W - 40, 10, thr, GOOD if thr >= 0 else BAD, centered=True)
            y += 22
            self._text("Steering  (L  <->  R)", x, y, self.font_small, TEXT_DIM)
            y += 18
            self._bar(x, y, PANEL_W - 40, 10, -car.steer / car.p.max_steer, ACCENT, centered=True)
            y += 24

        if car.sliding:
            y += self._text("TYRES AT THE LIMIT!", x, y, self.font, WARN) + 4
        if env.crashed:
            y += self._text("CRASH", x, y, self.font, BAD) + 4

        for line in extra_lines or []:
            y += self._text(line, x, y, self.font_small, TEXT) + 4

        # key help at the bottom
        help_lines = ["Arrows / WASD  drive", "R  restart", "C  switch view",
                      "V  sensors on/off", "ESC  quit"]
        yy = self.H - 20 - len(help_lines) * 19
        for line in help_lines:
            self._text(line, x, yy, self.font_small, TEXT_DIM)
            yy += 19

    def _draw_overlays(self):
        cx = PANEL_W + (self.W - PANEL_W) / 2
        if self.env.crashed:
            self._center_text("CRASH", cx, self.H / 2 - 30, BAD)
            self._center_text("press R to restart", cx, self.H / 2 + 30, TEXT, self.font)
        elif self._flash_text and pygame.time.get_ticks() < self._flash_until:
            self._center_text(self._flash_text, cx, 70, GOOD)

    def _center_text(self, txt, cx, cy, color, font=None):
        surf = (font or self.font_big).render(txt, True, color)
        box = pygame.Surface((surf.get_width() + 40, surf.get_height() + 16), pygame.SRCALPHA)
        box.fill((0, 0, 0, 150))
        self.screen.blit(box, (cx - box.get_width() / 2, cy - box.get_height() / 2))
        self.screen.blit(surf, (cx - surf.get_width() / 2, cy - surf.get_height() / 2))
