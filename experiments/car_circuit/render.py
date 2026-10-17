"""
Renderer pygame: vista dall'alto del circuito.

È solo un osservatore: legge lo stato dell'ambiente e lo disegna.
Non modifica mai la simulazione.

Mostra:
  - pista con cordoli e traguardo
  - macchina e raggi dei sensori (verde = libero, rosso = muro vicino)
  - la traiettoria del giro attuale e di quello precedente,
    colorata per velocità (rosso = lento, verde = veloce)
  - pannello con velocità, tempi sul giro, progresso, comandi

Tasti gestiti qui: C = cambia vista, V = mostra/nascondi sensori,
R = riparti, ESC = esci.
"""
import math

import numpy as np
import pygame

# ---------------------------------------------------------------- colori
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
OVERVIEW_CAR_SCALE = 2.5     # nella vista d'insieme la macchina è ingrandita per vederla
FOLLOW_SCALE = 6.0           # pixel per metro nella vista che segue la macchina


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
    def __init__(self, env, size=(1280, 820), title="MTAS - Auto su circuito"):
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

        # vista d'insieme: tutto il circuito nello spazio a destra del pannello
        lo, hi = self.track.bounds()
        self.pad = 20.0                                    # metri di erba attorno
        span = (hi - lo) + 2 * self.pad
        avail_w, avail_h = self.W - PANEL_W, self.H
        self.overview_scale = min(avail_w / span[0], avail_h / span[1])
        self.overview_center = (lo + hi) / 2
        self.bg_origin = np.array([lo[0] - self.pad, hi[1] + self.pad])  # angolo alto-sinistra
        self.bg_overview = self._render_track(self.overview_scale, span)
        self.bg_follow = self._render_track(FOLLOW_SCALE, span)

    # ================================================================ vista
    def _view(self):
        if self.follow:
            return FOLLOW_SCALE, self.env.car.pos, (PANEL_W + (self.W - PANEL_W) / 2, self.H / 2)
        return self.overview_scale, self.overview_center, (PANEL_W + (self.W - PANEL_W) / 2, self.H / 2)

    def w2s(self, p):
        s, c, sc = self._view_cache
        return (sc[0] + (p[0] - c[0]) * s, sc[1] - (p[1] - c[1]) * s)

    def w2s_many(self, pts):
        s, c, sc = self._view_cache
        pts = np.asarray(pts)
        x = sc[0] + (pts[:, 0] - c[0]) * s
        y = sc[1] - (pts[:, 1] - c[1]) * s
        return list(zip(x.tolist(), y.tolist()))

    # ======================================================== sfondo pista
    def _render_track(self, scale, span):
        """Disegna la pista una volta sola su una superficie (veloce da copiare ogni frame)."""
        t = self.track
        w, h = int(span[0] * scale) + 2, int(span[1] * scale) + 2
        surf = pygame.Surface((w, h))
        surf.fill(GRASS_A)
        stripe = max(8, int(20 * scale))
        for i in range(0, w, stripe * 2):
            pygame.draw.rect(surf, GRASS_B, (i, 0, stripe, h))

        def m(p):
            return ((p[0] - self.bg_origin[0]) * scale, (self.bg_origin[1] - p[1]) * scale)

        L, R, N = t.left, t.right, t.N
        for i in range(N):
            j = (i + 1) % N
            pygame.draw.polygon(surf, ASPHALT, [m(L[i]), m(L[j]), m(R[j]), m(R[i])])

        # linea centrale tratteggiata (solo riferimento visivo)
        for i in range(0, N, 6):
            j = (i + 3) % N
            pygame.draw.line(surf, ASPHALT_LINE, m(t.center[i]), m(t.center[j]), max(1, int(0.25 * scale)))

        # cordoli bianchi/rossi lungo i bordi (larghi 1 m, verso l'interno)
        kerb = 1.0
        Li, Ri = L - t.normal * kerb, R + t.normal * kerb
        for i in range(N):
            j = (i + 1) % N
            col = KERB_RED if (i // 3) % 2 == 0 else KERB_WHITE
            pygame.draw.polygon(surf, col, [m(L[i]), m(L[j]), m(Li[j]), m(Li[i])])
            pygame.draw.polygon(surf, col, [m(R[i]), m(R[j]), m(Ri[j]), m(Ri[i])])

        # traguardo a scacchi
        c0, n0, d0 = t.center[0], t.normal[0], t.tangent[0]
        cells = 8
        size = t.width / cells
        for row in range(2):
            for k in range(cells):
                base = c0 - n0 * t.half_width + n0 * (k * size) + d0 * (row * size)
                quad = [base, base + n0 * size, base + n0 * size + d0 * size, base + d0 * size]
                col = (240, 240, 240) if (k + row) % 2 == 0 else (25, 25, 25)
                pygame.draw.polygon(surf, col, [m(q) for q in quad])
        return surf

    # ============================================================== eventi
    def handle_events(self):
        """Ritorna False se l'utente vuole chiudere la finestra."""
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

    # ============================================================ disegno
    def _update_history(self):
        env = self.env
        if env.time < self._last_time:            # l'ambiente è stato resettato
            self.trail = []
            self._last_laps = 0
        self._last_time = env.time

        if env.laps > self._last_laps:            # giro completato
            self._last_laps = env.laps
            lap = env.last_lap_time
            record = self.best_lap is None or lap < self.best_lap
            if record:
                self.best_lap = lap
            self.prev_trail = self.trail
            self.trail = []
            self._flash_text = ("RECORD! " if record else "GIRO ") + fmt_time(lap)
            self._flash_until = pygame.time.get_ticks() + 2500

        pos = env.car.pos
        if not self.trail or np.hypot(*(pos - self.trail[-1][0])) > 1.5:
            self.trail.append((pos.copy(), env.car.speed * 3.6))

    def draw(self, action=None, extra_lines=None):
        self._update_history()
        self._view_cache = self._view()
        env, car = self.env, self.env.car

        self.screen.fill(GRASS_A)
        bg = self.bg_follow if self.follow else self.bg_overview
        self.screen.blit(bg, self.w2s(self.bg_origin))

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
        # parabrezza: indica il muso
        fl, fr, rr, rl = corners
        glass = [fl + (rl - fl) * 0.18, fr + (rr - fr) * 0.18, fr + (rr - fr) * 0.38, fl + (rl - fl) * 0.38]
        glass = [g + (fr - fl) * (0.12 if i in (0, 3) else -0.12) for i, g in enumerate(glass)]
        pygame.draw.polygon(self.screen, CAR_GLASS, self.w2s_many(glass))

    # -------------------------------------------------------------- pannello
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
            ("Giro", f"{env.laps + 1}"),
            ("Tempo giro", fmt_time(env.time - env.lap_start_time)),
            ("Ultimo giro", fmt_time(env.last_lap_time)),
            ("Miglior giro", fmt_time(self.best_lap)),
        ]
        for label, val in rows:
            self._text(label, x, y, self.font, TEXT_DIM)
            self._text(val, x + 150, y, self.font, TEXT)
            y += 24

        y += 8
        lap_frac = (env.progress / self.track.length) - env.laps
        self._text(f"Progresso giro  {max(0.0, lap_frac):5.1%}", x, y, self.font, TEXT_DIM)
        y += 22
        self._bar(x, y, PANEL_W - 40, 8, lap_frac, ACCENT)
        y += 24

        # comandi attuali
        if action is not None:
            steer, thr = float(action[0]), float(action[1])
            self._text("Acceleratore / freno", x, y, self.font_small, TEXT_DIM)
            y += 18
            self._bar(x, y, PANEL_W - 40, 10, thr, GOOD if thr >= 0 else BAD, centered=True)
            y += 22
            self._text("Sterzo  (sx  <->  dx)", x, y, self.font_small, TEXT_DIM)
            y += 18
            self._bar(x, y, PANEL_W - 40, 10, -car.steer / car.p.max_steer, ACCENT, centered=True)
            y += 24

        if car.sliding:
            y += self._text("GOMME AL LIMITE!", x, y, self.font, WARN) + 4
        if env.crashed:
            y += self._text("INCIDENTE", x, y, self.font, BAD) + 4

        for line in extra_lines or []:
            y += self._text(line, x, y, self.font_small, TEXT) + 4

        # legenda comandi in basso
        help_lines = ["Frecce / WASD  guida", "R  riparti", "C  cambia vista",
                      "V  sensori on/off", "ESC  esci"]
        yy = self.H - 20 - len(help_lines) * 19
        for line in help_lines:
            self._text(line, x, yy, self.font_small, TEXT_DIM)
            yy += 19

    def _draw_overlays(self):
        cx = PANEL_W + (self.W - PANEL_W) / 2
        if self.env.crashed:
            self._center_text("INCIDENTE", cx, self.H / 2 - 30, BAD)
            self._center_text("premi R per ripartire", cx, self.H / 2 + 30, TEXT, self.font)
        elif self._flash_text and pygame.time.get_ticks() < self._flash_until:
            self._center_text(self._flash_text, cx, 70, GOOD)

    def _center_text(self, txt, cx, cy, color, font=None):
        surf = (font or self.font_big).render(txt, True, color)
        box = pygame.Surface((surf.get_width() + 40, surf.get_height() + 16), pygame.SRCALPHA)
        box.fill((0, 0, 0, 150))
        self.screen.blit(box, (cx - box.get_width() / 2, cy - box.get_height() / 2))
        self.screen.blit(surf, (cx - surf.get_width() / 2, cy - surf.get_height() / 2))
