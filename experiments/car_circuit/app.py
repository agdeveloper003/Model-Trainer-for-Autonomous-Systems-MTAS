"""
MTAS app: one window, three screens.

  Setup     choose the circuit (classic or random from a seed), then
            PLAY it yourself or start NEUROEVOLUTION; a trained model can
            also be run on whatever track is set up here.
  Drive     you (keyboard) or a trained model drive the circuit.
  Training  every car of the population drives live on the map, with the
            leader's neural network, statistics and a progress chart.

    python app.py
"""
import os
import random
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from env import CarCircuitEnv
from netviz import NetworkView, draw_chart, draw_chart_general
from neuroevo import Net, list_models, load_model, model_path
from render import CAR_BODY, CAR_CRASH, FOLLOW_SCALE, TrackView, fmt_time, speed_color
from track import TrackSpec
from trainer import CRASHED, DONE, HEADING, LAP_TIME, PROGRESS, SPEED, X, Y, LiveTrainer
from ui import (ACCENT, BAD, BG, BORDER, GOOD, PANEL, TEXT, TEXT_DIM, WARN, Button, Cycler,
                NumberField, Segmented, Slider, bar, card, fonts, text, wrap)

TITLE = "MTAS - Model Trainer for Autonomous Systems"
MARGIN = 20


class Screen:
    """Base class: keeps a list of widgets and re-does the layout when the window is resized."""

    def __init__(self, app):
        self.app = app
        self.widgets = []
        self._size = None

    def handle(self, e):
        for w in self.widgets:
            if w.handle(e):
                return
        self.on_event(e)

    def on_event(self, e):
        pass

    def update(self, dt):
        pass

    def draw(self, surf):
        size = surf.get_size()
        if size != self._size:
            self._size = size
            self.layout(*size)
        surf.fill(BG)
        self.render(surf)
        for w in self.widgets:
            w.draw(surf)
        self.overlay(surf)

    def layout(self, w, h):
        pass

    def render(self, surf):
        pass

    def overlay(self, surf):
        pass

    def close(self):
        pass


def center_text(surf, txt, cx, cy, color, font=None):
    img = (font or fonts().big).render(txt, True, color)
    box = pygame.Surface((img.get_width() + 40, img.get_height() + 16), pygame.SRCALPHA)
    box.fill((0, 0, 0, 160))
    surf.blit(box, (cx - box.get_width() / 2, cy - box.get_height() / 2))
    surf.blit(img, (cx - img.get_width() / 2, cy - img.get_height() / 2))


def model_summary(m):
    """Two lines describing a saved model."""
    if m["general"]:
        done, spd = m["val_completed"], m["val_speed"]
        val = (f"unseen circuits completed {done:.0%}" + (f" · {spd:.0f} km/h avg" if spd else "")
               if done is not None else "not validated yet")
        return [f"{m['track_name']} · {m['generations']} generations", val]
    lap = fmt_time(m["lap_time"]) if m["lap_time"] > 0 else "never finished a lap"
    km = f" ({m['track_length'] / 1000:.2f} km)" if m["track_length"] else ""
    return [f"Trained on {m['track_name']}{km}", f"{m['generations']} generations · best lap {lap}"]


# ==========================================================================
# Setup
# ==========================================================================
class SetupScreen(Screen):
    def __init__(self, app):
        super().__init__(app)
        s = app.spec
        lim = TrackSpec.LIMITS
        self.kind = Segmented([("random", "Random circuit"), ("classic", "Classic")], s.kind, self._changed)
        self.seed = NumberField("SEED", s.seed, on_change=lambda _v: self._changed())
        self.dice = Button("New seed", self._new_seed)
        self.corners = Slider("Corners", *lim["corners"], 1, s.corners, "{:.0f}", self._changed)
        self.length = Slider("Length", *lim["length"], 100, s.length, "{:.0f} m", self._changed)
        self.width = Slider("Width", *lim["width"], 1, s.width, "{:.0f} m", self._changed)
        self.twist = Slider("Twistiness", *lim["twistiness"], 0.05, s.twistiness, "{:.2f}", self._changed)
        self.direction = Segmented([("ccw", "Counter-clockwise"), ("cw", "Clockwise")],
                                   "cw" if s.clockwise else "ccw", self._changed)
        self.random_widgets = [self.seed, self.dice, self.corners, self.length, self.width, self.twist]

        self.btn_play = Button("PLAY", self._play, "big", "drive this circuit yourself")
        self.btn_train = Button("NEUROEVOLUTION", self._train, "big", "")
        self.mode = Segmented([("track", "This track"), ("general", "Generalist")], app.train_mode,
                              self._mode_changed)
        self.grid = Segmented([("3", "3x3 · 6"), ("4", "4x4 · 8"), ("5", "5x5 · 10")], str(app.grid_n),
                              self._mode_changed)
        self.gens = Slider("Generations", 10, 1000, 10, app.generations, "{:.0f}")
        self.pop = Slider("Population", 16, 480, 16, app.population, "{:.0f} cars")

        self.models = list_models()
        idx = next((i for i, m in enumerate(self.models) if m["path"] == app.model_path), 0)
        self.model = Cycler([(m["path"], self._model_label(m)) for m in self.models], idx, self._pick_model)
        self.btn_model = Button("Run this model on the track", self._run_model, "primary")

        self.widgets = [self.kind, self.seed, self.dice, self.corners, self.length, self.width, self.twist,
                        self.direction, self.btn_play, self.btn_train, self.mode, self.grid, self.gens,
                        self.pop, self.model, self.btn_model]
        self.track, self.error, self.view = None, None, None
        self._rebuild_at = 0.0
        self._rebuild()

    @staticmethod
    def _model_label(m):
        if m["general"]:
            done = m["val_completed"]
            return f"Generalist  {done:.0%} unseen" if done is not None else "Generalist"
        lap = fmt_time(m["lap_time"]) if m["lap_time"] > 0 else "no full lap"
        return f"{m['track_name']}  {lap}"

    def _mode_changed(self, *_):
        self.app.train_mode, self.app.grid_n = self.mode.value, int(self.grid.value)

    # ---------------------------------------------------------- settings
    def _changed(self, *_):
        self._rebuild_at = time.perf_counter() + 0.2          # rebuild shortly after the last change

    def _new_seed(self):
        self.seed.set(random.randint(0, 999_999))
        self._changed()

    def _read_spec(self):
        cw = self.direction.value == "cw"
        if self.kind.value == "classic":
            return TrackSpec(kind="classic", clockwise=cw)
        return TrackSpec(kind="random", seed=int(self.seed.value or 0), corners=int(self.corners.value),
                         length=float(self.length.value), width=float(self.width.value),
                         twistiness=round(self.twist.value, 2), clockwise=cw).clamped()

    def _rebuild(self):
        self._rebuild_at = 0.0
        spec = self._read_spec()
        self.app.spec = spec
        try:
            self.track = self.app.track_for(spec)
            self.error = None
            self.view = TrackView(self.track)
            self._size = None                                  # re-layout places the new view
        except ValueError as e:
            self.track, self.view, self.error = None, None, str(e)

    def _pick_model(self, path):
        self.app.model_path = path
        self.app.hint_test = False

    def update(self, dt):
        if self._rebuild_at and time.perf_counter() >= self._rebuild_at:
            self._rebuild()
        for w in self.random_widgets:
            w.enabled = self.kind.value == "random"
        ok = self.track is not None
        general = self.mode.value == "general"
        self.grid.visible = general
        self.btn_play.enabled = ok
        self.btn_train.enabled = ok or general
        self.btn_train.sub = (f"generalist: {2 * int(self.grid.value)} new random tracks per generation"
                              if general else "train neural networks on this track, live")
        self.btn_model.enabled = ok and bool(self.models)
        self.model.enabled = bool(self.models)
        self.app.generations, self.app.population = int(self.gens.value), int(self.pop.value)

    # ----------------------------------------------------------- actions
    def _play(self):
        self.app.go(DriveScreen(self.app, self.app.spec, self.track))

    def _train(self):
        if self.mode.value == "general":
            self.app.go(TrainScreen(self.app, None, None, n_tracks=2 * int(self.grid.value)))
        else:
            self.app.go(TrainScreen(self.app, self.app.spec, self.track))

    def _run_model(self):
        model = next((m for m in self.models if m["path"] == self.model.value), None)
        if model:
            self.app.model_path = model["path"]
            self.app.go(DriveScreen(self.app, self.app.spec, self.track, model))

    def on_event(self, e):
        if e.type == pygame.KEYDOWN and e.key == pygame.K_RETURN and self.track is not None:
            self._play()

    # ------------------------------------------------------------ layout
    def layout(self, w, h):
        top = 92
        self.r_left = pygame.Rect(MARGIN, top, 360, h - top - MARGIN)
        self.r_right = pygame.Rect(w - MARGIN - 380, top, 380, h - top - MARGIN)
        self.r_preview = pygame.Rect(self.r_left.right + MARGIN, top,
                                     self.r_right.x - self.r_left.right - 2 * MARGIN, h - top - MARGIN)

        x, cw = self.r_left.x + 16, self.r_left.w - 32
        y = self.r_left.y + 40
        self.kind.rect = pygame.Rect(x, y, cw, 40)
        y += 56
        self.seed.rect = pygame.Rect(x, y, cw - 130, 40)
        self.dice.rect = pygame.Rect(x + cw - 120, y, 120, 40)
        y += 60
        for s in (self.corners, self.length, self.width, self.twist):
            s.rect = pygame.Rect(x, y, cw, 42)
            y += 60
        self.direction.rect = pygame.Rect(x, y, cw, 36)
        y += 50
        self.info_y = y + 4

        x, cw = self.r_right.x, self.r_right.w
        y = self.r_right.y
        self.btn_play.rect = pygame.Rect(x, y, cw, 96)
        y += 108
        self.btn_train.rect = pygame.Rect(x, y, cw, 96)
        y += 110
        self.r_evo = pygame.Rect(x, y, cw, 246)
        self.mode.rect = pygame.Rect(x + 16, y + 38, cw - 32, 36)
        self.grid.rect = pygame.Rect(x + 16, y + 82, cw - 32, 36)
        self.gens.rect = pygame.Rect(x + 16, y + 132, cw - 32, 42)
        self.pop.rect = pygame.Rect(x + 16, y + 188, cw - 32, 42)
        y += 260
        self.r_models = pygame.Rect(x, y, cw, max(150, self.r_right.bottom - y))
        self.model.rect = pygame.Rect(x + 16, y + 40, cw - 32, 38)
        self.btn_model.rect = pygame.Rect(x + 16, self.r_models.bottom - 54, cw - 32, 40)

        if self.view:
            self.view.set_rect(self.r_preview.inflate(-2, -2))

    def render(self, surf):
        f = fonts()
        w = surf.get_width()
        text(surf, "MTAS", (MARGIN, 22), f.title, ACCENT)
        text(surf, "Model Trainer for Autonomous Systems", (MARGIN + 110, 36), f.normal, TEXT_DIM)
        text(surf, "set up the circuit, then choose what to do", (w - MARGIN, 36), f.small, TEXT_DIM, "topright")

        card(surf, self.r_left, "Track")
        y = self.info_y
        if self.track is not None:
            t = self.track
            for line in (t.name, f"{t.length / 1000:.2f} km  ·  {t.width:g} m wide",
                         f"tightest corner radius {t.min_curve_radius():.0f} m"):
                text(surf, line, (self.r_left.x + 16, y), f.normal, TEXT if line == t.name else TEXT_DIM)
                y += 22
            if self.kind.value == "random":
                y += 8
                for line in wrap("Same settings + same seed = same circuit, every time.", f.small,
                                 self.r_left.w - 32):
                    text(surf, line, (self.r_left.x + 16, y), f.small, TEXT_DIM)
                    y += 18

        # preview
        pygame.draw.rect(surf, BORDER, self.r_preview, 1, border_radius=10)
        if self.view is not None:
            self.view.begin(surf)
            self._draw_direction(surf)
            self.view.end(surf)
            text(surf, f"{self.track.name}  ·  {self.track.length / 1000:.2f} km",
                 (self.r_preview.x + 14, self.r_preview.y + 12), f.mid, TEXT)
        else:
            pygame.draw.rect(surf, PANEL, self.r_preview.inflate(-2, -2), border_radius=10)
            yy = self.r_preview.centery - 30
            for line in wrap(self.error or "", f.normal, self.r_preview.w - 80):
                text(surf, line, (self.r_preview.centerx, yy), f.normal, BAD, "center")
                yy += 24

        card(surf, self.r_evo, "Neuroevolution settings")
        card(surf, self.r_models, "Trained models")
        x, y = self.r_models.x + 16, self.r_models.y + 92
        cw = self.r_models.w - 32
        if not self.models:
            lines = ["No trained models yet.", "Train one with NEUROEVOLUTION:",
                     "it can then drive any other track."]
        else:
            m = next(m for m in self.models if m["path"] == self.model.value)
            lines = model_summary(m)
            if self.app.hint_test:
                lines += ["", "Set up a different circuit on the left,", "then run the model on it."]
        for line in lines:
            for part in wrap(line, f.small, cw):
                if y < self.btn_model.rect.y - 16:
                    text(surf, part, (x, y), f.small, ACCENT if self.app.hint_test and line.startswith(("Set", "then"))
                         else TEXT_DIM)
                y += 18

    def _draw_direction(self, surf):
        """Arrow along the first metres of the track: the driving direction."""
        t = self.track
        k = min(t.N - 1, max(3, int(90 / (t.length / t.N))))
        pts = self.view.w2s_many(t.center[:k + 1])
        pygame.draw.lines(surf, (255, 255, 255), False, pts, 3)
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        a = np.arctan2(y1 - y0, x1 - x0)
        tip = np.array([x1, y1])
        head = [tip, tip - 14 * np.array([np.cos(a - 0.45), np.sin(a - 0.45)]),
                tip - 14 * np.array([np.cos(a + 0.45), np.sin(a + 0.45)])]
        pygame.draw.polygon(surf, (255, 255, 255), [tuple(p) for p in head])


# ==========================================================================
# Drive (you or a trained model)
# ==========================================================================
def keyboard_action():
    k = pygame.key.get_pressed()
    steer = float(k[pygame.K_LEFT] or k[pygame.K_a]) - float(k[pygame.K_RIGHT] or k[pygame.K_d])
    throttle = float(k[pygame.K_UP] or k[pygame.K_w]) - float(k[pygame.K_DOWN] or k[pygame.K_s])
    return np.array([steer, throttle])


class DriveScreen(Screen):
    PANEL_W = 300
    NET_W = 420

    def __init__(self, app, spec, track, model=None):
        super().__init__(app)
        self.spec, self.track, self.model = spec, track, model
        if model is None:
            # you drive: 60 physics steps per second
            self.env = CarCircuitEnv(track=track, dt=1 / 60, substeps=1, laps_to_finish=10_000,
                                     max_time=1e9, stall_time=1e9)
        else:
            # same time step as in training, so the model drives exactly as it learned
            self.env = CarCircuitEnv(track=track, max_time=1e9, stall_time=1e9, laps_to_finish=10_000)
            self.genome = model["genome"]
            self.params = Net().unpack(self.genome)
            self.netview = NetworkView(self.env.ray_angles)
        self.view = TrackView(track)
        self.obs = self.env.reset()
        self.action = np.zeros(2)
        self.show_rays = True
        self.acc = 0.0
        self.trail, self.prev_trail = [], []
        self.best_lap = None
        self._last_laps = 0
        self._flash, self._flash_until = None, 0
        self.btn_back = Button("< Menu  (Esc)", self._back)
        self.widgets = [self.btn_back]

    def _back(self):
        self.app.go(SetupScreen(self.app))

    def _restart(self):
        self.obs = self.env.reset()
        self.trail, self._last_laps, self.acc = [], 0, 0.0

    def on_event(self, e):
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                self._back()
            elif e.key == pygame.K_c:
                self.view.follow = not self.view.follow
            elif e.key == pygame.K_v:
                self.show_rays = not self.show_rays
            elif e.key == pygame.K_r:
                self._restart()

    def update(self, dt):
        env = self.env
        self.acc = min(self.acc + dt, 4 * env.dt)          # never more than 4 steps per frame
        while self.acc >= env.dt and not env.done:
            self.acc -= env.dt
            self.action = keyboard_action() if self.model is None else Net.act(self.params, self.obs)
            self.obs, *_ = env.step(self.action)
        if env.done:
            self.acc = 0.0

        if env.laps > self._last_laps:                     # lap completed
            self._last_laps = env.laps
            lap = env.last_lap_time
            record = self.best_lap is None or lap < self.best_lap
            if record:
                self.best_lap = lap
            self.prev_trail, self.trail = self.trail, []
            self._flash = ("RECORD! " if record else "LAP ") + fmt_time(lap)
            self._flash_until = pygame.time.get_ticks() + 2500
        pos = env.car.pos
        if not self.trail or np.hypot(*(pos - self.trail[-1][0])) > 1.5:
            self.trail.append((pos.copy(), env.car.speed * 3.6))

    def layout(self, w, h):
        right = self.NET_W if self.model is not None else 0
        self.r_map = pygame.Rect(self.PANEL_W, 0, w - self.PANEL_W - right, h)
        self.r_net = pygame.Rect(w - right, 0, right, h)
        self.view.set_rect(self.r_map)
        self.btn_back.rect = pygame.Rect(20, h - 162, self.PANEL_W - 40, 36)

    def render(self, surf):
        env, car = self.env, self.env.car
        v = self.view
        v.begin(surf, (car.x, car.y, car.heading))
        w = max(1, int(round(0.8 * FOLLOW_SCALE))) if v.follow else 2
        v.draw_trail(surf, self.prev_trail, max(1, w - 1))
        v.draw_trail(surf, self.trail, w)
        if self.show_rays:
            v.draw_rays(surf, car.pos, env.ray_dirs, env.ray_dist, env.max_range)
        v.draw_car(surf, car.x, car.y, car.heading, CAR_CRASH if env.crashed else CAR_BODY)
        v.end(surf)
        self._draw_panel(surf)
        if self.model is not None:
            self._draw_net_panel(surf)

    def _draw_panel(self, surf):
        env, car = self.env, self.env.car
        f = fonts()
        h = surf.get_height()
        pygame.draw.rect(surf, PANEL, (0, 0, self.PANEL_W, h))
        x, y = 20, 18
        y += text(surf, "MTAS", (x, y), f.mid, ACCENT).h + 2
        y += text(surf, f"{self.track.name}  {self.track.length / 1000:.2f} km", (x, y), f.small, TEXT_DIM).h + 4
        driver = "You (keyboard)" if self.model is None else "Neural network"
        y += text(surf, f"Driver: {driver}", (x, y), f.small, TEXT).h + 14

        kmh = car.speed * 3.6
        text(surf, f"{kmh:5.0f}", (x - 6, y), f.big, speed_color(kmh))
        text(surf, "km/h", (x + 175, y + 32), f.normal, TEXT_DIM)
        y += 70
        bar(surf, x, y, self.PANEL_W - 40, 8, car.speed / car.p.max_speed, speed_color(kmh))
        y += 26
        for label, val in (("Lap", f"{env.laps + 1}"), ("Lap time", fmt_time(env.time - env.lap_start_time)),
                           ("Last lap", fmt_time(env.last_lap_time)), ("Best lap", fmt_time(self.best_lap))):
            text(surf, label, (x, y), f.normal, TEXT_DIM)
            text(surf, val, (x + 150, y), f.normal, TEXT)
            y += 24
        y += 8
        lap_frac = env.progress / self.track.length - env.laps
        text(surf, f"Lap progress    {max(0.0, lap_frac):5.1%}", (x, y), f.normal, TEXT_DIM)
        y += 22
        bar(surf, x, y, self.PANEL_W - 40, 8, lap_frac, ACCENT)
        y += 24
        thr = float(self.action[1])
        text(surf, "Throttle / brake", (x, y), f.small, TEXT_DIM)
        y += 18
        bar(surf, x, y, self.PANEL_W - 40, 10, thr, GOOD if thr >= 0 else BAD, centered=True)
        y += 22
        text(surf, "Steering  (L  <->  R)", (x, y), f.small, TEXT_DIM)
        y += 18
        bar(surf, x, y, self.PANEL_W - 40, 10, -car.steer / car.p.max_steer, ACCENT, centered=True)
        y += 26
        if car.sliding:
            y += text(surf, "TYRES AT THE LIMIT!", (x, y), f.normal, WARN).h + 4
        if env.crashed:
            y += text(surf, "CRASH", (x, y), f.normal, BAD).h + 4

        help_lines = (["Arrows / WASD  drive"] if self.model is None else []) + \
                     ["R  restart", "C  switch view", "V  sensors on/off"]
        yy = h - 20 - len(help_lines) * 19
        for line in help_lines:
            text(surf, line, (x, yy), f.small, TEXT_DIM)
            yy += 19

    def _draw_net_panel(self, surf):
        f = fonts()
        r = self.r_net
        pygame.draw.rect(surf, PANEL, r)
        x, y = r.x + 20, 18
        y += text(surf, "NEURAL NETWORK", (x, y), f.mid, ACCENT).h + 4
        m = self.model
        for line in model_summary(m):
            y += text(surf, line, (x, y), f.small, TEXT_DIM).h + 2
        if not m["general"] and m["track_id"] != self.spec.track_id():
            y += text(surf, "Test on a track it has never seen", (x, y), f.small, WARN).h + 2
        self.netview.draw(surf, pygame.Rect(r.x + 10, y + 16, r.w - 20, r.h - y - 40), self.genome, self.obs)

    def overlay(self, surf):
        cx = self.r_map.centerx
        if self.env.crashed:
            center_text(surf, "CRASH", cx, surf.get_height() / 2 - 30, BAD)
            center_text(surf, "press R to restart", cx, surf.get_height() / 2 + 30, TEXT, fonts().normal)
        elif self._flash and pygame.time.get_ticks() < self._flash_until:
            center_text(surf, self._flash, cx, 70, GOOD)


# ==========================================================================
# Training (live neuroevolution)
# ==========================================================================
class TrainScreen(Screen):
    PANEL_W = 440                  # wide enough for the progress chart
    SIDE_W = 460
    STEPS_PER_SECOND = 30          # simulation steps in one simulated second (dt = 1/30)

    def __init__(self, app, spec, track, n_tracks=None):
        super().__init__(app)
        self.spec, self.track = spec, track
        self.general = n_tracks is not None
        self.n_tracks = n_tracks or 1
        self.grid_n = (n_tracks or 2) // 2            # generalist grid: N x N cells, 2N circuits
        self.trainer = None
        self.view = None if self.general else TrackView(track)
        self.views, self.big_views, self._views_gen = [], {}, None
        self.featured = 0                             # circuit shown large (generalist)
        self.follow = False
        probe = CarCircuitEnv(track=track) if track is not None else CarCircuitEnv()
        self.ray_angles, self.max_range = probe.ray_angles, probe.max_range
        self.netview = NetworkView(self.ray_angles)
        self.paused = False
        self.acc = 0.0
        self.max_steps = 20
        self.speed_hist = []                       # (real seconds, sim steps) for the speed readout
        self.small_rects = []

        self.speed = Segmented([("1", "1x"), ("4", "4x"), ("16", "16x"), ("max", "Max")], "max")
        self.btn_pause = Button("Pause", self._pause)
        self.btn_stop = Button("Stop training", self._stop)
        self.btn_watch = Button("Watch best model", self._watch, "primary")
        self.btn_other = Button("Try another track", self._other, "primary")
        self.btn_back = Button("< Menu", self._back)
        self.widgets = [self.speed, self.btn_pause, self.btn_stop, self.btn_watch, self.btn_other, self.btn_back]

    # ----------------------------------------------------------- actions
    def _pause(self):
        self.paused = not self.paused

    def _stop(self):
        if self.trainer:
            self.trainer.stop()

    def _model(self):
        return load_model(model_path(self.trainer.key)) if self.trainer else None

    def _watch(self):
        m = self._model()
        if not m:
            return
        if self.general:                            # watch it on the circuit shown large
            tr = self.trainer
            self.app.go(DriveScreen(self.app, tr.track_specs[self.featured], tr.tracks[self.featured], m))
        else:
            self.app.go(DriveScreen(self.app, self.spec, self.track, m))

    def _other(self):
        m = self._model()
        if m:
            self.app.model_path, self.app.hint_test = m["path"], True
        self.app.go(SetupScreen(self.app))

    def _back(self):
        self.app.go(SetupScreen(self.app))

    def close(self):
        if self.trainer:
            self.trainer.close()

    def on_event(self, e):
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                self._back()
            elif e.key == pygame.K_c:
                self.follow = not self.follow
            elif e.key == pygame.K_SPACE:
                self._pause()
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.general:
            for k, r in self.small_rects:                 # click a thumbnail: show that circuit large
                if r.collidepoint(e.pos):
                    self.featured = k
                    self._views_gen = None
                    return

    # ------------------------------------------------------------ update
    def update(self, dt):
        if self.trainer is None:
            if getattr(self, "_shown_once", False):          # first frame shows "starting..."
                self.trainer = LiveTrainer(self.app.generations, self.app.population, spec=self.spec,
                                           track=self.track, n_tracks=self.n_tracks if self.general else None)
                self.t_last = time.perf_counter()
            return
        tr = self.trainer
        done = tr.finished
        self.btn_pause.label = "Resume" if self.paused else "Pause"
        self.btn_pause.visible = self.btn_stop.visible = self.speed.visible = not done
        self.btn_watch.visible = self.btn_other.visible = done
        self.btn_watch.enabled = self.btn_other.enabled = tr.best_genome is not None
        self.btn_back.rect = self.r_back_done if done else self.r_back_running
        if done or self.paused:
            return

        if tr.gen != getattr(self, "_gen_seen", None):        # new generation: start small again
            self._gen_seen, self.max_steps = tr.gen, 20
        before = tr.total_steps
        if self.speed.value == "max":
            t0 = time.perf_counter()
            tr.advance(self.max_steps)
            took = time.perf_counter() - t0
            # aim for ~60 ms of simulation per frame (performance over smooth graphics):
            # more steps when few cars are left
            self.max_steps = int(np.clip(self.max_steps * 0.06 / max(took, 1e-3), 2, 2000))
            self.acc = 0.0
        else:
            self.acc += dt * self.STEPS_PER_SECOND * int(self.speed.value)
            k = min(int(self.acc), 400)
            self.acc -= k
            tr.advance(k)
        now = time.perf_counter()
        self.speed_hist.append((now - self.t_last, tr.total_steps - before))
        self.speed_hist = self.speed_hist[-60:]
        self.t_last = now

    # ------------------------------------------------------------ layout
    def layout(self, w, h):
        self.r_map = pygame.Rect(self.PANEL_W, 0, w - self.PANEL_W - self.SIDE_W, h)
        self.r_side = pygame.Rect(w - self.SIDE_W, 0, self.SIDE_W, h)
        if self.view:
            self.view.set_rect(self.r_map)
        if self.general:
            # N x N grid: the featured circuit takes (N-1) x (N-1) cells, the others one cell each
            n = self.grid_n
            cw, ch = self.r_map.w / n, self.r_map.h / n
            self.r_big = pygame.Rect(self.r_map.x, 0, int(cw * (n - 1)), int(ch * (n - 1)))
            cells = [(n - 1, r) for r in range(n - 1)] + [(c, n - 1) for c in range(n)]
            self.cell_rects = [pygame.Rect(int(self.r_map.x + c * cw) + 2, int(r * ch) + 2,
                                           int(cw) - 4, int(ch) - 4) for c, r in cells]
            self._views_gen = None                    # thumbnails are re-placed
        x, cw = 20, self.PANEL_W - 40
        # while training: speed switch, then Pause | Stop | Menu on one row
        y = h - 20 - 38 - 10 - 38
        self.speed.rect = pygame.Rect(x, y, cw, 38)
        third = (cw - 20) // 3
        self.btn_pause.rect = pygame.Rect(x, y + 48, third, 38)
        self.btn_stop.rect = pygame.Rect(x + third + 10, y + 48, third, 38)
        self.r_back_running = pygame.Rect(x + 2 * third + 20, y + 48, cw - 2 * third - 20, 38)
        # when done: Watch | Try another on one row, Menu below
        y = h - 20 - 38 - 10 - 40
        half = (cw - 10) // 2
        self.btn_watch.rect = pygame.Rect(x, y, half, 40)
        self.btn_other.rect = pygame.Rect(x + half + 10, y, cw - half - 10, 40)
        self.r_back_done = pygame.Rect(x, y + 50, cw, 38)

    def _grid_views(self):
        """Track views of the current generation's circuits (rebuilt when the circuits change)."""
        tr = self.trainer
        if getattr(self, "_views_tracks", None) is not tr.tracks:
            self.views = [TrackView(t, lite=True) for t in tr.tracks]
            self.big_views = {}
            self._views_tracks = tr.tracks
            self._views_gen = None
        if self._views_gen != tr.gen:
            self._views_gen = tr.gen
            others = [k for k in range(len(self.views)) if k != self.featured]
            self.small_rects = list(zip(others, self.cell_rects))
            for k, r in self.small_rects:
                self.views[k].set_rect(r)
        if self.featured not in self.big_views:
            self.big_views[self.featured] = TrackView(tr.tracks[self.featured])
        big = self.big_views[self.featured]
        big.set_rect(self.r_big)
        return big

    # ------------------------------------------------------------- drawing
    def render(self, surf):
        tr = self.trainer
        if tr is None:
            self._shown_once = True
            pygame.draw.rect(surf, PANEL, (0, 0, self.PANEL_W, surf.get_height()))
            n = max(1, min((os.cpu_count() or 2) - 2, self.app.population))
            center_text(surf, f"Starting {n} worker processes...", surf.get_width() / 2,
                        surf.get_height() / 2, TEXT, fonts().mid)
            return
        lead = tr.leader()
        if self.general:
            if self.featured >= len(tr.tracks):
                self.featured = 0
            big = self._grid_views()
            self._draw_population(surf, big, self.featured, lead)
            self._draw_label(surf, big.rect, self.featured, big=True)
            for k, _ in self.small_rects:
                self._draw_thumb(surf, k, lead)
        else:
            self._draw_population(surf, self.view, 0, lead)
        self._draw_stats(surf, lead)
        self._draw_side(surf, lead)

    def _draw_population(self, surf, v, k, lead):
        """Every car on circuit k, the leader highlighted with its sensors."""
        s = self.trainer.state[:, k]
        v.follow = self.follow
        v.begin(surf, (s[lead, X], s[lead, Y], s[lead, HEADING]))
        scale = 1.0 if v.follow else 1.8
        crashed = np.nonzero((s[:, DONE] == 1) & (s[:, CRASHED] == 1))[0]
        stopped = np.nonzero((s[:, DONE] == 1) & (s[:, CRASHED] == 0))[0]
        running = np.nonzero(s[:, DONE] == 0)[0]
        for i in crashed:
            v.draw_car(surf, s[i, X], s[i, Y], s[i, HEADING], (120, 50, 50), scale * 0.8, glass=False)
        for i in stopped:
            v.draw_car(surf, s[i, X], s[i, Y], s[i, HEADING], (110, 110, 120), scale * 0.8, glass=False)
        for i in running:
            if i != lead:
                v.draw_car(surf, s[i, X], s[i, Y], s[i, HEADING], (120, 160, 210), scale, glass=False)
        h = s[lead, HEADING] + self.ray_angles
        dirs = np.stack([np.cos(h), np.sin(h)], axis=1)
        dists = self.trainer.obs[lead, k, :len(self.ray_angles)] * self.max_range
        if s[lead, DONE] == 0:
            v.draw_rays(surf, np.array([s[lead, X], s[lead, Y]]), dirs, dists, self.max_range)
        v.draw_car(surf, s[lead, X], s[lead, Y], s[lead, HEADING], CAR_BODY, scale * 1.15)
        v.end(surf)

    def _draw_thumb(self, surf, k, lead):
        """Small circuit: only the leader's car and the best lap of the generation."""
        v = self.views[k]
        s = self.trainer.state[lead, k]
        v.follow = False
        v.begin(surf)
        col = CAR_CRASH if s[CRASHED] else ((110, 110, 120) if s[DONE] and s[LAP_TIME] <= 0 else CAR_BODY)
        scale = max(2.5, 9.0 / (4.5 * v.overview_scale))       # never smaller than ~9 px
        v.draw_car(surf, s[X], s[Y], s[HEADING], col, scale, glass=False)
        v.end(surf)
        hover = v.rect.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(surf, ACCENT if hover else BORDER, v.rect, 1)
        self._draw_label(surf, v.rect, k)

    def _draw_label(self, surf, r, k, big=False):
        f = fonts()
        tr = self.trainer
        laps = tr.state[:, k, LAP_TIME]
        best = laps[laps > 0].min() if (laps > 0).any() else None
        t = tr.tracks[k]
        if big:
            text(surf, t.name, (r.x + 12, r.y + 10), f.mid, TEXT)
            text(surf, f"{t.length / 1000:.2f} km", (r.x + 12, r.y + 40), f.bold, TEXT_DIM)
            if best:
                text(surf, f"best lap {fmt_time(best)}", (r.x + 100, r.y + 40), f.bold, GOOD)
            text(surf, "click a small circuit to show it here", (r.x + 12, r.bottom - 24), f.small, TEXT_DIM)
        else:
            cw = " cw" if "clockwise" in t.name else ""
            text(surf, f"{t.length / 1000:.1f} km{cw}", (r.x + 6, r.y + 4), f.small, TEXT_DIM)
            if best:
                text(surf, fmt_time(best), (r.right - 6, r.bottom - 4), f.bold, GOOD, "bottomright")

    def _draw_stats(self, surf, lead):
        tr = self.trainer
        f = fonts()
        h = surf.get_height()
        pygame.draw.rect(surf, PANEL, (0, 0, self.PANEL_W, h))
        x, y = 20, 18
        cw = self.PANEL_W - 40
        y += text(surf, "NEUROEVOLUTION", (x, y), f.mid, ACCENT).h + 2
        sub = (f"Generalist · {tr.n_tracks} new circuits per generation" if self.general
               else f"{self.track.name}  {self.track.length / 1000:.2f} km")
        y += text(surf, sub, (x, y), f.small, TEXT_DIM).h + 16

        text(surf, "Generation", (x, y), f.normal, TEXT_DIM)
        text(surf, f"{tr.gen} / {tr.generations}", (x + cw, y), f.bold, TEXT, "topright")
        y += 24
        frac = (tr.gen - 1 + (0 if not tr.finished else 1)) / tr.generations
        bar(surf, x, y, cw, 8, frac, ACCENT)
        y += 22

        s = tr.state
        laps = s[:, :, LAP_TIME]
        episodes = laps.size
        running = int((s[:, :, DONE] == 0).sum())
        if self.general:
            lead_done = int((laps[lead] > 0).sum())
            rv = tr.record_val
            last = next((e for e in reversed(tr.history) if e.get("val_completed") is not None), None)
            rows = [
                ("Cars running", f"{running} / {episodes}"),
                ("Laps done (gen)", f"{int((laps > 0).sum())}  ({(laps > 0).mean():.0%})"),
                ("Leader", f"{lead_done} / {tr.n_tracks} circuits done"),
                ("Validation (last)", f"{last['val_completed']:.0%} unseen done" if last else "every 5 gen"),
                ("Best validation", f"{rv['completed']:.0%}  (gen {tr.record_gen})" if rv else "-"),
            ]
        else:
            done_laps = laps[laps > 0]
            gen_best = float(done_laps.min()) if len(done_laps) else None
            rows = [
                ("Cars running", f"{running} / {tr.pop_size}"),
                ("Laps done (gen)", f"{len(done_laps)}"),
                ("Best lap (gen)", fmt_time(gen_best)),
                ("Record", fmt_time(tr.record_lap)),
                ("  found in gen", f"{tr.record_gen}" if tr.record_gen else "-"),
            ]
        rows.append(("Sim time (gen)", fmt_time(tr.sim_time)))
        if self.speed_hist and not tr.finished:
            real = sum(a for a, _ in self.speed_hist)
            sim = sum(b for _, b in self.speed_hist) / self.STEPS_PER_SECOND
            rows.append(("Speed", f"{sim / max(real, 1e-6):.0f}x real time"))
        rows.append(("Elapsed", fmt_time(time.perf_counter() - tr.started)[:-3]))
        rows.append(("Workers", f"{tr.n_workers} processes"))
        for label, val in rows:
            good = label in ("Record", "Best validation") and val not in ("-", "--:--.--")
            text(surf, label, (x, y), f.normal, TEXT_DIM)
            text(surf, val, (x + cw, y), f.normal, GOOD if good else TEXT, "topright")
            y += 23

        y += 12
        if not tr.finished:                                    # leader block only while cars are driving
            k = self.featured if self.general else 0
            label = "LEADER (large circuit)" if self.general else "LEADER"
            y += text(surf, label, (x, y), f.small, TEXT_DIM).h + 4
            kmh = s[lead, k, SPEED] * 3.6
            text(surf, f"{kmh:4.0f} km/h", (x, y), f.mid, speed_color(kmh))
            prog = max(0.0, s[lead, k, PROGRESS] / tr.tracks[k].length)
            text(surf, f"{min(prog, 1):.0%} of lap", (x + cw, y + 6), f.normal, TEXT, "topright")
            y += 34
            bar(surf, x, y, cw, 8, prog, speed_color(kmh))
            y += 26

        if tr.finished:
            bottom = self.btn_watch.rect.y - 70
            text(surf, "TRAINING COMPLETE", (x, bottom + 10), f.mid, GOOD)
            if self.general:
                rv = tr.record_val
                best = f"completes {rv['completed']:.0%} of unseen circuits" if rv else "not validated"
            else:
                best = fmt_time(tr.record_lap) if tr.record_lap else "no full lap"
            text(surf, f"best model: {best}", (x, bottom + 40), f.normal, TEXT)
        else:
            bottom = self.speed.rect.y - 30
            text(surf, "SPEED  (Space pause, C follow)", (x, self.speed.rect.y - 22), f.small, TEXT_DIM)

        # progress chart: takes all the height left between the stats and the controls
        text(surf, "PROGRESS PER GENERATION", (x, y), f.small, TEXT_DIM)
        chart = pygame.Rect(x - 4, y + 22, cw + 8, bottom - y - 30)
        if chart.h >= 120:
            (draw_chart_general if self.general else draw_chart)(surf, chart, tr.history, tr.generations)

    def _draw_side(self, surf, lead):
        tr = self.trainer
        f = fonts()
        r = self.r_side
        pygame.draw.rect(surf, PANEL, r)
        x = r.x + 20
        y = 18
        y += text(surf, "LEADER'S NEURAL NETWORK", (x, y), f.mid, ACCENT).h + 2
        y += text(surf, "11 sensors -> 16 neurons -> 2 controls", (x, y), f.small, TEXT_DIM).h
        net_rect = pygame.Rect(r.x + 10, y + 16, r.w - 20, r.h - y - 36)
        k = self.featured if self.general else 0
        self.netview.draw(surf, net_rect, tr.pop[lead], tr.obs[lead, k])


# ==========================================================================
class App:
    def __init__(self):
        pygame.init()
        info = pygame.display.Info()
        w = max(1180, min(1560, int(info.current_w * 0.92)))
        h = max(720, min(940, int(info.current_h * 0.86)))
        self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        pygame.display.set_caption(TITLE)
        pygame.key.set_repeat(400, 40)
        self.clock = pygame.time.Clock()

        self.spec = TrackSpec(kind="random", seed=1)
        self.generations, self.population = 150, 96
        self.model_path = None
        self.hint_test = False
        self.train_mode, self.grid_n = "track", 4
        self._tracks = {}
        self.view = SetupScreen(self)

    def track_for(self, spec):
        key = spec.track_id()
        if key not in self._tracks:
            self._tracks[key] = spec.build()            # raises ValueError for impossible settings
        return self._tracks[key]

    def go(self, screen):
        self.view.close()
        self.view = screen

    def run(self):
        running = True
        while running:
            dt = min(self.clock.tick(60) / 1000.0, 0.1)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                else:
                    self.view.handle(e)
            self.view.update(dt)
            self.view.draw(self.screen)
            pygame.display.flip()
        self.view.close()
        pygame.quit()


if __name__ == "__main__":
    App().run()
