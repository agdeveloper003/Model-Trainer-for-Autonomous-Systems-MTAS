"""
Drawings of the learning: the neural network with its weights and live
activations, and the per-generation training chart.
"""

import numpy as np
import pygame

from neuroevo import Net
from ui import BORDER, CARD, GOOD, TEXT, TEXT_DIM, WARN, fonts, lerp_color, text

POS_W = (90, 180, 255)       # positive weight
NEG_W = (255, 120, 70)       # negative weight
NODE_LO = (40, 44, 56)


class NetworkView:
    """
    Shows the network of one car: inputs (sensors) on the left, hidden layer
    in the middle, outputs (steering, throttle) on the right.

    Lines are the weights: blue = positive, orange = negative, thicker and
    brighter = larger. When the shown network changes (new leader, new
    generation) the weights blend smoothly into the new values.
    Circles are the neurons, lit by how strongly they fire right now.
    """

    def __init__(self, ray_angles):
        n = len(ray_angles)
        self.in_labels = []
        for a in np.degrees(ray_angles):
            side = "R" if a < -0.5 else ("L" if a > 0.5 else "")
            self.in_labels.append(f"ray {side}{abs(a):.0f}" if side else "ray 0")
        self.in_labels += ["speed", "steer"]
        self.out_labels = ["steer", "throttle"]
        self.net = Net(n_in=n + 2)
        self.shown = None                      # genome currently drawn (blends to the target)

    def draw(self, surf, rect, genome, obs, dt=1 / 60):
        f = fonts()
        genome = np.asarray(genome, float)
        if self.shown is None or self.shown.shape != genome.shape:
            self.shown = genome.copy()
        else:
            self.shown += (genome - self.shown) * min(1.0, dt * 4.0)
        params = self.net.unpack(self.shown)
        w1, b1, w2, b2 = params
        obs = np.asarray(obs, float)
        hid, out = Net.activations(params, obs)

        top, bottom = rect.y + 8, rect.bottom - 8
        x_in, x_hid, x_out = rect.x + 92, rect.centerx + 18, rect.right - 72

        def column(n, x):
            gap = (bottom - top) / n
            return [(x, top + gap * (i + 0.5)) for i in range(n)]

        p_in, p_hid, p_out = column(len(obs), x_in), column(len(hid), x_hid), column(len(out), x_out)

        for layer_w, src, dst in ((w1, p_in, p_hid), (w2, p_hid, p_out)):
            mag = np.abs(layer_w)
            scale = max(1e-6, np.percentile(mag, 95))
            # draw weak weights first so strong ones stay on top
            for i, j in zip(*np.unravel_index(np.argsort(mag, axis=None), mag.shape)):
                t = min(1.0, mag[i, j] / scale)
                col = lerp_color(CARD, POS_W if layer_w[i, j] > 0 else NEG_W, 0.15 + 0.85 * t)
                pygame.draw.line(surf, col, src[i], dst[j], 2 if t > 0.6 else 1)

        def node(p, v, r=7):
            col = lerp_color(NODE_LO, POS_W if v >= 0 else NEG_W, min(1.0, abs(v)))
            pygame.draw.circle(surf, col, (int(p[0]), int(p[1])), r)
            pygame.draw.circle(surf, BORDER, (int(p[0]), int(p[1])), r, 1)

        for p, v, lab in zip(p_in, obs, self.in_labels):
            node(p, v)
            text(surf, lab, (p[0] - 14, p[1]), f.small, TEXT_DIM, "midright")
        for p, v in zip(p_hid, hid):
            node(p, v, 6)
        for p, v, lab in zip(p_out, out, self.out_labels):
            node(p, v, 10)
            text(surf, lab, (p[0] + 16, p[1] - 9), f.small, TEXT_DIM)
            text(surf, f"{v:+.2f}", (p[0] + 16, p[1] + 7), f.small, TEXT)


SURVIVAL_COL = GOOD                  # left axis: share of cars that completed the lap
TIME_COL = (90, 180, 255)            # right axis: best lap time of the generation


def _nice_step(span, max_ticks, steps):
    """Smallest step from `steps` that gives at most `max_ticks` intervals."""
    for st in steps:
        if span / st <= max_ticks:
            return st
    return steps[-1]


def _short_time(t, step):
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:04.1f}" if step < 1 else f"{int(m)}:{int(round(s)):02d}"


def _axes(surf, rect, history, total_generations, left_label, right_label, right_col):
    """Background, legend, generation axis and the left 0..100 % axis. Returns (plot, x_of, y_pct, n)."""
    f = fonts()
    pygame.draw.rect(surf, (16, 18, 23), rect, border_radius=6)
    plot = pygame.Rect(rect.x + 50, rect.y + 30, rect.w - 50 - 66, rect.h - 30 - 38)
    text(surf, left_label, (plot.x, rect.y + 8), f.small, SURVIVAL_COL)
    text(surf, right_label, (plot.right, rect.y + 8), f.small, right_col, "topright")

    # x axis: generation 1 .. total (the chart fills up as training goes on)
    n = max(total_generations, len(history), 2)

    def x_of(g):
        return plot.x + (g - 1) / (n - 1) * plot.w

    gstep = _nice_step(n - 1, 6, [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000])
    for g in [1] + list(range(gstep, n + 1, gstep)):
        x = x_of(g)
        pygame.draw.line(surf, BORDER, (x, plot.bottom), (x, plot.bottom + 4))
        text(surf, str(g), (x, plot.bottom + 6), f.small, TEXT_DIM, "midtop")
    text(surf, "generation", (plot.centerx, plot.bottom + 21), f.small, TEXT_DIM, "midtop")

    # left axis: 0..100 %, with grid lines
    def y_pct(p):
        return plot.bottom - p * plot.h

    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y_pct(p)
        pygame.draw.line(surf, (32, 35, 44), (plot.x, y), (plot.right, y))
        text(surf, f"{p:.0%}", (plot.x - 6, y), f.small, SURVIVAL_COL, "midright")
    return plot, x_of, y_pct, n


def _frame_lines(surf, plot, right_col):
    pygame.draw.line(surf, BORDER, plot.bottomleft, plot.bottomright)
    pygame.draw.line(surf, SURVIVAL_COL, plot.topleft, plot.bottomleft)
    pygame.draw.line(surf, right_col, plot.topright, plot.bottomright)


def _survival_bars(surf, plot, history, x_of, n):
    """Share of episodes that completed the lap: one semi-transparent bar per generation."""
    bw = max(1.0, plot.w / (n - 1) * 0.8)
    bars = pygame.Surface(plot.size, pygame.SRCALPHA)
    for h in history:
        hgt = h["completed"] / max(1, h.get("pop", 1)) * plot.h
        if hgt > 0:
            x = x_of(h["gen"]) - plot.x - bw / 2
            bars.fill((*SURVIVAL_COL, 90), pygame.Rect(round(x), round(plot.h - hgt), max(1, round(bw)), round(hgt)))
    surf.blit(bars, plot.topleft)


def draw_chart(surf, rect, history, total_generations):
    """
    Cartesian chart, one point per generation (shared x axis):
      left axis  = survival rate, % of cars that completed the lap (green)
      right axis = best lap time of the generation (blue)
    """
    f = fonts()
    plot, x_of, y_pct, n = _axes(surf, rect, history, total_generations,
                                 "survival rate %", "best lap time", TIME_COL)
    laps = [h["best_lap"] for h in history if h["best_lap"]]

    # right axis: best lap time, range from the data
    laps = [h["best_lap"] for h in history if h["best_lap"]]
    if laps:
        lo, hi = min(laps), max(laps)
        pad = max((hi - lo) * 0.1, 0.5)
        lo, hi = lo - pad, hi + pad
        tstep = _nice_step(hi - lo, 4, [0.5, 1, 2, 5, 10, 15, 30, 60, 120])
        lo, hi = np.floor(lo / tstep) * tstep, np.ceil(hi / tstep) * tstep

        def y_time(t):
            return plot.bottom - (t - lo) / (hi - lo) * plot.h

        t = lo
        while t <= hi + 1e-9:
            y = y_time(t)
            pygame.draw.line(surf, BORDER, (plot.right, y), (plot.right + 4, y))
            text(surf, _short_time(t, tstep), (plot.right + 7, y), f.small, TIME_COL, "midleft")
            t += tstep

    _frame_lines(surf, plot, TIME_COL)
    if not history:
        text(surf, "waiting for the first generation...", plot.center, f.small, TEXT_DIM, "center")
        return
    _survival_bars(surf, plot, history, x_of, n)

    # best lap time: the line breaks where a generation completed no lap
    if laps:
        run = []
        for h in history + [{"best_lap": None, "gen": 0}]:
            if h["best_lap"]:
                run.append((x_of(h["gen"]), y_time(h["best_lap"])))
                continue
            if len(run) > 1:
                pygame.draw.lines(surf, TIME_COL, False, run, 2)
            for p in run:
                pygame.draw.circle(surf, TIME_COL, (int(p[0]), int(p[1])), 3)
            run = []
    else:
        best = history[-1]["best_progress"]
        text(surf, "no lap completed yet", (plot.centerx, plot.centery - 9), f.small, WARN, "center")
        text(surf, f"best {min(best, 1):.0%} of a lap", (plot.centerx, plot.centery + 9), f.small, WARN, "center")


VAL_COL = (240, 200, 90)             # left axis: unseen validation circuits completed
SPEED_COL = (90, 180, 255)           # right axis: average speed on the validation circuits


def draw_chart_general(surf, rect, history, total_generations, every=5):
    """
    Generalist training, shared generation axis:
      bars (left %)     survival rate on the training circuits of each generation
      yellow (left %)   validation circuits completed by the best network (every few generations)
      blue (right)      its average speed on those circuits, km/h
    """
    f = fonts()
    plot, x_of, y_pct, n = _axes(surf, rect, history, total_generations,
                                 "survival %", "speed km/h", SPEED_COL)
    text(surf, "unseen done %", (plot.x + f.small.size("survival %  ")[0], rect.y + 8), f.small, VAL_COL)
    vals = [h for h in history if h.get("val_completed") is not None]
    speeds = [h["val_speed"] for h in vals if h.get("val_speed")]
    if speeds:
        lo, hi = min(speeds), max(speeds)
        step = _nice_step(max(hi - lo, 1.0), 4, [2, 5, 10, 20, 25, 50, 100])
        lo, hi = np.floor((lo - 1) / step) * step, np.ceil((hi + 1) / step) * step

        def y_speed(v):
            return plot.bottom - (v - lo) / (hi - lo) * plot.h

        v = lo
        while v <= hi + 1e-9:
            y = y_speed(v)
            pygame.draw.line(surf, BORDER, (plot.right, y), (plot.right + 4, y))
            text(surf, f"{v:.0f}", (plot.right + 7, y), f.small, SPEED_COL, "midleft")
            v += step
    _frame_lines(surf, plot, SPEED_COL)
    if not history:
        text(surf, "waiting for the first generation...", plot.center, f.small, TEXT_DIM, "center")
        return
    _survival_bars(surf, plot, history, x_of, n)

    for pts, col in (([(x_of(h["gen"]), y_pct(h["val_completed"])) for h in vals], VAL_COL),
                     ([(x_of(h["gen"]), y_speed(h["val_speed"])) for h in vals if h.get("val_speed")]
                      if speeds else [], SPEED_COL)):
        if len(pts) > 1:
            pygame.draw.lines(surf, col, False, pts, 2)
        for p in pts:
            pygame.draw.circle(surf, col, (int(p[0]), int(p[1])), 3)
    if not vals:
        text(surf, "validation on unseen circuits", (plot.centerx, plot.centery - 9), f.small, VAL_COL, "center")
        when = "every generation" if every == 1 else f"every {every} generations"
        text(surf, when, (plot.centerx, plot.centery + 9), f.small, VAL_COL, "center")
