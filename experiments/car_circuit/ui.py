"""
Tiny pygame UI toolkit: theme, fonts and a handful of widgets
(button, slider, number field, segmented switch, option cycler).

Every widget has a `rect` that the screen sets in its layout, a
`handle(event)` method that returns True when it used the event, and
`draw(surface)`.
"""
import pygame

# ------------------------------------------------------------------ theme
BG = (14, 16, 20)
PANEL = (18, 20, 26)
CARD = (25, 28, 35)
CARD_HI = (34, 38, 47)
BORDER = (46, 50, 62)
TEXT = (230, 232, 238)
TEXT_DIM = (140, 146, 160)
ACCENT = (90, 180, 255)
ACCENT_DARK = (40, 92, 150)
WARN = (255, 170, 40)
BAD = (255, 80, 70)
GOOD = (90, 220, 120)

MONO = "consolas,menlo,dejavusansmono,couriernew,monospace"


class Fonts:
    def __init__(self):
        self.small = pygame.font.SysFont(MONO, 14)
        self.normal = pygame.font.SysFont(MONO, 17)
        self.bold = pygame.font.SysFont(MONO, 17, bold=True)
        self.mid = pygame.font.SysFont(MONO, 24, bold=True)
        self.title = pygame.font.SysFont(MONO, 34, bold=True)
        self.big = pygame.font.SysFont(MONO, 54, bold=True)


_fonts = None


def fonts():
    global _fonts
    if _fonts is None:
        _fonts = Fonts()
    return _fonts


# ---------------------------------------------------------------- drawing
def text(surf, txt, pos, font=None, color=TEXT, anchor="topleft"):
    img = (font or fonts().normal).render(str(txt), True, color)
    r = img.get_rect(**{anchor: (int(pos[0]), int(pos[1]))})
    surf.blit(img, r)
    return r


def card(surf, rect, title=None):
    pygame.draw.rect(surf, CARD, rect, border_radius=10)
    pygame.draw.rect(surf, BORDER, rect, 1, border_radius=10)
    if title:
        text(surf, title.upper(), (rect.x + 16, rect.y + 12), fonts().small, TEXT_DIM)
        return rect.y + 36
    return rect.y + 12


def lerp_color(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def bar(surf, x, y, w, h, frac, color, centered=False):
    pygame.draw.rect(surf, (45, 48, 58), (x, y, w, h))
    frac = max(-1.0, min(1.0, frac))
    if centered:
        mid = x + w / 2
        fw = abs(frac) * w / 2
        pygame.draw.rect(surf, color, (mid if frac >= 0 else mid - fw, y, fw, h))
        pygame.draw.line(surf, TEXT_DIM, (mid, y - 2), (mid, y + h + 2))
    else:
        pygame.draw.rect(surf, color, (x, y, max(0.0, frac) * w, h))


def wrap(txt, font, width):
    """Split `txt` into lines that fit `width` pixels."""
    words, lines, cur = txt.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if font.size(trial)[0] <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# ---------------------------------------------------------------- widgets
class Widget:
    def __init__(self):
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.enabled = True
        self.visible = True

    def active(self):
        return self.enabled and self.visible

    def handle(self, e):
        return False

    def draw(self, surf):
        pass


class Button(Widget):
    """kind: "normal", "primary" (filled accent) or "big" (large, with a subtitle)."""

    def __init__(self, label, on_click, kind="normal", sub=None):
        super().__init__()
        self.label, self.on_click, self.kind, self.sub = label, on_click, kind, sub

    def handle(self, e):
        if (self.active() and e.type == pygame.MOUSEBUTTONDOWN and e.button == 1
                and self.rect.collidepoint(e.pos)):
            self.on_click()
            return True
        return False

    def draw(self, surf):
        if not self.visible:
            return
        hover = self.enabled and self.rect.collidepoint(pygame.mouse.get_pos())
        f = fonts()
        if not self.enabled:
            bg, fg, border = CARD, (80, 84, 96), BORDER
        elif self.kind in ("primary", "big"):
            bg = lerp_color(ACCENT_DARK, ACCENT, 0.35 if hover else 0.0)
            fg, border = TEXT, ACCENT
        else:
            bg, fg, border = (CARD_HI if hover else CARD), TEXT, (ACCENT if hover else BORDER)
        pygame.draw.rect(surf, bg, self.rect, border_radius=10)
        pygame.draw.rect(surf, border, self.rect, 1 if self.kind != "big" else 2, border_radius=10)
        if self.kind == "big":
            cy = self.rect.centery - (12 if self.sub else 0)
            text(surf, self.label, (self.rect.centerx, cy), f.title, fg, "center")
            if self.sub:
                text(surf, self.sub, (self.rect.centerx, cy + 32), f.small,
                     (200, 220, 245) if self.enabled else fg, "center")
        else:
            text(surf, self.label, self.rect.center, f.bold if self.kind == "primary" else f.normal, fg, "center")


class Slider(Widget):
    """Label and value on top, draggable bar below. on_change fires on release."""

    def __init__(self, label, lo, hi, step, value, fmt="{:g}", on_change=None):
        super().__init__()
        self.label, self.lo, self.hi, self.step = label, lo, hi, step
        self.value, self.fmt, self.on_change = value, fmt, on_change
        self.dragging = False

    @property
    def bar_rect(self):
        return pygame.Rect(self.rect.x, self.rect.bottom - 14, self.rect.w, 14)

    def _set_from_x(self, x):
        r = self.bar_rect
        t = min(1.0, max(0.0, (x - r.x) / max(1, r.w)))
        v = self.lo + t * (self.hi - self.lo)
        self.value = min(self.hi, max(self.lo, round(v / self.step) * self.step))

    def _changed(self):
        if self.on_change:
            self.on_change(self.value)

    def handle(self, e):
        if not self.active():
            return False
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.rect.collidepoint(e.pos):
            self.dragging = True
            self._set_from_x(e.pos[0])
            return True
        if e.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(e.pos[0])
            return True
        if e.type == pygame.MOUSEBUTTONUP and e.button == 1 and self.dragging:
            self.dragging = False
            self._changed()
            return True
        if e.type == pygame.MOUSEWHEEL and self.rect.collidepoint(pygame.mouse.get_pos()):
            self.value = min(self.hi, max(self.lo, self.value + e.y * self.step))
            self._changed()
            return True
        return False

    def draw(self, surf):
        if not self.visible:
            return
        f = fonts()
        col = TEXT if self.enabled else (80, 84, 96)
        text(surf, self.label, (self.rect.x, self.rect.y), f.normal, TEXT_DIM if self.enabled else col)
        text(surf, self.fmt.format(self.value), (self.rect.right, self.rect.y), f.bold, col, "topright")
        r = self.bar_rect
        line_y = r.centery
        pygame.draw.line(surf, BORDER, (r.x, line_y), (r.right, line_y), 4)
        t = (self.value - self.lo) / (self.hi - self.lo) if self.hi > self.lo else 0
        kx = r.x + t * r.w
        if self.enabled:
            pygame.draw.line(surf, ACCENT, (r.x, line_y), (kx, line_y), 4)
        pygame.draw.circle(surf, ACCENT if self.enabled else BORDER, (int(kx), line_y), 7)


class NumberField(Widget):
    """Click to type a whole number (e.g. the seed)."""

    def __init__(self, label, value, max_len=9, on_change=None):
        super().__init__()
        self.label, self.value, self.max_len, self.on_change = label, str(value), max_len, on_change
        self.focused = False

    def handle(self, e):
        if not self.active():
            return False
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            was = self.focused
            self.focused = self.rect.collidepoint(e.pos)
            return self.focused or was
        if e.type == pygame.KEYDOWN and self.focused:
            if e.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_ESCAPE, pygame.K_TAB):
                self.focused = False
            elif e.key == pygame.K_BACKSPACE:
                self.value = self.value[:-1]
                self._changed()
            elif e.unicode.isdigit() and len(self.value) < self.max_len:
                self.value = (self.value + e.unicode).lstrip("0") or "0"
                self._changed()
            return True
        return False

    def _changed(self):
        if self.on_change:
            self.on_change(self.value)

    def set(self, value):
        self.value = str(value)

    def draw(self, surf):
        if not self.visible:
            return
        f = fonts()
        col = TEXT if self.enabled else (80, 84, 96)
        border = ACCENT if self.focused else BORDER
        pygame.draw.rect(surf, BG, self.rect, border_radius=6)
        pygame.draw.rect(surf, border, self.rect, 1, border_radius=6)
        r = text(surf, self.label, (self.rect.x + 10, self.rect.centery), f.small, TEXT_DIM, "midleft")
        v = text(surf, self.value or " ", (r.right + 10, self.rect.centery), f.bold, col, "midleft")
        if self.focused and (pygame.time.get_ticks() // 500) % 2 == 0:
            pygame.draw.line(surf, ACCENT, (v.right + 2, v.top + 2), (v.right + 2, v.bottom - 2), 2)


class Segmented(Widget):
    """A row of mutually exclusive options."""

    def __init__(self, options, value, on_change=None):
        super().__init__()
        self.options, self.value, self.on_change = options, value, on_change   # options: [(key, label)]

    def _cells(self):
        n = len(self.options)
        w = self.rect.w / n
        return [pygame.Rect(int(self.rect.x + i * w), self.rect.y, int(w), self.rect.h) for i in range(n)]

    def handle(self, e):
        if self.active() and e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            for (key, _), cell in zip(self.options, self._cells()):
                if cell.collidepoint(e.pos):
                    if key != self.value:
                        self.value = key
                        if self.on_change:
                            self.on_change(key)
                    return True
        return False

    def draw(self, surf):
        if not self.visible:
            return
        pygame.draw.rect(surf, BG, self.rect, border_radius=8)
        hover = pygame.mouse.get_pos()
        for (key, label), cell in zip(self.options, self._cells()):
            sel = key == self.value
            if sel:
                pygame.draw.rect(surf, ACCENT_DARK, cell.inflate(-4, -4), border_radius=6)
            elif self.enabled and cell.collidepoint(hover):
                pygame.draw.rect(surf, CARD_HI, cell.inflate(-4, -4), border_radius=6)
            text(surf, label, cell.center, fonts().bold if sel else fonts().normal,
                 TEXT if sel else TEXT_DIM, "center")
        pygame.draw.rect(surf, BORDER, self.rect, 1, border_radius=8)


class Cycler(Widget):
    """< option >  —  step through a list with the arrows."""

    def __init__(self, options, index=0, on_change=None):
        super().__init__()
        self.options, self.index, self.on_change = options, index, on_change   # options: [(key, label)]

    @property
    def value(self):
        return self.options[self.index][0] if self.options else None

    def _arrows(self):
        s = self.rect.h
        return pygame.Rect(self.rect.x, self.rect.y, s, s), pygame.Rect(self.rect.right - s, self.rect.y, s, s)

    def handle(self, e):
        if not (self.active() and self.options) or e.type != pygame.MOUSEBUTTONDOWN or e.button != 1:
            return False
        left, right = self._arrows()
        step = -1 if left.collidepoint(e.pos) else (1 if right.collidepoint(e.pos) else 0)
        if step:
            self.index = (self.index + step) % len(self.options)
            if self.on_change:
                self.on_change(self.value)
            return True
        return False

    def draw(self, surf):
        if not self.visible:
            return
        f = fonts()
        pygame.draw.rect(surf, BG, self.rect, border_radius=8)
        pygame.draw.rect(surf, BORDER, self.rect, 1, border_radius=8)
        left, right = self._arrows()
        hover = pygame.mouse.get_pos()
        for r, ch in ((left, "<"), (right, ">")):
            if self.options and r.collidepoint(hover):
                pygame.draw.rect(surf, CARD_HI, r.inflate(-6, -6), border_radius=6)
            text(surf, ch, r.center, f.bold, TEXT if len(self.options) > 1 else TEXT_DIM, "center")
        label = self.options[self.index][1] if self.options else "-"
        avail = self.rect.w - 2 * self.rect.h - 8
        while f.normal.size(label)[0] > avail and len(label) > 4:
            label = label[:-2] + "…"
        text(surf, label, self.rect.center, f.normal, TEXT, "center")
