"""
Circuito: geometria, progresso lungo il tracciato e sensori a raggi.

Unità di misura: metri. Asse y verso l'alto (come in matematica);
è il renderer a capovolgerlo per lo schermo.

Il circuito è descritto da pochi "punti di controllo". Una spline
Catmull-Rom chiusa li unisce in una curva morbida (la linea centrale),
che poi viene ricampionata a passo costante. Dalla linea centrale e
dalla larghezza si ricavano i due bordi (i "muri").
"""
import numpy as np


# --------------------------------------------------------------------------
# Circuito disegnato a mano (metri). Senso di marcia: antiorario.
# Il primo punto è sul rettilineo di partenza.
# --------------------------------------------------------------------------
DEFAULT_LAYOUT = [
    (0, 0), (120, 0), (240, 0), (330, 10),          # rettilineo principale
    (390, 50), (405, 115),                          # curvone a sinistra
    (375, 175), (310, 195),                         # verso la chicane
    (265, 175), (225, 195), (215, 245),             # chicane destra-sinistra
    (250, 300), (320, 320), (375, 350),             # esse veloce
    (385, 410), (340, 450),                         # tornante
    (250, 460), (150, 450), (70, 420),              # rettilineo opposto
    (20, 370), (5, 300), (35, 240),                 # discesa
    (60, 185), (40, 125), (-15, 90),                # curva a S lenta
    (-60, 50), (-55, 10),                           # ultima curva
]


def catmull_rom_closed(points, samples_per_segment=40):
    """Curva chiusa e morbida che passa per tutti i punti di controllo."""
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


def resample_closed(poly, spacing):
    """Ricampiona una polilinea chiusa a passo (circa) costante."""
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
    def __init__(self, control_points=None, width=14.0, spacing=2.0, name="Circuito MTAS 1"):
        self.name = name
        self.width = float(width)
        self.half_width = self.width / 2.0
        points = DEFAULT_LAYOUT if control_points is None else control_points

        c = resample_closed(catmull_rom_closed(points), spacing)
        self.center = c                                   # (N, 2) punti della linea centrale
        self.N = len(c)

        seg = np.roll(c, -1, axis=0) - c                  # segmento i: center[i] -> center[i+1]
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.tangent = seg / self.seg_len[:, None]
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)[:-1]])  # ascissa curvilinea
        self.length = float(self.seg_len.sum())

        # normale (verso sinistra) nei vertici: media delle tangenti adiacenti
        tv = self.tangent + np.roll(self.tangent, 1, axis=0)
        tv /= np.linalg.norm(tv, axis=1)[:, None]
        self.normal = np.stack([-tv[:, 1], tv[:, 0]], axis=1)
        self.left = c + self.normal * self.half_width
        self.right = c - self.normal * self.half_width

        # muri come segmenti A->B (bordo sinistro + bordo destro)
        self.wall_a = np.vstack([self.left, self.right])
        self.wall_b = np.vstack([np.roll(self.left, -1, 0), np.roll(self.right, -1, 0)])
        self.wall_mid = 0.5 * (self.wall_a + self.wall_b)
        self.wall_reach = 0.5 * np.linalg.norm(self.wall_b - self.wall_a, axis=1).max()

        self.start_pos = c[0].copy()
        self.start_heading = float(np.arctan2(self.tangent[0, 1], self.tangent[0, 0]))

    # ------------------------------------------------------------------
    def closest(self, pos, hint=None, window=40):
        """
        Punto della linea centrale più vicino a `pos`.

        Ritorna (indice_segmento, s, laterale):
          s        = metri percorsi lungo il circuito dal traguardo
          laterale = distanza con segno dalla linea centrale (+ = sinistra)
        `hint` è l'indice trovato al passo precedente: si cerca solo lì
        intorno, così è veloce. Senza hint si cerca su tutto il circuito.
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
        Lancia raggi da `origin` nelle `directions` (R, 2 versori) e
        ritorna la distanza dal primo muro colpito (max_range se nessuno).
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
        """Raggio di curva più stretto (utile per verificare il tracciato)."""
        dth = np.arctan2(_cross(np.roll(self.tangent, 1, 0), self.tangent),
                         (np.roll(self.tangent, 1, 0) * self.tangent).sum(1))
        k = np.abs(dth) / self.seg_len
        return float(1.0 / k.max())

    def bounds(self):
        pts = np.vstack([self.left, self.right])
        return pts.min(axis=0), pts.max(axis=0)
