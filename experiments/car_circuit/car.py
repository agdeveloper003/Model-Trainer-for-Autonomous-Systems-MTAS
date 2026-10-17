"""
Fisica della macchina: modello "bicicletta" cinematico con limite di aderenza.

Comandi (entrambi tra -1 e +1):
  steer    : +1 = tutto a sinistra, -1 = tutto a destra
  throttle : +1 = acceleratore a fondo, -1 = freno a fondo

L'idea chiave per la "traiettoria efficiente": le gomme hanno un limite
di accelerazione laterale. Se entri in curva troppo veloce la macchina
non riesce a girare quanto vorresti (sottosterzo) e va verso il muro.
Quindi per essere veloci bisogna frenare prima delle curve e usare
tutta la larghezza della pista: proprio quello che il modello dovrà imparare.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CarParams:
    length: float = 4.5          # m
    width: float = 2.0           # m
    wheelbase: float = 2.8       # m, distanza tra gli assi
    max_speed: float = 70.0      # m/s (252 km/h)
    engine_accel: float = 9.0    # m/s^2 a bassa velocità, cala verso max_speed
    brake_decel: float = 16.0    # m/s^2 a freno pieno
    rolling_decel: float = 0.6   # m/s^2, attrito quando non acceleri
    max_steer: float = 0.5       # rad (~29°) angolo massimo delle ruote
    steer_rate: float = 2.5      # rad/s, velocità con cui il volante si muove
    max_lat_accel: float = 13.0  # m/s^2 (~1.3 g), limite di aderenza in curva
    scrub_decel: float = 6.0     # m/s^2 persi quando le gomme slittano


class Car:
    def __init__(self, params=None):
        self.p = params or CarParams()
        self.reset(0.0, 0.0, 0.0)

    def reset(self, x, y, heading, speed=0.0):
        self.x, self.y = float(x), float(y)
        self.heading = float(heading)   # rad, 0 = verso +x, antiorario positivo
        self.speed = float(speed)       # m/s, solo marcia avanti
        self.steer = 0.0                # rad, angolo attuale delle ruote
        self.sliding = False            # True se stiamo superando l'aderenza

    @property
    def pos(self):
        return np.array([self.x, self.y])

    def step(self, steer_cmd, throttle, dt):
        p = self.p
        steer_cmd = max(-1.0, min(1.0, float(steer_cmd)))
        throttle = max(-1.0, min(1.0, float(throttle)))

        # sterzo: le ruote raggiungono l'angolo richiesto con velocità limitata
        target = steer_cmd * p.max_steer
        delta = max(-p.steer_rate * dt, min(p.steer_rate * dt, target - self.steer))
        self.steer += delta

        # velocità longitudinale
        if throttle >= 0:
            accel = throttle * p.engine_accel * (1.0 - self.speed / p.max_speed)
        else:
            accel = throttle * p.brake_decel
        accel -= p.rolling_decel

        # rotazione: quanto vorremmo girare vs quanto le gomme permettono
        yaw_rate = self.speed / p.wheelbase * math.tan(self.steer)
        max_yaw = p.max_lat_accel / max(self.speed, 0.1)
        self.sliding = abs(yaw_rate) > max_yaw
        if self.sliding:
            yaw_rate = math.copysign(max_yaw, yaw_rate)
            accel -= p.scrub_decel

        self.speed = max(0.0, min(p.max_speed, self.speed + accel * dt))
        self.heading = (self.heading + yaw_rate * dt + math.pi) % (2 * math.pi) - math.pi
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt

    def corners(self):
        """I 4 angoli della carrozzeria (per collisioni e disegno)."""
        c, s = math.cos(self.heading), math.sin(self.heading)
        hl, hw = self.p.length / 2, self.p.width / 2
        local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + self.pos
