"""
Car physics: kinematic "bicycle" model with a grip limit.

Commands (both between -1 and +1):
  steer    : +1 = full left, -1 = full right
  throttle : +1 = full throttle, -1 = full brake

The key idea behind the "efficient racing line": tyres have a limit on
lateral acceleration. Enter a corner too fast and the car cannot turn as
much as you ask (understeer), so it drifts towards the wall.
To be fast you must brake before corners and use the whole width of the
track: exactly what the model has to learn.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CarParams:
    length: float = 4.5          # m
    width: float = 2.0           # m
    wheelbase: float = 2.8       # m, distance between the axles
    max_speed: float = 70.0      # m/s (252 km/h)
    engine_accel: float = 9.0    # m/s^2 at low speed, fades towards max_speed
    brake_decel: float = 16.0    # m/s^2 at full brake
    rolling_decel: float = 0.6   # m/s^2, drag when not accelerating
    max_steer: float = 0.5       # rad (~29°) maximum wheel angle
    steer_rate: float = 2.5      # rad/s, how fast the steering wheel turns
    max_lat_accel: float = 13.0  # m/s^2 (~1.3 g), cornering grip limit
    scrub_decel: float = 6.0     # m/s^2 lost while the tyres are sliding


class Car:
    def __init__(self, params=None):
        self.p = params or CarParams()
        self.reset(0.0, 0.0, 0.0)

    def reset(self, x, y, heading, speed=0.0):
        self.x, self.y = float(x), float(y)
        self.heading = float(heading)   # rad, 0 = towards +x, counter-clockwise positive
        self.speed = float(speed)       # m/s, forward only
        self.steer = 0.0                # rad, current wheel angle
        self.sliding = False            # True while exceeding the grip limit

    @property
    def pos(self):
        return np.array([self.x, self.y])

    def step(self, steer_cmd, throttle, dt):
        p = self.p
        steer_cmd = max(-1.0, min(1.0, float(steer_cmd)))
        throttle = max(-1.0, min(1.0, float(throttle)))

        # steering: the wheels reach the requested angle at a limited rate
        target = steer_cmd * p.max_steer
        delta = max(-p.steer_rate * dt, min(p.steer_rate * dt, target - self.steer))
        self.steer += delta

        # longitudinal speed
        if throttle >= 0:
            accel = throttle * p.engine_accel * (1.0 - self.speed / p.max_speed)
        else:
            accel = throttle * p.brake_decel
        accel -= p.rolling_decel

        # rotation: how much we would like to turn vs how much the tyres allow
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
        """The 4 corners of the body (for collisions and drawing)."""
        c, s = math.cos(self.heading), math.sin(self.heading)
        hl, hw = self.p.length / 2, self.p.width / 2
        local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + self.pos
