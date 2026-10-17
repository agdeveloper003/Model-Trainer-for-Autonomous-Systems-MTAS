"""
Ambiente di simulazione: unisce circuito + macchina + sensori + ricompensa.

Interfaccia nello stile Gymnasium (lo standard del reinforcement learning):

    obs = env.reset()
    obs, reward, terminated, truncated, info = env.step([steer, throttle])

Non contiene NESSUNA grafica: può girare migliaia di volte più veloce
del tempo reale durante l'addestramento. Il renderer lo osserva soltanto.
"""
import math

import numpy as np

from car import Car, CarParams
from track import Track


class CarCircuitEnv:
    def __init__(self, track=None, car_params=None, dt=1 / 30, substeps=2,
                 n_rays=9, ray_fov_deg=180.0, max_range=150.0,
                 max_time=240.0, stall_time=6.0, laps_to_finish=1,
                 crash_penalty=10.0, time_penalty=0.01):
        self.track = track or Track()
        self.car = Car(car_params or CarParams())
        self.dt = dt
        self.substeps = substeps
        self.max_range = max_range
        self.max_time = max_time
        self.stall_time = stall_time
        self.laps_to_finish = laps_to_finish
        self.crash_penalty = crash_penalty
        self.time_penalty = time_penalty

        # angoli dei raggi relativi al muso: da -fov/2 (destra) a +fov/2 (sinistra)
        half = math.radians(ray_fov_deg) / 2
        self.ray_angles = np.linspace(-half, half, n_rays)

        # osservazione = distanze dei raggi (0..1) + velocità (0..1) + sterzo (-1..1)
        self.obs_size = n_rays + 2
        self.action_size = 2
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        t = self.track
        self.car.reset(t.start_pos[0], t.start_pos[1], t.start_heading)
        self.seg, self.prev_s, self.lateral = t.closest(self.car.pos)
        self.progress = 0.0          # metri percorsi in avanti (somma con segno)
        self.best_progress = 0.0
        self.time = 0.0
        self.last_improve_time = 0.0
        self.laps = 0
        self.lap_start_time = 0.0
        self.last_lap_time = None
        self.lap_times = []
        self.crashed = False
        self.done = False
        self._sense()
        return self._obs()

    # ------------------------------------------------------------------
    def step(self, action):
        if self.done:
            raise RuntimeError("Episodio finito: chiama reset()")
        steer, throttle = float(action[0]), float(action[1])
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            self.car.step(steer, throttle, h)
        self.time += self.dt

        # progresso lungo il circuito (gestisce il passaggio sul traguardo)
        t = self.track
        self.seg, s, self.lateral = t.closest(self.car.pos, hint=self.seg)
        ds = s - self.prev_s
        if ds < -t.length / 2:
            ds += t.length
        elif ds > t.length / 2:
            ds -= t.length
        self.prev_s = s
        self.progress += ds

        reward = ds / 10.0 - self.time_penalty      # +1 ogni 10 m in avanti
        terminated = truncated = False
        lap_completed = False

        # giro completato?
        if self.progress >= (self.laps + 1) * t.length:
            self.laps += 1
            self.last_lap_time = self.time - self.lap_start_time
            self.lap_times.append(self.last_lap_time)
            self.lap_start_time = self.time
            lap_completed = True
            if self.laps >= self.laps_to_finish:
                terminated = True

        # collisione: un angolo della macchina fuori dai bordi
        self.crashed = any(abs(t.closest(c, hint=self.seg)[2]) > t.half_width
                           for c in self.car.corners())
        if self.crashed:
            reward -= self.crash_penalty
            terminated = True

        # macchina ferma o che non avanza: tronca l'episodio
        if self.progress > self.best_progress + 0.5:
            self.best_progress = self.progress
            self.last_improve_time = self.time
        if self.time - self.last_improve_time > self.stall_time or self.time >= self.max_time:
            truncated = not terminated

        self.done = terminated or truncated
        self._sense()
        info = {
            "crashed": self.crashed,
            "lap_completed": lap_completed,
            "laps": self.laps,
            "lap_time": self.last_lap_time,
            "current_lap_time": self.time - self.lap_start_time,
            "progress": self.progress / t.length,     # in giri
            "speed_kmh": self.car.speed * 3.6,
            "sliding": self.car.sliding,
            "time": self.time,
        }
        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _sense(self):
        h = self.car.heading + self.ray_angles
        self.ray_dirs = np.stack([np.cos(h), np.sin(h)], axis=1)
        self.ray_dist = self.track.raycast(self.car.pos, self.ray_dirs, self.max_range)

    def _obs(self):
        p = self.car.p
        return np.concatenate([
            self.ray_dist / self.max_range,
            [self.car.speed / p.max_speed, self.car.steer / p.max_steer],
        ]).astype(np.float32)
