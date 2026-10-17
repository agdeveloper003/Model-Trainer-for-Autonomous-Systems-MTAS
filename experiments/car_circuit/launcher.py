"""
MTAS launcher: choose the circuit, then drive it or train a neural network on it.

    python launcher.py

The left side sets up the track. A random track is fully defined by its
settings and seed: the same values always give the same circuit, so a
result can be reproduced or shared just by writing down the numbers.

Play, training and replay each run in their own process: the pygame
windows and the training (which uses every CPU core) never freeze this
window.
"""
import os
import queue
import random
import re
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from neuroevo import load_checkpoint
from track import TrackSpec

def fmt_time(t):
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:05.2f}"


HERE = os.path.dirname(os.path.abspath(__file__))
NO_CONSOLE = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

BG = "#121418"
GRASS = "#365c34"
ASPHALT = "#3a3c42"
EDGE = "#c4342e"
TEXT_BAD = "#c0392b"


class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MTAS - Model Trainer for Autonomous Systems")
        self.minsize(980, 640)
        self.track = None
        self.spec = None
        self.train_proc = None
        self.stop_file = None
        self.lines = queue.Queue()
        self._rebuild_job = None

        self._build_ui()
        self._schedule_rebuild()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_training)

    # ================================================================ layout
    def _build_ui(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 12))

        # ---------------------------------------------------------- track
        tf = ttk.LabelFrame(left, text="Track", padding=10)
        tf.pack(fill="x")
        d = TrackSpec(kind="random")
        lim = TrackSpec.LIMITS
        self.kind = tk.StringVar(value="random")
        self.seed = tk.StringVar(value=str(d.seed))
        self.corners = tk.IntVar(value=d.corners)
        self.length = tk.IntVar(value=int(d.length))
        self.width = tk.IntVar(value=int(d.width))
        self.twist = tk.DoubleVar(value=d.twistiness)

        ttk.Radiobutton(tf, text="Random circuit", value="random", variable=self.kind,
                        command=self._on_kind).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(tf, text="Classic circuit (hand drawn)", value="classic", variable=self.kind,
                        command=self._on_kind).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 8))

        self.random_widgets = []

        def row(r, label, widget, extra=None):
            lab = ttk.Label(tf, text=label)
            lab.grid(row=r, column=0, sticky="w", pady=3)
            widget.grid(row=r, column=1, sticky="ew", pady=3)
            self.random_widgets.append(widget)
            if extra:
                extra.grid(row=r, column=2, sticky="w", padx=(6, 0))
                self.random_widgets.append(extra)

        row(2, "Seed", ttk.Entry(tf, textvariable=self.seed, width=12),
            ttk.Button(tf, text="Random", width=8, command=self._random_seed))
        row(3, "Corners", ttk.Spinbox(tf, from_=lim["corners"][0], to=lim["corners"][1],
                                      textvariable=self.corners, width=10))
        row(4, "Length (m)", ttk.Spinbox(tf, from_=lim["length"][0], to=lim["length"][1], increment=100,
                                         textvariable=self.length, width=10))
        row(5, "Width (m)", ttk.Spinbox(tf, from_=lim["width"][0], to=lim["width"][1],
                                        textvariable=self.width, width=10))
        self.twist_label = ttk.Label(tf, width=5)
        row(6, "Twistiness", ttk.Scale(tf, from_=0.0, to=1.0, variable=self.twist,
                                       command=lambda _v: self._on_twist()), self.twist_label)
        self._on_twist(rebuild=False)
        tf.columnconfigure(1, weight=1)

        self.track_info = ttk.Label(tf, text="", wraplength=280)
        self.track_info.grid(row=7, column=0, columnspan=3, sticky="w", pady=(8, 0))

        for v in (self.seed, self.corners, self.length, self.width):
            v.trace_add("write", lambda *_: self._schedule_rebuild())

        # ------------------------------------------------- neuroevolution
        nf = ttk.LabelFrame(left, text="Neuroevolution", padding=10)
        nf.pack(fill="x", pady=(12, 0))
        self.generations = tk.IntVar(value=150)
        self.population = tk.IntVar(value=96)
        self.workers = tk.IntVar(value=os.cpu_count() or 4)
        for r, (label, var, lo, hi, inc) in enumerate([
                ("Generations", self.generations, 10, 5000, 10),
                ("Population", self.population, 16, 1000, 16),
                ("CPU processes", self.workers, 1, os.cpu_count() or 4, 1)]):
            ttk.Label(nf, text=label).grid(row=r, column=0, sticky="w", pady=3)
            ttk.Spinbox(nf, from_=lo, to=hi, increment=inc, textvariable=var, width=10).grid(
                row=r, column=1, sticky="ew", pady=3)
        nf.columnconfigure(1, weight=1)
        self.ckpt_info = ttk.Label(nf, text="", wraplength=280)
        self.ckpt_info.grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # -------------------------------------------------------- buttons
        bf = ttk.Frame(left)
        bf.pack(fill="x", pady=(14, 0))
        self.btn_play = ttk.Button(bf, text="Play  (drive it yourself)", command=self._play)
        self.btn_train = ttk.Button(bf, text="Train neuroevolution", command=self._train)
        self.btn_watch = ttk.Button(bf, text="Watch trained network", command=self._watch)
        self.btn_stop = ttk.Button(bf, text="Stop training", command=self._stop)
        for b in (self.btn_play, self.btn_train, self.btn_watch, self.btn_stop):
            b.pack(fill="x", pady=3, ipady=4)

        # ------------------------------------------------- preview + log
        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=3)
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(right, bg=BG, highlightthickness=0, width=560, height=420)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _e: self._draw_preview())

        self.progress = ttk.Progressbar(right, mode="determinate")
        self.progress.grid(row=1, column=0, sticky="ew", pady=(10, 4))
        logf = ttk.Frame(right)
        logf.grid(row=2, column=0, sticky="nsew")
        logf.rowconfigure(0, weight=1)
        logf.columnconfigure(0, weight=1)
        self.log = tk.Text(logf, height=9, bg=BG, fg="#d0d4dc", insertbackground="#d0d4dc",
                           font=("Consolas", 9), relief="flat", state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.log["yscrollcommand"] = sb.set

        self._update_buttons()

    # ========================================================= track setup
    def _on_kind(self):
        state = ["!disabled"] if self.kind.get() == "random" else ["disabled"]
        for w in self.random_widgets:
            w.state(state)
        self._schedule_rebuild()

    def _on_twist(self, rebuild=True):
        v = round(self.twist.get() / 0.05) * 0.05
        self.twist_label.configure(text=f"{v:.2f}")
        if rebuild:
            self._schedule_rebuild()

    def _random_seed(self):
        self.seed.set(str(random.randint(0, 999_999)))

    def _read_spec(self):
        if self.kind.get() == "classic":
            return TrackSpec(kind="classic")
        try:
            return TrackSpec(kind="random", seed=int(self.seed.get()), corners=int(self.corners.get()),
                             length=float(self.length.get()), width=float(self.width.get()),
                             twistiness=round(self.twist.get() / 0.05) * 0.05).clamped()
        except (ValueError, tk.TclError):
            raise ValueError("Seed, corners, length and width must be whole numbers.")

    def _schedule_rebuild(self):
        if self._rebuild_job:
            self.after_cancel(self._rebuild_job)
        self._rebuild_job = self.after(250, self._rebuild)

    def _rebuild(self):
        self._rebuild_job = None
        try:
            self.spec = self._read_spec()
            self.track = self.spec.build()
            self.error = None
            t = self.track
            self.track_info.configure(
                foreground="", text=f"{t.name}: {t.length / 1000:.2f} km, {t.width:g} m wide, "
                                    f"tightest corner radius {t.min_curve_radius():.0f} m")
        except ValueError as e:
            self.spec, self.track, self.error = None, None, str(e)
            self.track_info.configure(foreground=TEXT_BAD, text=str(e))
        self._draw_preview()
        self._refresh_checkpoint()
        self._update_buttons()

    def _refresh_checkpoint(self):
        data = load_checkpoint(self.spec) if self.spec else None
        self.has_ckpt = data is not None
        if data is None:
            self.ckpt_info.configure(text="No trained network for this track yet.")
        elif data["lap_time"] > 0:
            self.ckpt_info.configure(text=f"Trained network: best lap {fmt_time(data['lap_time'])} "
                                          f"after {data['generations']} generations.")
        else:
            self.ckpt_info.configure(text=f"Trained network: no full lap yet "
                                          f"({data['generations']} generations).")

    # ============================================================= preview
    def _draw_preview(self):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if self.track is None:
            c.create_text(w / 2, h / 2, text=getattr(self, "error", "") or "", fill="#ff7a6e",
                          width=w - 60, font=("Segoe UI", 11))
            return
        t = self.track
        lo, hi = t.bounds()
        pad = 24
        s = min((w - 2 * pad) / (hi[0] - lo[0]), (h - 2 * pad) / (hi[1] - lo[1]))
        ox = w / 2 - (lo[0] + hi[0]) / 2 * s
        oy = h / 2 + (lo[1] + hi[1]) / 2 * s

        def pts(p):
            p = np.vstack([p, p[:1]])
            return np.stack([ox + p[:, 0] * s, oy - p[:, 1] * s], axis=1).ravel().tolist()

        c.create_rectangle(0, 0, w, h, fill=GRASS, outline="")
        c.create_line(pts(t.center), fill=EDGE, width=max(2, t.width * s + 3), joinstyle="round")
        c.create_line(pts(t.center), fill=ASPHALT, width=max(1, t.width * s), joinstyle="round")

        # start line and driving direction
        p0, n0 = t.center[0], t.normal[0]
        a, b = p0 + n0 * t.half_width * 1.6, p0 - n0 * t.half_width * 1.6
        c.create_line(ox + a[0] * s, oy - a[1] * s, ox + b[0] * s, oy - b[1] * s, fill="white", width=3)
        k = min(t.N - 1, max(3, int(60 / (t.length / t.N))))
        arrow = t.center[:k + 1]
        c.create_line(np.stack([ox + arrow[:, 0] * s, oy - arrow[:, 1] * s], axis=1).ravel().tolist(),
                      fill="white", width=2, arrow="last", arrowshape=(12, 14, 5))
        c.create_text(12, 12, anchor="nw", fill="white", font=("Segoe UI", 11, "bold"),
                      text=f"{t.name}  ·  {t.length / 1000:.2f} km")

    # ============================================================ launching
    def _run(self, script, extra=()):
        cmd = [sys.executable, "-u", os.path.join(HERE, script), *self.spec.to_args(), *extra]
        return cmd

    def _play(self):
        if self.spec:
            subprocess.Popen(self._run("play.py"), cwd=HERE, creationflags=NO_CONSOLE)

    def _watch(self):
        if self.spec and self.has_ckpt:
            subprocess.Popen(self._run("neuroevo.py", ["--play"]), cwd=HERE, creationflags=NO_CONSOLE)

    def _train(self):
        if not self.spec or self.training:
            return
        try:
            gens, pop, workers = int(self.generations.get()), int(self.population.get()), int(self.workers.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("MTAS", "Generations, population and CPU processes must be whole numbers.")
            return
        if self.has_ckpt and not messagebox.askyesno(
                "MTAS", "This track already has a trained network.\n"
                        "Training again replaces it with the new result. Continue?"):
            return
        fd, self.stop_file = tempfile.mkstemp(prefix="mtas_stop_")
        os.close(fd)
        os.remove(self.stop_file)                      # training stops when this file appears
        cmd = self._run("neuroevo.py", ["--generations", str(gens), "--pop", str(pop),
                                        "--workers", str(workers), "--stop-file", self.stop_file])
        self._log_clear()
        self.progress.configure(maximum=gens, value=0)
        self.train_proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", errors="replace",
                                           creationflags=NO_CONSOLE)
        threading.Thread(target=self._reader, args=(self.train_proc,), daemon=True).start()
        self._update_buttons()

    def _reader(self, proc):
        for line in proc.stdout:
            self.lines.put(line.rstrip("\n"))
        proc.wait()
        self.lines.put(None)                           # end of training

    def _poll_training(self):
        try:
            while True:
                line = self.lines.get_nowait()
                if line is None:
                    self._training_finished()
                    continue
                m = re.match(r"gen\s+(\d+)/(\d+)", line)
                if m:
                    self.progress.configure(value=int(m.group(1)))
                self._log_append(line)
        except queue.Empty:
            pass
        self.after(100, self._poll_training)

    def _training_finished(self):
        code = self.train_proc.returncode if self.train_proc else 0
        if code not in (0, None):
            self._log_append(f"\nTraining exited with error code {code}.")
        self.train_proc = None
        if self.stop_file and os.path.exists(self.stop_file):
            os.remove(self.stop_file)
        self.stop_file = None
        self._refresh_checkpoint()
        self._update_buttons()

    def _stop(self):
        if self.training and self.stop_file:
            open(self.stop_file, "w").close()          # stops cleanly after the current generation
            self._log_append("Stopping after this generation...")
            self.btn_stop.configure(state="disabled")

    @property
    def training(self):
        return self.train_proc is not None

    def _update_buttons(self):
        ok = self.track is not None
        self.btn_play.configure(state="normal" if ok else "disabled")
        self.btn_train.configure(state="normal" if ok and not self.training else "disabled")
        self.btn_watch.configure(state="normal" if ok and getattr(self, "has_ckpt", False) else "disabled")
        self.btn_stop.configure(state="normal" if self.training else "disabled")

    # ================================================================= log
    def _log_append(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _log_clear(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _on_close(self):
        if self.training:
            if not messagebox.askyesno("MTAS", "Training is running. Stop it and quit?"):
                return
            if sys.platform == "win32":                # also ends the training's worker processes
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.train_proc.pid)],
                               capture_output=True, creationflags=NO_CONSOLE)
            else:
                self.train_proc.terminate()
        self.destroy()


if __name__ == "__main__":
    Launcher().mainloop()
