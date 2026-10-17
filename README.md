# MTAS — Model Trainer for Autonomous Systems

A collection of experiments: in each one an agent learns to drive a simulated car.

## Experiment 1 — Car on a circuit (`experiments/car_circuit`)

Goal: complete the lap without ever touching the walls, in the shortest possible time.

### Installation (once)

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

### Start

```bash
cd experiments/car_circuit
python launcher.py
```

The launcher window lets you:

- **set up the circuit:** the classic hand-drawn one, or a random one generated from a **seed**, number of corners, length, width and twistiness. The same settings always give the same track, so a circuit can be shared just by writing down the numbers. A live preview shows the layout and the driving direction;
- **Play:** drive it yourself;
- **Train neuroevolution:** evolve a neural network on that track, with live progress and log;
- **Watch trained network:** see the best network drive. Every track keeps its own trained network.

In the game window: **Arrows / WASD** drive · **R** restart · **C** overview / close-up view that rotates with the car · **V** sensors on/off · **ESC** quit.

### Command line

Everything the launcher does is also available from the terminal. Every script accepts the same track options:
`--track classic|random --seed N --corners N --length M --width M --twistiness 0..1`.

```bash
python play.py --track random --seed 42          # drive yourself
python bot.py --render                           # watch the rule-based driver
python bot.py --laps 3                           # same driver without graphics: prints lap times
python neuroevo.py --track random --seed 42      # train (about 3 minutes for 150 generations)
python neuroevo.py --play --track random --seed 42   # watch the best network
python track.py --track random --seed 42         # print the stats of a track
```

### How it is built

| File | What it does |
|---|---|
| `launcher.py` | Start window: track settings with preview, buttons for play / train / watch |
| `track.py` | Circuit: control points → spline → centre line, edges, lap progress, ray sensors; seeded random track generator |
| `car.py` | Physics: bicycle model with a grip limit (too fast in a corner and the car slides wide) |
| `env.py` | Gymnasium-style environment: `reset()`, `step([steer, throttle])` → observation, reward, end of episode |
| `render.py` | Pygame graphics. Observer only: it never touches the simulation |
| `play.py` | Keyboard driving |
| `bot.py` | Rule-based driver (no learning): checks the environment and sets a time to beat |
| `neuroevo.py` | Neuroevolution: a population of neural networks that improves generation after generation |

**What the car sees (observation, 11 numbers):** 9 rays from right to left (distance to the wall, 0–1), speed (0–1), steering angle (−1…1).

**Controls (action, 2 numbers):** steering (−1 right … +1 left), throttle/brake (−1 full brake … +1 full throttle).

**Reward:** +1 every 10 m travelled forward, −0.01 per step (time costs), −10 for a crash (which ends the episode).
We never tell the car which line to follow: by rewarding metres per second, it has to discover the efficient racing line by itself.

### Random tracks

Control points are placed around a circle (counter-clockwise) with a random angle and radius;
higher twistiness means the radii vary more, so the layout gets more bends and hairpins.
The shape is stretched, rotated and scaled to the requested length, and rejected if a corner is
too tight or two parts of the track overlap; the next candidate from the same seeded generator is
then tried. Very dense settings (many corners on a short track) may not have a valid layout:
the launcher says so and suggests what to change.

### Neuroevolution

A population of 96 small neural networks (11 → 16 → 2, 226 weights) each drives one lap.
The best 6 pass unchanged to the next generation; the others are replaced by "children" of the best
(half the weights from one parent, half from the other, plus a little random noise).

Score: metres covered until the lap is completed; after that only the lap time counts.
The best network is saved in `checkpoints/neuroevo_<track id>.npz` and the progress in `runs/neuroevo_<track id>.csv`.

Training runs on the **CPU**, in parallel on every core: the time goes into the physics simulation, not into
the tiny network, so a GPU would not help (it would only add CPU↔GPU copies at every step).

### Results (classic circuit, 1.93 km, same simulation time step)

| Driver | Lap time |
|---|---|
| Rule-based (`bot.py`) | 1:37.3 |
| Neuroevolution, 150 generations (`neuroevo.py`) | **1:01.3** |

### Next steps

1. Reinforcement learning (PPO with Stable-Baselines3) through a Gymnasium wrapper.
2. Train on many random tracks at once, so the network learns to drive instead of memorising one layout.
