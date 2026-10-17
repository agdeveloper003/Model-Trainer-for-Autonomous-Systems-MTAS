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
python app.py
```

Everything happens in one window:

1. **Setup** — choose the circuit: the classic hand-drawn one, or a random one built from a **seed**,
   number of corners, length, width and twistiness. The same settings always give the same track, so a
   circuit can be shared just by writing down the numbers. Circuits can be driven in both directions.
   Then press one of the two big buttons:
   - **PLAY** — drive it yourself;
   - **NEUROEVOLUTION** — train neural networks, live, in one of two modes:
     - *This track*: every network drives the circuit set up here;
     - *Generalist*: every generation brings new random circuits (6, 8 or 10), so the networks have
       to learn to drive instead of memorising a layout.

   The *Trained models* box lets you run any saved model on the track currently set up, also one it
   has never seen, to check whether it learned to *drive* or only memorised its own circuit.
2. **Drive** — you (arrows / WASD) or a trained model drive the circuit. With a model, its neural
   network is shown live next to the map.
3. **Training** — every car of the population drives on the map at the same time
   (blue = running, grey = stopped, dark red = crashed; the leader is highlighted with its sensors).
   On the right: the leader's neural network (weights as coloured lines, neurons lit by their live
   activation) and a chart of the progress per generation. Speed 1x / 4x / 16x / Max, pause, stop.
   In *Generalist* mode the map becomes an N×N grid: one circuit large with the whole population,
   the others small with only the leader's car and the best lap of the generation (click one to
   enlarge it).
   When training ends you can watch the best model or try it on another track.

Keys: **C** overview / close-up view that rotates with the car · **V** sensors on/off · **R** restart ·
**Space** pause training · **Esc** back to the menu.

### Command line

```bash
python neuroevo.py --track random --seed 42      # headless training on one track (same results as the app)
python neuroevo.py --general --tracks 8          # headless generalist training
python neuroevo.py --general --device gpu        # ... with the simulation on the GPU
python benchmark.py                              # every saved model on the validation circuits
python bot.py --track random --seed 42           # rule-based driver: prints a reference lap time
python track.py --track random --seed 42         # print the stats of a track
```

Every script accepts the same track options:
`--track classic|random --seed N --corners N --length M --width M --twistiness 0..1 --clockwise`.

### How it is built

| File | What it does |
|---|---|
| `app.py` | The app: one window with the Setup, Drive and Training screens |
| `ui.py` | Small pygame UI toolkit: theme, buttons, sliders, number field, switches |
| `render.py` | Track view: draws the circuit and any number of cars (overview or rotating close-up) |
| `netviz.py` | Drawings of the learning: neural network with weights and activations, training chart |
| `trainer.py` | Training engine: worker processes drive the population on one or many circuits, in step |
| `neuroevo.py` | Network, scores, genetic algorithm, validation set, saved models; command-line training |
| `benchmark.py` | Measures saved models on circuits they have never seen, against the rule-based bot |
| `gpusim.py` | The same simulation for every car at once, on the GPU (PyTorch + a CUDA kernel) |
| `track.py` | Circuit: control points → spline → centre line, edges, lap progress, ray sensors; seeded random track generator |
| `car.py` | Physics: bicycle model with a grip limit (too fast in a corner and the car slides wide) |
| `env.py` | Gymnasium-style environment: `reset()`, `step([steer, throttle])` → observation, reward, end of episode |
| `bot.py` | Rule-based driver (no learning): checks the environment and sets a time to beat |

**What the car sees (observation, 11 numbers):** 9 rays from right to left (distance to the wall, 0–1), speed (0–1), steering angle (−1…1).

**Controls (action, 2 numbers):** steering (−1 right … +1 left), throttle/brake (−1 full brake … +1 full throttle).

**Reward:** +1 every 10 m travelled forward, −0.01 per step (time costs), −10 for a crash (which ends the episode).
We never tell the car which line to follow: by rewarding metres per second, it has to discover the efficient racing line by itself.

### Random tracks

Control points are placed around a circle (reversed for a clockwise track) with a random angle and radius;
higher twistiness means the radii vary more, so the layout gets more bends and hairpins.
The shape is stretched, rotated and scaled to the requested length, and rejected if a corner is
too tight or two parts of the track overlap; the next candidate from the same seeded generator is
then tried. Very dense settings (many corners on a short track) may not have a valid layout:
the setup screen says so and suggests what to change.

### Neuroevolution

A population of 96 small neural networks (11 → 16 → 2, 226 weights) each drives one lap.
The best 6 pass unchanged to the next generation; the others are replaced by "children" of the best
(half the weights from one parent, half from the other, plus a little random noise).

Score on a circuit: the share of the lap covered (0–1); a completed lap scores 1 + average speed / top
speed, so any completed lap beats any incomplete one and faster is better. It does not depend on the
circuit's length, so scores of different circuits can be averaged.

- **This track:** fitness = score on that circuit. The best network survives unchanged and the circuit
  never changes, so the best score can only stay equal or improve from one generation to the next.
- **Generalist:** fitness = average score over K new random circuits (both directions) every generation.
  Every 5 generations the 3 best networks drive the **validation circuits** — 12 random ones plus the
  classic circuit, fixed and never used for training — and the model is saved only when it completes
  more of them (ties: higher average score). This measures driving, not memory.

Models are saved in `checkpoints/neuroevo_<track id>.npz` (generalist: `neuroevo_general-k<K>.npz`);
the command-line trainer also logs the progress in `runs/`.

### Compute: CPU or GPU

The time goes into simulating the cars (physics, sensors, track), not into the tiny network. Two engines,
same rules (choose in the setup screen, or `--device cpu|gpu` on the command line):

- **CPU** — one worker process per core, each driving a slice of the population one car at a time
  (`env.py`). Best for *This track* (~100 cars: too few to keep a GPU busy).
- **GPU** — `gpusim.py`: every car of the generation at once, as arrays in PyTorch on the graphics card.
  Best for *Generalist* (networks × circuits = hundreds or thousands of cars). Three tricks make it fast:
  each car only tests the walls within sensor range (listed in advance for every point of the track),
  one simulation step is recorded once as a CUDA graph and replayed, and the sensors run in a custom CUDA
  kernel compiled at start-up with NVRTC (shipped with PyTorch, no CUDA toolkit needed).

The GPU simulation is checked against the CPU one: on 624 episodes (48 networks × 13 circuits, including
crashes and stalls) the outcome is the same in 624/624, lap times identical or within one time step.
Bit-for-bit equality is not possible (the GPU's maths library rounds a few functions differently in the
last digit), so a training run on the GPU follows a slightly different path than the same run on the CPU.
Everything stays in float64 like the CPU version: in float32 a few borderline cars crash or survive differently.

| Measured on 32 cores + RTX 5070 Ti | car-steps per second |
|---|---|
| CPU, 30 processes | 0.10 M |
| GPU, 1,000 cars | ~1.7 M |
| GPU, 5,000+ cars | ~5 M |

Generalist training, 150 generations, 8 circuits, population 96: **40 min on the CPU, 8.4 min on the GPU**
(the GPU is not full with 768 cars, and at the end of each generation only a few cars are still running).
A model trained on the GPU drives the same in the CPU simulation used by the app (table below).

The GPU needs PyTorch with CUDA (`pip install torch` from pytorch.org); without it the app uses the CPU.

### Results (classic circuit, 1.93 km, same simulation time step)

| Driver | Lap time |
|---|---|
| Rule-based (`bot.py`) | 1:37.3 |
| Neuroevolution, 150 generations (`neuroevo.py`) | **1:01.3** |
| Generalist, 150 generations, never trained on it | 1:03.1 |

### Results on circuits never seen in training

Generalist: 150 generations, 8 new random circuits per generation, population 96 (about 40 minutes on 32 cores).

| Driver | Validation circuits (13) | Test circuits (30 new ones) | Lap time vs bot |
|---|---|---|---|
| Rule-based bot | 100% | 97% | 1.00 |
| Generalist (`general-k8`, trained on CPU) | **100%**, no crash | **100%**, no crash | **0.56–0.59** |
| Generalist (same settings, trained on GPU) | **100%**, no crash | **100%**, no crash | 0.57–0.61 |
| Specialists (trained on one circuit) | 31–85% | – | 0.55–0.80 where they finish |

The validation circuits also choose which model is saved, so the test circuits (a third set, used only
for this table) are the honest measure.

### Next steps

1. Reinforcement learning (PPO with Stable-Baselines3) through a Gymnasium wrapper.
2. Use the GPU's speed: bigger populations and more circuits per generation.
3. Richer inputs (more forward rays, slip) and better optimisers (CMA-ES), measured on the validation circuits.
