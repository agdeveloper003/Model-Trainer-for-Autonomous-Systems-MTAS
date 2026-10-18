# MTAS — Model Trainer for Autonomous Systems

MTAS is a collection of projects in which simulated autonomous systems **learn to act by themselves**.
Each project builds a simulation of one system — its physics, its sensors, its goal — and trains
agents that control it only from what their sensors perceive, without being told how.

Every project lives in its own folder under [`projects/`](projects/), with its own README, its own
requirements and its own code, and can be used on its own.

## Projects

| Project | What learns | How | Status |
|---|---|---|---|
| [**Self-Driving Track Car**](projects/self-driving-track-car/) | A car driving race tracks from 9 distance sensors, fast and without touching the walls — also on tracks it has never seen | Neuroevolution of small neural networks, on CPU or GPU; live training app | Active |

## Repository layout

```
MTAS/
├── README.md            this overview
├── requirements.txt     installs the dependencies of every project
└── projects/
    └── <project-name>/
        ├── README.md        what the project does and how to use it
        ├── requirements.txt its own dependencies
        └── ...              code
```

Trained models and training logs (`checkpoints/`, `runs/` inside each project) stay on your computer
and are not uploaded.

## Getting started

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt                              # every project
# or only one:  pip install -r projects/<project-name>/requirements.txt
```

Then open the project's folder and follow its README.

## Adding a project

1. Create `projects/<project-name>/` with the code, a `README.md` and a `requirements.txt`.
2. Add a row to the **Projects** table above.
3. Add a line `-r projects/<project-name>/requirements.txt` to the root `requirements.txt`.
