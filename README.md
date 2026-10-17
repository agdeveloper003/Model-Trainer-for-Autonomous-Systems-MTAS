# MTAS — Model Trainer for Autonomous Systems

Raccolta di esperimenti: in ognuno un agente impara a guidare una macchina simulata.

## Esperimento 1 — Auto su circuito (`experiments/car_circuit`)

Obiettivo: completare il giro senza mai toccare i muri, nel minor tempo possibile.

### Installazione (una volta sola)

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

### Comandi

```bash
cd experiments/car_circuit
python play.py           # guidi tu (frecce o WASD)
python bot.py --render   # guarda il pilota a regole
python bot.py --laps 3   # stesso pilota senza grafica: stampa i tempi
```

In finestra: **R** riparti · **C** vista d'insieme / vista che segue la macchina · **V** sensori on/off · **ESC** esci.

### Come è fatto

| File | Cosa fa |
|---|---|
| `track.py` | Circuito: punti di controllo → spline → linea centrale, bordi, progresso sul giro, sensori a raggi |
| `car.py` | Fisica: modello a bicicletta con limite di aderenza (in curva troppo veloce la macchina scivola fuori) |
| `env.py` | Ambiente stile Gymnasium: `reset()`, `step([sterzo, gas])` → osservazione, ricompensa, fine episodio |
| `render.py` | Grafica pygame. Solo osservatore: non tocca la simulazione |
| `play.py` | Guida da tastiera |
| `bot.py` | Pilota a regole (nessun apprendimento): verifica l'ambiente e dà un tempo da battere |

**Cosa vede la macchina (osservazione, 11 numeri):** 9 raggi da destra a sinistra (distanza dal muro, 0–1), velocità (0–1), angolo di sterzo (−1…1).

**Comandi (azione, 2 numeri):** sterzo (−1 destra … +1 sinistra), gas/freno (−1 freno pieno … +1 gas pieno).

**Ricompensa:** +1 ogni 10 m percorsi in avanti, −0,01 per passo (il tempo costa), −10 se sbatte (e l'episodio finisce).
Non diciamo mai alla macchina quale traiettoria seguire: premiando i metri al secondo, la traiettoria efficiente deve scoprirla da sola.

### Riferimento attuale

Pilota a regole: **~1:36** al giro (1,93 km). È il tempo da battere con il machine learning.

### Prossimi passi

1. Neuroevoluzione: una popolazione di piccole reti neurali, si tengono le migliori e si mutano.
2. Reinforcement learning (PPO con Stable-Baselines3) tramite un wrapper Gymnasium.
3. Generatore di circuiti casuali, per non far imparare a memoria un solo tracciato.
