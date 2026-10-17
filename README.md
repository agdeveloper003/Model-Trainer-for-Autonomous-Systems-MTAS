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
python neuroevo.py       # allena le reti neurali (circa 3 minuti)
python neuroevo.py --play  # guarda guidare la rete migliore
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
| `neuroevo.py` | Neuroevoluzione: popolazione di reti neurali che migliora generazione dopo generazione |

**Cosa vede la macchina (osservazione, 11 numeri):** 9 raggi da destra a sinistra (distanza dal muro, 0–1), velocità (0–1), angolo di sterzo (−1…1).

**Comandi (azione, 2 numeri):** sterzo (−1 destra … +1 sinistra), gas/freno (−1 freno pieno … +1 gas pieno).

**Ricompensa:** +1 ogni 10 m percorsi in avanti, −0,01 per passo (il tempo costa), −10 se sbatte (e l'episodio finisce).
Non diciamo mai alla macchina quale traiettoria seguire: premiando i metri al secondo, la traiettoria efficiente deve scoprirla da sola.

### Neuroevoluzione

Una popolazione di 96 piccole reti neurali (11 → 16 → 2, 226 pesi) guida un giro ciascuna.
Le 6 migliori passano intatte alla generazione successiva; le altre vengono rimpiazzate da
"figli" delle migliori (metà pesi da un genitore, metà dall'altro, più un po' di rumore casuale).

Punteggio: metri percorsi finché non completa il giro; dopo conta solo il tempo sul giro.
La rete migliore viene salvata in `checkpoints/neuroevo_best.npz` e l'andamento in `runs/neuroevo_log.csv`.

### Risultati (1,93 km, stesso passo di simulazione)

| Pilota | Tempo sul giro |
|---|---|
| A regole (`bot.py`) | 1:37.3 |
| Neuroevoluzione, 150 generazioni (`neuroevo.py`) | **1:01.3** |

### Prossimi passi

1. Reinforcement learning (PPO con Stable-Baselines3) tramite un wrapper Gymnasium.
2. Generatore di circuiti casuali, per non far imparare a memoria un solo tracciato.
