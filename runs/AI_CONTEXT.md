# AI_CONTEXT.md — `r2dreamer/runs/`

> **Zweck:** KI-Kontext-Datei für den `runs/`-Ordner.  
> **Letztes Update:** 2026-06-20

---

## 1. Zweck des Ordners

`runs/` enthält alle **Shell-Skripte zum Starten von Trainingsläufen** für R2Dreamer. Es gibt zwei klar getrennte Kategorien:

| Kategorie | Skripte | Zweck |
|---|---|---|
| **FPV-Produktion** | `train_fpv_pipeline.sh` | 3-phasiges FPV-Drohnen-Training mit Multi-GPU + Checkpoint-Resume |
| **Benchmark/Research** | `dmc.sh`, `atari.sh`, `metaworld.sh`, `memorymaze.sh`, `dmc_subtle.sh`, `crafter.sh` | Reproduzierbare Evaluation auf Standard-RL-Benchmarks (single-GPU, Multi-Seed) |

Alle Skripte rufen ausschließlich **`train.py`** im Root von `r2dreamer/` auf — entweder direkt (`python train.py`) oder über `accelerate launch`.

---

## 2. Tech-Stack & Tools

| Tool | Verwendung |
|---|---|
| `accelerate` (HuggingFace) | Multi-GPU-Start in `train_fpv_pipeline.sh` |
| `torch.compile` | Aktiviert via `model.compile=True` in allen Skripten |
| MuJoCo + EGL | `MUJOCO_GL=egl` für headless GPU-Rendering (DMC, MetaWorld, MemoryMaze) |
| `tee` | Logging in `train_fpv_pipeline.sh` → `train_drone_script.log` |
| TCP-Socket-Check | Colosseum-Simulator-Erkennung (Port 41451) in Phase 3 |

---

## 3. Skript-Referenz

### 3.1 `train_fpv_pipeline.sh` — FPV-Produktionstraining (primär)

Das einzige Skript mit **Multi-GPU-Support**, **automatischer Checkpoint-Wiederherstellung** und **3-Phasen-Logik**.

```bash
# Aufruf:
./runs/train_fpv_pipeline.sh [phase1|phase2|phase3|all] [Extra-Args...]

# Beispiele:
./runs/train_fpv_pipeline.sh all
./runs/train_fpv_pipeline.sh phase3 trainer.steps=100000
./runs/train_fpv_pipeline.sh phase1 use_depth=False  # ohne Tiefenkanal
```

#### Konfiguration (oben im Skript anpassen)

| Variable | Default | Bedeutung |
|---|---|---|
| `MODEL_SIZE` | `size100M` | Modell-Config aus `configs/model/` |
| `LOGDIR_BASE` | `./logdir` | Basis-Verzeichnis für Checkpoints + Logs |
| `MIXED_PRECISION` | `fp16` | Half-Precision für `accelerate launch` |
| `STEPS_PHASE1` | `20000` | World-Model-Training (RSSM + Encoder) |
| `STEPS_PHASE2` | `5000` | BC + SafetyNet Pre-Training |
| `STEPS_PHASE3` | `50000` | Online RL mit Simulator |
| `NUM_GPUS` | Auto-detect | Wird zur Laufzeit via `torch.cuda.device_count()` gesetzt |

#### Phase-Logik & Checkpoint-Vererbung

```
Phase 1 → logdir/phase1/latest.pt
     └─► (wenn vorhanden: resume; sonst: fresh start)

Phase 2 → logdir/phase2/latest.pt
     └─► (wenn vorhanden: resume; sonst: erbt von phase1/latest.pt)
     └─► FEHLER wenn phase1/latest.pt fehlt

Phase 3 → logdir/phase3/latest.pt
     └─► (wenn vorhanden: resume; sonst: erbt von phase2/latest.pt)
     └─► FEHLER wenn phase2/latest.pt fehlt
     └─► Colosseum-Check: TCP-Port 41451 → env.colosseum.enabled=true/false
```

#### `accelerate launch`-Parameter

```bash
accelerate launch \
    --num_processes "$NUM_GPUS" \
    --mixed_precision "fp16" \
    --dynamo_backend "no" \   # kein torch.compile via accelerate (separat gehandelt)
    train.py ...
```

> **Hinweis:** `--dynamo_backend "no"` deaktiviert accelerate's eigenes Dynamo — `model.compile=True` wird von `train.py` intern gehandhabt.

#### Colosseum-Simulator-Erkennung (Phase 3)

```bash
python3 -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', 41451)); s.close()"
```

Port `41451` ist der Standard-Port von AirSim/Colosseum. Ist kein Simulator aktiv, setzt das Skript `env.colosseum.enabled=false` und Phase 3 läuft ohne Live-Simulator.

---

### 3.2 Benchmark-Skripte (Forschung / Evaluation)

Alle Benchmark-Skripte folgen dem gleichen Muster:

```bash
# Gemeinsame Parameter:
model.compile=True
device=cuda:0
buffer.storage_device=cuda:0
model.rep_loss=r2dreamer
seed=$seed  # 0, 100, 200, 300, 400 → 5 Seeds total

# Logdir-Namensschema:
logdir=logdir/${DATE}_r2dreamer_${task}_seed${seed}
# DATE = aktuelles Datum (MMTT)
```

| Skript | Env-Config | Model | MuJoCo-GL | Aufgaben |
|---|---|---|---|---|
| `dmc.sh` | `dmc_vision` | Default | `egl` | 20 DMC-Tasks |
| `dmc_subtle.sh` | `dmc_vision` | Default | `egl` | 5 DMC-Subtle-Tasks |
| `metaworld.sh` | `metaworld` | Default | `egl` | 50 MetaWorld-Tasks |
| `memorymaze.sh` | `memorymaze` | Default | `egl` | 4 MemoryMaze-Sizes |
| `atari.sh` | `atari100k` | `size200M` | — | 26 Atari-Tasks |
| `crafter.sh` | `crafter` | `size200M` | — | Crafter |

> **Atari und Crafter erzwingen `model=size200M`** — diese Benchmarks brauchen mehr Kapazität als der Default.

#### Vollständige Task-Listen

**DMC (20 Tasks):** acrobot_swingup, ball_in_cup_catch, cartpole_balance, cartpole_balance_sparse, cartpole_swingup, cartpole_swingup_sparse, cheetah_run, finger_spin, finger_turn_easy, finger_turn_hard, hopper_hop, hopper_stand, pendulum_swingup, quadruped_run, quadruped_walk, reacher_easy, reacher_hard, walker_run, walker_stand, walker_walk

**DMC Subtle (5 Tasks):** ball_in_cup_catch_subtle, cartpole_swingup_subtle, finger_turn_subtle, point_mass_subtle, reacher_subtle

**Atari-100k (26 Tasks):** alien, amidar, assault, asterix, bank_heist, battle_zone, boxing, breakout, chopper_command, crazy_climber, demon_attack, freeway, frostbite, gopher, hero, jamesbond, kangaroo, krull, kung_fu_master, ms_pacman, pong, private_eye, qbert, road_runner, seaquest, up_n_down

**MetaWorld (50 Tasks):** assembly, basketball, bin-picking, box-close, button-press-topdown, button-press-topdown-wall, button-press, button-press-wall, coffee-button, coffee-pull, coffee-push, dial-turn, disassemble, door-close, door-lock, door-open, door-unlock, hand-insert, drawer-close, drawer-open, faucet-open, faucet-close, hammer, handle-press-side, handle-press, handle-pull-side, handle-pull, lever-pull, pick-place-wall, pick-out-of-hole, pick-place, plate-slide, plate-slide-side, plate-slide-back, plate-slide-back-side, peg-insert-side, peg-unplug-side, soccer, stick-push, stick-pull, push, push-wall, push-back, reach, reach-wall, shelf-place, sweep-into, sweep, window-open, window-close

**MemoryMaze (4 Tasks):** 9x9, 11x11, 13x13, 15x15

---

## 4. Wichtige Abhängigkeiten

| Abhängigkeit | Verwendung in `runs/` |
|---|---|
| `r2dreamer/train.py` | Einziger Einstiegspunkt für alle Skripte |
| `r2dreamer/configs/model/size100M.yaml` | Standard-Modellgröße für FPV-Skript |
| `r2dreamer/configs/model/size200M.yaml` | Erzwungen für Atari + Crafter |
| `r2dreamer/configs/env/` | Jedes `env=xyz`-Argument matcht eine YAML-Datei dort |
| `logdir/phase{1,2,3}/latest.pt` | Checkpoint-Pfade für `train_fpv_pipeline.sh` |
| AirSim/Colosseum (Port 41451) | Optional für Phase 3; Fallback: kein Simulator |

---

## 5. Limitierungen & Fallstricke

### ❌ VERBOTEN

| Aktion | Grund |
|---|---|
| Phase 2 ohne Phase-1-Checkpoint starten | Skript bricht mit `exit 1` ab (`[[ -z "$p1_ckpt" ]]`) |
| Phase 3 ohne Phase-2-Checkpoint starten | Gleicher Fehler-Check |
| `train_fpv_pipeline.sh` ohne Argument aufrufen | Gibt Usage-Meldung + `exit 1` |
| `MODEL_SIZE` auf `size200M` für FPV setzen | Kein RAM-Budget-Problem, aber 200M wurde nie für FPV validiert |

### ⚠️ EDGE CASES & GOTCHAS

- **Multi-GPU-Erkennung:** `NUM_GPUS` wird zur Laufzeit via Python ermittelt (`torch.cuda.device_count()`). Der Top-Wert im Skript (`NUM_GPUS=2`) ist ein Fallback, der überschrieben wird. Auf Single-GPU-Systemen läuft `accelerate launch --num_processes 1` normal.

- **`set -euo pipefail`** in `train_fpv_pipeline.sh`: Jeder Fehler bricht das Skript sofort ab. Das bedeutet:
  - Schlägt `pip install` fehl → Stop
  - Schlägt Phase 1 fehl → Phase 2 startet nie
  - Verwende `phase3` direkt (statt `all`) wenn Phase 1+2 bereits abgeschlossen sind

- **Checkpoint-Resume:** `get_checkpoint()` sucht exakt nach `${logdir}/latest.pt`. Wenn `train.py` den Checkpoint anders benennt (z.B. `checkpoint_20000.pt`), greift das Resume nicht. Das Namens-Matching ist hartkodiert.

- **`tee`-Logging:** `exec > >(tee -i train_drone_script.log) 2>&1` schreibt in `./train_drone_script.log` (relativ zum CWD beim Skript-Aufruf, nicht relativ zum Skript selbst). Das Skript sollte aus `r2dreamer/` heraus aufgerufen werden.

- **MuJoCo EGL:** `MUJOCO_GL=egl` und `MUJOCO_EGL_DEVICE_ID=$GPU_ID` werden in den DMC/MetaWorld/MemoryMaze-Skripten gesetzt — **nicht** in Atari und Crafter (die kein MuJoCo brauchen). Vergisst man `MUJOCO_GL=egl` auf einem headless-Server, stürzt MuJoCo mit `DISPLAY not set`-Fehler ab.

- **Benchmark-Skripte laufen sequenziell:** Task A → Seed 0, 100, 200, 300, 400, dann Task B → Seed 0... usw. Für parallele Experimente auf mehreren GPUs muss `GPU_ID` manuell überschrieben oder das Skript geteilt werden.

- **`--dynamo_backend "no"` in `accelerate`:** Verhindert Konflikte zwischen accelerate's Dynamo-Integration und dem internen `torch.compile` in `train.py`. Niemals `--dynamo_backend "inductor"` setzen ohne `model.compile=False` in `train.py`.

- **Colosseum Port 41451:** Ist der Port besetzt (andere Prozesse), kann es zu einem False-Positive kommen (`colosseum="true"` obwohl kein Simulator läuft). Vor Phase-3-Start den Simulator-Status manuell prüfen.
