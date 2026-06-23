# AI_CONTEXT.md — `r2dreamer/docs/`

> **Zweck:** KI-Kontext-Datei für den `docs/`-Ordner.  
> **Letztes Update:** 2026-06-20

---

## 1. Zweck des Ordners

`docs/` ist die **entwicklerbezogene Referenz-Dokumentation** für R2Dreamer. Sie enthält keine ausführbaren Dateien und ist ausschließlich für Menschen (und KIs) gedacht, die den Quellcode lesen oder erweitern.

Aktuell enthält der Ordner zwei Dateien:

| Datei | Inhalt |
|---|---|
| `tensor_shapes.md` | Authoritative Kürzel-Legende für alle Tensor-Shape-Annotationen im Codebase |
| `docker.md` | Build- und Run-Workflow für das containerisierte Training mit GPU-Support |

---

## 2. Tech-Stack & Tools

| Tool | Verwendung |
|---|---|
| Docker | Containerisiertes Training; Basis-Image aus `r2dreamer/Dockerfile` |
| NVIDIA Container Toolkit | GPU-Weiterleitung in den Container (`--gpus=all`) |
| TensorBoard | Training-Monitoring via `localhost:6006` (läuft im Container) |

---

## 3. Tensor-Shape-Legende (`tensor_shapes.md`)

Diese Symbole werden als Inline-Kommentare **überall im Codebase** verwendet — in `rssm.py`, `networks.py`, `dreamer.py`, `trainer.py` usw. Sie sind die **einzige autoritative Quelle** für deren Bedeutung.

| Symbol | Bedeutung | Beispiel-Wert (100M-Config) |
|---|---|---|
| `B` | Batch-Größe | 16 (Replay-Batch) |
| `T` | Zeitschritte / Sequenzlänge | 64 (Trainings-Horizon) |
| `T_imag` | Imagination-Horizon (Latent-Rollout im Actor-Critic) | 15 |
| `A` | Action-Dimension | 4 (FPV: roll, pitch, yaw, throttle) |
| `E` | Encoder-Embedding-Dimension | Ausgabe des FPN-Encoders |
| `F` | RSSM-Feature-Dimension = `S*K + D` | z.B. 32×32 + 512 = 1536 |
| `S` | Stochastische Gruppen (RSSM) | 32 |
| `K` | Diskrete Kategorien pro Gruppe (RSSM) | 32 |
| `D` | Deterministische Zustands-Dimension (RSSM GRU) | 512 |
| `G` | Anzahl Groups/Blocks (z.B. BlockLinear) | variabel |
| `U` | Hidden Units (generische Zwischen-Dimension) | 512 / 1024 |
| `H`, `W` | Bild-Höhe und -Breite (Eingangsauflösung) | 288, 512 (RSSM) |
| `C` | Bild-Kanäle | 3 (RGB) |
| `H_feat`, `W_feat`, `C_feat` | Feature-Map-Dimensionen innerhalb CNN/FPN | variabel |

### Kritische Beziehungen

```
F = S * K + D          # RSSM Feature Vector = Stochastic + Deterministic
                       # Wer F als Eingabe braucht, braucht S, K, D aus der Config

actor_input = F + E    # Policy-Netz: RSSM-Features + Encoder-Embedding (in manchen Configs)
```

---

## 4. Docker-Workflow (`docker.md`)

### Build

```bash
# Aus dem Repo-Root (wo Dockerfile liegt):
docker build -f Dockerfile -t r2dreamer:local .
```

### Run

```bash
docker run -it -d --rm \
    --gpus=all \              # NVIDIA Container Toolkit erforderlich
    --network=host \          # Host-Netzwerk (wichtig für TensorBoard, AirSim etc.)
    --volume=$PWD:/workspace \
    --name=r2dreamer-container \
    r2dreamer:local
```

### Training starten

```bash
docker exec -it r2dreamer-container bash
python3 train.py env=dmc_vision env.task=dmc_walker_walk

# Oder detached:
docker exec -it -d r2dreamer-container bash -c "python3 train.py ..."
```

### TensorBoard

```bash
# Im Container starten, auf Host unter localhost:6006 erreichbar:
docker exec -it r2dreamer-container tensorboard --logdir ./logdir
```

---

## 5. Wichtige Abhängigkeiten

- **`r2dreamer/Dockerfile`** (im Eltern-Verzeichnis): Definiert das Container-Image. Docker-Dokumentation und Dockerfile müssen synchron bleiben.
- **Gesamter Codebase**: `tensor_shapes.md` ist der Kommentar-Standard für alle Shape-Annotationen. Neue Module sollten diese Konventionen übernehmen.
- **`r2dreamer/configs/`**: Legt die konkreten Werte für S, K, D, T, T_imag, B pro Konfiguration fest.

---

## 6. Limitierungen & Fallstricke

- `docs/` enthält **keine** API-Referenz oder Architektur-Diagramme — nur die zwei o.g. Dateien.
- Die Tensor-Shape-Legende in `tensor_shapes.md` ist **statisch** — sie wird nicht automatisch aus dem Code generiert. Bei neuen Symbolen muss die Datei manuell erweitert werden.
- `--network=host` im Docker-Befehl ist für AirSim-Verbindungen (falls verwendet) und TensorBoard erforderlich; auf macOS funktioniert `--network=host` nicht (Docker Desktop Einschränkung).
- Neue Dateien in `docs/` sollten ausschließlich `.md`-Dokumentation sein — keine ausführbaren Skripte oder Konfigurations-Dateien gehören hierher.
