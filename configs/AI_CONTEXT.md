# AI_CONTEXT.md — r2dreamer/configs/

> **Für zukünftige KIs:** Dieser Ordner ist die Hydra-Konfigurationszentrale des gesamten
> r2dreamer-Trainingsstacks. Jede hier definierte Datei wird per `@hydra.main` von
> `train.py` bzw. `fly_real.py` geladen. Diese Datei deckt nur die zwei Dateien
> **direkt** in `configs/` ab (`configs.yaml`, `fly_real.yaml`) — die Unterordner
> `env/` und `model/` haben eigene, separate `AI_CONTEXT.md`-Dateien (siehe Abschnitt 4).

---

## 1. Zweck des Ordners

`configs/` ist die **Single-Entry-Point-Konfiguration** für alle vier Trainings-/
Deployment-Phasen des Projekts. Es gibt zwei Top-Level-Configs:

| Datei            | Verwendet von     | Zweck                                                          |
|-------------------|-------------------|------------------------------------------------------------------|
| `configs.yaml`     | `train.py`         | Phasen 1–3 (Pretraining, Policy-Adaptation, Online-Sim). `device='cuda'`. |
| `fly_real.yaml`    | `fly_real.py`       | Phase 4 (Real-Deployment auf Apple Silicon). `device='mps'`.    |

Beide Dateien komponieren via Hydra `defaults:` eine Modellgrößen-Config aus
`configs/model/` (Standard: `size100M`) und enthalten zusätzlich Override-Werte, die
direkt im Trainings-/Flugkontext relevant sind (Auflösungen, Achsenzuordnung,
Sicherheitsschwellen).

---

## 2. Tech-Stack & Tools

| Komponente   | Verwendung hier                                                          |
|---------------|---------------------------------------------------------------------------|
| **Hydra 1.3.2** | `defaults:`-Liste für Config-Composition, `@hydra.main`-Konsum in `train.py`/`fly_real.py` |
| **OmegaConf**   | Interpolation (`${batch_size}`, `${device}`, `${now:%Y-%m-%d}`)         |

Keine weiteren Libraries — reine YAML-Konfigurationsdateien.

---

## 3. Architektur-Regeln (KRITISCH)

### 3.1 Reihenfolge der `defaults:`-Liste ist verhaltensrelevant

```yaml
defaults:
  - model: size100M
  - override hydra/job_logging: disabled
  - _self_              # MUSS zuletzt stehen!
```

`_self_` steht bewusst **am Ende** der Liste. Das bedeutet: Werte, die direkt in
`configs.yaml` (z. B. der `model:`-Block am Dateiende) gesetzt sind, **überschreiben**
die aus `model/size100M.yaml` importierten Werte. Würde `_self_` nach oben verschoben,
würde sich diese Override-Reihenfolge unbemerkt umkehren.

### 3.2 Auflösungs-Duplikation (3 Quellen, kein Single-Source-of-Truth)

```
configs.yaml:    model.img_height=288, img_width=512, safety_img_height=720, safety_img_width=1280
fly_real.yaml:   identische Werte, separat dupliziert
pipeline/pipeline.py (Projekt-Root): out_h=288, out_w=512, safety_h=720, safety_w=1280
```

Alle drei Stellen müssen manuell synchron gehalten werden. Hydra validiert das **nicht**.

### 3.3 `throttle_index` — Code-Default ist gefährlich

```python
# dreamer.py L44
throttle_index = int(getattr(cfg, "throttle_index", 0))  # Fallback = 0!
```

Beide Configs hier setzen explizit `throttle_index: 3` (Action-Dim 3 = Throttle, bei
`act_dim=4`: roll, pitch, yaw, throttle). **Wird `throttle_index` in einer neuen/eigenen
Config vergessen, fällt der Code still auf Index 0 zurück** — das wäre dann der Roll-Kanal,
nicht Throttle. Kein Fehler, nur falsches Verhalten.

### 3.4 `fly_real.yaml` — RSSM-Dimensionen sind NICHT automatisch abgeleitet

```yaml
# RSSM state dims — must match the size config used during training.
stoch: 32
discrete: 48
deter: 6144
```

Diese Werte sind als reiner Kommentar-Verweis auf `size100M` hart codiert. Wechselt man
die für das Training verwendete Modellgröße (z. B. auf `size25M`: `deter=3072`), **muss
dieser Block in `fly_real.yaml` manuell angepasst werden** — sonst Dimension-Mismatch
beim Checkpoint-Laden für den CoreML-Export.

### 3.5 `device` unterscheidet sich bewusst zwischen den Dateien

```
configs.yaml:   device: 'cuda'   # Training (T4/Cloud-GPU)
fly_real.yaml:  device: 'mps'    # Real-Flight (Apple Silicon)
```

Niemals den `device`-Wert zwischen den beiden Dateien kopieren — das sind unterschiedliche
Zielplattformen mit unterschiedlichen AMP-Strategien (siehe `r2dreamer/AI_CONTEXT.md` 3.7).

### 3.6 `require_osd: false` ist eine bewusste Entscheidung

```yaml
require_osd: false  # OSD-Werte werden in der Loss-Funktion nicht genutzt;
                     # true würde ~90% der Daten filtern
```

Nicht versehentlich auf `true` setzen, um "saubere" Daten zu erzwingen — das würde den
nutzbaren Trainingsdatensatz auf ~10% reduzieren, ohne einen Nutzen für die Loss-Berechnung
zu bringen.

### 3.7 `safety_in_channels` ist obsolet — Kommentar beachten

```yaml
# safety_in_channels is now auto-derived from dataset.raw_image_mode
# (grayscale→1, rgb→3). Remove or ignore this value — it is no longer read.
```

Falls dieser Key irgendwo in einer Config noch gesetzt ist: ignorieren, er wird vom Code
nicht gelesen. Steuerung erfolgt ausschließlich über `dataset.raw_image_mode`.

### 3.8 `hf_repo` ist ein Platzhalter

```yaml
hf_repo: 'dein_benutzername/dein_fpv_dataset' # Hier den Pfad zu deinem HF-Dataset einfügen
```

Muss vor jedem echten Trainingslauf per CLI-Override (`dataset.hf_repo=...`) oder
Datei-Edit ersetzt werden.

---

## 4. Wichtige Abhängigkeiten

```
configs.yaml
  ├── model/size100M.yaml (+ model/_base_.yaml)
  └── konsumiert von train.py (@hydra.main)

fly_real.yaml
  ├── model/size100M.yaml (+ model/_base_.yaml)
  └── konsumiert von fly_real.py (@hydra.main)

env/*.yaml  → NICHT von configs.yaml/fly_real.yaml importiert (kein `defaults:`-Eintrag
              für env: hier). Werden separat von runs/*.sh für klassische Benchmarks
              (DMC, Atari, Crafter, MemoryMaze, MetaWorld) referenziert.

Auflösungswerte ↔ pipeline/pipeline.py (Projekt-Root, siehe Root-AI_CONTEXT.md 3.1)
```

**Hinweis:** Die `env:`-Sektion in `configs.yaml` selbst (`encoder.cnn_keys: 'image'`,
`decoder.cnn_keys: 'image'`) ist NICHT identisch mit den Dateien im Unterordner `env/` —
letztere sind eigenständige Environment-Presets für andere Benchmark-Suiten, nicht für den
FPV-Pfad.

---

## 5. Limitierungen & Fallstricke

### ❌ NIEMALS tun

1. **`_self_` in der `defaults:`-Liste verschieben** — kehrt die Override-Reihenfolge
   zwischen Modell-Config und Top-Level-Config unbemerkt um.
2. **Auflösungswerte nur in einer der drei Dateien ändern** (`configs.yaml`,
   `fly_real.yaml`, `pipeline/pipeline.py`) — keine Validierung, Mismatch zeigt sich erst
   als Shape-Error beim Training/Export.
3. **`throttle_index` in einer neuen Config weglassen** — fällt still auf `0` zurück
   (falsche Steuerachse, kein Fehler).
4. **Modellgröße in `fly_real.yaml` (`defaults: model: size100M`) wechseln, ohne den
   `stoch/discrete/deter`-Block manuell anzupassen** — Checkpoint-Load-Mismatch beim
   CoreML-Export.
5. **`require_osd` auf `true` setzen** in der Annahme, das verbessere die Datenqualität —
   filtert ~90% der Trainingsdaten ohne Loss-Nutzen.
6. **`hf_repo`-Platzhalter unverändert lassen** und Training starten.

### ⚠️ Vorsicht

- `device` ist je nach Datei fest auf `cuda` bzw. `mps` gesetzt — bei neuen Configs
  (z. B. für CPU-Debugging) bewusst überschreiben, nicht einfach eine der beiden Dateien
  kopieren und unverändert lassen.
- `logdir: logdir/${now:%Y-%m-%d}/${now:%H-%M-%S}` erzeugt bei jedem Lauf automatisch
  einen neuen Zeitstempel-Ordner — kein Resume in denselben `logdir` ohne expliziten
  Override.
