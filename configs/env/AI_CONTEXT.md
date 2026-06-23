# AI_CONTEXT.md — r2dreamer/configs/env/

> **Für zukünftige KIs:** Dieser Ordner enthält zwei klar getrennte Arten von Configs,
> die NICHT verwechselt werden dürfen: sechs generische DreamerV3-Benchmark-Presets
> (akademische Sanity-Checks, nicht Teil des FPV-Produktionspfads) und eine einzige
> FPV-relevante Datei (`colosseum_shm.yaml`) für die Phase-3-Simulator-Anbindung.

---

## 1. Zweck des Ordners

| Datei                | Zweck                                                                       | FPV-relevant? |
|------------------------|------------------------------------------------------------------------------|----------------|
| `atari100k.yaml`        | Atari-100k-Benchmark-Preset (26 Spiele, siehe `runs/atari.sh`)              | ❌ Nein         |
| `crafter.yaml`          | Crafter-Survival-Benchmark-Preset                                           | ❌ Nein         |
| `dmc_proprio.yaml`       | DeepMind-Control-Suite, Zustandsraum-basiert (proprioceptiv, kein Bild)     | ❌ Nein         |
| `dmc_vision.yaml`        | DeepMind-Control-Suite, bildbasiert                                         | ❌ Nein         |
| `memorymaze.yaml`        | MemoryMaze-Benchmark (Langzeit-Gedächtnis-Test)                             | ❌ Nein         |
| `metaworld.yaml`         | MetaWorld-Manipulationsbenchmark                                            | ❌ Nein         |
| `colosseum_shm.yaml`     | **Aktiviert die Shared-Memory-IPC-Bridge zum Colosseum-Simulator** (Unreal-Engine-basierter AirSim-Fork) für Phase-3-Online-Training der Drohne | ✅ **Ja** |

Die ersten sechs Dateien dienen ausschließlich der **wissenschaftlichen Validierung der
RSSM/Rep-Loss-Architektur** gegen Standard-DreamerV3-Benchmarks — sie werden von den
Batch-Skripten in `r2dreamer/runs/` aufgerufen (siehe Abschnitt 4) und haben mit dem
eigentlichen FPV-Drohnenprojekt nichts zu tun. `colosseum_shm.yaml` ist der einzige Eintrag
mit direktem Produktionsbezug.

---

## 2. Tech-Stack & Tools

| Komponente            | Verwendung                                                                |
|--------------------------|--------------------------------------------------------------------------|
| **Hydra Config Group `env:`** | Wird NICHT über `defaults:` in `configs.yaml` statisch eingebunden, sondern per CLI-Override (`env=atari100k`, `env=dmc_vision`, ...) aus `runs/*.sh` ausgewählt |
| **POSIX Shared Memory (`shm_open`)** | Grundlage der `colosseum_shm.yaml`-IPC-Bridge — Python-Seite via `posix_ipc` in `envs/drone_sim.py` |
| **Gymnasium-Env-Wrapper (DMC, Atari, Crafter, MemoryMaze, MetaWorld)** | Referenziert in `envs/` (eigene `AI_CONTEXT.md` ausstehend) |

---

## 3. Architektur-Regeln (KRITISCH)

### 3.1 `mlp_keys`/`cnn_keys`-Regex-Konvention — `'$^'` ist kein Tippfehler

```yaml
encoder:
  mlp_keys: '$^'      # = "matched nichts" (Regex: Zeilenende direkt nach Zeilenanfang)
  cnn_keys: 'image'    # = matched den Key "image"
```

`'$^'` ist ein bewusster Regex-Trick, um eine Modalität vollständig zu deaktivieren — KEIN
Platzhalter und KEIN Fehler. Beispiel `dmc_proprio.yaml` vs. `dmc_vision.yaml`:

```
dmc_proprio:  mlp_keys: '.*'   cnn_keys: '$^'    → nur Zustandsvektor, kein Bild
dmc_vision:   mlp_keys: '$^'   cnn_keys: 'image'  → nur Bild, kein Zustandsvektor
```

Beim Hinzufügen neuer Benchmark-Presets diese Konvention exakt beibehalten.

### 3.2 Auflösung `[64, 64]` — NICHT mit FPV-Auflösungen verwechseln

Alle sechs klassischen Benchmark-Presets verwenden `size: [64, 64]`. Das hat **keinerlei**
Bezug zur FPV-Dual-Auflösung (288×512 RSSM / 720×1280 SafetyNet, siehe
`configs/AI_CONTEXT.md` 3.2). Wird versehentlich `env=dmc_vision` o. ä. in einem
FPV-Trainingslauf referenziert, bricht die Pipeline an der Bild-Shape.

### 3.3 `colosseum_shm.yaml` — Drei-Wege-String-Kontrakt ohne Typsicherheit

```yaml
colosseum:
  enabled: true
  shm_name: "/colosseum_ipc"
  sem_ue:   "/colosseum_ue_wrote"
  sem_py:   "/colosseum_py_wrote"
```

Diese drei String-Literale müssen **exakt** mit drei unabhängigen Stellen übereinstimmen:

1. Den hartcodierten Fallback-Konstanten in `r2dreamer/envs/drone_sim.py`
   (`_SHM_NAME`, `_SEM_UE`, `_SEM_PY` — Zeilen 288–290, bestätigt per Code-Scan)
2. Der `ShmIPCLayout.h` im (nicht in diesem Repo enthaltenen) Unreal-Engine-Plugin
3. Diesem Config-File selbst

**Keine dieser drei Stellen validiert die anderen** — ein Tippfehler in `shm_name` führt
zu einem stillen Hänger beim `posix_ipc.SharedMemory(...)`-Aufruf (Object-not-found), nicht
zu einem Hydra-Validierungsfehler.

### 3.4 macOS Shared-Memory-Hinweis (nur Dokumentation, kein Code)

Der Kommentar in `colosseum_shm.yaml` stellt klar: POSIX `shm_open` (verwendet hier) hat
**keine** 4-MB-Grenze — die gilt nur für das ältere System-V-Interface (`shmget`). Keine
`sysctl`-Anpassung nötig, außer das System-V-Interface wird explizit gebraucht.

---

## 4. Wichtige Abhängigkeiten

```
runs/atari.sh        --[CLI: env=atari100k, env.task=$task]-->     configs/env/atari100k.yaml
runs/crafter.sh       --[CLI: env=crafter]-->                       configs/env/crafter.yaml
runs/dmc.sh           --[CLI: env=dmc_${MODAL}]-->                  configs/env/dmc_vision.yaml | dmc_proprio.yaml
runs/dmc_subtle.sh    --[CLI: env=dmc_${MODAL}]-->                  (gleiche Dateien, andere Task-Liste)
runs/memorymaze.sh    --[CLI: env=memorymaze]-->                    configs/env/memorymaze.yaml
runs/metaworld.sh     --[CLI: env=metaworld]-->                     configs/env/metaworld.yaml

configs/env/colosseum_shm.yaml  --[String-Kontrakt]-->  r2dreamer/envs/drone_sim.py (ColosseumBridge-Klasse)
                                  --[String-Kontrakt]-->  Unreal-Engine-Plugin ShmIPCLayout.h (außerhalb dieses Repos)
```

**Wichtig:** Keine dieser sieben Dateien wird über die `defaults:`-Liste in
`configs/configs.yaml` automatisch geladen — alle werden ad hoc per Hydra-CLI-Override
(`env=<preset>`) ausgewählt.

---

## 5. Limitierungen & Fallstricke

### ❌ NIEMALS tun

1. **Eine der sechs Benchmark-Presets (`atari100k`, `crafter`, `dmc_*`, `memorymaze`,
   `metaworld`) für einen FPV-Trainingslauf verwenden** — Auflösung (`64×64`) und
   Observation-Keys passen nicht zum FPV-Datenformat (`image` 6-Kanal, `raw_image`
   für SafetyNet).
2. **`shm_name`/`sem_ue`/`sem_py` in `colosseum_shm.yaml` ändern, ohne gleichzeitig
   `envs/drone_sim.py` UND das Unreal-Engine-Plugin anzupassen** — alle drei Stellen
   müssen synchron bleiben, es gibt keine automatische Validierung.
3. **`'$^'` als vermeintlichen Fehler "korrigieren"** (z. B. zu `''` oder `'.*'` ändern) —
   das deaktiviert dann ungewollt die jeweilige Modalität nicht mehr oder komplett anders
   als beabsichtigt.

### ⚠️ Vorsicht

- `colosseum_shm.yaml` setzt `enabled: true` direkt — wird diese Datei nicht explizit per
  `env=colosseum_shm` eingebunden, bleibt Colosseum standardmäßig deaktiviert (Fallback: Mock-Modus, siehe `envs/AI_CONTEXT.md` Abschnitt 3.3).
- Die Benchmark-Presets sind reine Forschungs-/Validierungs-Infrastruktur. Änderungen hier
  betreffen NIE den produktiven FPV-Trainingspfad (`train.py phase=1/2/3` ohne `env=`-
  Override nutzt implizit die FPV-Datenpipeline aus `dataset:` in `configs.yaml`, nicht
  diesen Ordner).
