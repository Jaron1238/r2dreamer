# AI_CONTEXT.md — r2dreamer/envs/

> **Für zukünftige KIs:** `drone_sim.py` enthält die Phase-3-Colosseum-Bridge. Die Benchmark-Suite
> (`dmc.py`, `atari.py` etc.) ist nahezu 1:1 aus dem DreamerV3-Referenz-Code übernommen und dient
> ausschließlich der wissenschaftlichen Validierung. Lies Abschnitt 3 zuerst.

---

## 1. Zweck des Ordners

Der Ordner zerfällt in zwei funktional getrennte Bereiche:

| Bereich                       | Dateien                                                                 | FPV-relevant? |
|---------------------------------|--------------------------------------------------------------------------|----------------|
| **DreamerV3-Referenz-Benchmark-Suite** | `__init__.py`, `atari.py`, `crafter.py`, `dmc.py`, `dmc_subtle.py`, `memorymaze.py`, `metaworld.py`, `wrappers.py`, `parallel.py` | ❌ Nein — wissenschaftliche Validierung |
| **FPV-Drohnen-Simulation**       | `drone_sim.py`                                                            | ✅ **Ja** — Phase-3-Online-Training |

Die Benchmark-Suite folgt nahezu 1:1 dem öffentlichen Referenz-Code des originalen
DreamerV3-Repos (danijar/dreamerv3) — das ist kein Custom-Code für dieses Projekt,
sondern die Standard-Infrastruktur, um die RSSM/Rep-Loss-Architektur gegen bekannte
Benchmarks zu validieren.

---

## 2. Tech-Stack & Tools

| Komponente              | Verwendung                                                              |
|---------------------------|--------------------------------------------------------------------------|
| **gymnasium**              | Moderne Env-API, von fast allen Dateien genutzt                        |
| **gym (legacy, "alt")**    | NUR in `memorymaze.py` (`import gym as old_gym`) — das `memory_maze`-Package registriert sich ausschließlich in der alten `gym`-Registry |
| **ale_py**                 | Atari Learning Environment Backend für `atari.py`                       |
| **dm_control**             | DeepMind Control Suite, `dmc.py` + `dmc_subtle.py`                       |
| **crafter, metaworld**     | Jeweils eigenes Drittanbieter-Package                                    |
| **multiprocessing (spawn) + cloudpickle** | `parallel.py` — Subprozess-RPC für parallele Env-Instanzen     |
| **ctypes + mmap + posix_ipc** | `drone_sim.py` — Shared-Memory-IPC-Bridge zu Colosseum/Unreal Engine |
| **OpenCV (cv2)**           | Bildskalierung in `drone_sim.py` (`ColosseumBridge.read_observation`)   |

---

## 3. Architektur-Regeln (KRITISCH)

### 3.1 Zwei inkompatible Step-API-Konventionen im selben Ordner

```
Benchmark-Suite (atari/crafter/dmc/memorymaze/metaworld/wrappers): 
    step() → (obs, reward, done, info)            ← alte 4-Tupel-API, BEWUSST so gehalten

drone_sim.py DroneSimEnv:
    step() → (obs, reward, terminated, truncated, info)   ← moderne Gymnasium-5-Tupel-API
```

Das ist kein Versehen: `DroneSimEnv` wird NIE durch `wrappers.TimeLimit`/`wrappers.Dtype`
geschickt (die für 4-Tupel gebaut sind), weil es nicht über `envs/__init__.py::make_env()`
instanziiert wird, sondern direkt in `train.py` (Phase 3). **`DroneSimEnv` niemals mit
den 4-Tupel-Wrappern aus `wrappers.py` kombinieren** — würde beim Unpacking sofort
crashen (`too many/few values to unpack`).

### 3.2 `make_env()`/`make_envs()` kennt KEINE Drohnen-Suite

`envs/__init__.py::make_env()` dispatcht nur auf `dmc`/`atari`/`memorymaze`/`crafter`/
`metaworld` (Suite-Präfix aus `config.task.split("_", 1)`). Es gibt keinen `drone`-Zweig.
`DroneSimEnv` wird stattdessen direkt in `train.py` instanziiert
(`from envs.drone_sim import DroneSimEnv`). Diese Trennung ist beabsichtigt — die
Parallelisierungs-Infrastruktur (`ParallelEnv`/`Worker`) wird für Phase 3 NICHT verwendet.

### 3.3 `DroneSimEnv` hat zwei Betriebsmodi — Default ist ein synthetischer Mock

```python
self._colosseum = ColosseumBridge(config)
if self._colosseum.enabled:
    self._colosseum.connect()
```

- **Standardmodus (Colosseum deaktiviert):** `_render_rgb()` generiert eine rein
  prozedurale 2D-Korridor-Szene (Himmel/Boden-Gradient, Korridorwände, ein
  "Hindernis-Blob", Motion-Blur-Streifen) — **keine echte Physik, keine echten Bilder.**
  Reine Smoke-Test-Infrastruktur.
- **Colosseum-Modus** (`config.colosseum.enabled=true`, siehe
  `configs/env/colosseum_shm.yaml`): Echte Bilder + Sensordaten kommen per Shared-Memory
  vom Unreal-Engine-Simulator.

Beide Modi liefern ein **6-Kanal-Bild** (`[R,G,B, dR,dG,dB]` — RGB + zeitliche Differenz),
konsistent mit `shapes["image"]=(H,W,6)` im RSSM.

### 3.4 `ColosseumBridge` — Stale-Frame-Schutz, kein Crash bei SHM-Problemen

`read_observation()` vergleicht `seq_ue` (Frame-Sequenznummer aus dem SHM-Header) gegen
den zuletzt gesehenen Wert. Hat UE noch keinen neuen Frame geschrieben, wird **kein
Fehler geworfen**, sondern `_fallback_obs()` (schwarzes Bild) zurückgegeben — nur ein
Konsolen-Print, kein Exception. Gleiches gilt, wenn `sem_ue_obj.acquire(timeout=0.1)`
abläuft. **Bei Phase-3-Trainingsläufen unbedingt die Logs auf wiederholte
`[ColosseumBridge] Warte auf UE...`-Meldungen prüfen** — das Training läuft sonst
unbemerkt mit Schwarzbild-Beobachtungen weiter.

### 3.5 `ParallelEnv`/`Worker` — Spawn-Context, "thread"-Strategie existiert nicht

```python
self.impl = {
    "process": bind(ProcessPipeWorker, initializers=inits),
    "daemon":  bind(ProcessPipeWorker, initializers=inits, daemon=True),
}[strategy](fn)
```

`Worker.__init__` hat `strategy="thread"` als Default-Parameter — es gibt aber **keine
`"thread"`-Implementierung** im `impl`-Dict. Würde `Worker(fn)` ohne expliziten
`strategy="process"` aufgerufen, crasht es sofort mit `KeyError: 'thread'`. Aktuell
harmlos, weil der einzige Aufrufer (`Parallel.__init__` via `ParallelEnv`) immer
`"process"` explizit übergibt — aber eine Falle für neuen Code, der sich auf den
Default verlässt.

### 3.6 `np.bool = np.bool_`-Monkeypatch in `crafter.py`

Kompatibilitäts-Shim für das `crafter`-Package, das noch das in NumPy ≥1.24 entfernte
`np.bool`-Alias nutzt. Modulweite Monkeypatch-Zeile — beeinflusst den gesamten
Python-Prozess, nicht nur `crafter.py` selbst.

---

## 4. Wichtige Abhängigkeiten

```
envs/__init__.py (make_env/make_envs)
  ├── envs/dmc.py, atari.py, memorymaze.py, crafter.py, metaworld.py
  ├── envs/wrappers.py (TimeLimit, NormalizeActions, OneHotAction, Dtype, ...)
  ├── envs/parallel.py (ParallelEnv)
  └── konsumiert von runs/*.sh über configs/env/*.yaml (CLI-Override `env=<preset>`)

envs/dmc.py
  └── envs/dmc_subtle.py (dynamischer Import bei Task-Namen mit "_subtle"-Suffix)

envs/wrappers.py, envs/parallel.py
  └── tools.py (Root) — tools.convert(), tools.to_np()

envs/drone_sim.py
  ├── reward.py (Root) — DroneRewardFunction
  ├── configs/env/colosseum_shm.yaml — String-Kontrakt für shm_name/sem_ue/sem_py
  │     (siehe configs/env/AI_CONTEXT.md Abschnitt 3.3)
  └── konsumiert DIREKT von train.py (Phase 3), NICHT über make_env()
```

---

## 5. Limitierungen & Fallstricke

### ❌ NIEMALS tun

1. **`DroneSimEnv` mit den 4-Tupel-Wrappern aus `wrappers.py` (`TimeLimit`, `Dtype`,
   etc.) kombinieren** — inkompatible Step-API (5-Tupel vs. 4-Tupel).
2. **`Worker(fn)` ohne expliziten `strategy="process"`-Parameter aufrufen** — der
   Default `"thread"` existiert nicht im `impl`-Dict, sofortiger `KeyError`.
3. **Das `np.bool = np.bool_`-Monkeypatch aus `crafter.py` entfernen**, ohne die
   `crafter`-Package-Version zu prüfen — bricht sonst beim nächsten Crafter-Lauf mit
   `AttributeError: module 'numpy' has no attribute 'bool'`.

### ⚠️ Vorsicht

- `memorymaze.py` benötigt **sowohl** das alte `gym`-Package als auch `gymnasium`
  gleichzeitig installiert (`old_gym.make(...)`). Eine Dependency-Bereinigung, die das
  alte `gym` entfernt, bricht MemoryMaze still beim nächsten `make_env()`-Aufruf.
- `_render_rgb()`'s synthetischer Korridor-Mock (Default-Modus von `DroneSimEnv`) ist
  KEINE validierte Flugphysik — nur ein Platzhalter für Smoke-Tests ohne laufenden
  Unreal-Engine-Simulator. Ergebnisse aus diesem Modus dürfen nicht als
  Politik-Validierung für echtes Fliegen interpretiert werden.
- `ColosseumBridge.disconnect()` löscht den Shared-Memory-Block selbst NICHT (laut
  Kommentar "Sache von UE") — das Aufräumen liegt vollständig beim Unreal-Engine-Prozess.
