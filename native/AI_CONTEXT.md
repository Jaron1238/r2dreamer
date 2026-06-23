# AI_CONTEXT.md — r2dreamer/native/

> **Für zukünftige KIs:** Dieser Ordner enthält ZWEI unabhängige Deployment-Pfade für
> Apple-Silicon-Hardware, die nicht verwechselt werden dürfen. Lies Abschnitt 1 zuerst,
> bevor du irgendeine Datei hier änderst.

---

## 1. Zweck des Ordners

| Datei                  | Pfad                  | Zweck                                                                 |
|--------------------------|------------------------|--------------------------------------------------------------------------|
| `coreml_export.py`        | **Pfad A: CoreML-Export** | Tracet das *bestehende, trainierte PyTorch-Dreamer-Modell* per `torch.jit.trace` und konvertiert es zu einem `.mlpackage` (CoreML). Konsumiert von `fly_real.py`. Kein eigenständiges Modell — reiner Konverter. |
| `mlx_types.py`            | **Pfad B: MLX-Port**     | Dataclasses (`RSSMState`, `ContextState`, `TransitionBatch`, `LoaderReport`) |
| `mlx_distributions.py`     | **Pfad B: MLX-Port**     | 1:1-Neuimplementierung von `r2dreamer/distributions.py` in MLX (OneHotDist, TwoHot, SymlogDist, BernoulliDist, BoundedNormalDist) |
| `mlx_utils.py`             | **Pfad B: MLX-Port**     | PyTorch→MLX Gewichts-Loader (Name-Remapping, Tensor-Layout-Konvertierung) |
| `mlx_models.py`            | **Pfad B: MLX-Port**     | Vollständige Neuimplementierung des gesamten Dreamer/RSSM-Stacks in MLX (1156 Zeilen) |
| `mlx_trainer.py`           | **Pfad B: MLX-Port**     | Eigenständiger Offline-/Online-Trainer für das MLX-Modell |

**Pfad A und Pfad B sind komplett unabhängig.** Pfad A nimmt das fertig trainierte
PyTorch-Modell und exportiert es 1:1. Pfad B ist eine **komplette Neuimplementierung**
der Architektur — kein Code wird zwischen PyTorch- und MLX-Seite geteilt, nur Gewichte
werden (teilweise) per `mlx_utils.py` übertragen.

---

## 2. Tech-Stack & Tools

| Komponente              | Verwendung                                                              |
|---------------------------|--------------------------------------------------------------------------|
| **MLX (`mlx.core`, `mlx.nn`, `mlx.optimizers`)** | Apples natives Array-/NN-Framework für Apple Silicon (unified memory, Metal-Backend) |
| **CoreMLTools (`coremltools`)** | `torch.jit.trace` → `ct.convert(..., convert_to="mlprogram")`, nur in `coreml_export.py`, lazy-importiert (`try/except ImportError`) |
| **`posix_ipc`** (indirekt, nicht hier) | Nicht in diesem Ordner, aber `mlx_*` und `coreml_export.py` zielen beide letztlich auf dieselbe Zielplattform (macOS/Apple Silicon) wie `envs/drone_sim.py`'s Colosseum-Bridge |

---

## 3. Architektur-Regeln (KRITISCH)

### 3.1 `mlx_utils.load_pytorch_to_mlx()` — Gewichtsübertragung ist NUR partiell möglich

Der Loader macht drei Dinge: (1) Regex-basiertes Renaming von PyTorch-Parameternamen auf
MLX-Konvention (`_remap_pytorch_name_to_mlx`), (2) Tensor-Layout-Konvertierung für Conv-
Gewichte (`(out,in,kh,kw)` → `(out,kh,kw,in)` für Conv2d; umgekehrtes Layout für
ConvTranspose2d), (3) **Linear-Gewichte werden NICHT transponiert**
(`_map_linear_weight_torch_to_mlx` ist Identity) — MLX erwartet dasselbe
`(out_features, in_features)`-Layout wie PyTorch für `nn.Linear`.

**Wichtig:** Shape- oder Namens-Mismatches führen NICHT zu einem Fehler, sondern werden
in einem `LoaderReport` (`loaded`/`skipped`/`missing`/`details`) protokolliert und
stillschweigend übersprungen. Nach jedem Lade-Versuch MUSS `report["skipped"]` und
`report["missing"]` geprüft werden — ansonsten trainiert/inferiert das MLX-Modell
unbemerkt mit zufällig initialisierten statt geladenen Gewichten für die übersprungenen
Layer.

### 3.2 Encoder-Architektur weicht zwischen PyTorch und MLX fundamental ab

```
PyTorch (networks.py ConvEncoder):  torchvision.efficientnet_v2_s (PRETRAINED, Fused-MBConv)
MLX (mlx_models.py MLXConvEncoder): Hand-geschriebenes EfficientNet-B0 (_B0_STAGES, KEIN Pretraining)
```

Das sind architektonisch unterschiedliche Netze (andere Stage-Konfiguration, andere
Block-Typen in den frühen Stages: Fused-MBConv bei V2-S vs. klassisches MBConv bei B0).
**Eine Gewichtsübertragung für den Encoder-Teil ist strukturell unmöglich** —
`_remap_efficientnet_encoder_name()` macht nur einen Prefix-Strip
(`encoder.encoders.0.` entfernen), kein architektur-bewusstes Remapping. In der Praxis
werden die meisten Encoder-Keys vom Loader als `missing_in_mlx` oder `shape_mismatch`
markiert. Nur die RSSM-Kernkomponenten (`deter_net`, `obs_net`, `img_net`) sind
tatsächlich 1:1 übertragbar.

### 3.3 `BlockLinear` und `gumbel_onehot` — bewusst dieselben Hailo-inkompatiblen Ops

```python
# mlx_models.py BlockLinear.__call__
x = mx.einsum("...gi,oig->...go", x, self.weight)   # Ellipsis-Notation

# mlx_models.py MLXRSSM.gumbel_onehot
return hard + y - mx.stop_gradient(y)                 # Straight-Through Hard-Sample
```

Diese zwei Operationen sind laut `r2dreamer/AI_CONTEXT.md` (Chip-Auswahl-Historie) exakt
die Gründe, warum Hailo-8 als NPU-Ziel verworfen wurde
(`torch.einsum`-Ellipsis-Notation + `F.gumbel_softmax(hard=True)` sind DFC-inkompatibel).
Der MLX-Port behält dieselben Operationen bei MIT Absicht, weil MLX/Apple Silicon (Metal)
diese Einschränkung nicht hat. **Das ist der architektonische Beleg dafür, warum die
Apple-Silicon/MLX-Schiene statt Hailo gewählt wurde** — nicht zufällig identisch.

### 3.4 `MLXRSSM.obs_step()` hat eine andere Signatur als `rssm.py` (PyTorch)

```python
# PyTorch (rssm.py):
obs_step(self, stoch, deter, prev_action, embed, reset, d_emb=None, prev_filtered_action=None, alpha=None)

# MLX (mlx_models.py):
obs_step(self, state: RSSMState, embed, reset, d_emb=None, alpha=None)
```

Die MLX-Version bündelt `stoch/deter/prev_action/prev_filtered_action` in einem
`RSSMState`-Dataclass statt als lose Positionsargumente. **Code, der zwischen beiden
Implementierungen kopiert wird, muss diese Signatur-Differenz manuell anpassen** — ein
direkter Copy-Paste zwischen `rssm.py` und `mlx_models.py` funktioniert nicht.

### 3.5 `coreml_export.py` — Deterministische Mode-Inferenz, kein Sampling

```python
raw_out = self.actor.last(self.actor.mlp(policy_input))   # interne Layer direkt, nicht actor.forward()
mean, _ = torch.chunk(raw_out, 2, dim=-1)
action  = torch.tanh(mean)                                  # NUR der Mode, kein Sampling
```

Der CoreML-Export greift bewusst auf interne Actor-Layer zu (`.mlp`, `.last`) statt die
volle Forward-Logik (inkl. stochastischem Sampling der Aktionsverteilung) zu durchlaufen
— Deployment nutzt ausschließlich den deterministischen Modus (`tanh(mean)`), kein
`rsample()`. Das ist beabsichtigtes Verhalten für reproduzierbare Edge-Inferenz, keine
Vereinfachung aus Versehen.

### 3.6 `prev_filtered_act` wird im CoreML-Graph doppelt verwendet (beabsichtigt)

```python
self.rssm.obs_step(
    stoch, prev_deter, prev_filtered_act,    # ← als prev_action
    embed, reset, self.d_emb, prev_filtered_act,  # ← UND als prev_filtered_action
)
```

Der exportierte CoreML-Graph hat **kein separates `prev_action`-Eingabe-Tensor** — nur
`prev_filtered_act` wird als externer State zurückgeführt. Beide Parameter-Positionen
bekommen denselben Wert, weil bei Streaming-Inferenz die zuletzt ausgeführte
(gefilterte) Aktion gleichzeitig die "rohe" letzte Aktion für den nächsten Schritt ist.
**Kein Bug** — bewusste I/O-Vereinfachung für den State-Machine-Charakter des
CoreML-Modells (minimale externe State-Variablen).

### 3.7 Int8-Quantisierung beim CoreML-Export ist automatisch

```python
op_cfg = ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=opt_cfg)
```

JEDER Export quantisiert die Gewichte automatisch zu int8 (linear-symmetrisch), bei
gleichzeitig `compute_precision=ct.precision.FLOAT16` für die Aktivierungen. Mixed
Precision ist Standard, nicht optional. `minimum_deployment_target=ct.target.macOS14`
ist eine harte Mindestanforderung.

### 3.8 MLX-Trainer: zwei inkompatible Konstruktor-Signaturen in einer Klasse

`MLXOnlineTrainer.__init__` unterscheidet per `hasattr(model_or_config, "steps")`
zwischen zwei Aufrufmustern ("Signatur A": Config-Objekt + Logger + Logdir, Engine wird
erst bei `begin()` erstellt; "Signatur B": Modell + Optimizer, Engine wird sofort
erstellt). `update()` wirft `RuntimeError("update() requires Signature A.")`, wenn
`self._engine is None` — diese Fehlermeldung ist leicht irreführend: Sie bedeutet
eigentlich "Engine noch nicht initialisiert, `begin()` aufrufen", nicht "Signatur A
verwenden" (Signatur B hat die Engine ja schon im Konstruktor).

### 3.9 `mx.compile`-Trainingsschritt erfordert explizite Mutable-State-Deklaration

```python
_mutable = [model.trainable_parameters(), optimizer.state, mx.random.state]
self._compiled = mx.compile(_step, inputs=_mutable, outputs=_mutable)
```

Anders als PyTorch (eager by default) MUSS bei MLX jeder veränderliche State, der
innerhalb der kompilierten Funktion gelesen/geschrieben wird, explizit als
`inputs`/`outputs` deklariert werden. Wird ein neuer State (z. B. ein zusätzlicher
Buffer) hinzugefügt, ohne ihn hier zu registrieren, wird er nicht korrekt zwischen
Compile-Aufrufen aktualisiert.

---

## 4. Wichtige Abhängigkeiten

```
fly_real.py
  └── native/coreml_export.py (ExportConfig, export_to_coreml)
        └── nutzt dreamer.encoder/rssm/actor/safety_net/drone_embed (PyTorch, aus dreamer.py)

native/mlx_trainer.py (_begin_offline)
  └── from trainer import FPVDataset          ← QUERABHÄNGIGKEIT zur PyTorch-Trainer-Seite!
        (numpy-Batches werden per _np2mx() nach MLX konvertiert)

native/mlx_models.py
  └── from native.mlx_distributions import kl, symexp_twohot, binary, bounded_normal, ...
  └── from native.mlx_types import RSSMState

native/mlx_utils.py
  └── konsumiert torch.Tensor state_dicts (Brücke PyTorch → MLX), import torch nötig
```

**Bemerkenswert:** `mlx_trainer.py` ist NICHT vollständig von der PyTorch-Seite
entkoppelt — es importiert `FPVDataset` direkt aus dem Root-Modul `trainer.py` für das
Offline-Training. Der MLX-Pfad ist also kein 100% eigenständiges System.

---

## 5. Limitierungen & Fallstricke

### ❌ NIEMALS tun

1. **Annehmen, dass `load_pytorch_to_mlx()` den Encoder vollständig überträgt** — die
   Architekturen (EfficientNet-V2-S vs. handgeschriebenes B0) sind strukturell
   verschieden; Encoder-Gewichte werden größtenteils übersprungen.
2. **`obs_step()`-Aufrufmuster zwischen `rssm.py` (PyTorch) und `mlx_models.py` (MLX)
   1:1 kopieren** — unterschiedliche Signaturen (lose Tensoren vs. `RSSMState`-Dataclass).
3. **`coreml_export.py` ohne installiertes `coremltools` aufrufen** — wirft kontrolliert
   `ImportError` mit Installationshinweis, kein stiller Fehlschlag.
4. **Den `LoaderReport` von `load_pytorch_to_mlx()` ignorieren** — `skipped`/`missing`
   müssen geprüft werden, sonst bleiben Teile des MLX-Modells unbemerkt zufällig
   initialisiert.

### ⚠️ Vorsicht

- `_begin_online` (mlx_trainer.py) verwendet `obs_shape = getattr(model, "obs_shape",
  (84, 84, 3))` als Fallback — das ist ein generischer Atari-Platzhalterwert (84×84×3),
  NICHT die tatsächliche FPV-Auflösung. Falls `model.obs_shape` nicht explizit gesetzt
  ist, allokiert der `MLXReplayBuffer` Arrays mit falscher Shape.
- `_memory_guard()` nutzt `mx.metal.get_active_memory()` mit `try/except AttributeError`
  — funktioniert nur auf Metal-Backend (Apple Silicon); auf anderen Backends wird die
  Speicherprüfung stillschweigend übersprungen, kein Fehler.
- `@dataclass`-Decorator auf `class MLXActor(nn.Module)` (Zeile 403) ist ungewöhnlich/
  vermutlich ein Überbleibsel — die Klasse hat keine annotierten Felder und definiert
  `__init__` manuell, daher funktional folgenlos, aber potenziell verwirrend bei
  zukünftigen Refactorings.
- `coreml_export.py` exportiert immer für genau EINEN `drone_id` (gebacken als Buffer,
  `self.register_buffer("d_emb", d_emb)`) — für ein anderes Drohnen-Embedding muss neu
  exportiert werden, kein Laufzeit-Switch im `.mlpackage` möglich.
