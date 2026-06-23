# AI_CONTEXT.md — r2dreamer/configs/model/

> **Für zukünftige KIs:** Dieser Ordner definiert die Modellarchitektur- und
> Hyperparameter-Configs für den DreamerV3/RSSM-Stack. Er wird ausschließlich über die
> Hydra-Config-Group `model:` aus `configs/configs.yaml` bzw. `configs/fly_real.yaml`
> referenziert (`defaults: - model: sizeXXM`). Diese Dateien sind NICHT eigenständig
> lauffähig.

---

## 1. Zweck des Ordners

`model/` enthält die vollständige Architektur- und Optimizer-Konfiguration für sechs
Modellgrößen-Varianten (12M bis 400M Parameter) des RSSM-World-Models. Das Pattern ist
strikt **Base + Override**:

- **`_base_.yaml`** definiert ALLE größenunabhängigen Hyperparameter: Optimizer (lr, agc,
  beta1/2, warmup), Loss-Gewichte (`loss_scales`), Submodul-Configs (rssm, encoder,
  decoder, reward, cont, actor, critic) sowie die drei alternativen
  Repräsentationslern-Blöcke (`r2dreamer`, `nedreamer`, `dreamer_pro`).
- **`sizeXXM.yaml`** (12M/25M/50M/100M/200M/400M) setzt NUR die größenabhängigen Werte
  (`deter`, `hidden`, `discrete`, `depth`, `units`, `act`, `norm`) und importiert
  `_base_` über `defaults: [_base_]`, sodass beide in denselben `model:`-Namespace
  gemergt werden.

`_base_.yaml` ist **nicht eigenständig funktionsfähig** — es enthält Interpolationen wie
`rssm.deter: ${model.deter}`, die nur aufgelöst werden können, wenn eine `sizeXXM.yaml`
diese Schlüssel zusätzlich liefert.

---

## 2. Tech-Stack & Tools

| Komponente              | Verwendung                                                              |
|---------------------------|----------------------------------------------------------------------|
| **Hydra Config Groups**    | `model:` ist eine Hydra-Config-Group; `defaults: - _base_` in jeder Size-Datei mergt beide Dateien in denselben Namespace |
| **OmegaConf-Interpolation**| Intra-Namespace-Referenzen (`${model.deter}`, `${model.units}`) und Cross-Namespace (`${device}`, `${env.encoder.mlp_keys}`) |

---

## 3. Architektur-Regeln (KRITISCH)

### 3.1 Größen-Skalierungstabelle

| Variante | deter  | hidden | discrete | depth | units | act    | norm |
|----------|--------|--------|----------|-------|-------|--------|------|
| 12M      | 2048   | 256    | 16       | 16    | 256   | SiLU   | True |
| 25M      | 3072   | 384    | 24       | 24    | 384   | SiLU   | True |
| 50M      | 4096   | 512    | 32       | 32    | 512   | SiLU   | True |
| 100M     | 6144   | 768    | 48       | 48    | 768   | SiLU   | True |
| 200M     | 8192   | 1024   | 64       | 64    | 1024  | SiLU   | True |
| 400M     | 12288  | 1536   | 96       | 96    | 1536  | SiLU   | True |

`act='SiLU'` und `norm=True` sind über alle Größen hinweg konstant — bislang keine
Variante weicht davon ab. `hidden` und `units` sind in jeder Zeile identisch (gleicher
Wert, zwei verschiedene Config-Keys).

### 3.2 Interpolationsabhängigkeit (`_base_.yaml` → `sizeXXM.yaml`)

```yaml
# _base_.yaml
rssm:
  deter: ${model.deter}      # MUSS von sizeXXM.yaml geliefert werden
  hidden: ${model.hidden}
  discrete: ${model.discrete}
encoder:
  cnn:
    depth: ${model.depth}
decoder:
  mlp: { units: ${model.units} }
```

**Jede neue Size-Variante MUSS exakt die sieben Keys `deter, hidden, discrete, depth,
units, act, norm` setzen** — fehlt einer, schlägt die Hydra/OmegaConf-Auflösung mit einem
Interpolation-Error fehl, sobald der entsprechende Submodul-Block instanziiert wird.

### 3.3 `rep_loss`-Block-Selektion (drei Blöcke, nur einer aktiv)

`_base_.yaml` enthält drei alternative Repräsentationslern-Konfigurationsblöcke:

```yaml
rep_loss: "r2dreamer"   # steuert, welcher Block unten tatsächlich konsumiert wird

r2dreamer:    { lambd: 5e-4 }
nedreamer:    { hidden_dim: 256, transformer_layers: 2, ... }
dreamer_pro:  { num_prototypes: 2500, sinkhorn_eps: 0.05, ... }
```

Nur der zu `rep_loss` passende Block wird vom Code tatsächlich gelesen (siehe
`r2dreamer/AI_CONTEXT.md` Abschnitt 3.8 — Tabelle der `rep_loss`-Module). Die anderen
beiden Blöcke liegen syntaktisch im Config, werden aber ignoriert. **`rep_loss` darf
laut Root-Doc nicht zwischen Checkpoints gewechselt werden** — andere Module werden
instanziiert, alter Checkpoint passt nicht mehr.

### 3.4 `loss_scales` ist global, nicht pro Größe

Alle sechs Größenvarianten teilen sich dieselben `loss_scales`-Gewichte aus `_base_.yaml`
(barlow, infonce, recon, rew, con, dyn, rep, policy, value, repval, bc, safety, swav,
temp, norm, nedreamer, depth_aux, inv_dyn). Es gibt keine größenspezifische
Loss-Gewichtung — wird ein größeres Modell instabiler, muss das manuell über CLI-Override
angepasst werden, nicht über eine neue Size-Datei.

---

## 4. Wichtige Abhängigkeiten

```
configs/configs.yaml     --[defaults: model: size100M]-->  model/size100M.yaml --[defaults: _base_]--> model/_base_.yaml
configs/fly_real.yaml     --[defaults: model: size100M]-->  model/size100M.yaml --[defaults: _base_]--> model/_base_.yaml

_base_.yaml interpoliert zusätzlich:
  ${device}                  ← aus configs.yaml / fly_real.yaml Top-Level
  ${env.encoder.mlp_keys}    ← aus configs.yaml Top-Level `env:`-Block
  ${env.encoder.cnn_keys}    ← aus configs.yaml Top-Level `env:`-Block
```

Konsumiert zur Laufzeit von `dreamer.py`, `networks.py`, `rssm.py` (siehe
`r2dreamer/AI_CONTEXT.md` Abschnitt 4 für die vollständige Code-Abhängigkeitskette).

---

## 5. Limitierungen & Fallstricke

### ❌ NIEMALS tun

1. **Eine neue Size-Variante anlegen, ohne alle sieben Pflicht-Keys
   (`deter, hidden, discrete, depth, units, act, norm`) zu setzen** — Interpolation
   bricht mit `InterpolationResolutionError`.
2. **`model/size50M.yaml` standalone laden** (z. B. in einem eigenen Skript ohne
   `configs.yaml` als überlagernde `_self_`-Quelle) — siehe Fund unten, enthält veraltete
   Auflösungswerte.
3. **`rep_loss` zwischen Trainingsläufen wechseln, ohne das Modell neu zu
   initialisieren** — bereits in `r2dreamer/AI_CONTEXT.md` als kritisch dokumentiert,
   gilt genauso hier auf Config-Ebene.

### ⚠️ Gefundene Inkonsistenzen — Vorsicht


- **`safety_in_channels: 1` in `_base_.yaml` ist totes Config-Feld.** Laut Kommentar in
  `configs/configs.yaml` wird dieser Wert nicht mehr gelesen — er wird automatisch aus
  `dataset.raw_image_mode` abgeleitet. Nicht als wirksamen Override missverstehen.
- **`ctx_len: 128` in `_base_.yaml` ist der tatsächlich wirksame Wert** — nicht `16`.
  Der Code-Fallback in `dreamer.py` (`getattr(cfg, "ctx_len", 16)`) wird in der Praxis
  immer von diesem Config-Wert überschrieben. Die Tabelle in `r2dreamer/AI_CONTEXT.md`
  Abschnitt 5 listet nur den Code-internen Default (16) — das ist NICHT der Wert, mit dem
  tatsächlich trainiert wird. Kommentar bestätigt die Absicht:
  `ctx_len passt zu batch_length=256 (erste Hälfte = Kontext, zweite = Prediction)`.
- Gleiches Muster gilt vermutlich für weitere `getattr(cfg, "X", default)`-Fallbacks im
  Code — der jeweils **wirksame** Wert steht immer in der Config, nicht im Python-Default.
  Bei Unsicherheit über einen Laufzeitwert immer zuerst hier in `model/` (bzw. die
  überlagernden Top-Level-Configs) prüfen, nicht nur den Python-Quellcode.
