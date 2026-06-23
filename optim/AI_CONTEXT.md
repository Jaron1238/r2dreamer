# AI_CONTEXT.md — r2dreamer/optim/

> **Für zukünftige KIs:** Dieser Ordner enthält zwei eigenständige PyTorch-Optimierungsbausteine
> (LaProp-Optimizer + AGC Gradient Clipping), die direkt aus der DreamerV3-Architektur stammen.

---

## 1. Zweck des Ordners

`optim/` enthält zwei eigenständige PyTorch-Optimierungs-Bausteine, beide direkt aus der
DreamerV3-Originalarchitektur übernommen (DreamerV3 nutzt im offiziellen JAX-Code exakt
diese Kombination):

| Datei         | Inhalt                                                                    |
|-----------------|------------------------------------------------------------------------------|
| `laprop.py`     | `LaProp` — Adam-Variante, bei der das Momentum NACH statt VOR der RMS-Normalisierung des Gradienten berechnet wird |
| `agc.py`        | `clip_grad_agc_` — Adaptive Gradient Clipping (NFNets-Paper), klippt pro Parameter-Tensor statt global |

Beide werden ausschließlich von `dreamer.py` konsumiert (`from optim import LaProp,
clip_grad_agc_`) — `optim/` selbst hat keine Kenntnis vom Rest des Projekts.

---

## 2. Tech-Stack & Tools

| Komponente                          | Verwendung                                                         |
|----------------------------------------|------------------------------------------------------------------------|
| **`torch.optim.Optimizer`**             | Basisklasse für `LaProp`                                            |
| **`torch.utils._foreach_utils`** (privat)| `agc.py` nutzt interne, unterstrichene PyTorch-Hilfsfunktionen für vektorisierte Multi-Tensor-Operationen |
| **`torch._foreach_*`**                  | Low-Level vektorisierte Tensor-Ops für performantes Batch-Clipping über alle Parameter eines Device/Dtype-Buckets |

---

## 3. Architektur-Regeln (KRITISCH)

### 3.1 LaProp ≠ Adam — Momentum-Reihenfolge ist invertiert

```python
denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)   # RMS-Normalisierung ZUERST
step_of_this_grad = grad / denom                               # normalisierter Gradient
exp_avg.mul_(beta1).add_(... step_of_this_grad)                 # Momentum DANACH über den normalisierten Gradienten
```

Bei Adam wird zuerst der Momentum-Term (`exp_avg`) über dem rohen Gradienten gebildet und
erst am Ende durch `sqrt(exp_avg_sq)` geteilt. LaProp dreht das um: Der Gradient wird
**zuerst** durch seine RMS-Schätzung normalisiert, **dann** läuft das Momentum über den
bereits normalisierten Wert. Das entkoppelt die effektive Lernrate von der
Momentum-Schätzung — der namensgebende Effekt ("Learning-rate-adaptive Momentum").

### 3.2 Bias-Korrektur über EMA der tatsächlichen Lernrate, nicht über `beta^t`

```python
state["exp_avg_lr_1"] = state["exp_avg_lr_1"] * beta1 + (1 - beta1) * group["lr"]
bias_correction1 = state["exp_avg_lr_1"] / group["lr"] if group["lr"] != 0.0 else 1.0
```

Anders als Standard-Adam (`bias_correction1 = 1 - beta1**step`) verwendet LaProp eine
laufende EMA der tatsächlich angewendeten Lernrate selbst als Korrekturnenner. Das ist
**wichtig bei sich änderndem LR-Schedule** (Warmup/Decay) — die Bias-Korrektur passt sich
automatisch an, statt nur von der Schrittzahl abzuhängen.

### 3.3 `centered`-Modus erst nach 10 Schritten aktiv

```python
self.steps_before_using_centered = 10
...
if state["step"] > self.steps_before_using_centered:
    mean = exp_mean_avg_beta2**2
    denom = denom - mean
```

In den ersten 10 Schritten verhält sich `centered=True` identisch zu `centered=False`
(reines RMSProp-artiges Verhalten) — verhindert, dass eine zu früh instabile
Varianzschätzung (`exp_mean_avg_beta2`) zu negativen oder nahezu-Null-`denom`-Werten
führt.

### 3.4 `amsgrad` + `centered` gleichzeitig — Sonderfall in Zeile 82

```python
if amsgrad and not (centered and state["step"] <= self.steps_before_using_centered):
    torch.max(max_exp_avg_sq, denom, out=max_exp_avg_sq)
    denom = max_exp_avg_sq
```

Ist `centered=True` UND wir sind noch innerhalb der ersten 10 Schritte, wird `amsgrad`
faktisch übersprungen (`max_exp_avg_sq` wird nicht aktualisiert/verwendet). Erst danach
greifen beide Mechanismen zusammen. Bei der Fehlersuche von frühen
Trainings-Instabilitäten (erste ~10 Schritte) relevant.

### 3.5 AGC klippt PRO PARAMETER-TENSOR, nicht global

```python
upper = clip * max(||W||_2, pmin)
scale = 1 / max(||grad||_2 / upper, 1.0)
```

Jeder Parameter-Tensor bekommt seine eigene Clip-Schwelle relativ zu seiner eigenen Norm
— im Gegensatz zu `torch.nn.utils.clip_grad_norm_`, das einen einzigen globalen Norm-Wert
über alle Parameter berechnet. `pmin` ist ein Floor, der verhindert, dass
Parameter mit sehr kleiner Norm (z. B. Bias-Vektoren nahe Null) übermäßig oder durch
Division-nahe-Null instabil geklippt werden.

---

## 4. Wichtige Abhängigkeiten

```
dreamer.py
  ├── from optim import LaProp, clip_grad_agc_
  ├── Zeile ~368: LaProp(params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps, weight_decay=cfg.weight_decay, ...)
  └── Zeile ~253: clip_grad_agc_(params, float(cfg.agc), float(cfg.pmin), foreach=True)

configs/model/_base_.yaml
  └── liefert lr, betas (beta1/beta2), eps, weight_decay, agc, pmin als Hyperparameter
```

`optim/` selbst importiert nichts aus dem restlichen Projekt — reine, in sich
geschlossene PyTorch-Bausteine.

---

## 5. Limitierungen & Fallstricke

### ⚠️ Vorsicht

- `agc.py` hängt von privaten, unterstrichenen PyTorch-Internals ab
  (`_group_tensors_by_device_and_dtype`, `_has_foreach_support`,
  `_device_has_foreach_support` aus `torch.utils._foreach_utils`). Diese sind **keine
  öffentlich garantierte API** — ein PyTorch-Minor-Update kann diese Funktionen
  umbenennen oder entfernen, ohne dass es im Changelog als Breaking Change auftaucht.
- `clip_grad_agc_` mit `foreach=False` UND einem Device ohne Foreach-Unterstützung
  (`elif not foreach:` Zweig) läuft pro-Parameter in einer Python-Schleife — bei vielen
  kleinen Parametern spürbar langsamer als der `foreach=True`-Pfad. `dreamer.py` ruft
  bewusst `foreach=True` auf.
