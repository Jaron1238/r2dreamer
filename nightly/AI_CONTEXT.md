# AI_CONTEXT.md — `r2dreamer/nightly/`

> **Zweck:** KI-Kontext-Datei für den `nightly/`-Ordner.  
> **Status:** Experimentell — kein Einfluss auf den stabilen Training-Loop.  
> **Letztes Update:** 2026-06-20

---

## 1. Zweck des Ordners

`nightly/` ist die **VLA-Erweiterungsschicht (Vision-Language-Action)** für R2Dreamer. Sie ergänzt die stabilen Actor-, Reward- und Value-Heads um **sprachliche Konditionierung** via `z_text`-Vektoren, ohne den bestehenden DreamerV3-Kern anzutasten.

**Aktivierung:** Ausschließlich über `cfg.vla_enabled = True` in `dreamer.py`. Standard: `False` → kein Code aus diesem Ordner wird ausgeführt.

### Modul-Übersicht

| Datei | Rolle | Pattern |
|---|---|---|
| `film.py` | FiLM-Primitive (γ⊙x + β) | Konditionierung über Skalar-Gates |
| `cross_attn.py` | Multi-Head Cross-Attention + KVCache | Konditionierung über Token-Sequenzen |
| `vla_heads.py` | VLA-Heads (Actor, Reward, Value) + TextEncoder | Ersetzt stabile Heads in Phase 3 |
| `navigator.py` | Reasoning-Navigator (LLM → JSON → z_text) | ~1Hz High-Level-Planer |
| `relabeling.py` | History-as-Intent Relabeling (VLM → z_text) | Background-Thread, Phase 3 |
| `__init__.py` | Öffentliche API des Pakets | Alle Symbole re-exportiert |

### 3-Phasen-Trainingsplan

| Phase | Was trainiert | Nightly-Code aktiv |
|---|---|---|
| 1 — Physik | RSSM, Encoder, alle Heads | ❌ Nein (vla_enabled = False) |
| 2 — Base Pilot | Actor via BC, SafetyNet | ⚠️ Optional: FiLM eingefügt, aber Identity-Init → nulleffekt |
| 3 — Language Coupling | **Nur FiLM-Params + Heads + StructuredProjector** | ✅ Vollständig aktiv |

---

## 2. Tech-Stack & Tools

| Library | Verwendung |
|---|---|
| `torch` / `torch.nn` | Alle Netzwerk-Klassen (Module, Parameter, Autograd) |
| `transformers` (>=4.36) | `TextEncoder` (MiniLM-L6), `MoondreamLabeler`, `ReasoningLLM` |
| `Pillow` | Frame-Collaging für Moondream (horizontal concat) |
| `numpy` | Frame-Arrays in `relabeling.py` |
| `threading` | Background-Worker in `RelabelBuffer._label_loop` |
| `queue.Queue` | Entkopplung Labeling-Queue (maxsize=32) |

> **Keine dieser Abhängigkeiten ist in `r2dreamer/requirements.txt`.**  
> Nightly-spezifische Pakete müssen separat installiert werden:
> ```bash
> pip install transformers>=4.36 sentence-transformers Pillow
> ```

---

## 3. Architektur-Regeln

### 3.1 `z_text`-Interface (KRITISCH — überall einheitlich)

```
z_text shape:  (B, 1, text_dim)        # für FiLMConditioner / StructuredProjector
z_tokens shape: (B, seq_len, text_dim)  # für CrossAttnMLP (TextEncoder.encode())
key_padding_mask: (B, seq_len) bool     # True = ignoriere diesen Token (HuggingFace-Konvention!)
```

- `TextEncoder.encode()` gibt **alle Token-Embeddings** zurück, **kein CLS-Pooling**.
- `CrossAttnMLP` attended über alle `seq_len=max_length=32` Tokens.
- `TextEncoder.make_null_tokens()` gibt Null-Tensor + Full-True-Maske → Cross-Attention-Contribution = 0 (Null-Fallback).

### 3.2 FiLM-Initialisierung = Identität (UNVERÄNDERLICH)

```python
# FiLMConditioner._init_identity()
bias[:units * n_layers] = 1.0   # γ-Bias → γ ≈ 1 bei z_text ≈ 0
bias[units * n_layers:] = 0.0   # β-Bias → β ≈ 0
nn.init.zeros_(last_linear.weight)  # Gradient startet bei 0
```

**Warum:** Phase 2 (BC Pre-Training) läuft durch FiLMedMLP hindurch, **ohne das Verhalten zu ändern**. Erst in Phase 3 (wenn `z_text ≠ 0`) beginnen FiLM-Parameter zu lernen.

**Niemals** die `_init_identity()`-Methode entfernen oder `FiLMConditioner` mit anderen Initialwerten starten — das würde Phase-2-Training brechen.

### 3.3 Cross-Attention `zero_gate`

```python
self.zero_gate = nn.Parameter(torch.zeros(1))
# In forward:
x = self.norm_attn(x + self.zero_gate * attn_out)
```

Gleiche Idee wie FiLM: Gate startet bei 0 → Cross-Attention-Contribution = 0 am Anfang. In Phase 3 lernt `zero_gate` einen positiven Wert.

### 3.4 KVCache — Inference-Optimierung

```python
# Einmal pro Navigator-Step (~1 Hz) precomputen:
cache = head.precompute_kv(z_tokens, key_padding_mask)

# Viele Male pro Step aufrufen (~50 Hz Actor-Calls):
action_dist = head.forward_cached(policy_input, cache)
```

`KVCache` ist **nicht** ein nn.Module — es ist ein einfaches Daten-Container. Es darf **nicht** in den Optimizer eingegeben werden.

### 3.5 VLA-Head Dimensionen

| Head | out_dim | dist_fn | symlog_inputs |
|---|---|---|---|
| `VLAActorHead` | `act_dim * 2` (µ+σ) | `bounded_normal` | False |
| `VLARewardHead` | `bin_num` (=255) | `symexp_twohot` | True |
| `VLAValueHead` | `bin_num` (=255) | `symexp_twohot` | True |

`VLAValueHead.outscale = 0.0` → Zero-Initialisierung des letzten Layers (DreamerV3-Konvention für Value-Heads).

### 3.6 NavigatorOutput-Struktur (vereinfacht!)

```python
@dataclass
class NavigatorOutput:
    intent:        str       # Imperativ-Satz ≤ 16 Wörter
    focus_objects: list[str] # Max 3 Objekte
    confidence:    float     # [0, 1]
```

> **⚠️ ACHTUNG:** Die README beschreibt eine reichere Struktur mit `roll_bias`, `pitch_bias`, `urgency` etc. Diese Felder existieren **nicht** im Code. `to_vector()` gibt nur `[confidence]` zurück (1D). Das `VECTOR_DIM = 1` muss mit `StructuredProjector(inp = base_text_dim + 1)` übereinstimmen.

### 3.7 StructuredProjector — Zero-Init + Phase-3-Optimizer

```python
# Zero-init final layer → Projector startet als Null-Signal
nn.init.zeros_(self.net[-1].weight)
nn.init.zeros_(self.net[-1].bias)
```

Der Projector muss **explizit** in den Phase-3-Optimizer eingetragen werden:
```python
optimizer.add_param_group({"params": projector.parameters(), "lr": 1e-4})
```
Er wird nicht automatisch durch `dreamer.py` trainiert.

### 3.8 MoondreamLabeler — Hardkodierte Revision

```python
_DEFAULT_ID  = "vikhyatk/moondream2"
_DEFAULT_REV = "2025-01-09"
```

**Niemals** die Revision ändern ohne zu testen — Moondream-API-Signaturen (`encode_image`, `answer_question`) können zwischen Revisions-Versionen brechen.

### 3.9 RelabelBuffer Threading-Regeln

```
Main Thread:    sample() → _enqueue_random_clip()  → _lookup_z_text()
Worker Thread:  _label_loop() → Moondream.label() → TextEncoder() → _cache
Lock:           threading.Lock() schützt _cache (dict)
```

- `_step` ist **nicht** thread-sicher, aber da er nur im Main-Thread inkrementiert wird ist das kein Problem.
- `queue.put_nowait()` **wirft keine Exception** bei vollem Queue — Clips werden still verworfen (Queue maxsize=32). Das ist **intentional**.
- `z_text` wird als `detach().cpu()` gecacht — kein GPU-Speicher-Leak in langen Trainings.

---

## 4. Wichtige Abhängigkeiten

### 4.1 Stabile Basis (`r2dreamer/`)

| Abhängigkeit | Verwendung in `nightly/` |
|---|---|
| `r2dreamer/distributions.py` | `bounded_normal`, `symexp_twohot` — von `vla_heads.py` importiert |
| `r2dreamer/dreamer.py` | Konsumiert `build_vla_heads()`, injiziert `z_text` in Heads |
| `r2dreamer/buffer.py` | `RelabelBuffer` wraps die Base-Buffer-Klasse |
| `r2dreamer/configs/configs.yaml` | `cfg.vla.text_dim`, `cfg.actor.units/layers/num_heads` |

### 4.2 Import-Pfad-Hack in `vla_heads.py`

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import distributions as dists
```

Dieser Hack setzt voraus, dass `vla_heads.py` von `r2dreamer/` aus aufgerufen wird (oder von einem Elternverzeichnis mit `r2dreamer/` im Suchpfad). **Nicht** von einem anderen Verzeichnis aus importieren.

### 4.3 Externe Modelle (Lazy-Loaded)

| Modell | Trigger | Größe |
|---|---|---|
| `all-MiniLM-L6-v2` (Sentence Transformers) | `TextEncoder.encode()` erster Aufruf | ~22 MB |
| `vikhyatk/moondream2` (Rev. 2025-01-09) | `MoondreamLabeler.label()` erster Aufruf | ~1.8 GB |
| `Qwen/Qwen2.5-3B-Instruct` | `ReasoningLLM.reason()` erster Aufruf | ~6 GB |

Alle Modelle cachen ihren Zustand in `self._model` — `_ensure_loaded()` ist idempotent.

### 4.4 Integration in `dreamer.py` (Referenz)

```python
# In __init__:
if cfg.get("vla_enabled", False):
    from nightly.vla_heads import build_vla_heads
    self.actor, self.reward, self.value = build_vla_heads(
        cfg,
        actor_input_dim  = actor_input_dim,
        reward_input_dim = self.rssm.feat_size,
        value_input_dim  = actor_input_dim,
        text_dim         = int(cfg.vla.text_dim),   # z.B. 384
    )

# In forward / imagine-Loop:
z_text = batch.get("z_text",
    torch.zeros(B, 1, self._vla_text_dim, device=self.device))
action_dist = self.actor(policy_input, z_text)       # CrossAttnMLP erwartet (B,1,384)
reward_dist = self.reward(feat, z_text)
value_dist  = self.value(policy_input, z_text)
```

---

## 5. Limitierungen & Fallstricke

### ❌ VERBOTEN

| Aktion | Grund |
|---|---|
| `_init_identity()` entfernen | Bricht Phase-2-Training (FiLM würde random stören) |
| `cross_attn`'s `zero_gate` entfernen | Gleicher Grund: Identitäts-Garantie beim Start |
| `moondream2`-Revision ändern (2025-01-09) | API-Signaturen können brechen |
| `z_text`-Shape von `(B,1,384)` ändern | Bricht FiLMConditioner und alle VLA-Heads |
| `StructuredProjector` vergessen im Phase-3-Optimizer | Projector trainiert sich nicht → Navigator ohne Effekt |
| `TextEncoder` mit CLS-Pooling wrappen | CrossAttnMLP braucht alle Token, kein CLS-Pool |
| Modell von außerhalb `r2dreamer/` importieren | sys.path-Hack in `vla_heads.py` bricht |

### ⚠️ EDGE CASES & GOTCHAS

- **`NavigatorOutput` vs. README:** Die README beschreibt `roll_bias`, `pitch_bias`, `urgency` — diese Felder **existieren nicht** im `@dataclass`. `to_vector()` gibt nur `[confidence]` zurück. `VECTOR_DIM = 1`.
- **`TextEncoder` max_length=32:** Befehle länger als 32 Tokens werden truncated. Für kurze Piloten-Befehle ist das ausreichend; für längere Navigations-Beschreibungen kann Information verloren gehen.
- **`ReasoningNavigator.step()` warnt bei >900ms:** Bei langsamem LLM (kein MPS, großes Modell) wird nur gewarnt, nicht abgebrochen. Der Caller muss eigenständig ein Timeout implementieren.
- **`_extract_json()`:** Nimmt immer den **letzten** `{...}`-Block im LLM-Output. Wenn das Modell kein valides JSON ausgibt, gibt es `{}` zurück → `NavigatorOutput.identity()`.
- **`RelabelBuffer._lookup_z_text()` Fallback:** Wenn keine Segment-Labels vorhanden sind (Queue leer, erste Schritte), gibt die Funktion `torch.zeros(B, 1, 384)` zurück — FiLM/Cross-Attn wirken als Identität.
- **`MoondreamLabeler._frames_to_pil()`:** Frames werden **horizontal** konkateniert (axis=1). Für `T=16` Frames bei `H=288, W=512` ergibt das ein `288 × 8192`-Bild — potentiell sehr breit. Moondream2 kann mit solchen Aspektverhältnissen umgehen, aber `T` sollte ≤ 16 bleiben.
- **Phase-3-Freeze:** RSSM und Encoder müssen explizit eingefroren werden (`param.requires_grad = False`). `nightly/` selbst enthält keine Freeze-Logik.
- **`CrossAttnMLP.forward()` vs. `.forward_cached()`:** Training → `forward()` (kv wird jedes Mal neu projiziert). Inference mit festem z_text → `precompute_kv()` einmal aufrufen, dann `forward_cached()` für jeden Actor-Aufruf.
