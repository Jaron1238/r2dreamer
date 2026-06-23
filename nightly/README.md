# R2Dreamer — Nightly: VLA Extension

> **Status:** experimental — zero changes to the stable training loop.  
> Opt in via `cfg.vla_enabled = True`.

---

## Konzept: Hierarchische Intelligenz

| Ebene | Gerät | Frequenz | Aufgabe |
|-------|-------|----------|---------|
| High-Level | MacBook | ~1 Hz | LLM/VLM als Navigator — übersetzt Befehle in `z_text` |
| Low-Level | Pi Zero / Drohne | 50 Hz | R2Dreamer-Actor als Pilot — setzt `z_text` in Motorsteuerung um |

---

## Dateien

```
nightly/
├── film.py         FiLM-Primitive (FiLMLayer, FiLMConditioner, FiLMedMLP)
├── vla_heads.py    Language-konditionierte Heads + TextEncoder
├── relabeling.py   History-as-Intent Relabeling (Moondream + RelabelBuffer)
└── README.md       Diese Datei
```

---

## Architektur: FiLM-Modulation

Anstatt `z_text` einfach zu konkatenieren, wirkt es über **Feature-wise Linear
Modulation** als „Mischpult" auf die Hidden States der Heads:

```
z_text ──► FiLMConditioner ──► (γ_l, β_l) für jeden Layer l
                                         │
policy_feat ──► Linear ──► Norm ──► Act ──► γ_l ⊙ x + β_l ──► ...
```

**Warum das funktioniert:**  
- γ ≈ 1, β ≈ 0 bei Initialisierung → FiLM startet als Identität.
- Phase 2 (BC Pre-Training) läuft unverändert durch.
- In Phase 3 werden *nur* FiLM-Parameter + Heads trainiert; RSSM + Encoder bleiben eingefroren.

---

## Integration in `dreamer.py`

```python
# __init__  (nach bestehenden Head-Konstruktionen)
if cfg.get("vla_enabled", False):
    from nightly.vla_heads import build_vla_heads
    text_dim = int(cfg.vla.text_dim)          # z.B. 384 (MiniLM-L6)
    self.actor, self.reward, self.value = build_vla_heads(
        cfg,
        actor_input_dim  = actor_input_dim,
        reward_input_dim = self.rssm.feat_size,
        value_input_dim  = actor_input_dim,
        text_dim         = text_dim,
    )
    self._vla_text_dim = text_dim

# forward / imagine-Loop  (überall wo actor/reward/value aufgerufen werden)
z_text = batch.get("z_text",
    torch.zeros(B, 1, self._vla_text_dim, device=self.device))

action_dist  = self.actor(policy_input,  z_text)
reward_dist  = self.reward(feat,         z_text)
value_dist   = self.value(policy_input,  z_text)
```

---

## History-as-Intent Relabeling

```
Drohne fliegt (Sim / Phase 2)
        │
        ▼
ClipSampler  ──►  Moondream2  ──►  „Fly a left arc around the tree"
                                          │
                                          ▼
                                   TextEncoder  ──►  z_text (384-d)
                                          │
                                          ▼
                              RelabelBuffer.cache[ep_id]
                                          │
                             injiziert z_text in jeden Batch
```

### Beispiel

```python
from nightly.relabeling import MoondreamLabeler, RelabelBuffer
from nightly.vla_heads   import TextEncoder

labeler  = MoondreamLabeler(device="mps")
text_enc = TextEncoder(device="mps")
rbuf     = RelabelBuffer(base_buffer, labeler, text_enc,
                         label_every=500, text_dim=384)

# Im Trainings-Loop (unverändert, außer Buffer-Swap):
batch = rbuf.sample(batch_size)
z_text = batch["z_text"]          # (B, 1, 384) — oder Null-Fallback
```

---

## 3-Phasen-Trainingsplan

| Phase | Was trainiert | FiLM |
|-------|--------------|------|
| 1 — Physik | RSSM, Encoder | nicht vorhanden |
| 2 — Base Pilot | Actor via BC, SafetyNet | als Identität eingefügt (keine Verhaltensänderung) |
| 3 — Language Coupling | **Nur FiLM-Parameter + Heads** | aktiv, via VLM-Relabeling |

---

---

## Option B: Reasoning Navigator (structured JSON → z_text)

```
Telemetry + command
      │
      ▼
ReasoningLLM  (Qwen2.5-3B / Phi-4-mini, ~300–500 ms)
      │  denkt nach, gibt JSON aus
      ▼
NavigatorOutput
  primary_intent:  "fly through the left gap"
  roll_bias:       -0.25
  pitch_bias:       0.10
  urgency:          0.7
  focus_objects:   ["gap", "branch"]
  ...
      │
      ├──► TextEncoder(primary_intent)   → text_emb  (1, 384)
      └──► to_vector()                   → num_vec   (1, 9)
                          │
                          ▼
                 StructuredProjector    (trainable, ~200k params)
                          │
                          ▼
                      z_text  (1, 1, 384)  ──►  FiLM heads
```

### Warum nicht plain text?

| | Plain sentence | Structured JSON |
|--|--|--|
| Biases | implizit in Wortbedeutung | explizit als Float |
| Gradients | durch Embedding, verlustreich | durch Projektor, direkt |
| Debugging | Black box | voll lesbar |
| Latenz | TextEncoder reicht | +LLM, aber ≤500 ms |

### Nutzung

```python
from nightly.navigator import build_reasoning_navigator

nav, projector = build_reasoning_navigator(device="mps")
# projector in Phase 3 optimizer registrieren:
optimizer.add_param_group({"params": projector.parameters(), "lr": 1e-4})

# jede Sekunde:
z_text, nav_out = nav.step(
    command    = "fly through the gap on the left",
    speed_mps  = 3.2,
    roll_deg   = -4.1,
    crash_prob = 0.12,
)
# z_text (1, 1, 384) → TCP → Drohne → FiLM heads
print(nav_out.primary_intent)   # "fly through left gap"
print(nav_out.roll_bias)        # -0.25
```

---

## Abhängigkeiten (nur für Nightly)

```
sentence-transformers   # TextEncoder (MiniLM-L6, ~22 MB)
transformers>=4.36      # MoondreamLabeler
Pillow                  # Clip → PIL für Moondream
```

Keine dieser Abhängigkeiten wird von der stabilen R2Dreamer-Basis importiert.
