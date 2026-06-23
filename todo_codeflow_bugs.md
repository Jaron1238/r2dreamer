# Dateiname: todo_codeflow_bugs.md

> Vollständiger Code-Flow-Trace ab den Einstiegspunkten `train.py`, `trainer.py`, `dreamer.py`, `rssm.py`, `networks.py`, `envs/drone_sim.py`, `optim/`, `native/mlx_*`, `native/coreml_export.py`, `nightly/`, `fly_real.py`, `tools.py` und `distributions.py`. 18 Befunde, sortiert nach Schweregrad.

## 🔴 Kritische Fehler (Crashes, NaNs, Deadlocks)

### 1. `optim/laprop.py` — `LaProp.step()` fehlt `closure`-Parameter → TypeError bei **jedem** Optimizer-Schritt
- **Execution Path:** `train.py` → `OfflineTrainer.begin()` (Phase 1/2) **und** `OnlineTrainer.begin()` (Phase 3) → `self.accelerator.prepare(agent, optimizer)` → `optimizer.step()` (`trainer.py:752` bzw. via `Dreamer.train_step()` → `dreamer.py:667` mit `scaler=None`, explizit so von `OnlineTrainer.begin()` übergeben, `trainer.py:856-863`) → `accelerate.optimizer.AcceleratedOptimizer.step(closure=None)` → da `mixed_precision="fp16"` (`trainer.py:613`/`796`) ist `self.scaler is not None` → `self.scaler.step(self.optimizer, closure)` = `GradScaler.step(optimizer, None)` → `optimizer.step(None)`.
- **Der Fehler:** `LaProp.step(self)` (Zeile 22) akzeptiert **keinerlei** Argument außer `self`. `GradScaler.step()` reicht aber bei aktivem Scaler immer `closure` (hier `None`) **positional** an `optimizer.step()` durch (verifiziert anhand des `accelerate`-Quellcodes, `accelerate/optimizer.py::AcceleratedOptimizer.step`). Das Ergebnis ist `TypeError: step() takes 1 positional argument but 2 were given` — und das bereits beim **allerersten** Gradientenschritt. Da sowohl `OfflineTrainer` als auch `OnlineTrainer` `mixed_precision="fp16"` fest verdrahten und beide letztlich `optimizer.step()` ohne Argumente aufrufen (was durch die Accelerate-Wrapper-Schicht zu `optimizer.step(None)` wird), ist **kein einziger Trainingslauf über `train.py` lauffähig**, unabhängig von Phase.
- **Zeile/Ort:** `optim/laprop.py:22`
- **Lösung:**
  ```python
  def step(self, closure=None):
      loss = None
      if closure is not None:
          with torch.enable_grad():
              loss = closure()
      for group in self.param_groups:
          ...
      return loss
  ```

### 2. `native/mlx_trainer.py` + `train.py` — MLX-Trainingspfad ist vollständig nicht-funktional (falscher Modell-Typ wird übergeben)
- **Execution Path:** `train.py:39` `agent = Dreamer(config).to(config.device)` (**PyTorch** `nn.Module`) → `train.py:60-63` `elif bool(getattr(config, "use_mlx", False)): trainer = MLXOnlineTrainer(...); trainer.begin(agent)` → `MLXOnlineTrainer.begin(self, model: nn.Module, env=None)` (`mlx_trainer.py:218`, hier ist `nn` = `mlx.nn`!) → `self._engine = _MLXUpdateEngine(model, optimizer, ...)` → `_MLXUpdateEngine.__init__` (`mlx_trainer.py:111-117`) `model.trainable_parameters()`.
- **Der Fehler:** `agent` ist und bleibt durchgehend die **PyTorch**-`Dreamer`-Instanz aus `train.py:39` — es wird an keiner Stelle ein tatsächliches MLX-Modell (`mlx_models.py`) instanziiert oder per `mlx_utils.load_pytorch_to_mlx()` befüllt. `torch.nn.Module` besitzt keine Methode `trainable_parameters()` (das ist MLX-API). Erster Aufruf von `model.trainable_parameters()` wirft `AttributeError: 'Dreamer' object has no attribute 'trainable_parameters'`. Der gesamte `use_mlx`-Zweig stürzt also bereits in der Konstruktion der Update-Engine ab, bevor auch nur ein Batch geladen wird.
- **Zeile/Ort:** `train.py:60-63` (fehlende Modell-Konstruktion) i.V.m. `native/mlx_trainer.py:117`
- **Lösung:** In `train.py` im `use_mlx`-Zweig ein echtes `mlx_models`-Modell aufbauen und die PyTorch-Gewichte via `mlx_utils.load_pytorch_to_mlx()` übertragen, statt der PyTorch-`agent`-Instanz:
  ```python
  elif bool(getattr(config, "use_mlx", False)):
      from native.mlx_models import MLXDreamer          # exakter Klassenname prüfen
      from native.mlx_utils import load_pytorch_to_mlx
      mlx_agent = MLXDreamer(config)
      report = load_pytorch_to_mlx(agent.state_dict(), mlx_agent)
      print(report["skipped"], report["missing"])        # Pflicht-Check laut native/AI_CONTEXT.md §3.1
      trainer = MLXOnlineTrainer(config.trainer, logger, logdir)
      trainer.begin(mlx_agent)
  ```

### 3. `dreamer.py` — `compute_losses()` Phase ≥ 3: Shape-Mismatch zwischen `boot` und `last/term/reward/value` in `_lambda_return()` bei aktivem `burn_in_mask`
- **Execution Path:** `compute_losses()` (Phase ≥ 3, `dreamer.py:963`) → `flat_mask = valid_mask.reshape(-1)` mit `valid_mask = burn_in_mask` (falls vorhanden, `dreamer.py:962`) → `start = (flat_post_stoch[flat_mask], flat_post_deter[flat_mask])` → `self._imagine(start, ...)` liefert `N = B·T_valid` Rollouts (`T_valid = T - burn_in_steps < T`) → `ret = self._lambda_return(...)` mit Shape `(N, imag_horizon, 1)` → `boot = ret[:, 0].reshape(B, -1, 1)[:, :T]` (`dreamer.py:1024`) → zweiter Aufruf `ret = self._lambda_return(last, term, reward, value, boot, disc, self.lamb)` (`dreamer.py:1030`) mit `last/term/reward/value` in Shape `(B, T, 1)`.
- **Der Fehler:** `ret[:, 0].reshape(B, -1, 1)` ergibt exakt `(B, T_valid, 1)` (da `N = B·T_valid`). Der nachfolgende Slice `[:, :T]` ist dann **kein** echtes Trimmen, sondern liefert wegen `T_valid < T` weiterhin nur `(B, T_valid, 1)` zurück (Python-Slicing über das Tensor-Ende hinaus wirft keinen Fehler, sondern clamped). `boot` hat damit eine andere Shape als `last/term/reward/value` (`(B, T, 1)`), wodurch der `assert last.shape == term.shape == reward.shape == value.shape == boot.shape` in `_lambda_return()` (`dreamer.py:1203`) fehlschlägt → `AssertionError`, Training bricht ab.
  Aktuell ist dieser Pfad **latent**, weil `OnlineTrainer.begin()` (`trainer.py:820-851`) beim Aufbau der Transition **kein** `burn_in_mask`-Feld setzt — `data.get("burn_in_mask", None)` ist also für den heutigen Phase-3-Online-Pfad immer `None`, wodurch `valid_mask` auf "alles gültig" fällt und `T_valid == T` gilt (kein Crash). Der Code an `dreamer.py:793-795/871-872/963` behandelt `burn_in_mask` aber explizit als phasen-übergreifendes, generisches Konzept — sobald `burn_in_mask` (z. B. nach einer naheliegenden Erweiterung des Online-Buffers um Burn-in-Markierung nach `reset()`, oder bei Wiederverwendung von `compute_losses()` mit Offline-Daten in Phase ≥ 3) tatsächlich nicht-trivial gesetzt ist, crasht dieser Pfad zuverlässig.
- **Zeile/Ort:** `dreamer.py:1024` (Bootstrap-Reshape) i.V.m. `dreamer.py:1203` (Assert)
- **Lösung:** `boot` muss auf volle Breite `T` zurück-gepaddet werden (z. B. mit dem letzten gültigen Wert oder 0 für die Burn-in-Positionen), statt mit `[:, :T]` zu hoffen, dass die Dimension bereits passt:
  ```python
  boot_valid = ret[:, 0].reshape(B, T_valid, 1)
  boot = torch.zeros(B, T, 1, device=boot_valid.device, dtype=boot_valid.dtype)
  boot[:, -T_valid:] = boot_valid   # oder analoge Platzierung passend zu valid_mask
  ```
  Alternativ defensiv: `assert boot.shape[1] == T, f"boot hat {boot.shape[1]} statt {T} Zeitschritte – burn_in_mask invalidiert die Reshape-Annahme"` direkt nach Zeile 1024 einfügen, damit der Fehler nicht erst tief in `_lambda_return()` sichtbar wird.

---

## 🟠 Logik- & Silent-Bugs (Falsche Berechnungen, Shape-Mismatches die broadcasten)

### 4. `trainer.py` — `FPVDataset`: reale Aktionen werden bei Standard-Config **immer** durch Nullen ersetzt
- **Execution Path:** `train.py` → `FPVDataset(config, ..., require_osd=bool(config.dataset.get("require_osd", False)))` (Default `False`, `configs.yaml`) → `FPVDataset.__iter__()` → `trainer.py:241-245`.
- **Der Fehler:**
  ```python
  if self.require_osd and sample.get("has_osd", False):
      actions_full = torch.from_numpy(np.array(sample["actions"][0])).float()
      actions = actions_full[start : start + self.batch_length]
  else:
      actions = torch.zeros((self.batch_length, self.action_dim), dtype=torch.float32)
  ```
  Die Bedingung verknüpft `self.require_osd` (ein einmaliger, globaler Dataset-Flag) **UND** `sample.get("has_osd")` (pro Sample). Bei Standard-Config (`require_osd=False`, explizit so gewählt, weil sonst laut Kommentar "~90% der Daten gefiltert" würden) ist die erste Bedingung **immer** `False` — unabhängig davon, ob das einzelne Sample tatsächlich reale OSD-Telemetrie besitzt. Damit ist `data["action"]` für **jeden** Trainingslauf mit Default-Config ein Null-Tensor. Das zerstört direkt: RSSM-Dynamics-Konditionierung (`dyn`/`rep`-Loss lernt "Aktion → Nicht-Veränderung"), `InverseDynamicsHead`, BC-Loss in Phase 2 (`losses["bc"]`, trainiert den Actor auf konstante Null-Aktion = "Drohne tut nichts"), und die Reward-/Cont-Konditionierung in Phase 3. Zum Vergleich: `_extract_osd()` (`trainer.py:299` ff.) prüft korrekt **nur** `sample.get("has_osd", False)`, ohne `self.require_osd` — das bestätigt, dass die Verknüpfung an Zeile 241 ein Copy-Paste-/Logikfehler ist und nicht beabsichtigt.
- **Zeile/Ort:** `trainer.py:241`
- **Lösung:**
  ```python
  if sample.get("has_osd", False):
      actions_full = torch.from_numpy(np.array(sample["actions"][0])).float()
      actions = actions_full[start : start + self.batch_length]
  else:
      actions = torch.zeros((self.batch_length, self.action_dim), dtype=torch.float32)
  ```
  `self.require_osd` sollte ausschließlich für das optionale Pre-Filtering des Datasets (`self.ds = self.ds.filter(...)`) verwendet werden, nicht für die Per-Sample-Aktionsextraktion.

### 5. `trainer.py` — `FPVDataset`: `drone_id` ist wegen nie gesetztem `self.model_num_drones` immer `0`
- **Execution Path:** `FPVDataset.__iter__()` → `trainer.py:271`.
- **Der Fehler:**
  ```python
  drone_id = int(drone_value) % self.model_num_drones if hasattr(self, "model_num_drones") else 0
  ```
  `self.model_num_drones` wird **nirgendwo** in `FPVDataset.__init__` (oder sonst in `trainer.py`) gesetzt. `hasattr(self, "model_num_drones")` ist daher für jede Instanz und jedes Sample `False`, sodass `drone_id` immer `0` ist — der tatsächlich aus den Metadaten gelesene `drone_value` wird komplett verworfen. Das torpediert das gesamte Multi-Drone-Disentanglement-Konzept: `self.drone_embed(data["drone_id"])` (`dreamer.py:98-99`, genutzt als "warmer" Teacher für den `ContextEncoder` in Phase 1, s. `dreamer.py` `else`-Zweig um Zeile 760 ff.) liefert für **alle** Drohnen denselben konstanten Embedding-Vektor (Zeile 0 der Embedding-Tabelle), wodurch der Context-Encoder nie lernt, drohnenspezifische Eigenheiten aus der Trajektorie zu inferieren.
- **Zeile/Ort:** `trainer.py:271`
- **Lösung:** `self.model_num_drones` im Konstruktor setzen, z. B. `self.model_num_drones = int(getattr(config.model, "num_drones", 100))`, und den `hasattr`-Fallback entfernen:
  ```python
  drone_id = int(drone_value) % self.model_num_drones
  ```

### 6. `trainer.py` + `dreamer.py` — Stale Config-Key `use_depth` statt `use_depth_aux` → Depth-Aux-Head bekommt nie echte Targets
- **Execution Path:** `train.py` → `FPVDataset.__init__` (`trainer.py:106`) `self.use_depth = bool(getattr(config, "use_depth", False))` → steuert, ob `self.depth_preprocessor` (DepthAnythingV2, On-the-fly-Tiefenschätzung) überhaupt instanziiert wird → `__iter__()`: ohne precomputed `frames_depth` im Dataset fällt der Code auf `depth = torch.zeros((self.batch_length, self.img_height, self.img_width, 1))` zurück → `data["depth_target"]` ist konstant Null → `dreamer.py:1095` `if bool(getattr(self.config.model, "use_depth_aux", False)) and "depth_target" in data:` (hier korrekt `use_depth_aux`!) berechnet `losses["depth_aux"]` gegen dieses Null-Target.
- **Der Fehler:** Der Konfig-Schlüssel heißt überall in `configs.yaml` **`use_depth_aux`** (`configs.yaml:14`, interpoliert nach `model.use_depth_aux`), wurde aber in `trainer.py:106`, `trainer.py:607` (Augmentation-Konstruktion) sowie `dreamer.py:40` (totes Attribut `self.use_depth`, wird nirgends weiter benutzt) noch mit dem alten/falschen Namen `use_depth` abgefragt. Da dieser Schlüssel nie existiert, liefert `getattr(..., "use_depth", False)` **immer** `False` — selbst wenn der Nutzer `use_depth_aux: True` setzt, bleibt `FPVDataset.use_depth` auf `False`, der DepthPreprocessor wird nie aktiviert, und der Depth-Aux-Head wird (sofern kein precomputed `frames_depth` im HF-Dataset vorliegt) ausschließlich gegen ein Null-Bild trainiert — eine stillschweigend nutzlose Hilfsaufgabe trotz korrekt aktivierter Config.
- **Zeile/Ort:** `trainer.py:106`, `trainer.py:607`, `dreamer.py:40`
- **Lösung:** Alle drei Stellen auf `use_depth_aux` umstellen:
  ```python
  self.use_depth = bool(getattr(config, "use_depth_aux", False))   # trainer.py:106
  use_depth=bool(getattr(config, "use_depth_aux", False)),          # trainer.py:607
  self.use_depth = bool(getattr(config, "use_depth_aux", getattr(cfg, "use_depth_aux", False)))  # dreamer.py:40
  ```

### 7. `trainer.py` — `Augmentation.phase` liest aus dem falschen Config-Scope → ist immer `1`, Flip-Augmentation wird in Phase 2 nie deaktiviert
- **Execution Path:** `train.py` → `OfflineTrainer(config.trainer, dataset, logger, logdir)` → `OfflineTrainer.__init__` (`trainer.py:604-608`) `self.augmentation = Augmentation(config.augmentation, phase=int(getattr(config, "phase", 1)), ...)`. Hier ist `config` **innerhalb von `OfflineTrainer.__init__`** bereits `config.trainer` (so von `train.py` übergeben) — `config.trainer` besitzt aber kein `phase`-Feld (das liegt nur top-level in `configs.yaml`, z. B. `phase: 2`).
- **Der Fehler:** `getattr(config.trainer, "phase", 1)` liefert deshalb für **jede** Phase immer den Default `1`. `Augmentation.__init__` setzt `self.phase = int(phase)` direkt aus diesem Argument (`trainer.py:508`), und `self.use_flip = bool(config.use_flip) and self.phase != 2` (`trainer.py:514`) wertet `self.phase != 2` daher **immer als `True`** aus. Horizontal-Flip-Augmentation wird also auch während tatsächlichem Phase-2-BC-Training nie deaktiviert, obwohl genau das offensichtlich beabsichtigt war (gespiegelte Bilder ohne mitgespiegelte Aktionslabels verfälschen das Steuerungssignal asymmetrisch, z. B. "links/rechts"-Roll-Aktionen).
- **Zeile/Ort:** `trainer.py:606`
- **Lösung:** Den top-level `phase`-Wert separat durchreichen, statt ihn aus `config.trainer` zu lesen — z. B. `OfflineTrainer.__init__` um einen expliziten `phase`-Parameter erweitern, den `train.py` mit `config.phase` befüllt:
  ```python
  # train.py
  trainer = OfflineTrainer(config.trainer, dataset, logger, logdir, phase=int(config.phase))
  # trainer.py OfflineTrainer.__init__(self, config, dataset, logger, logdir, phase=1):
  self.augmentation = Augmentation(config.augmentation, phase=phase, use_depth=...)
  ```

### 8. `networks.py` + `dreamer.py` — Differenzierte Lernrate für den Encoder-Backbone wird nie angewendet (String-Mismatch `backbone_early`/`backbone_late` vs. `.backbone.`)
- **Execution Path:** `dreamer.py:321` `_get_encoder_param_groups()` → filtert `self._named_params` nach Substring `".backbone."` → `ConvEncoder` (`networks.py:317` ff.) benennt die Submodule jedoch `self.backbone_early` (Zeile 375) und `self.backbone_late` (Zeile 376), nicht `self.backbone`.
- **Der Fehler:**
  ```python
  backbone_params = [param for name, param in encoder_named_params if ".backbone." in name]
  adapter_params  = [param for name, param in encoder_named_params if ".backbone." not in name]
  ```
  Da die tatsächlichen Parameter-Pfade `...backbone_early...` / `...backbone_late...` lauten, enthält die Substring `".backbone."` (mit Punkt **vor und nach**) niemals einen Treffer — `".backbone_early."` ≠ `".backbone."`. `backbone_params` ist daher **immer leer**, `if backbone_params:` (Zeile 339) greift nie, und **alle** Encoder-Parameter landen in `adapter_params` mit der vollen Lernrate `self._base_lr`. Die explizit vorgesehene 0.3×-Lernrate für den vortrainierten EfficientNetV2-S-Backbone (Schutz vor Catastrophic Forgetting der ImageNet-Gewichte während des Fine-Tunings) wird dadurch komplett umgangen — der gesamte Backbone wird mit der vollen Adapter-Lernrate trainiert.
- **Zeile/Ort:** `dreamer.py:330` (`".backbone." in name`) i.V.m. `networks.py:375-376`
- **Lösung:**
  ```python
  backbone_params = [param for name, param in encoder_named_params
                      if ".backbone_early." in name or ".backbone_late." in name]
  ```

### 9. `envs/drone_sim.py` + `reward.py` — Online-Reward-Curriculum (`DroneRewardFunction._phase_weights[0..2]`) ist durch Namespace-Kollision und Clip auf Preset `2` eingefroren
- **Execution Path:** `train.py:58` `if int(getattr(config, "phase", 1)) >= 3: ... env = DroneSimEnv(config)` → `DroneSimEnv.__init__` (`envs/drone_sim.py:41`) `self.phase = int(np.clip(int(getattr(config, "phase", 0)), 0, 2))` → `self.reward_fn.set_phase(self.phase)` (Zeile 42-43, **einmalig**, kein weiterer Aufrufer im gesamten Repo — verifiziert via Grep).
- **Der Fehler:** `DroneSimEnv` wird laut `train.py` ausschließlich für globale Trainings-Phase ≥ 3 instanziiert; `config.phase` ist an dieser Stelle also immer ≥ 3. Der `np.clip(..., 0, 2)` kappt diesen Wert aber **immer** auf `2` (den höchsten der drei in `reward.py` definierten `_phase_weights`-Presets — den "aggressivsten", geschwindigkeitsfokussiertesten). Da `set_phase()` sonst nirgends erneut aufgerufen wird, ist das vorgesehene Online-Curriculum (Preset 0 → 1 → 2, vermutlich konservativ/explorativ → aggressiv im Laufe des Online-Trainings) faktisch tot: Presets `0` und `1` (niedrigere `r_vel`-Gewichtung, höhere Survival-/Explorationsgewichtung) sind über den aktuellen Code-Pfad **nicht erreichbar**. Es handelt sich um eine Namespace-Kollision zwischen der globalen Dreamer-Trainings-Phase (1-4, immer ≥3 hier) und dem env-internen Reward-Curriculum-Konzept (0-2), die versehentlich denselben Config-Schlüssel `phase` verwenden.
- **Zeile/Ort:** `envs/drone_sim.py:41-43`
- **Lösung:** Eigenen, von der globalen `config.phase` unabhängigen Curriculum-Zähler einführen (z. B. `config.env.reward_curriculum_phase`) und `set_phase()` aus `OnlineTrainer.begin()` heraus periodisch (step- oder episodenbasiert) aufrufen, statt ihn einmalig aus der globalen Phase abzuleiten:
  ```python
  self.reward_curriculum_phase = int(np.clip(int(getattr(config, "env", {}).get("reward_curriculum_phase", 0) if hasattr(config, "env") else 0, 0), 0, 2))
  ```

### 10. `envs/drone_sim.py` — `ColosseumBridge.step()`: Reward/Termination werden bei Fallback-Frames aus veralteten Sensordaten berechnet
- **Execution Path:** `ColosseumBridge.step()` (`envs/drone_sim.py:538` ff.) → `obs_img, speed, info = self.read_observation(...)` → bei Timeout (`self._sem_ue_obj.acquire(timeout=0.1)` schlägt fehl) oder dupliziertem Frame (`seq == self._last_seq_ue`) liefert `read_observation()` `self._fallback_obs(...)` (Schwarzbild, `speed=0`) zurück, **ohne** dass `step()` davon erfährt → direkt danach liest `step()` unbedingt `s = self._ipc.sensors; collision = bool(s.has_collided); vel = ...; pos = ...` aus der **aktuellen** (potenziell veralteten) SHM-Region und reicht das an `reward_fn(...)` weiter.
- **Der Fehler:** Es gibt keinerlei Kopplung zwischen "war der zurückgegebene Frame frisch oder Fallback" und "aus welchem Sensor-Snapshot stammt die Reward-/Kollisions-Information". Bei jedem Frame-Drop (UE kurzzeitig langsamer als die 100 ms-Timeout-Grenze, oder Verbindungsabbruch mitten in der Episode) sieht der Agent ein Schwarzbild als Observation, während Reward und `terminated` aus dem **letzten bekannten** (ggf. mehrere Steps alten) Sensorzustand berechnet werden. Im Extremfall (UE-Prozess stirbt, SHM-Segment bleibt aber gemappt) "friert" `self._ipc.sensors` ein, und die Episode liefert beliebig viele Steps mit Schwarzbild + eingefrorenem `collision=False`-Reward, ohne dass dies erkannt oder die Episode terminiert wird — fehlerhafte Transitionen landen unbemerkt im Replay-Buffer.
- **Zeile/Ort:** `envs/drone_sim.py:538-545`
- **Lösung:** `read_observation()` soll einen `is_stale`-Flag zurückgeben; `step()` soll bei `is_stale=True` weder neue Sensordaten lesen noch einen validen Reward erzeugen, sondern den letzten validen Reward wiederholen/0 zurückgeben und nach N aufeinanderfolgenden Stale-Frames die Episode aktiv truncaten:
  ```python
  obs_img, speed, info, is_stale = self.read_observation(self.obs_h, self.obs_w, is_first=False)
  if is_stale:
      self._stale_count += 1
      truncated = self._stale_count > self._max_stale_frames
      return obs, 0.0, False, truncated, {**info, "stale": True}
  self._stale_count = 0
  ...
  ```

### 11. `trainer.py` — `OfflineTrainer.begin()` umgeht `Dreamer.train_step()` komplett und damit auch das konfigurierte AGC-Clipping
- **Execution Path:** `OfflineTrainer.begin()` (`trainer.py:617` ff.) ruft **nicht** `agent.train_step()`/`agent.update()` auf, sondern reimplementiert die Trainingsschleife manuell: `loss = self._compute_loss(...)` → `self.accelerator.backward(loss)` → `self.accelerator.clip_grad_norm_(agent.parameters(), max_norm=100.0)` (Zeile 750) → `optimizer.step()` (Zeile 752).
- **Der Fehler:** `Dreamer.train_step()` (`dreamer.py:602` ff.) implementiert explizit `self._agc(grad_params)` (Adaptive Gradient Clipping, `optim/agc.py`, parametrisiert über `agc: 0.3` in `configs/model/size100M.yaml`) — laut `r2dreamer/AI_CONTEXT.md` der bewusst gewählte Clipping-Mechanismus für dieses Modell (explizit **nicht** Standard-Global-Norm-Clipping). `OfflineTrainer.begin()` (Phase 1 & 2, der Hauptpfad für den Großteil des Trainings) ruft diesen Pfad nie auf und ersetzt ihn durch ein triviales, sehr permissives `clip_grad_norm_(..., max_norm=100.0)` — bei typischen Gradientennormen weit unterhalb von 100 ist das de facto kein Clipping. Das dokumentierte Stabilitätsdesign (AGC) greift damit im überwiegenden Teil des Trainings überhaupt nicht.
- **Zeile/Ort:** `trainer.py:617-755` (gesamte manuelle Trainingsschleife), insb. Zeile 750
- **Lösung:** `OfflineTrainer.begin()` auf `agent.train_step(data, initial=None, optimizer=optimizer, scheduler=scheduler, scaler=None, backward_fn=self.accelerator.backward, autocast_enabled=False)` umstellen (analog zu `OnlineTrainer.begin()`, `trainer.py:856-863`), damit `self._agc(...)` konsistent in allen Phasen greift.

### 12. `trainer.py` — `_eval()` evaluiert auf demselben Stream wie das Training (kein echtes Train/Val-Split, deterministisch wiederholte Eval-Daten)
- **Execution Path:** `OfflineTrainer.begin()` (`trainer.py:706-718`) erstellt `loader` **und** `self._eval_loader` aus demselben `self.dataset`-Objekt (identische `FPVDataset`-Instanz, die intern einen geseedeten `self.ds.shuffle(seed=42, ...)`-Stream hält) → `_eval()` (`trainer.py:665`) ruft `eval_iter = iter(self._eval_loader)` bei **jedem** Eval-Aufruf neu auf.
- **Der Fehler:** Ein `IterableDataset`, das einen HF-Streaming-Datensatz wrapt, startet bei jedem `iter(...)`-Aufruf wieder am Anfang des (mit festem `seed=42`) geshuffelten Streams. Da `loader` und `self._eval_loader` dieselbe zugrunde liegende `self.ds`-Referenz teilen, liest `_eval()` (a) tendenziell dieselben Samples, die auch fürs Training verwendet werden (kein Hold-out-Set), und (b) bei **jedem** Eval-Call exakt dieselben ersten `eval_batches`-Batches (da der Seed fix ist und der Iterator immer neu von vorne beginnt) — die Eval-Metrik bewegt sich also nie über unterschiedliche Daten und misst keine echte Generalisierung. Zusätzlich verwenden `loader` und `self._eval_loader` unterschiedliche `num_workers` (`self.num_workers` vs. `max(1, self.num_workers // 2)`), was bei einer Sharding-Logik, die von `worker_info.num_workers` abhängt (`FPVDataset.__iter__`), zu einer anderen Shard-Aufteilung führt als beim Training.
- **Zeile/Ort:** `trainer.py:706-718` (geteiltes `self.dataset`), `trainer.py:665-669` (`_eval()`)
- **Lösung:** Separates Hold-out-Dataset (z. B. eigener HF-Split oder `self.dataset.ds.take(N)`/`.skip(N)`-Split) für Eval verwenden, statt derselben `FPVDataset`-Instanz; alternativ einen fortlaufenden Eval-Offset/Seed pro Aufruf variieren, damit zumindest unterschiedliche Daten gesehen werden.

---

## 🟡 System- & Config-Risiken (Stille Fallbacks, Race Conditions)

### 13. `nightly/` — VLA-Heads-Integration aus `nightly/AI_CONTEXT.md` existiert nicht in `dreamer.py` (Feature ist nicht aktivierbar)
- **Execution Path:** `nightly/AI_CONTEXT.md` §4.4 beschreibt eine Integration über `if cfg.get("vla_enabled", False): from nightly.vla_heads import build_vla_heads; self.actor, self.reward, self.value = build_vla_heads(...)` innerhalb von `Dreamer.__init__`.
- **Der Fehler:** Eine Volltextsuche über `dreamer.py` nach `vla_enabled`, `build_vla_heads`, `nightly` und `z_text` liefert **keinen einzigen Treffer**. Der komplette `nightly/`-Ordner (FiLM-Conditioning, Cross-Attention-Heads, TextEncoder, Navigator, RelabelBuffer) ist vom restlichen Code vollständig entkoppelt — es gibt keinen Config-Schalter, über den er jemals aktiviert werden könnte. Das widerspricht der in `nightly/AI_CONTEXT.md` als "✅ Vollständig aktiv" beschriebenen Phase-3-Sprachkopplung.
- **Zeile/Ort:** fehlt in `dreamer.py` (erwartete Stelle: `__init__`, nahe den `self.actor`/`self.reward`/`self.value`-Zuweisungen)
- **Lösung:** Entweder die Integration gemäß `nightly/AI_CONTEXT.md` tatsächlich in `dreamer.py` ergänzen, oder die Doku korrigieren, falls `nightly/` bewusst (noch) nicht verdrahtet werden soll.

### 14. `r2dreamer/pipeline.py` — Verwaister Duplikat-/Altlasten-Pfad
- **Execution Path:** —
- **Der Fehler:** `r2dreamer/pipeline.py` existiert parallel zu `pipeline/pipeline.py` (dem tatsächlich verwendeten Daten-Pipeline-Skript, dessen `out_h/out_w/safety_h/safety_w` korrekt mit `configs.yaml` übereinstimmen — geprüft). Eine Volltextsuche (`import pipeline` / `from pipeline`) im gesamten Repository ergibt **keinen** Treffer auf `r2dreamer/pipeline.py` — die Datei wird von keinem aktiven Pfad importiert. Das deckt sich mit dem bereits in `r2dreamer/AI_CONTEXT.md` dokumentierten Hinweis auf einen veralteten Duplikat-Stand, ist hier aber durch eine frische Grep-Prüfung bestätigt statt nur übernommen. Risiko: zukünftige Edits an der falschen Datei.
- **Zeile/Ort:** `r2dreamer/pipeline.py` (gesamte Datei)
- **Lösung:** Datei löschen oder klar als `pipeline_DEPRECATED.py` kennzeichnen, um versehentliche Bearbeitung zu verhindern.

### 15. `networks.py` — `ContextEncoder` im GRU-Modus ignoriert `padding_mask` (nur transformer-Pfad maskiert)
- **Execution Path:** `ContextEncoder.forward(..., valid_len=...)` → `padding_mask` wird unabhängig vom `encoder_type` berechnet, aber nur im `if self.encoder_type == "transformer":`-Zweig als `src_key_padding_mask` verwendet. Im (laut `_base_.yaml` per Default aktiven) `"gru"`-Zweig läuft `self.rnn(x)` über die volle, ggf. Zero-gepaddete Sequenz ohne Maskierung.
- **Der Fehler:** Für den Standard-`act()`-Aufruf (`dreamer.py`) ist das folgenlos, da das Ergebnis bei `ctx_valid_steps < ctx_len` ohnehin über `torch.where(ctx_ready, ctx_d_emb, teacher_d_emb)` verworfen wird. Es bleibt aber eine stille Inkonsistenz im Modul selbst (Maske wird berechnet, aber nur für einen von zwei `encoder_type`-Pfaden tatsächlich genutzt) — bei künftiger Wiederverwendung von `ContextEncoder` mit `valid_len` außerhalb von `act()` (z. B. in einer Eval-Routine) würde der GRU-Pfad mit Padding kontaminierte States liefern, ohne dass dies offensichtlich wäre.
- **Zeile/Ort:** `networks.py`, `ContextEncoder.forward` (GRU-Zweig)
- **Lösung:** GRU-Pfad auf `nn.utils.rnn.pack_padded_sequence`/`pad_packed_sequence` mit `valid_len` umstellen, oder zumindest einen Kommentar ergänzen, der die GRU-Einschränkung explizit macht.

---

## 🟠 Logik- & Silent-Bugs — Teil 2

### 16. `native/coreml_export.py` + `fly_real.py` — Deployment: `prev_action` und `prev_filtered_action` werden auf dieselbe Variable kollabiert → motorgefilterte Aktion friert für den gesamten Flug bei `0` ein
- **Execution Path:** `coreml_export.py` `CoreMLDreamerWrapper.forward()` (Zeile 68 ff.) → Zeile 83-86:
  ```python
  next_stoch, next_deter, _, next_filtered_act = self.rssm.obs_step(
      stoch, prev_deter, prev_filtered_act,
      embed, reset, self.d_emb, prev_filtered_act,
  )
  ```
  Referenz-Implementierung in `dreamer.py::act()` (Zeile 457-461, der Python-seitige, **korrekte** Online-Inferenzpfad) zum Vergleich:
  ```python
  prev_stoch, prev_deter, prev_action, prev_filtered_action = (
      state["stoch"], state["deter"], state["prev_action"], state["prev_filtered_action"],
  )
  stoch, deter, _, filtered_action = self._frozen_rssm.obs_step(
      prev_stoch, prev_deter, prev_action, embed, obs["is_first"], d_emb, prev_filtered_action
  )
  ```
  `RSSM.obs_step(self, stoch, deter, prev_action, embed, reset, d_emb, prev_filtered_action, alpha)` berechnet intern `filtered_action = self._filter_action(prev_action, prev_filtered_action, alpha) = alpha·prev_action + (1-alpha)·prev_filtered_action` (`rssm.py`).
- **Der Fehler:** In `dreamer.py::act()` und in `get_initial_state()` (Zeile 543-558) werden `prev_action` (die rohe, ungefilterte Aktion des letzten Schritts) und `prev_filtered_action` (der motorgefilterte Zustand) korrekt als **zwei getrennte** State-Variablen geführt. `CoreMLDreamerWrapper.forward()` besitzt aber **keinen eigenen Eingabe-/Ausgabeslot für die rohe `prev_action`** — als CoreML-I/O-State existiert ausschließlich `prev_filtered_act` (siehe `dummy_inputs`/`ct.TensorType`-Liste, nur 6 Inputs, kein `prev_action`). Der Wrapper übergibt deshalb **denselben Wert `prev_filtered_act` für beide Parameter** (`prev_action` UND `prev_filtered_action`) an `obs_step()`. Damit gilt für `_filter_action`:
  ```
  filtered_action = alpha·prev_filtered_act + (1-alpha)·prev_filtered_act = prev_filtered_act   (für JEDES alpha)
  ```
  Das ist eine Identitätsabbildung — die tatsächlich vom Actor gewählte Aktion (`action`, erst **nach** diesem `obs_step()`-Aufruf berechnet, Zeile 90-93) fließt nie in `filtered_action`/`next_filtered_act` ein. Da `next_filtered_act` 1:1 als `self.prev_filtered_act` für den **nächsten** Inferenz-Call zurückgereicht wird (`fly_real.py:357-359`), bleibt der gefilterte Aktionszustand für den **gesamten Flug** bei seinem Initialwert `np.zeros((1, A))` (`fly_real.py:308`) eingefroren. Die RSSM-`deter`-Rekursion (`self._deter_net(stoch, deter, filtered_action, d_emb)`) bekommt dadurch während des **gesamten realen Flugs** effektiv immer "Null-Aktion" als Konditionierung — die zeitliche Aktions-Dynamik des Weltmodells ist im exportierten/deployten Modell faktisch abgeschaltet, obwohl Training und Python-seitige Online-Inferenz (`act()`) korrekt implementiert sind. Das ist ein reiner Export-/Wrapper-Bug, der nur im CoreML-Deployment-Pfad auftritt, dort aber sicherheitskritisch ist (degradierte Weltmodell-Vorhersagen während des Echtflugs, ohne jedes Fehler-Signal).
- **Zeile/Ort:** `native/coreml_export.py:68-95` (fehlender `prev_action`-I/O), `native/coreml_export.py:83-86` (doppelte Verwendung von `prev_filtered_act`); spiegelbildlich `fly_real.py:306-359` (`CoreMLPolicy` führt ebenfalls nur `self.prev_filtered_act`, kein `self.prev_action`)
- **Lösung:** Zusätzlichen State-Slot `prev_action` ergänzen — sowohl im `CoreMLDreamerWrapper` (neuer Input `prev_action`, neuer Output `next_action` = der in Zeile 92 berechnete `action`-Tensor) als auch in `CoreMLPolicy` (`self.prev_action = np.zeros((1, self.A), ...)`, in `reset()` und nach jedem `act()`-Aufruf aktualisiert):
  ```python
  # forward(..., prev_action: torch.Tensor, prev_filtered_act: torch.Tensor, ...)
  next_stoch, next_deter, _, next_filtered_act = self.rssm.obs_step(
      stoch, prev_deter, prev_action,        # <- rohe vorherige Aktion, NICHT prev_filtered_act
      embed, reset, self.d_emb, prev_filtered_act,
  )
  ...
  return action, next_stoch_flat, next_deter, next_filtered_act, action, safety_score, safe_action
  #                                                                ^^^^^^ neuer next_action-Output, als prev_action für den nächsten Call
  ```

### 17. `nightly/relabeling.py` — `RelabelBuffer`/`relabel_dataset` rufen `text_encoder(...)` auf, aber `vla_heads.TextEncoder` definiert weder `forward()` noch `__call__()`
- **Execution Path:** `RelabelBuffer._label_loop()` (`nightly/relabeling.py`, im Worker-Thread) → `z = self._text_enc(seg.instruction)` — analog `relabel_dataset()`'s `worker()` → `z = text_encoder(instruction).detach().cpu()`. Die einzige im Repository vorhandene Text-Encoder-Implementierung ist `nightly/vla_heads.py::TextEncoder`, deren öffentliche API ausschließlich `.encode(command: str) -> tuple[torch.Tensor, torch.Tensor]` ist (token-level Embeddings **plus** Padding-Maske als Tupel) — eine `forward()`-Methode ist nicht definiert.
- **Der Fehler:** Ruft man die `TextEncoder`-Instanz `text_encoder(instruction)` direkt auf (statt `.encode(instruction)`), greift PyTorchs `nn.Module.__call__` → `self.forward(instruction)` → die Default-Implementierung von `nn.Module.forward` wirft `NotImplementedError`. Selbst wenn `forward` existierte, würde `.encode()` ein **Tupel** zurückgeben, auf dem `z.detach().cpu()` (Singular-Tensor-API) ebenfalls fehlschlagen würde. `RelabelBuffer` ist damit, sobald es mit der einzigen vorhandenen `TextEncoder`-Klasse instanziiert wird, beim ersten gelabelten Clip funktionsunfähig (Exception wird zwar in `_label_loop`'s `except Exception as exc: print(...)` abgefangen und der Worker-Thread läuft weiter — der Fehler bleibt also "silent", produziert aber dauerhaft keine `z_text`-Embeddings).
- **Zeile/Ort:** `nightly/relabeling.py` (`_label_loop`, `relabel_dataset`) i.V.m. `nightly/vla_heads.py::TextEncoder.encode` (kein `forward`)
- **Lösung:** In `relabeling.py` konsequent `self._text_enc.encode(seg.instruction)` statt `self._text_enc(seg.instruction)` aufrufen und das zurückgegebene Tupel `(token_embs, mask)` korrekt weiterverarbeiten (z. B. mean-pooling über `token_embs` unter Berücksichtigung von `mask`, um einen einzelnen `(text_dim,)`-Vektor für `seg.z_text` zu erhalten) — oder `TextEncoder` um eine `forward()`-Methode ergänzen, die genau das tut.

---

## 🟡 System- & Config-Risiken — Teil 2

### 18. `native/mlx_*` — Auch bei korrektem Modell-Typ bliebe der MLX-Pfad unverifiziert (kein Smoke-Test, keine Schema-Prüfung gegen Fund #2)
- **Hinweis:** Da Fund #2 (`MLXOnlineTrainer.begin()` erhält ein PyTorch- statt MLX-Modell) den gesamten Pfad bereits beim ersten Aufruf zum Absturz bringt, konnte der **restliche** MLX-Code (`mlx_models.py`, `mlx_utils.py::load_pytorch_to_mlx`) im Rahmen dieser Trace-Analyse nicht durch tatsächliche Ausführung verifiziert werden. Vor einer Behebung von Fund #2 sollte zusätzlich geprüft werden, ob `load_pytorch_to_mlx()` mit den aktuellen `state_dict()`-Schlüsselnamen des PyTorch-`Dreamer` (insb. `encoder.encoders.0.backbone_early.*` / `backbone_late.*`, siehe Fund #8) tatsächlich übereinstimmt — ein erneutes Namens-Mismatch an dieser Schnittstelle ist angesichts des bereits gefundenen `backbone_early`/`backbone_late`-Bugs nicht unwahrscheinlich.
- **Zeile/Ort:** `native/mlx_utils.py` (ungeprüft, abhängig von Fix zu Fund #2)
- **Lösung:** Nach Behebung von Fund #2 einen Smoke-Test ergänzen, der `load_pytorch_to_mlx()` gegen ein frisch initialisiertes `Dreamer`-`state_dict()` laufen lässt und `report["missing"]`/`report["skipped"]` auf Leere prüft (CI-Gate).

---

*Ende der Analyse. Alle oben aufgeführten Befunde wurden durch tatsächliches Lesen des Codes (inkl. exakter Shape-/Aufruf-Verifikation und – bei Fund #1 – Abgleich mit dem `accelerate`-Quellcode) bestätigt, nicht aus der vorhandenen `AI_CONTEXT.md`-Dokumentation übernommen.*

---
## ✅ FIX-PROTOKOLL (alle 18 Bugs behoben)

| # | Datei(en) | Status |
|---|-----------|--------|
| 1 | `optim/laprop.py` | ✅ `step(closure=None)` + `return loss` |
| 2 | `train.py` | ✅ `MLXDreamer` instanziiert + Gewichtstransfer via `load_pytorch_to_mlx()` |
| 3 | `dreamer.py` | ✅ `boot` via `flat_mask`-Scatter statt `reshape(B,-1,1)[:,:T]` |
| 4 | `trainer.py` | ✅ `require_osd`-Gate entfernt — nur noch `sample.get("has_osd")` |
| 5 | `trainer.py` | ✅ `self.model_num_drones` in `__init__` gesetzt; `hasattr`-Fallback entfernt |
| 6 | `trainer.py` + `dreamer.py` | ✅ `use_depth` → `use_depth_aux` (3 Stellen) |
| 7 | `trainer.py` + `train.py` | ✅ `OfflineTrainer.__init__(phase=1)` + Weitergabe von `config.phase` |
| 8 | `dreamer.py` | ✅ `.backbone.` → `.backbone_early.` / `.backbone_late.` |
| 9 | `envs/drone_sim.py` + `configs/configs.yaml` | ✅ `reward_curriculum_phase` entkoppelt von globalem `phase` |
| 10 | `envs/drone_sim.py` | ✅ `read_observation()` gibt 4-Tuple zurück; stale-Frames ergeben `reward=0.0` + Truncation |
| 11 | `trainer.py` | ✅ `clip_grad_norm_` → `train_agent._agc()` in `OfflineTrainer` |
| 12 | `trainer.py` | ✅ `FPVDataset.clone_for_eval(seed=137)` für separaten Eval-Stream |
| 13 | `dreamer.py` | ✅ VLA-Heads-Hook via `cfg.vla_enabled` |
| 14 | `r2dreamer/pipeline.py` | ✅ → `pipeline_DEPRECATED.py` mit Warnheader |
| 15 | `networks.py` | ✅ GRU-Pfad: `x.masked_fill(padding_mask, 0.0)` vor `self.rnn(x)` |
| 16 | `native/coreml_export.py` + `fly_real.py` | ✅ `prev_action` als separater I/O-Slot; `CoreMLPolicy` trackt beide Aktionen |
| 17 | `nightly/vla_heads.py` | ✅ `TextEncoder.forward()` ergänzt (mean-pool über valide Tokens) |
| 18 | `native/tests/smoke_mlx.py` | ✅ Smoke-Test: Encoder → obs_step → Actor/Value → SafetyNet |
