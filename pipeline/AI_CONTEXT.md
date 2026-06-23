# AI_CONTEXT.md — `pipeline/`

> **Zweck:** KI-Kontext-Datei für den `pipeline/`-Ordner. Erstellt durch systematische Code-Analyse.  
> **Letztes Update:** 2026-06-20

---

## 1. Zweck des Ordners

`pipeline/` ist die **vollständig autonome FPV-Videodatenerfassungs- und Verarbeitungs-Engine** des R2Dreamer-Projekts.  
Sie läuft ausschließlich auf **kostenlosen GitHub Actions Runnern** (ubuntu-latest, 2 vCPUs, 7 GB RAM) und produziert das **Parquet-Trainings-Dataset** für `r2dreamer`'s `FPVDataset`- und `SafetyNet`-Trainer.

Das Verzeichnis enthält genau zwei Dateien:

| Datei | Rolle |
|---|---|
| `fpv_pipeline.yml` | GitHub Actions Workflow: Orchestrierung, Sharding, Secrets, Gist-Persistenz |
| `pipeline.py` | Python-Verarbeitungs-Engine: Download → Dekodierung → Analyse → Parquet-Export |

**End-to-End-Datenfluss:**
```
fpv_videos_augmented_v2.csv
    │
    ├─► [Sharding] df.iloc[shard::num_shards]
    │
    ├─► [Download] yt-dlp / HuggingFace Hub / GitHub API / HTTP-Streaming
    │
    ├─► [Dekodierung] PyAV → mmap (kein subprocess-Deadlock)
    │
    ├─► [Analyse]
    │      TakeoffDetector   → start_sec (überspringt Boden-Footage)
    │      OSDAutoDetector   → battery/speed/altitude ROIs (Tesseract OCR, optional)
    │      StickCamDetector  → Stick-Cam-Overlay-Maske (Ecken-Flow-Vergleich)
    │      SceneSegmenter    → Shot-Grenzen (Mean-Abs-Diff Grayscale)
    │      CrashDetector     → is_terminal + crash_indices (Flow + Brightness + Sharpness)
    │      DroneClassifier   → drone_id (GMM auf Flow-Fingerprint, optional)
    │
    ├─► [Inpainting] Mean-Fill der Overlay-Pixel (KEIN cv2.inpaint!)
    │
    ├─► [Export] ParquetExporter → *_seg{N:04d}.parquet
    │             CrashDatasetExporter → crash_{N:06d}.parquet
    │
    └─► [Upload] HuggingFace Hub (jaron12/fpv-dataset + jaron12/fpv-crash-dataset)
                  via pack-and-upload (Chunks ≤ 1 GB)
```

---

## 2. Tech-Stack & Tools

### Python-Verarbeitung (`pipeline.py`)
| Library | Verwendung |
|---|---|
| `cv2` (OpenCV) | Optischer Fluss (Farneback), Resize, Graustufen, Laplacian |
| `av` (PyAV) | Haupt-Decoder (kein `subprocess`/`image2pipe`) → mmap |
| `ffmpeg` (subprocess) | Fallback-Decoder und kleine Sample-Reads (Takeoff, OSD-Scan) |
| `ffprobe` (subprocess) | Ermittlung von Breite, Höhe, FPS |
| `numpy` | Frame-Arrays, Vektorisierung |
| `pyarrow` / `pyarrow.parquet` | Parquet-Lesen/Schreiben, Schema-Erstellung |
| `pandas` | CSV-Laden, DataFrame-Zwischenschicht für Parquet |
| `huggingface_hub` | `HfApi.upload_file()`, `hf_hub_download()`, Repo-Erstellung |
| `yt_dlp` | YouTube / AirVuz Download |
| `pytesseract` | Tesseract OCR für OSD-Erkennung (optional, teuer) |
| `PIL` (Pillow) | PNG-Encoding für Crash-Dataset-Frames |
| `scipy.interpolate.interp1d` | NaN-Interpolation von Telemetriewerten |
| `sklearn` | PCA + GaussianMixture für DroneClassifier (optional) |
| `requests` | HTTP-Downloads mit Stall-Timeout |
| `tqdm` | Progress-Bars |
| `multiprocessing` | Worker-Prozesse + Queue + Value (Progress) |
| `threading` | ThreadPoolExecutor für HF-Uploads |

### CI-Infrastruktur (`fpv_pipeline.yml`)
| Komponente | Detail |
|---|---|
| GitHub Actions Matrix | 10 Shards parallel (`shard: [0..9]`) |
| Timeout | 340 min (Hard-Cap), Python stoppt bei 330 min (5.5 h) |
| Persistenz | GitHub Gist (`processed_urls.txt`) + Artifact-Upload pro Shard |
| Merge-Job | `merge-progress` dedupliziert alle Shard-Artifacts → Gist |
| Secrets | `HF_TOKEN`, `GH_TOKEN`, `GH_GIST_ID`, `YT_COOKIES` (optional) |
| Disk-Freigabe | `~30 GB` durch Löschen von Android SDK, .NET, Haskell, Hostedtoolcache |

---

## 3. Architektur-Regeln

### 3.1 Auflösungs-Constraints (KRITISCH — müssen mit `configs/configs.yaml` übereinstimmen)

```python
# In PipelineConfig:
out_h:    288   # RSSM-Encoder-Höhe  (img_height in configs.yaml)
out_w:    512   # RSSM-Encoder-Breite (img_width  in configs.yaml)
safety_h: 720   # SafetyNet-Höhe     (model.safety_img_height)
safety_w: 1280  # SafetyNet-Breite   (model.safety_img_width)
depth_h:  1080  # Depth-Quelle       (kein Modell-Constraint)
depth_w:  1920
```

**Regel:** `out_h/out_w` und `safety_h/safety_w` dürfen **nur** geändert werden, wenn `configs.yaml` synchron aktualisiert wird. Die Pipeline und der Trainer müssen immer die gleichen Werte verwenden.

### 3.2 Inpainting — Mean-Fill (KEIN `cv2.inpaint`)

```python
# KORREKT (schnell, intentional):
fill = frame[~bool_mask].mean(axis=0).astype(np.uint8)
frame[bool_mask] = fill

# VERBOTEN (war ~10 000× langsamer auf CPU-Only-Runner):
# cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
```

R2Dreamer empfängt die `cam_overlay_mask` und lernt, dass maskierte Pixel irrelevant sind. Inpainting-Qualität ist nicht erforderlich.

### 3.3 Decoder-Hierarchie

```
1. PyAV  → read_video_to_mmap()    (Primär: kein Pipe-Deadlock)
2. FFmpeg → read_video_frames_ffmpeg() (Fallback/kleine Reads)
```

`forkserver` als Multiprocessing-Startmethode ist **Pflicht** (verhindert Deadlocks beim Forken mit offenen File-Handles).

### 3.4 CI-Ressourcen-Budget

| Ressource | Limit | Constraint |
|---|---|---|
| RAM | 7 GB | `hf_upload_threshold_bytes = 3 GB` (< RAM) |
| Disk | ~50 GB (nach Cleanup) | `max_single_file_gb = 4.0` |
| vCPUs | 2 | `ffmpeg -threads 2`, `cv2.setNumThreads(0)`, `n_workers = 1` |
| Zeit | 6 h (hard) | `GRACEFUL_SHUTDOWN_SECONDS = 5.5 * 3600` (Python-soft-stop) |
| Pack-Chunks | `pack_chunk_gb = 1.0` | Peak-RAM innerhalb Budget |

### 3.5 Threading-Regeln

- **Hauptprozess:** Orchestrierung, tqdm-Anzeige
- **Worker-Prozesse** (`multiprocessing.Process`): Video-Verarbeitung (einer pro vCPU)
- **Upload-Threads** (`ThreadPoolExecutor`): HF-Upload parallel zu Worker-Prozessen
- `URLRegistry` verwendet `multiprocessing.Lock` (cross-process), `SegmentRegistry` verwendet `threading.Lock` (in-process)

### 3.6 Parquet-Schema (unveränderlich — muss `FPVDataset` entsprechen)

```python
record = {
    "frames_rgb":   np.ndarray,  # (T, 288, 512, 3)  uint8  RGB, inpainted
    "frames_gray":  np.ndarray,  # (T, 720, 1280, 1) uint8  Grayscale, SafetyNet-Res
    "cam_overlay":  np.ndarray,  # (288, 512)         uint8  0/255 binary
    "osd":          np.ndarray,  # (T, 8)             float32 [speed,alt,bat,0,0,0,0,0]
    "actions":      np.ndarray,  # (T, 4)             float32
    "speeds":       np.ndarray,  # (T,)               float32
    "altitudes":    np.ndarray,  # (T,)               float32
    "batteries":    np.ndarray,  # (T,)               float32
    "is_terminal":  np.ndarray,  # (T,)               bool
    "n_frames":     int,
    "drone_id":     int,
    "has_osd":      bool,
    "fps":          float,
    "stem":         str,
    "segment_id":   int,
}
```

OSD-Slots (Index): 0=speed, 1=altitude, 2=battery, 3–7=reserviert (Null).

---

## 4. Wichtige Abhängigkeiten

### 4.1 Input-CSV
- **Pfad:** `fpv_videos_augmented_v2.csv` (relativ zum Runner-Workspace)
- **Pflicht-Spalten:** `url`, `platform`, `title`
- **Platform-Werte:** `"YouTube"`, `"YouTube Playlist"`, `"AirVuz"`, `"HuggingFace"`, `"GitHub"`, (direkter HTTP-Link)
- Die Sharding-Logik (`df.iloc[shard::num_shards]`) setzt voraus, dass die CSV-Reihenfolge stabil ist.

### 4.2 R2Dreamer-Konsument
- **`r2dreamer/pipeline.py`** → `FPVDataset`: Liest die Parquet-Felder aus `jaron12/fpv-dataset`
- **`r2dreamer/envs/drone_sim.py`** → `SafetyRawImageSource`: Liest `jaron12/fpv-crash-dataset`
- **`r2dreamer/configs/configs.yaml`**: `img_height`, `img_width`, `model.safety_img_height`, `model.safety_img_width` **müssen** mit `PipelineConfig` übereinstimmen

### 4.3 Gist-Persistenz
- GitHub Gist (ID in Secret `GH_GIST_ID`) speichert `processed_urls.txt`
- Shard lädt beim Start die aktuelle Gist-Version → überspringt bereits verarbeitete URLs
- Race-Condition-Schutz: Jeder Shard uploaded sein eigenes Artifact → `merge-progress`-Job dedupliziert und schreibt einmalig zurück

### 4.4 Externe Services
| Service | Verwendung |
|---|---|
| `jaron12/fpv-dataset` (HF Dataset) | Ziel für Trainings-Parquets |
| `jaron12/fpv-crash-dataset` (HF Dataset) | Ziel für Crash-Safety-Parquets |
| GitHub Gist (`GH_GIST_ID`) | URL-Fortschrittspersistenz über Runs |
| GitHub Issues | Automatisches Issue-Tracking bei Video-Fehlern (`gh issue create`) |

---

## 5. Limitierungen & Fallstricke

### ❌ VERBOTEN

| Aktion | Grund |
|---|---|
| `cv2.inpaint()` verwenden | ~10.000× langsamer als Mean-Fill auf CPU-Only-Runnern |
| `cancel-in-progress: true` in YAML | Würde aktive HF-Uploads abrechen → inkomplette Datasets |
| `n_workers > 1` ohne RAM-Check | Verdoppelt RAM-Verbrauch auf einem 7 GB-Runner |
| `multiprocessing.set_start_method("fork")` | Deadlocks bei geöffneten Sockets/File-Handles |
| `enable_osd: True` (Standard: False) | Tesseract OCR ist sehr teuer; nur für Debugging |
| `enable_drone_clustering: True` (Standard: False) | Phase-2 Barlow-Twins ignoriert `drone_id`; Phase-1 braucht es nicht |
| `out_h/out_w` oder `safety_h/safety_w` ändern | Bricht Kompatibilität mit `FPVDataset` + Trainer-Configs |
| `max_single_file_gb > 4.0` | Kann Runner-OOM verursachen |
| `hf_upload_threshold_bytes > 5 GB` | Überschreitet Runner-RAM (7 GB) |
| Gist direkt von mehreren Shards beschreiben | Race-Condition → Datenverlust; immer den Merge-Job nutzen |
| `pack_chunk_gb > 2.0` | PyArrow hält Tisch-Daten im RAM → OOM-Risiko |

### ⚠️ EDGE CASES & GOTCHAS

- **`read_video_to_mmap` vs. `read_video_frames_ffmpeg`:** Der mmap-Decoder ist der primäre Pfad; `ffmpeg`-Pipe kann bei großen Videos deadlocken (OS pipe buffer voll). Beide müssen exakt die gleiche Auflösung und FPS produzieren.
- **`interp_nan()`:** Telemetrie-Arrays können `nan` enthalten (OCR-Fehler). Die Funktion interpoliert linear; wenn der gesamte Array `nan` ist, gibt sie Nullen zurück.
- **`SceneSegmenter.min_scene_len_frames = 256`:** Segmente kürzer als 256 Frames (~4.3s bei 60fps) werden verworfen. Änderung beeinflusst direkt, wie viele Trainingssequenzen aus einem Video entstehen.
- **`crash_context_frames = 30`:** 30 Frames VOR einem Crash werden als `crash=1.0` gelabelt. Das entspricht 0.5s bei 60fps.
- **`mmap_max_frames = 60`:** Maximale Frames, die gleichzeitig im mmap-Buffer gehalten werden — RAM-Budget-Constraint, nicht inhaltlich.
- **`max_segment_frames = 3600`:** Sicherheits-Cap (60 Sekunden bei 60fps). Verhindert mono-lith-Parquets, die den RAM sprengen.
- **`forkserver`-Start:** Der Aufruf `multiprocessing.set_start_method("forkserver", force=True)` steht im `if __name__ == "__main__":`-Guard. Er darf nicht in Worker-Prozesse propagieren.
- **OSD-ROI-Skalierung:** `pad = max(8, int(h * 0.01))` ist relativ zur Frame-Höhe, nicht hardkodiert. Änderungen an `out_h` wirken sich auf die Pad-Berechnung aus.
- **`gh`-CLI:** Muss im PATH der Runner vorhanden sein (vorinstalliert auf GitHub-Runnern). Fehlt sie, werden Gist-Saves und Issue-Tracking stumm übersprungen (kein `sys.exit`).
- **YouTube Cookies:** `cookies.txt` wird aus dem Secret `YT_COOKIES` injiziert. Fehlt das Secret, läuft yt-dlp ohne Cookies — age-gated Videos schlagen fehl, öffentliche Videos funktionieren.
- **`GRACEFUL_SHUTDOWN_SECONDS = 5.5 * 3600`:** Der Shutdown-Check muss in der Worker-Loop implementiert sein. Ein hängendes `ffmpeg`-Subprocess verhindert den graceful shutdown.
