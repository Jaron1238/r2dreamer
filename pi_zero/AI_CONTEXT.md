# AI_CONTEXT.md — r2dreamer/pi_zero/

> **Für zukünftige KIs:** Dieser Ordner läuft NICHT auf dem Inferenz-Rechner (Apple
> Silicon, siehe `native/`). Er enthält Deployment-Code für ein separates
> **Companion-Board (Raspberry Pi Zero)**, das ausschließlich als Kamera-Streaming-Quelle
> für das SafetyNet dient. Kein Python-Import-Bezug zum Rest von `r2dreamer/` — die
> Kopplung läuft rein über Netzwerk/Protokoll (GStreamer H.264/RTP/UDP).

---

## 1. Zweck des Ordners

`pi_zero/stream_server.py` ist ein eigenständiges CLI-Skript, das auf einem Raspberry Pi
Zero läuft und dessen Kamerabild per H.264 über RTP/UDP an den Hauptinferenzrechner
streamt. Es ist die **Producer-Seite** einer dedizierten Hochauflösungs-Kamera für das
SafetyNet — getrennt von der Haupt-RGB-Kamera, die für die RSSM-Steuerung verwendet wird.

Die **Consumer-Seite** ist `r2dreamer/fly_real.py::PiVideoStream` (Root-Ordner, eigene
`AI_CONTEXT.md` noch ausstehend), die per GStreamer auf demselben Port lauscht und das
empfangene Bild für das SafetyNet bereitstellt.

---

## 2. Tech-Stack & Tools

| Komponente              | Verwendung                                                              |
|---------------------------|--------------------------------------------------------------------------|
| **GStreamer (`gst-launch-1.0`)** | Aufgerufen via `subprocess.Popen`, NICHT über ein Python-Binding (kein PyGObject/gst-python) |
| **`libcamerasrc`**         | GStreamer-Quellelement für den modernen Raspberry-Pi-Kamera-Stack (libcamera, nicht das alte raspivid/MMAL) |
| **`x264enc`**               | Software-H.264-Encoder, `tune=zerolatency`, `speed-preset=ultrafast`     |
| **`rtph264pay` + `udpsink`** | RTP-Payload-Verpackung + UDP-Versand                                    |

---

## 3. Architektur-Regeln (KRITISCH)

### 3.1 Pipeline-String-Kontrakt mit `fly_real.py::PiVideoStream`

```python
# pi_zero/stream_server.py (Sender)
"... x264enc tune=zerolatency ... rtph264pay config-interval=1 pt=96 ! udpsink host={host} port={port}"

# fly_real.py PiVideoStream.PIPELINE (Empfänger)
"udpsrc port={port} ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! h264parse ! avdec_h264 ..."
```

Beide Seiten müssen exakt zusammenpassen: **Payload-Type `pt=96`/`payload=96`** und der
**Port** (Default hier `5600`, identisch zu `configs/fly_real.yaml`'s `port: 5600`).
Ändert sich der Payload-Type oder das Encoding auf einer Seite, ohne die andere
anzupassen, bekommt der Empfänger keine decodierbaren Frames.

### 3.2 Latenz-Priorisierung vor Bildqualität

`tune=zerolatency` + `speed-preset=ultrafast` + feste `bitrate=2000` (kbps) sind bewusst
auf **minimale Latenz** statt Bildqualität optimiert — konsistent mit dem
Echtzeit-Steuerungs-/Sicherheits-Anwendungsfall (SafetyNet braucht aktuelle Frames, keine
verlustfreie Aufnahme).

### 3.3 Auflösung wird empfängerseitig IMMER erzwungen — Mismatch crasht nicht, degradiert aber still

`fly_real.py::PiVideoStream` ruft in jeder Frame-Verarbeitungsschleife
`cv2.resize(frame, (W, H))` auf die in `resolution=(1280, 720)` konfigurierte Zielgröße
auf. Das bedeutet: **egal welche Auflösung `stream_server.py` tatsächlich sendet, der
Empfänger akzeptiert sie immer** — es gibt keine Shape-Validierung. Siehe Abschnitt 5 für
die daraus resultierende konkrete Falle.

---

## 4. Wichtige Abhängigkeiten

```
pi_zero/stream_server.py  --[GStreamer H.264/RTP/UDP, Port 5600]-->  r2dreamer/fly_real.py::PiVideoStream
                                                                        (konfiguriert über configs/fly_real.yaml: port: 5600)
```

**Keine Python-Importbeziehung** — die einzige Kopplung ist der Netzwerk-/
Protokollvertrag (Host, Port, Payload-Type, Encoding). Es gibt keinen automatisierten
Test, der diesen Vertrag absichert; ein Bruch zeigt sich erst beim echten Flugtest als
fehlendes oder leeres Kamerabild.

---

## 5. Limitierungen & Fallstricke

### ❌ KRITISCH — Stille Qualitätsdegradation bei Standard-Aufruf

**`stream_server.py`'s Default-Auflösung (`--width 256 --height 256`) passt NICHT zur
vom SafetyNet erwarteten Auflösung (`1280×720`, laut Kommentar in `fly_real.py` "die
größte Auflösung").** Wird das Skript auf dem Pi Zero ohne explizite
`--width 1280 --height 720`-Argumente gestartet, sendet es 256×256-Frames. Der Empfänger
**crasht nicht** (er skaliert per `cv2.resize` immer auf die Zielgröße hoch) — aber das
SafetyNet erhält dann ein künstlich von 256×256 hochskaliertes Bild statt eines echten
1280×720-Bildes. **Keine Fehlermeldung, keine Warnung** — die Sicherheitswahrnehmung ist
dadurch massiv degradiert, ohne dass dies im Log sichtbar wird.

→ **Beim Start auf dem Pi Zero immer explizit angeben:**
```bash
python3 stream_server.py --host <inferenz-rechner-ip> --width 1280 --height 720 --fps 30
```

### ❌ NIEMALS tun

1. **`stream_server.py` mit Default-Argumenten für den echten Flugbetrieb starten** —
   führt zur oben beschriebenen stillen Auflösungs-Degradation.
2. **`pt=96`/Payload-Type oder Port nur auf einer Seite (Sender ODER Empfänger) ändern**
   — die GStreamer-Pipelines müssen exakt synchron bleiben.

### ⚠️ Vorsicht

- **`host` Default `"192.168.1.100"` ist ein Platzhalter** — eine hartkodierte private
  IP, die vor jedem Einsatz an die tatsächliche IP des Inferenzrechners angepasst werden
  muss.
- **Kein Reconnect-/Retry-Mechanismus:** Bricht die GStreamer-Pipeline ab (z. B.
  Kameraverlust, WLAN-Aussetzer), beendet sich `start_stream()` einfach über
  `proc.wait()` — kein automatischer Neustart. Für den Produktiveinsatz wäre ein
  Supervisor (z. B. `systemd`-Service mit `Restart=always`) ratsam, ist hier aber nicht
  implementiert.
- **Feste Bitrate (2000 kbps), kein adaptives Bitrate-Management** — bei schwacher
  WLAN-Verbindung kann das zu Paketverlust/Rucklern führen, ohne dass die Pipeline sich
  automatisch anpasst.
- **`libcamerasrc` setzt ein passendes GStreamer-Pi-Camera-Setup voraus**
  (`gstreamer1.0-libcamera` o. ä.) — auf älteren Raspberry-Pi-OS-Images (vor dem
  libcamera-Wechsel) oder Nicht-Pi-Geräten nicht garantiert verfügbar.
