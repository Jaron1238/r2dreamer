

import collections
import pathlib
import socket
import threading
import time

import coremltools as ct
import cv2
import hydra
import numpy as np

from native.coreml_export import ExportConfig

try:
    import av
    _PYAV_AVAILABLE = True
except ImportError:
    _PYAV_AVAILABLE = False

class PiVideoStream:
    

    PIPELINE = (
        "udpsrc port={port} "
        "! application/x-rtp,encoding-name=H264,payload=96 "
        "! rtph264depay ! h264parse ! avdec_h264 "
        "! videoconvert ! appsink drop=true max-buffers=1"
    )

    def __init__(self, port: int = 5600, resolution: tuple[int, int] = (1280, 720)):
        # resolution = (height, width) for the safety-net stream (largest resolution)
        self.resolution = resolution   # (H, W) for safety-net
        self._frame = None
        self._lock = threading.Lock()
        self._running = True

        
        pipeline = self.PIPELINE.format(port=port)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        
        
        ret, _ = cap.read()
        if ret:
            print("[VideoStream] GStreamer backend active.")
            self.cap = cap
            self._use_pyav = False
            threading.Thread(target=self._loop_cv, daemon=True).start()
            return

        cap.release()

        
        if _PYAV_AVAILABLE:
            print(
                "[VideoStream] GStreamer not available → PyAV UDP fallback. "
                f"Listening on UDP port {port}."
            )
            self._port = port
            self._use_pyav = True
            threading.Thread(target=self._loop_pyav, daemon=True).start()
        else:
            print(
                "[VideoStream] WARNING: GStreamer unavailable and PyAV not installed. "
                "Falling back to webcam (index 0).  "
                "For the real drone, run: pip install av"
            )
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._use_pyav = False
            threading.Thread(target=self._loop_cv, daemon=True).start()

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        H, W = self.resolution
        frame = cv2.resize(frame, (W, H))   # cv2.resize wants (width, height)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame.astype(np.float32) / 255.0

    def _loop_cv(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = self._preprocess(frame)

    def _loop_pyav(self):
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", self._port))
        sock.settimeout(1.0)

        codec = av.CodecContext.create("h264", "r")
        buf = b""
        while self._running:
            try:
                data, _ = sock.recvfrom(65536)
            except socket.timeout:
                continue
            
            if len(data) > 12:
                buf += data[12:]
            
            if len(buf) < 4096:
                continue
            try:
                packets = codec.parse(buf)
                buf = b""
                for pkt in packets:
                    frames = codec.decode(pkt)
                    for f in frames:
                        img = f.to_ndarray(format="bgr24")
                        with self._lock:
                            self._frame = self._preprocess(img)
            except Exception:
                buf = b""  

    def read(self) -> np.ndarray | None:
        with self._lock:
            if self._frame is None:
                return None
            frame = self._frame.copy()
            self._frame = None
            return frame

    def close(self):
        self._running = False
        if not self._use_pyav:
            self.cap.release()

class DroneInterface:
    

    def __init__(self, config):
        self.protocol = str(config.protocol)
        self.connection = str(config.connection)
        self._mav = None
        self._serial = None
        self._telemetry: dict[str, float] = {}

    def connect(self):
        if self.protocol == "mavlink":
            from pymavlink import mavutil

            self._mav = mavutil.mavlink_connection(self.connection)
            self._mav.wait_heartbeat()
            print(f"[MAVLink] Verbunden: {self.connection}")
            self._mav.mav.request_data_stream_send(
                self._mav.target_system,
                self._mav.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                10,
                1,
            )
            threading.Thread(target=self._telem_loop, daemon=True).start()
        elif self.protocol == "msp":
            import serial

            self._serial = serial.Serial(self.connection, baudrate=115200, timeout=0.1)
            print(f"[MSP] Verbunden: {self.connection}")

    def _telem_loop(self):
        while True:
            try:
                msg = self._mav.recv_match(blocking=True, timeout=0.1)
                if msg is None:
                    continue
                mt = msg.get_type()
                if mt == "ATTITUDE":
                    self._telemetry.update({"roll": msg.roll, "pitch": msg.pitch, "yaw": msg.yaw})
                elif mt == "VFR_HUD":
                    self._telemetry.update({"speed": msg.groundspeed, "altitude": msg.alt})
                elif mt == "VIBRATION":
                    self._telemetry["vibration"] = max(msg.vibration_x, msg.vibration_y, msg.vibration_z)
                elif mt == "SYS_STATUS":
                    self._telemetry["battery"] = msg.battery_remaining
            except Exception:
                pass

    def get_telemetry(self) -> dict[str, float]:
        return dict(self._telemetry)

    def send_action(self, action: np.ndarray):
        a = np.asarray(action, dtype=np.float32).reshape(-1)

        def to_pwm(v: float) -> int:
            return int(np.clip((v + 1.0) / 2.0 * 1000 + 1000, 1000, 2000))

        if self.protocol == "mavlink":
            channels = [to_pwm(float(a[i])) if i < len(a) else 1500 for i in range(8)]
            self._mav.mav.rc_channels_override_send(
                self._mav.target_system,
                self._mav.target_component,
                *channels,
            )
        elif self.protocol == "msp":
            channels = b"".join(to_pwm(float(a[i])).to_bytes(2, "little") for i in range(min(len(a), 8)))
            header = b"$M<" + bytes([len(channels), 200])
            crc = 0
            for byte in bytes([len(channels), 200]) + channels:
                crc ^= byte
            self._serial.write(header + channels + bytes([crc]))

    def emergency_stop(self):
        print("[EMERGENCY STOP]")
        if self.protocol == "mavlink" and self._mav is not None:
            from pymavlink import mavutil

            self._mav.mav.command_long_send(
                self._mav.target_system,
                self._mav.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            )

    def send_brake(self):
        if self.protocol == "mavlink" and self._mav is not None:
            from pymavlink import mavutil

            self._mav.mav.command_long_send(
                self._mav.target_system,
                self._mav.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,
                1,
                17,  
                0, 0, 0, 0, 0,
            )
        else:
            self.send_action(np.zeros((4,), dtype=np.float32))

    def restore_guided(self):
        

        if self.protocol == "mavlink" and self._mav is not None:
            from pymavlink import mavutil

            
            
            self._mav.mav.set_mode_send(
                self._mav.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                4,  
            )
            print("[MAVLink] Flight mode restored to GUIDED.")

class SafetyMonitor:
    

    def __init__(self, config):
        self.max_vibration = float(config.max_vibration)
        self.min_altitude = float(config.min_altitude)
        self.min_battery = float(config.min_battery)

    def is_safe(self, telemetry: dict[str, float]) -> bool:
        if telemetry.get("vibration", 0.0) > self.max_vibration:
            print("[Safety] Vibrations-Spike → Emergency Stop")
            return False
        if telemetry.get("altitude", 99.0) < self.min_altitude:
            print("[Safety] Bodennähe → Emergency Stop")
            return False
        if telemetry.get("battery", 100) < self.min_battery:
            print("[Safety] Akku kritisch → Emergency Stop")
            return False
        return True

def build_model_image(frame: np.ndarray, prev_frame: np.ndarray | None) -> np.ndarray:
    

    prev_rgb = prev_frame if prev_frame is not None else np.zeros_like(frame)
    diff = np.clip((frame - prev_rgb + 1.0) * 0.5, 0.0, 1.0)   
    return np.concatenate([frame, diff], axis=-1).astype(np.float32)  

class CoreMLPolicy:
    

    def __init__(self, model_path: pathlib.Path, export_cfg: "ExportConfig"):
        self.model = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.ALL)
        self.SF  = export_cfg.stoch_flat_dim
        self.D   = export_cfg.deter_dim
        self.A   = export_cfg.action_dim
        self.C   = export_cfg.img_channels
        self.fs  = export_cfg.safety_frame_stack
        # RSSM resolution (H, W) and SafetyNet resolution (SH, SW) may differ
        self.rssm_h   = export_cfg.img_height
        self.rssm_w   = export_cfg.img_width
        self.safety_h = export_cfg.safety_h
        self.safety_w = export_cfg.safety_w
        # _safety_gray: True wenn das SafetyNet pro Frame 1 Kanal erwartet (Graustufen-Modus).
        # Nutzt safety_channels_per_frame aus ExportConfig; falls 0, fallback auf img_channels.
        _safety_ch = export_cfg.safety_channels_per_frame if export_cfg.safety_channels_per_frame > 0 else export_cfg.img_channels
        self._safety_gray = (_safety_ch == 1)
        self.safety_threshold = 0.4
        self.reset()

    def reset(self):
        self.prev_stoch_flat   = np.zeros((1, self.SF), dtype=np.float32)
        self.prev_deter        = np.zeros((1, self.D),  dtype=np.float32)
        self.prev_action       = np.zeros((1, self.A),  dtype=np.float32)  # Bug #16 fix
        self.prev_filtered_act = np.zeros((1, self.A),  dtype=np.float32)
        
        self.frame_buf: collections.deque = collections.deque(maxlen=self.fs)

    def _build_stacked_safety_image(self, current_frame: np.ndarray) -> np.ndarray:
        # Auto-convert RGB→Grayscale wenn das SafetyNet mit 1-Kanal trainiert wurde
        if self._safety_gray and current_frame.ndim == 3 and current_frame.shape[-1] != 1:
            current_frame = current_frame.mean(axis=-1, keepdims=True).astype(np.float32)

        self.frame_buf.append(current_frame)
        frames = list(self.frame_buf)

        while len(frames) < self.fs:
            frames.insert(0, np.zeros_like(current_frame))

        return np.concatenate(frames, axis=-1).astype(np.float32)  # [H, W, C*fs]

    def act(
        self, image: np.ndarray, raw_frame: np.ndarray, speed: float, is_first: bool
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """
        image     : [rssm_h, rssm_w, 6]  — RGB+diff frame for the RSSM encoder
        raw_frame : [safety_h, safety_w, 3] — full-res RGB frame for SafetyNet stacking
        """

        if is_first:
            self.reset()

        # SafetyNet gets its own stacked buffer at safety resolution
        safety_image = self._build_stacked_safety_image(raw_frame)

        feed = {
            "image":             image[None, ...].astype(np.float32),          
            "safety_image":      safety_image[None, ...].astype(np.float32),   
            "prev_stoch_flat":   self.prev_stoch_flat,
            "prev_deter":        self.prev_deter,
            "prev_action":       self.prev_action,           # Bug #16 fix
            "prev_filtered_act": self.prev_filtered_act,
            "speed":             np.array([[speed]], dtype=np.float32),
        }
        out = self.model.predict(feed)

        action           = np.asarray(out["action"],           dtype=np.float32)
        next_stoch_flat  = np.asarray(out["next_stoch_flat"],  dtype=np.float32)
        next_deter       = np.asarray(out["next_deter"],       dtype=np.float32)
        next_filtered    = np.asarray(out["next_filtered_act"],dtype=np.float32)
        next_action      = np.asarray(out["next_action"],      dtype=np.float32)  # Bug #16 fix
        safety_score     = float(np.asarray(out["safety_score"]).reshape(-1)[0])
        safe_action      = np.asarray(out["safe_action"],      dtype=np.float32).reshape(-1)

        
        self.prev_stoch_flat   = next_stoch_flat.reshape(1, self.SF)
        self.prev_deter        = next_deter.reshape(1, self.D)
        self.prev_action       = next_action.reshape(1, self.A)   # Bug #16 fix
        self.prev_filtered_act = next_filtered.reshape(1, self.A)

        return action.reshape(-1), safety_score, safe_action

@hydra.main(version_base=None, config_path="configs", config_name="fly_real")
def main(config):
    model_path = pathlib.Path(getattr(config, "coreml_model", "logdir/policy.mlpackage")).expanduser()

    
    
    stoch = int(getattr(config.model, "stoch", 32))
    discrete = int(getattr(config.model, "discrete", 48))
    deter = int(getattr(config.model, "deter", 6144))

    _safety_frame_stack = int(getattr(config.model, "safety_frame_stack", 3))
    # Auto-detect per-frame channel count from dataset.raw_image_mode,
    # identical logic to dreamer.py — no manual safety_in_channels needed.
    _RAW_MODE_CHANNELS = {"grayscale": 1, "rgb": 3}
    _raw_image_mode = str(getattr(getattr(config, "dataset", object()), "raw_image_mode", "grayscale")).lower()
    _safety_channels_per_frame = _RAW_MODE_CHANNELS.get(
        _raw_image_mode,
        int(getattr(config.model, "safety_in_channels", 6)) // _safety_frame_stack,  # legacy fallback
    )

    export_cfg = ExportConfig(
        img_height=int(config.model.img_height),
        img_width=int(config.model.img_width),
        img_channels=6,
        action_dim=int(config.model.act_dim),
        stoch_flat_dim=stoch * discrete,
        deter_dim=deter,
        drone_id=int(getattr(config, "drone_id", 0)),
        safety_frame_stack=_safety_frame_stack,
        safety_img_height=int(getattr(config.model, "safety_img_height", config.model.img_height)),
        safety_img_width=int(getattr(config.model, "safety_img_width",  config.model.img_width)),
        safety_channels_per_frame=_safety_channels_per_frame,
    )

    policy = CoreMLPolicy(model_path, export_cfg=export_cfg)
    policy.safety_threshold = float(getattr(config.model, "safety_threshold", 0.4))

    stream = PiVideoStream(
        port=config.stream.port,
        resolution=(export_cfg.safety_h, export_cfg.safety_w),  # stream at SafetyNet res
    )
    drone = DroneInterface(config.drone)
    safety = SafetyMonitor(config.safety)
    drone.connect()

    step = 0
    prev_rssm_frame = None   # tracks previous RSSM-resolution frame for diff channel
    next_is_first = True
    rssm_h = export_cfg.img_height
    rssm_w = export_cfg.img_width
    
    
    
    
    SAFETY_ENGAGE  = float(getattr(config.model, "safety_threshold",      0.4))
    SAFETY_RELEASE = float(getattr(config.model, "safety_release_threshold", 0.15))
    is_evading     = False   

    print("[fly_real] CoreML Inference-Loop gestartet. Ctrl+C zum Beenden.")
    try:
        while True:
            t0 = time.perf_counter()

            frame = stream.read()
            if frame is None:
                time.sleep(0.005)
                continue

            telem = drone.get_telemetry()
            if not safety.is_safe(telem):
                drone.send_brake()
                stable_count = 0
                while stable_count < 10:
                    time.sleep(0.05)
                    t = drone.get_telemetry()
                    speed_ok = float(t.get("speed", 99.0)) < 0.5
                    roll_ok  = abs(float(t.get("roll",  9.9))) < 0.2
                    pitch_ok = abs(float(t.get("pitch", 9.9))) < 0.2
                    stable_count = stable_count + 1 if (speed_ok and roll_ok and pitch_ok) else 0

                
                
                drone.restore_guided()

                is_evading = False   # reset neural-safety state after emergency recovery
                next_is_first = True
                prev_rssm_frame = None
                policy.reset()
                continue

            speed = float(telem.get("speed", 0.0))
            # frame is at safety resolution; resize to RSSM resolution for encoder
            rssm_frame = cv2.resize(frame, (rssm_w, rssm_h))  # cv2: (width, height)
            model_image = build_model_image(rssm_frame, prev_rssm_frame)
            prev_rssm_frame = rssm_frame

            action, safety_score, safe_action = policy.act(
                model_image, raw_frame=frame, speed=speed, is_first=next_is_first
            )
            next_is_first = False

            
            
            
            
            if not is_evading and safety_score > SAFETY_ENGAGE:
                is_evading = True
                print(
                    f"[Neural Safety] ENGAGE  score={safety_score:.3f} > "
                    f"{SAFETY_ENGAGE:.2f} → Neural Evasion"
                )
            elif is_evading and safety_score < SAFETY_RELEASE:
                is_evading = False
                print(
                    f"[Neural Safety] RELEASE score={safety_score:.3f} < "
                    f"{SAFETY_RELEASE:.2f} → Actor resumed"
                )

            if is_evading:
                drone.send_action(safe_action)
            else:
                drone.send_action(action)

            dt_ms = (time.perf_counter() - t0) * 1000.0
            if dt_ms > 10.0:
                print(f"[LATENCY WARN] {dt_ms:.2f}ms > 10.00ms")
            if step % 50 == 0:
                mode_str = "EVADE" if is_evading else "plan "
                print(
                    f"[{step:5d}] {dt_ms:.1f}ms | "
                    f"alt={telem.get('altitude', 0):.1f}m | "
                    f"bat={telem.get('battery', 0)}% | "
                    f"vib={telem.get('vibration', 0):.1f} | "
                    f"safety={safety_score:.3f} [{mode_str}]"
                )
            step += 1

    except KeyboardInterrupt:
        print("\n[fly_real] Manueller Abbruch.")
    finally:
        drone.emergency_stop()
        stream.close()

if __name__ == "__main__":
    main()
