"""fly_real.py — Hardware-Inference-Loop für echte Drohne.

Hardware-Stack:
  Video:     Raspberry Pi Zero 2W → GStreamer UDP:5600 → MacBook
  Steuerung: pymavlink MAVLink (ArduPilot) / MSP (Betaflight-Fallback)
  Inference: Core ML (MLProgram) via coremltools auf Apple Silicon

Starten:
  python fly_real.py --config-name fly_real
"""

import pathlib
import threading
import time

import coremltools as ct
import cv2
import hydra
import numpy as np

from native.coreml_export import ExportConfig


class PiVideoStream:
    """H.264 UDP-Stream vom Pi Zero 2W via GStreamer. Eigener Thread."""

    PIPELINE = (
        "udpsrc port={port} "
        "! application/x-rtp,encoding-name=H264,payload=96 "
        "! rtph264depay ! h264parse ! avdec_h264 "
        "! videoconvert ! appsink drop=true max-buffers=1"
    )

    def __init__(self, port: int = 5600, resolution: tuple[int, int] = (256, 256)):
        self.resolution = resolution
        pipeline = self.PIPELINE.format(port=port)
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            print("[VideoStream] GStreamer nicht verfügbar → DirectCapture Fallback")
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, self.resolution)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._frame = frame.astype(np.float32) / 255.0

    def read(self) -> np.ndarray | None:
        with self._lock:
            if self._frame is None:
                return None
            frame = self._frame.copy()
            self._frame = None
            return frame

    def close(self):
        self._running = False
        self.cap.release()


class DroneInterface:
    """MAVLink (ArduPilot primär) / MSP (Betaflight-Fallback)."""

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
                0,
                0,
                0,
                0,
                0,
            )
        else:
            self.send_action(np.zeros((4,), dtype=np.float32))


class SafetyMonitor:
    """Unabhängige Hardware-Sicherheitsprüfung."""

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


def build_model_image(frame: np.ndarray, prev_frame: np.ndarray | None, use_depth: bool) -> np.ndarray:
    """Build model input image in expected channel layout using NumPy only."""
    if use_depth:
        depth = np.mean(frame, axis=-1, keepdims=True)
        prev_depth = np.mean(prev_frame, axis=-1, keepdims=True) if prev_frame is not None else np.zeros_like(depth)
        diff = depth - prev_depth
        return np.concatenate([depth, diff], axis=-1).astype(np.float32)

    prev_rgb = prev_frame if prev_frame is not None else np.zeros_like(frame)
    diff = frame - prev_rgb
    return np.concatenate([frame, diff], axis=-1).astype(np.float32)


class CoreMLPolicy:
    def __init__(self, model_path: pathlib.Path, state_dim: int):
        self.model = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.ALL)
        self.state = np.zeros((1, state_dim), dtype=np.float32)

    def reset(self):
        self.state.fill(0.0)

    def act(self, image: np.ndarray, speed: float, is_first: bool) -> np.ndarray:
        if is_first:
            self.reset()
        feed = {
            "image": image[None, ...].astype(np.float32),
            "speed": np.array([[speed]], dtype=np.float32),
            "state": self.state.astype(np.float32),
        }
        out = self.model.predict(feed)
        action = np.asarray(out.get("action", np.zeros((1, 4), dtype=np.float32)), dtype=np.float32)
        if "next_state" in out:
            self.state = np.asarray(out["next_state"], dtype=np.float32)
        return action.reshape(-1)


@hydra.main(version_base=None, config_path="configs", config_name="fly_real")
def main(config):
    model_path = pathlib.Path(getattr(config, "coreml_model", "logdir/policy.mlpackage")).expanduser()
    export_cfg = ExportConfig(
        img_height=int(config.model.img_height),
        img_width=int(config.model.img_width),
        img_channels=2 if bool(config.model.use_depth) else 6,
        action_dim=int(config.model.act_dim),
        state_dim=int(getattr(config.model, "coreml_state_dim", 7696)),
    )
    state_dim = export_cfg.state_dim
    policy = CoreMLPolicy(model_path, state_dim=state_dim)

    stream = PiVideoStream(
        port=config.stream.port,
        resolution=(config.model.img_height, config.model.img_width),
    )
    drone = DroneInterface(config.drone)
    safety = SafetyMonitor(config.safety)
    drone.connect()

    step = 0
    prev_frame = None
    next_is_first = True

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
                    roll_ok = abs(float(t.get("roll", 9.9))) < 0.2
                    pitch_ok = abs(float(t.get("pitch", 9.9))) < 0.2
                    stable_count = stable_count + 1 if (speed_ok and roll_ok and pitch_ok) else 0
                next_is_first = True
                prev_frame = None
                policy.reset()
                continue

            speed = float(telem.get("speed", 0.0))
            model_image = build_model_image(frame, prev_frame, use_depth=bool(config.model.use_depth))
            prev_frame = frame

            action = policy.act(model_image, speed=speed, is_first=next_is_first)
            next_is_first = False
            drone.send_action(action)

            dt_ms = (time.perf_counter() - t0) * 1000.0
            if dt_ms > 10.0:
                print(f"[LATENCY WARN] {dt_ms:.2f}ms > 10.00ms")
            if step % 50 == 0:
                print(
                    f"[{step:5d}] {dt_ms:.1f}ms | "
                    f"alt={telem.get('altitude', 0):.1f}m | "
                    f"bat={telem.get('battery', 0)}% | "
                    f"vib={telem.get('vibration', 0):.1f}"
                )
            step += 1

    except KeyboardInterrupt:
        print("\n[fly_real] Manueller Abbruch.")
    finally:
        drone.emergency_stop()
        stream.close()


if __name__ == "__main__":
    main()
