"""fly_real.py — Hardware-Inference-Loop für echte Drohne.

Hardware-Stack:
  Video:     Raspberry Pi Zero 2W → GStreamer UDP:5600 → MacBook
  Steuerung: pymavlink MAVLink (ArduPilot) / MSP (Betaflight-Fallback)
  Inference: MacBook M2 (MPS), ~5–8 ms

Starten:
  python fly_real.py --config-name fly_real
"""

import pathlib
import threading
import time

import cv2
import hydra
import numpy as np
import torch

from dreamer import Dreamer


class PiVideoStream:
    """H.264 UDP-Stream vom Pi Zero 2W via GStreamer. Eigener Thread."""

    PIPELINE = (
        "udpsrc port={port} "
        "! application/x-rtp,encoding-name=H264,payload=96 "
        "! rtph264depay ! h264parse ! avdec_h264 "
        "! videoconvert ! appsink drop=true max-buffers=1"
    )

    def __init__(self, port: int = 5600, resolution: tuple = (256, 256)):
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
                    self._frame = frame

    def read(self):
        with self._lock:
            if self._frame is None:
                return None
            return torch.from_numpy(self._frame.copy()).float() / 255.0

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
        self._telemetry = {}

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

    def get_telemetry(self) -> dict:
        return dict(self._telemetry)

    def send_action(self, action: torch.Tensor):
        a = action.cpu().numpy().flatten()

        def to_pwm(v):
            return int(np.clip((v + 1.0) / 2.0 * 1000 + 1000, 1000, 2000))

        if self.protocol == "mavlink":
            channels = [to_pwm(a[i]) if i < len(a) else 1500 for i in range(8)]
            self._mav.mav.rc_channels_override_send(
                self._mav.target_system,
                self._mav.target_component,
                *channels,
            )
        elif self.protocol == "msp":
            channels = b"".join(to_pwm(a[i]).to_bytes(2, "little") for i in range(min(len(a), 8)))
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
            self.send_action(torch.zeros(4))


class SafetyMonitor:
    """Unabhängige Hardware-Sicherheitsprüfung."""

    def __init__(self, config):
        self.max_vibration = float(config.max_vibration)
        self.min_altitude = float(config.min_altitude)
        self.min_battery = float(config.min_battery)

    def is_safe(self, telemetry: dict) -> bool:
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


def build_model_image(frame: torch.Tensor, prev_frame: torch.Tensor | None, use_depth: bool) -> torch.Tensor:
    """Build Dreamer input image in expected channel layout.

    - use_depth=True  -> 2 channels: [depth(gray), temporal_diff]
    - use_depth=False -> 6 channels: [rgb, temporal_diff]
    """
    if use_depth:
        depth = frame.mean(dim=-1, keepdim=True)
        prev_depth = prev_frame.mean(dim=-1, keepdim=True) if prev_frame is not None else torch.zeros_like(depth)
        diff = depth - prev_depth
        return torch.cat([depth, diff], dim=-1)

    prev_rgb = prev_frame if prev_frame is not None else torch.zeros_like(frame)
    diff = frame - prev_rgb
    return torch.cat([frame, diff], dim=-1)


@hydra.main(version_base=None, config_path="configs", config_name="fly_real")
def main(config):
    device = torch.device(config.device)

    agent = Dreamer(config.model).to(device)
    ckpt = torch.load(pathlib.Path(config.checkpoint).expanduser(), map_location=device)
    ckpt_phase = ckpt.get("phase", None)
    if ckpt_phase is not None and ckpt_phase < 3:
        print(f"[WARNUNG] Checkpoint ist aus Phase {ckpt_phase} — unvollständiges Modell!")
    agent.load_state_dict(ckpt["model"])
    agent.eval()
    if device.type == "cuda":
        agent.act = torch.compile(agent.act, mode="reduce-overhead")
    elif device.type == "mps":
        agent.act = torch.compile(agent.act)

    stream = PiVideoStream(
        port=config.stream.port,
        resolution=(config.model.img_height, config.model.img_width),
    )
    drone = DroneInterface(config.drone)
    safety = SafetyMonitor(config.safety)
    drone.connect()

    state = agent.get_initial_state(B=1).to(device)
    step = 0
    prev_frame = None
    next_is_first = True

    print("[fly_real] Inference-Loop gestartet. Ctrl+C zum Beenden.")
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
                state = agent.get_initial_state(B=1).to(device)
                next_is_first = True
                prev_frame = None
                continue

            speed = torch.tensor([[telem.get("speed", 0.0)]], dtype=torch.float32, device=device)
            model_image = build_model_image(frame, prev_frame, use_depth=bool(config.model.use_depth))
            prev_frame = frame
            obs = {
                "image": model_image.unsqueeze(0).to(device),
                "speed": speed,
                "is_first": torch.tensor([next_is_first], dtype=torch.bool, device=device),
            }
            next_is_first = False

            action, state = agent.act(obs, state, eval=True)
            drone.send_action(action.squeeze(0))

            dt = (time.perf_counter() - t0) * 1000
            if step % 50 == 0:
                print(
                    f"[{step:5d}] {dt:.1f}ms | "
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
