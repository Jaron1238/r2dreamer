#!/usr/bin/env python3
"""
Läuft auf dem Raspberry Pi Zero 2W.
Streamt H.264 via GStreamer UDP an MacBook.
"""

import argparse
import subprocess


def start_stream(host: str, port: int, width: int, height: int, fps: int):
    pipeline = (
        f"libcamerasrc ! "
        f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
        f"videoconvert ! "
        f"x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast ! "
        f"rtph264pay config-interval=1 pt=96 ! "
        f"udpsink host={host} port={port}"
    )
    print(f"[Pi Stream] → {host}:{port} @ {width}x{height} {fps}fps")
    proc = subprocess.Popen(["gst-launch-1.0"] + pipeline.split())
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.1.100")
    parser.add_argument("--port", type=int, default=5600)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    start_stream(args.host, args.port, args.width, args.height, args.fps)
