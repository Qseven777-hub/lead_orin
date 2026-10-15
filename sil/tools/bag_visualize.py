#!/usr/bin/env python3
"""Turn a recorded SIL rosbag into visual artifacts.

Reads a ROS1 bag of the SIL topics (``std_msgs/UInt8MultiArray`` holding msgpack
bytes), decodes them with :mod:`lead.evaluation.sil.codec`, and writes:

- ``input.mp4``     the camera images of ``lead/sensor_frame`` (cameras tiled),
- ``controls.csv``  per-step steer/throttle/brake/infer_ms,
- ``controls.png``  those controls as curves,
- ``plan_grid.png`` samples of the ``lead/plan`` trajectory (ego frame).

Reading ROS1 bags needs the pure-python ``rosbags`` package::

    pip install rosbags

Usage::

    python sil/tools/bag_visualize.py <bag file or directory> [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

from lead.evaluation.sil import codec

try:
    from rosbags.rosbag1 import Reader
except ImportError:  # pragma: no cover - dependency hint
    sys.exit("need the 'rosbags' package: pip install rosbags")


def _payload(rawdata: bytes) -> bytes:
    """msgpack bytes inside a serialized ``std_msgs/UInt8MultiArray``."""
    # ROS1 serializes ``uint8[] data`` as a uint32 length prefix + the bytes.
    return rawdata[4:] if len(rawdata) >= 4 else rawdata


def _bag_files(path: str) -> list[str]:
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.bag")))
    else:
        files = [path]
    if not files:
        sys.exit(f"no .bag files under {path}")
    return files


def _read_messages(path: str):
    """Yield (topic, decoded message) for every SIL payload in the bag(s)."""
    for bag in _bag_files(path):
        with Reader(bag) as reader:
            for connection, _timestamp, rawdata in reader.messages():
                if connection.msgtype != "std_msgs/UInt8MultiArray":
                    continue
                yield connection.topic, codec.decode(_payload(rawdata))


def _collect(path: str) -> dict:
    session = None
    frames: list[np.ndarray] = []
    controls: list[dict] = []
    plans: list[dict] = []
    for topic, message in _read_messages(path):
        if topic == "/lead/session":
            session = message
        elif topic == "/lead/sensor_frame":
            sensors = message.get("sensors", {})
            images = [
                sensors[key]
                for key in sorted(sensors)
                if key.startswith("rgb_") and isinstance(sensors[key], np.ndarray)
            ]
            if images:
                frames.append(np.concatenate(images, axis=1))
        elif topic == "/lead/control":
            controls.append(message)
        elif topic == "/lead/plan":
            plans.append(message)
    return {
        "session": session,
        "frames": frames,
        "controls": controls,
        "plans": plans,
    }


def _write_video(frames: list[np.ndarray], out_dir: str, fps: float) -> str | None:
    import cv2

    path = os.path.join(out_dir, "input.mp4")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():  # some aarch64 builds lack mp4v
        path = os.path.join(out_dir, "input.avi")
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height),
        )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return path


def _write_controls(controls: list[dict], out_dir: str) -> str:
    fields = ["step", "steer", "throttle", "brake", "infer_ms"]
    csv_path = os.path.join(out_dir, "controls.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for control in controls:
            writer.writerow(control)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [c["step"] for c in controls]
        figure, axis = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        for name in ("steer", "throttle", "brake"):
            axis[0].plot(steps, [c[name] for c in controls], label=name)
        axis[0].set_ylabel("control")
        axis[0].legend()
        axis[1].plot(
            steps,
            [c.get("infer_ms") or 0.0 for c in controls],
            color="tab:red",
        )
        axis[1].set_ylabel("infer_ms")
        axis[1].set_xlabel("step")
        figure.tight_layout()
        png_path = os.path.join(out_dir, "controls.png")
        figure.savefig(png_path, dpi=120)
        plt.close(figure)
        return png_path
    except ImportError:
        return csv_path


def _write_plan_grid(plans: list[dict], out_dir: str, max_plots: int = 6) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not plans:
        return None
    indices = np.linspace(0, len(plans) - 1, min(max_plots, len(plans))).astype(int)
    figure, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4.2))
    axes = np.atleast_1d(axes)
    for axis, index in zip(axes, indices, strict=True):
        plan = plans[index].get("plan", {})
        waypoints = plan.get("future_waypoints")
        if waypoints is not None:
            axis.plot(waypoints[:, 0], waypoints[:, 1], "-o", ms=3, label="waypoints")
        route = plan.get("route")
        if route is not None:
            axis.plot(route[:, 0], route[:, 1], "--", color="gray", label="route")
        for key, color in (
            ("previous_target_point", "tab:green"),
            ("target_point", "tab:orange"),
            ("next_target_point", "tab:red"),
        ):
            point = plan.get(key)
            if point is not None:
                axis.scatter(point[0], point[1], color=color, label=key)
        axis.scatter(0.0, 0.0, color="black", marker="s", label="ego")
        axis.set_title(f"step {plans[index].get('step')}")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper left")
    figure.tight_layout()
    path = os.path.join(out_dir, "plan_grid.png")
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", help="bag file, or a directory holding *.bag")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--fps", type=float, default=10.0, help="video fps")
    args = parser.parse_args()

    out_dir = args.out or os.path.join(os.path.dirname(args.bag) or ".", "viz")
    os.makedirs(out_dir, exist_ok=True)

    data = _collect(args.bag)
    session = data["session"]
    print(
        f"session route={session.get('route_id') if session else None} "
        f"map={session.get('map_name') if session else None}",
    )
    print(
        f"frames={len(data['frames'])} controls={len(data['controls'])} "
        f"plans={len(data['plans'])}",
    )

    if data["frames"]:
        print("video  :", _write_video(data["frames"], out_dir, args.fps))
    if data["controls"]:
        print("control:", _write_controls(data["controls"], out_dir))
    if data["plans"]:
        print("plan   :", _write_plan_grid(data["plans"], out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
