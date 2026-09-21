"""Thin XRoboToolkit client for Pico controller and headset state."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_SERVICE = "/opt/apps/roboticsservice/runService.sh"


def _load_xrt():
    try:
        import xrobotoolkit_sdk as xrt
    except ImportError as exc:
        raise RuntimeError(
            "Pico teleop requires xrobotoolkit_sdk. Install XRoboToolkit PC Service "
            "and its Python bindings before setting interactive.pico=true."
        ) from exc
    return xrt


class XrClient:
    """Client for the XRoboToolkit SDK (same surface as GR00T's Pico XrClient)."""

    def __init__(self) -> None:
        self._xrt = _load_xrt()
        self._xrt.init()
        print("[pico-play] XRoboToolkit SDK initialized.")

    def get_pose_by_name(self, name: str) -> np.ndarray:
        if name == "left_controller":
            return self._xrt.get_left_controller_pose()
        if name == "right_controller":
            return self._xrt.get_right_controller_pose()
        if name == "headset":
            return self._xrt.get_headset_pose()
        raise ValueError(
            f"Invalid name: {name}. Valid names are: "
            "'left_controller', 'right_controller', 'headset'."
        )

    def get_key_value_by_name(self, name: str) -> float:
        if name == "left_trigger":
            return float(self._xrt.get_left_trigger())
        if name == "right_trigger":
            return float(self._xrt.get_right_trigger())
        if name == "left_grip":
            return float(self._xrt.get_left_grip())
        if name == "right_grip":
            return float(self._xrt.get_right_grip())
        raise ValueError(
            f"Invalid name: {name}. Valid names are: "
            "'left_trigger', 'right_trigger', 'left_grip', 'right_grip'."
        )

    def get_joystick_state(self, controller: str) -> list[float]:
        if controller.lower() == "left":
            axis = self._xrt.get_left_axis()
        elif controller.lower() == "right":
            axis = self._xrt.get_right_axis()
        else:
            raise ValueError("Invalid controller. Valid controllers are: 'left', 'right'.")
        values = np.asarray(axis, dtype=np.float64).reshape(-1)
        if values.size < 2:
            return [0.0, 0.0]
        return [float(values[0]), float(values[1])]

    def close(self) -> None:
        self._xrt.close()


class PicoDevice:
    """Owns the XR client and optional PC-service process."""

    def __init__(
        self,
        *,
        client: XrClient | None = None,
        service_path: str | None = _DEFAULT_SERVICE,
    ) -> None:
        self._service_proc: subprocess.Popen[Any] | None = None
        self.client: XrClient | None = None
        if service_path:
            self._start_service(service_path)
        try:
            self.client = client if client is not None else XrClient()
        except Exception:
            self.close()
            raise

    def _start_service(self, service_path: str) -> None:
        path = Path(service_path)
        if not path.is_file():
            print(f"[pico-play] Pico service not found at {service_path}; assuming it is running.")
            return
        self._service_proc = subprocess.Popen(["bash", str(path)])
        print(f"[pico-play] Pico service running with pid {self._service_proc.pid}")

    def get(self) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("Pico device is closed")
        return {
            "left_pose": np.asarray(
                self.client.get_pose_by_name("left_controller"), dtype=np.float64
            ),
            "right_pose": np.asarray(
                self.client.get_pose_by_name("right_controller"), dtype=np.float64
            ),
            "head_pose": np.asarray(self.client.get_pose_by_name("headset"), dtype=np.float64),
            "right_grip": float(self.client.get_key_value_by_name("right_grip")),
            "right_trigger": float(self.client.get_key_value_by_name("right_trigger")),
            "left_joystick": self.client.get_joystick_state("left"),
        }

    def close(self) -> None:
        if self.client is not None:
            try:
                self.client.close()
            finally:
                self.client = None
        if self._service_proc is not None:
            self._service_proc.kill()
            print(f"[pico-play] Pico service killed with pid {self._service_proc.pid}")
            self._service_proc = None
