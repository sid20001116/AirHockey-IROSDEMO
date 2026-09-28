"""Camera-based puck sensing (vision assignment): pinhole camera model, red/blue
segmentation, and the sensor that turns offscreen renders into world-frame puck
measurements for the EKF.

``camera`` and ``sensor`` import **mujoco** (they render and query the sim scene);
``detector`` deliberately does not -- it is numpy/scipy/cv2 only, so the hardware and
offline tools can segment real frames on a box with no sim stack (``hw/real_camera.py``,
``hw/capture_puck_rolls.py``, ``hw/tune_puck.py``). The re-exports below are therefore
**lazy** (PEP 562): ``from mini_platform.vision.detector import detect_puck`` costs no mujoco,
while ``from mini_platform.vision import PinholeCamera`` still works exactly as before.
"""

_LAZY = {
    "PUCK_TOP_Z": "mini_platform.vision.camera",
    "PinholeCamera": "mini_platform.vision.camera",
    "PuckMeasurement": "mini_platform.vision.sensor",
    "PuckVisionSensor": "mini_platform.vision.sensor",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
