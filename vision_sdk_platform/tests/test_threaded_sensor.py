"""Unit tests for ``ThreadedPuckSensor`` (hw/real_sensor.py).

The wrapper runs an inner puck sensor's ~40 ms ``measure()`` off the control thread so the
agent loop never blocks on vision. The contract these pin:

  * a fresh frame's measurement is handed to the control thread **exactly once**, then
    ``None`` until a newer frame lands (so the EKF corrects once per frame, never twice on the
    same detection);
  * ``None`` before the first frame and after a no-detection frame;
  * unknown attributes and ``close()`` delegate to the inner sensor.

No camera, mujoco or SDK needed -- the inner sensor is a plain stand-in.
"""

import threading
import time

from mini_platform.hw.real_sensor import ThreadedPuckSensor


class _FakeSensor:
    """Stand-in inner sensor: ``measure()`` returns whatever ``value`` is set to and counts
    calls; carries an arbitrary attribute to prove delegation."""

    def __init__(self):
        self.value = None
        self.calls = 0
        self.closed = False
        self.camera = object()          # a stand-in for the delegated `.camera`
        self._lock = threading.Lock()

    def measure(self):
        with self._lock:
            self.calls += 1
            return self.value

    def close(self):
        self.closed = True


def _wait(pred, timeout=2.0):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.005)
    return False


def test_none_before_any_frame():
    s = ThreadedPuckSensor(_FakeSensor())
    # No thread started yet: nothing has been produced.
    assert s.measure() is None


def test_measurement_delivered_exactly_once():
    inner = _FakeSensor()
    inner.value = "M1"
    s = ThreadedPuckSensor(inner)
    s.start()
    try:
        # The background thread should produce M1; the control thread gets it once.
        assert _wait(lambda: s.measure() == "M1")
        # ...and only once -- subsequent reads are None until a NEWER frame lands, even
        # though the inner sensor keeps returning M1 (same detection re-processed).
        nones = [s.measure() for _ in range(5)]
        assert all(m is None for m in nones), nones
    finally:
        s.close()


def test_new_value_supersedes():
    inner = _FakeSensor()
    inner.value = "M1"
    s = ThreadedPuckSensor(inner)
    s.start()
    try:
        assert _wait(lambda: s.measure() == "M1")
        inner.value = "M2"                       # a new frame with a new detection
        assert _wait(lambda: s.measure() == "M2")
    finally:
        s.close()


def test_none_frame_is_forwarded():
    inner = _FakeSensor()
    inner.value = None                           # no puck detected
    s = ThreadedPuckSensor(inner)
    s.start()
    try:
        # Frames are being produced (calls climb) but every measure() is None: a dropout
        # is indistinguishable from "no new frame", which is exactly what the EKF wants.
        start_calls = inner.calls
        assert _wait(lambda: inner.calls > start_calls + 2)
        assert s.measure() is None
    finally:
        s.close()


def test_close_stops_thread_and_delegates():
    inner = _FakeSensor()
    s = ThreadedPuckSensor(inner)
    s.start()
    assert _wait(lambda: inner.calls > 0)
    s.close()
    assert inner.closed is True
    assert s._thread is None
    # Delegation of unknown attributes to the inner sensor.
    assert s.camera is inner.camera


def test_start_is_idempotent():
    inner = _FakeSensor()
    s = ThreadedPuckSensor(inner)
    s.start()
    t = s._thread
    s.start()                                    # second start must not spawn a second thread
    assert s._thread is t
    s.close()


class _FakeSeqSensor(_FakeSensor):
    """Inner sensor that also exposes ``frame_seq`` (like ``RealPuckSensor`` over a camera), so
    the frame-novelty gate can be exercised without a real camera."""

    def __init__(self):
        super().__init__()
        self.value = "M"
        self.frame_seq = 0


def test_coasts_between_camera_frames():
    """With a frame_seq, the thread detects each frame ONCE and coasts (control-thread measure()
    -> None, no re-detect) until the camera frame advances -- the fix for the cv2 over-fusing."""
    inner = _FakeSeqSensor()
    inner.frame_seq = 5
    s = ThreadedPuckSensor(inner)
    s.start()
    try:
        assert _wait(lambda: s.measure() == "M")     # first frame delivered once
        # frame_seq frozen: control gets None AND the thread stops re-detecting the same frame
        base_calls = inner.calls
        time.sleep(0.1)
        assert s.measure() is None
        assert inner.calls <= base_calls + 1, inner.calls
        # advance the camera frame -> a fresh detection is produced and delivered once
        inner.frame_seq = 6
        assert _wait(lambda: s.measure() == "M")
    finally:
        s.close()
