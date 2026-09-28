"""Low-level hardware backend: a thin wrapper over the vendor ``metal_sdk``.

The vendor SDK (``metal_sdk.MetalSDKInterface``) owns the smooth low-level joint control --
it runs the arm's own NRT/RT joint-position law (the exact law ``core/sdk_control.py``
reproduces in simulation) with the KDL gravity/Coriolis feedforward and the MIT inner loop
on the motors. This module does **not** reimplement any of that; it opens the SDK, streams
joint targets to it, reads feedback, watches the joint error/temperature codes, and catches
the arm under gravity compensation on exit.

Nothing in ``manipulator_lib`` is imported.

Requirements:
  * CPython **3.10** (the ``metal_sdk`` ``.so`` is built ``cp310``).
  * ``can0`` up (``metal_ros2/can_scripts/start_can0.sh``) and the arm powered.

The vendor API this wraps (``metal_sdk/metal_sdk/metal_sdk_interface.h`` +
``metal_sdk/metal_sdk_interface.cpp``):
  * ``MetalSDKInterface(can_id, urdf_path, arm_end_type, enable_arm)`` -> ``Init()``
  * ``SetArmControlMode(ControlMode.{GRAVITY_COMPENSATION,RT_JOINT_POSITION,NRT_JOINT_POSITION})``
  * ``SetArmJointPosition(array[6], velocity_ratio=5)``   -- NRT, interpolated, for slow moves
  * ``SetFollowerArmJointPosition(list)``                  -- RT streaming, no interpolation
  * ``SetGripperStroke(mm in [0,80], velocity_ratio=5)``
  * ``GetJointPosition()/GetJointVelocity()/GetJointEffort()`` (6, or 7 incl. gripper)
  * ``GetJointErrorCode()`` / ``GetRotorTemperature()``
"""

import os
import threading

import numpy as np

#: Number of arm joints the platform controls (J1-J6). The gripper (motor 7) is held, not
#: part of the control action -- matching the sim, where the grip is a constant clamp.
N_ARM = 6

#: GetJointErrorCode legend (metal_sdk_interface.h:44-49). 0/1 are healthy; >=2 are faults.
ERROR_LEGEND = {
    0: "Disabled", 1: "Enabled", 2: "MotorDisconnected", 3: "OverVoltage",
    4: "UnderVoltage", 5: "Overcurrent", 6: "MosOverTemperature",
    7: "RotorOverTemperature", 8: "Overload",
}


class HardwareFault(RuntimeError):
    """Raised when the arm reports a joint error code or over-temperature."""


def urdf_search_dirs():
    """Directories :func:`default_urdf_path` looks in, in order.

    Exposed so the not-found error can say where it actually looked.

    Deliberately NOT included: ``metal_ros2/src/metal_description/urdf``. It holds URDFs of the
    same name but with ``package://`` meshes and a different mass distribution, so silently
    falling back to it would change the model the SDK's gravity feedforward is built from
    without anyone asking for it. Point ``--urdf`` at it explicitly if you want it.
    """
    dirs = []
    env = os.environ.get("METAL_URDF_DIR")
    if env:
        dirs.append(env)
    try:
        import metal_sdk
        # Only hits for an EDITABLE metal_sdk install, where the package still sits in the
        # checkout next to example/: .../metal_sdk/metal_sdk/__init__.py -> .../metal_sdk/
        pkg_dir = os.path.dirname(os.path.abspath(metal_sdk.__file__))
        dirs.append(os.path.join(os.path.dirname(pkg_dir), "example", "urdf"))
    except Exception:
        pass
    # The documented install is `cd metal_sdk && pip install .` -- NON-editable, so the package
    # lands in site-packages while example/urdf/ stays behind in the checkout and the guess
    # above resolves to site-packages/example/urdf. Walk up from this file to the repo instead,
    # the same way run_hw_match/calibrate_robot already derive their DEFAULT_URDF.
    # .../<repo>/vision_sdk_platform/mini_platform/hw/sdk_backend.py -> <repo>
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    dirs.append(os.path.join(repo_root, "metal_sdk", "example", "urdf"))
    return dirs


def default_urdf_path(arm_end_type):
    """Best-effort locate the vendor example URDF that matches ``arm_end_type``.

    arm_end_type 0 -> no-gripper URDF; 1/2/3 -> with-gripper URDF (matches
    ``metal_sdk/example/single_arm_control.py``). Returns ``None`` if it cannot be found,
    in which case the caller must pass ``urdf_path`` explicitly. See :func:`urdf_search_dirs`.
    """
    name = urdf_filename(arm_end_type)
    for d in urdf_search_dirs():
        cand = os.path.join(d, name)
        if os.path.exists(cand):
            return cand
    return None


def urdf_filename(arm_end_type):
    return "metal_no_gripper.urdf" if arm_end_type == 0 else "metal_with_gripper.urdf"


class MetalArm:
    """Thin, safe wrapper over ``metal_sdk.MetalSDKInterface``.

    Typical use::

        with MetalArm(can_id="can0", arm_end_type=1) as arm:   # __enter__ calls .init()
            arm.set_gripper(HANDLE_STROKE_MM)                   # close on the mallet
            arm.set_mode_rt()
            while running:
                arm.stream_rt(q_des)                            # 6-vector, ~200-400 Hz
                arm.check_health()
        # __exit__ hands the arm to GRAVITY_COMPENSATION so it holds, not collapses.
    """

    def __init__(self, can_id="can0", urdf_path=None, arm_end_type=1,
                 stream_include_gripper=None, temp_limit_c=80.0):
        # Lazy import: the .so is cp310-only, and importing at module load would break the
        # geometry/vision unit tests on any other interpreter.
        from metal_sdk import MetalSDKInterface, ControlMode

        self._ControlMode = ControlMode
        if urdf_path is None:
            urdf_path = default_urdf_path(arm_end_type)
        if urdf_path is None or not os.path.exists(urdf_path):
            searched = "\n  ".join(urdf_search_dirs())
            raise FileNotFoundError(
                f"Could not locate {urdf_filename(arm_end_type)!r}. Pass urdf_path=... (--urdf) "
                f"explicitly, or set METAL_URDF_DIR. Searched:\n  {searched}")

        self._iface = MetalSDKInterface(can_id=can_id, urdf_path=urdf_path,
                                        arm_end_type=arm_end_type, enable_arm=True)
        # Every call into the vendor .so goes through the CAN bus. The RT stream (a
        # background thread, see run_hw_match._JointStreamer) writes setpoints while the
        # main thread reads feedback + health, so the two would otherwise issue overlapping
        # SDK calls onto the same bus. Serialise all iface access so no two calls interleave.
        self._io_lock = threading.RLock()
        self.arm_end_type = arm_end_type
        #: The RT follower command (SetFollowerArmJointPosition) binds to the SDK's
        #: vector<double> overload, documented "all joint position, include gripper". With a
        #: gripper attached (arm_end_type != 0) that vector MUST be length 7 -- streaming 6
        #: trips a fatal C++ CHECK ("joint position size must be 7") that core-dumps the
        #: process, bypassing the gravity-comp catch. So default to including the gripper
        #: whenever one is attached; pass True/False explicitly to override. The 7th element
        #: is the gripper stroke in mm (example/one_slave.py), which stream_rt appends.
        self.stream_include_gripper = ((arm_end_type != 0)
                                       if stream_include_gripper is None
                                       else bool(stream_include_gripper))
        self.temp_limit_c = temp_limit_c
        self._mode = None
        self._gripper_stroke_mm = 0.0
        self._initialized = False

    # -- lifecycle --------------------------------------------------------------------
    def init(self):
        with self._io_lock:
            ok = self._iface.Init()
        if not ok:
            raise RuntimeError(
                "metal_sdk Init() failed -- check can0 is up (start_can0.sh), the arm is "
                "powered, and no other MIT master (ROS node / another script) owns the bus.")
        self._initialized = True
        return self

    def __enter__(self):
        return self.init()

    def __exit__(self, *exc):
        self.catch_and_close()
        return False  # never suppress exceptions -- a fault must propagate after the catch

    # -- modes ------------------------------------------------------------------------
    def set_mode_nrt(self):
        """Non-real-time interpolated joint moves (SetArmJointPosition). For slow/discrete moves."""
        with self._io_lock:
            self._iface.SetArmControlMode(self._ControlMode.NRT_JOINT_POSITION)
        self._mode = "nrt"

    def set_mode_rt(self):
        """Real-time follower streaming (SetFollowerArmJointPosition). For the smooth control loop."""
        with self._io_lock:
            self._iface.SetArmControlMode(self._ControlMode.RT_JOINT_POSITION)
        self._mode = "rt"

    def set_mode_gravity(self):
        """Gravity compensation -- the arm holds itself, accepts no position command."""
        with self._io_lock:
            self._iface.SetArmControlMode(self._ControlMode.GRAVITY_COMPENSATION)
        self._mode = "grav"

    @property
    def mode(self):
        return self._mode

    # -- commands ---------------------------------------------------------------------
    def move_nrt(self, q6, velocity_ratio=3):
        """Interpolated move to ``q6`` (rad). velocity_ratio in [1,10]; 1 slowest. NRT mode."""
        q = np.asarray(q6, dtype=float)[:N_ARM]
        with self._io_lock:
            self._iface.SetArmJointPosition([float(x) for x in q], int(velocity_ratio))

    def stream_rt(self, q6):
        """Push one raw joint setpoint (rad) to the RT follower. Call at ~200-400 Hz."""
        q = [float(x) for x in np.asarray(q6, dtype=float)[:N_ARM]]
        if self.stream_include_gripper:
            q = q + [float(self._gripper_stroke_mm)]
        with self._io_lock:
            self._iface.SetFollowerArmJointPosition(q)

    def set_gripper(self, stroke_mm, velocity_ratio=5):
        """Set the gripper opening (mm, [0,80]). Called once to clamp the mallet handle."""
        stroke_mm = float(np.clip(stroke_mm, 0.0, 80.0))
        self._gripper_stroke_mm = stroke_mm
        with self._io_lock:
            self._iface.SetGripperStroke(stroke_mm, int(velocity_ratio))

    # -- feedback ---------------------------------------------------------------------
    def q(self):
        with self._io_lock:
            v = self._iface.GetJointPosition()
        return np.asarray(v, dtype=float)[:N_ARM]

    def qd(self):
        with self._io_lock:
            v = self._iface.GetJointVelocity()
        return np.asarray(v, dtype=float)[:N_ARM]

    def tau(self):
        with self._io_lock:
            v = self._iface.GetJointEffort()
        return np.asarray(v, dtype=float)[:N_ARM]

    def end_pose(self):
        """Arm end pose [x,y,z,roll,pitch,yaw] as the SDK's own FK reports it."""
        with self._io_lock:
            v = self._iface.GetArmEndPose()
        return np.asarray(v, dtype=float)

    def error_codes(self):
        with self._io_lock:
            return list(self._iface.GetJointErrorCode())

    def temperatures(self):
        with self._io_lock:
            return list(self._iface.GetRotorTemperature())

    # -- safety -----------------------------------------------------------------------
    def check_health(self):
        """Raise ``HardwareFault`` on any joint error code >= 2 or over-temperature.

        Also caches the two readbacks it just paid for on ``last_codes`` / ``last_temps``, so the
        telemetry row can record them without issuing CAN reads of its own. The poll already
        shares ``_io_lock`` with the RT stream writes -- reading them a second time per tick just
        to log them is exactly the contention this poll was moved off the tick rate to avoid.
        Without these columns a log cannot say whether the arm heated up over a long match, which
        is the one monotone physical degradation the banner warns about.
        """
        codes = self.error_codes()
        self.last_codes = [int(c) for c in codes[:N_ARM]]
        bad = [(j, ERROR_LEGEND.get(c, f"code{c}")) for j, c in enumerate(codes[:N_ARM]) if c >= 2]
        if bad:
            raise HardwareFault(f"joint error codes: {bad}")
        temps = self.temperatures()
        self.last_temps = [np.nan if t is None else float(t) for t in temps[:N_ARM]]
        hot = [(j, round(t, 1)) for j, t in enumerate(temps[:N_ARM])
               if t is not None and t > self.temp_limit_c]
        if hot:
            raise HardwareFault(f"rotor over temperature (> {self.temp_limit_c} C): {hot}")

    def catch_and_close(self):
        """Hand the arm to gravity compensation so it holds instead of collapsing.

        Idempotent and exception-safe -- meant to run from ``finally``/``__exit__`` even when
        the process is already failing. SDK-native replacement for manipulator_lib's
        ``hw_guard.sh`` (which we do not use).

        Uses a BOUNDED lock acquire: if the RT stream thread is wedged in a CAN write while
        holding ``_io_lock`` (a hung bus after a hard fault), a plain ``with self._io_lock`` here
        would DEADLOCK -- the arm would never get gravity comp and the process would never exit.
        Time out instead and warn loudly, so this ALWAYS returns and the program can exit.
        """
        if not self._initialized:
            return
        got = self._io_lock.acquire(timeout=2.0)
        if not got:
            print("WARNING: SDK bus lock is wedged (RT stream stuck mid-write) -- cannot command "
                  "gravity compensation. E-STOP or power-cycle the arm NOW. Exiting anyway so the "
                  "process does not hang.")
            return
        try:
            self.set_mode_gravity()          # _io_lock is a reentrant RLock, so this re-acquire is fine
        except Exception:
            pass
        finally:
            self._io_lock.release()


class MockMetalArm:
    """No-hardware stand-in for :class:`MetalArm` with the identical surface.

    Echoes commanded positions back as feedback (with a light first-order lag) so the
    runtime loop, interpolation and safety plumbing can be exercised without an arm or the
    cp310 SDK. Reports all joints healthy. Use ``run_hw_match --mock``.
    """

    def __init__(self, q0=None, lag=0.4, **_):
        self._q = np.zeros(N_ARM) if q0 is None else np.asarray(q0, float)[:N_ARM].copy()
        self._q_cmd = self._q.copy()
        self._qd = np.zeros(N_ARM)
        self._lag = float(lag)
        self._mode = None
        self._gripper_stroke_mm = 0.0
        self._initialized = False
        self.arm_end_type = 1
        self.stream_include_gripper = False
        self.temp_limit_c = 80.0

    def init(self):
        self._initialized = True
        return self

    def __enter__(self):
        return self.init()

    def __exit__(self, *exc):
        self.catch_and_close()
        return False

    def set_mode_nrt(self):
        self._mode = "nrt"

    def set_mode_rt(self):
        self._mode = "rt"

    def set_mode_gravity(self):
        self._mode = "grav"

    @property
    def mode(self):
        return self._mode

    def _advance(self, q_cmd):
        q_cmd = np.asarray(q_cmd, float)[:N_ARM]
        new_q = self._q + self._lag * (q_cmd - self._q)
        self._qd = (new_q - self._q) / (1 / 250.0)  # nominal stream period; magnitude only
        self._q = new_q
        self._q_cmd = q_cmd

    def move_nrt(self, q6, velocity_ratio=3):
        self._advance(q6)
        self._q = np.asarray(q6, float)[:N_ARM].copy()  # NRT "reaches" the target
        self._qd = np.zeros(N_ARM)

    def stream_rt(self, q6):
        self._advance(q6)

    def set_gripper(self, stroke_mm, velocity_ratio=5):
        self._gripper_stroke_mm = float(np.clip(stroke_mm, 0.0, 80.0))

    def q(self):
        return self._q.copy()

    def qd(self):
        return self._qd.copy()

    def tau(self):
        return np.zeros(N_ARM)

    def end_pose(self):
        return np.zeros(6)

    def error_codes(self):
        return [1] * N_ARM

    def temperatures(self):
        return [25.0] * N_ARM

    def check_health(self):
        # Cache the same way the real arm does, so --mock logs sane health columns instead of
        # holes and the telemetry path is exercised without hardware.
        self.last_codes = [int(c) for c in self.error_codes()[:N_ARM]]
        self.last_temps = [float(t) for t in self.temperatures()[:N_ARM]]
        return None

    def catch_and_close(self):
        self._mode = "grav"
