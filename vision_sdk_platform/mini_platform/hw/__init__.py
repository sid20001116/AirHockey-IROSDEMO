"""Hardware bridge for the mini air-hockey platform.

This package lets the platform's *brain* (agent + planner + FSM + EKF, unchanged) drive the
**physical** Metal arm and read a **real** fixed camera, instead of the MuJoCo simulation.

Design (see the approved plan and ``README.md`` here):

* Low-level joint control is owned by the **vendor** ``metal_sdk`` -- it runs the arm's own
  smooth NRT/RT joint-position law (the exact law ``core/sdk_control.py`` reproduces in sim),
  so nothing here re-implements torque control. We stream planned joint targets to it.
* The puck sensor is swapped for a real, ChArUco-calibrated fixed camera that returns a
  ``PuckMeasurement`` in the *same* world frame ``env_info['robot']['base_frame']`` uses, so
  the agent's world->robot->virtual frame chain is untouched.
* Nothing in ``manipulator_lib`` is imported.

The submodules split by dependency so the geometry/vision math stays testable without the
sim stack:

* ``sdk_backend``  -- ``MetalArm`` / ``MockMetalArm`` over ``metal_sdk`` (needs CPython 3.10).
* ``real_camera``  -- ``V4L2Camera`` capture + ``RealPinholeCamera`` (numpy + cv2 only).
* ``real_sensor``  -- ``RealPuckSensor`` (subclasses ``vision.sensor.PuckVisionSensor``; pulls
  mujoco through it, as the sim vision code does).
* ``calibrate_extrinsics`` -- one-shot camera->world extrinsic calibration from a board.
* ``capture_puck_rolls`` -- camera-only roll logger for the mini-table EKF refit (numpy +
  cv2 only, like ``real_camera``: no arm, no sim, so it runs on a box with just a camera).
* ``run_hw_match`` -- the real-time runtime loop and staged bring-up CLI.
"""
