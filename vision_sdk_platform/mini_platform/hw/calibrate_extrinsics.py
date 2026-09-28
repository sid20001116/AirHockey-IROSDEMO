"""One-shot camera->world extrinsic calibration for the fixed table camera.

Given intrinsics from ``cam_tests.py`` (``calibration.npz``: ``K``, ``dist``) and a ChArUco
board placed at a **known pose in the world/table frame**, this computes the fixed
camera->world extrinsic (``R_wc``, ``C_world``) and writes a combined ``hw_calib.npz``
(:class:`~mini_platform.hw.real_camera.CameraCalibration`) that the runtime loads.

World frame convention (must match the MJCF / ``env_info['robot']['base_frame']``): origin at
the **table centre**, z up out of the playing surface, x/y in the table plane. Place the
board flat on the table and tell this tool where its origin sits in that frame
(``--board-x/-y/-z`` metres, ``--board-yaw`` degrees about +z). Default: board origin at the
world origin, axes aligned (``0,0,0`` / ``0 deg``).

The board geometry below MUST match the printed board (kept in sync with ``cam_tests.py``).

Usage::

    # intrinsics first (from repo root): python cam_tests.py capture && python cam_tests.py calibrate
    python -m mini_platform.hw.calibrate_extrinsics \
        --intrinsics calibration.npz --camera-index 0 --resolution 640x480 \
        --board-x 0 --board-y 0 --board-z 0 --board-yaw 0 --out hw_calib.npz
"""

import argparse
import time

import numpy as np

from mini_platform.hw.real_camera import CameraCalibration, load_intrinsics

# --- ChArUco board geometry (MUST match the printed board AND cam_tests.py) --------------
import cv2

ARUCO_DICT = cv2.aruco.DICT_5X5_100
SQUARES_X = 5            # columns  -- MUST match cam_tests.py and your printed board
SQUARES_Y = 5            # rows
SQUARE_LENGTH_MM = 41.5  # real measured square edge; sets the metric scale of the extrinsic
MARKER_LENGTH_MM = 31.0  # marker edge (< square). UNVERIFIED -- measure it with calipers.
# The GRID must match cam_tests.py exactly, not just the dictionary: squares_x/y decide which
# marker id sits on which cell. Point a 10x10 detector at this 5x5 board and every ArUco marker
# is still found (same dict) but charuco interpolation returns ZERO corners -- which surfaces
# down in capture_correspondences as "board never detected", never as a geometry complaint.
#
# SIZE: from the table mount (0.62 m, fx = 341 px) a 31 mm marker spans ~17 px, i.e. ~2.4 px per
# ArUco module against the ~3 px/module ArUco needs to decode -- workable but marginal. If the
# board will not detect from the mount, move it nearer the camera or fall back to
# hw.calibrate_robot (gripped puck, 50 mm -> ~26 px). ~60 mm squares would be comfortable.

_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
_board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH_MM, MARKER_LENGTH_MM, _dict)
_detector = cv2.aruco.CharucoDetector(_board)


def _rot_z(yaw_rad):
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _compose(R_bc, t_bc_m, board_R_wb, board_t_wb):
    """Camera world pose (R_wc, C_world) from board->camera and board->world."""
    R_cw = R_bc @ board_R_wb.T
    t_cw = t_bc_m - R_cw @ np.asarray(board_t_wb, float)
    R_wc = R_cw.T
    C_world = -R_wc @ t_cw
    return R_wc, C_world


def solve_camera_pose(K, dist, obj_pts, img_pts, board_R_wb, board_t_wb):
    """Compose the board->camera PnP result with the board's world pose -> (R_wc, C_world).

    obj_pts are in the board frame in **mm** (as ``matchImagePoints`` returns them);
    ``board_t_wb`` is in **metres**. See module docstring for the frame convention.

    A flat ChArUco board's object frame follows OpenCV's y-down convention, so its +z points
    *into* the printed surface (away from an overhead camera). Composed against a "z-up" board
    pose that mirrors the camera through the table plane -- C_world lands BELOW the surface and
    every back-projected ray points up, so pixel_to_plane returns None for every pixel. A fixed
    table-view camera is always ABOVE the surface, so if the first solve puts it below we flip
    the assumed board normal (180 deg about the board x-axis: y,z -> -y,-z) and re-solve. Returns
    ``(R_wc, C_world, flipped)``.
    """
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
    if not ok:
        raise RuntimeError("solvePnP failed on the accumulated board correspondences")
    R_bc, _ = cv2.Rodrigues(rvec)                 # board -> camera rotation
    t_bc_m = tvec.ravel() / 1000.0                # mm -> m

    R_wc, C_world = _compose(R_bc, t_bc_m, board_R_wb, board_t_wb)
    flipped = False
    if C_world[2] <= board_t_wb[2]:               # camera ended up at/below the table plane
        flip = np.diag([1.0, -1.0, -1.0])         # 180 deg about board x: reorient the normal up
        R_wc, C_world = _compose(R_bc, t_bc_m, board_R_wb @ flip, board_t_wb)
        flipped = True
    return R_wc, C_world, flipped


def capture_correspondences(camera_index, resolution, n_frames, fourcc="MJPG", timeout_s=30.0):
    """Accumulate ChArUco corner correspondences of a STATIC board over ``n_frames``.

    Paced by DISTINCT camera frames (``frame_seq``) under a wall-clock deadline, never by loop
    iterations. V4L2Camera grabs on a background thread: it returns None until the first frame
    lands (~0.5 s after open, measured) and then re-returns the SAME buffer until the next one
    arrives. An iteration-counted budget therefore does not measure frames at all -- the old
    ``tries < n_frames * 10`` spent all 300 tries in 0.08 ms, before the camera had produced
    anything, and reported "board never detected" having examined ZERO frames. Gating on seq
    also stops one buffer being re-detected into n duplicate correspondence sets, which would
    weight a single view n times in the solve instead of averaging n independent looks.
    """
    from mini_platform.hw.real_camera import V4L2Camera

    cam = V4L2Camera(index=camera_index, resolution=resolution, fourcc=fourcc)
    all_obj, all_img, seen = [], [], 0
    examined, detected, last_seq = 0, 0, -1
    deadline = time.monotonic() + timeout_s
    try:
        while seen < n_frames and time.monotonic() < deadline:
            rgb, seq, _ = cam.read_rgb_stamped()
            if rgb is None or seq == last_seq:
                time.sleep(0.005)          # no NEW frame yet -- yield rather than spin
                continue
            last_seq = seq
            examined += 1
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            ch_corners, ch_ids, _, _ = _detector.detectBoard(gray)
            if ch_ids is None or len(ch_ids) < 4:
                continue
            detected += 1
            obj_pts, img_pts = _board.matchImagePoints(ch_corners, ch_ids)
            if obj_pts is None or len(obj_pts) < 4:
                continue
            all_obj.append(obj_pts)
            all_img.append(img_pts)
            seen += 1
        actual_res = cam.resolution
    finally:
        cam.close()
    if seen == 0:
        # Separate "the camera gave us nothing" from "we looked and the board was not there";
        # the old single message blamed the board for both.
        if examined == 0:
            raise RuntimeError(
                f"camera index {camera_index} delivered NO frames in {timeout_s:.0f}s -- a "
                f"capture problem, not a board problem. Check the index against "
                f"`ls -l /dev/v4l/by-id/` and that nothing else holds the device.")
        raise RuntimeError(
            f"board not detected in any of {examined} camera frames ({detected} had >=4 charuco "
            f"corners but too few matched points). Check lighting and placement, and that the "
            f"print matches the constants in this module ({SQUARES_X}x{SQUARES_Y}, "
            f"{SQUARE_LENGTH_MM}/{MARKER_LENGTH_MM} mm) -- a mismatched GRID finds every ArUco "
            f"marker and still yields zero charuco corners.")
    obj = np.vstack(all_obj)
    img = np.vstack(all_img)
    print(f"accumulated {len(obj)} corner correspondences over {seen} distinct frames "
          f"({examined} examined)")
    return obj, img, actual_res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fixed-camera extrinsic calibration.")
    ap.add_argument("--intrinsics", required=True, help="calibration.npz from cam_tests.py")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--resolution", default="640x480", help="WxH; MUST match the intrinsics "
                                                            "AND the runtime feed")
    ap.add_argument("--frames", type=int, default=30, help="board frames to accumulate")
    ap.add_argument("--board-x", type=float, default=0.0, help="board origin world x (m)")
    ap.add_argument("--board-y", type=float, default=0.0, help="board origin world y (m)")
    ap.add_argument("--board-z", type=float, default=0.0, help="board origin world z (m)")
    ap.add_argument("--board-yaw", type=float, default=0.0, help="board yaw about +z (deg)")
    ap.add_argument("--out", default="hw_calib.npz")
    args = ap.parse_args(argv)

    w, h = (int(x) for x in args.resolution.lower().split("x"))
    K, dist = load_intrinsics(args.intrinsics)

    board_R_wb = _rot_z(np.radians(args.board_yaw))
    board_t_wb = np.array([args.board_x, args.board_y, args.board_z])

    obj, img, actual_res = capture_correspondences(args.camera_index, (w, h), args.frames)
    if actual_res != (w, h):
        print(f"WARNING: camera opened at {actual_res}, requested {(w, h)}. Intrinsics are "
              f"resolution-specific -- recalibrate at the resolution you actually run.")

    R_wc, C_world, flipped = solve_camera_pose(K, dist, obj, img, board_R_wb, board_t_wb)

    calib = CameraCalibration(K=K, dist=dist, R_wc=R_wc, C_world=C_world, resolution=(w, h))
    calib.save(args.out)

    print("\n================ EXTRINSICS ================")
    print(f"camera centre in world (m): {np.round(C_world, 4)}")
    print(f"R_wc (world<-camera):\n{np.round(R_wc, 4)}")
    print(f"height above surface (m)  : {C_world[2]:.4f}")
    if flipped:
        print("NOTE: auto-flipped the board normal so the camera is above the table (the "
              "flat board's frame had +z into the surface). This also flips the board's in-"
              "plane y-axis, so VERIFY orientation: run `check_calib`, then `--stage vision "
              "--view` and confirm the reported world-xy sign matches the table. If x/y are "
              "mirrored or swapped vs the MJCF, correct with --board-yaw (0/90/180/270).")
    if C_world[2] <= 0:
        print("WARNING: camera still at/below the surface -- the solve is unreliable. Re-check "
              "the board is flat, fully visible, and the geometry constants match the print.")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
