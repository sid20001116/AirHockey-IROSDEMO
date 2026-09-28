"""Point-based camera->world extrinsic calibration from the PUCK at known table points.

Use this instead of ``calibrate_extrinsics`` (the ChArUco-board method) when the board frame's
orientation vs the sim/robot world frame is ambiguous -- which is exactly the failure that put
the arm off-board: the board extrinsic came out with the world **x-axis inverted**, so the
defender read the puck's down-table position backwards.

This method has no board-frame ambiguity: you place the PUCK at a handful of points whose world
(table-centre-frame) coordinates you MEASURE with a tape, the tool detects the puck pixel at
each, and ``cv2.solvePnP`` recovers the camera pose directly in that measured frame. World-x
comes out however you measured it -- there is nothing to flip.

World frame convention (identical to ``calibrate_extrinsics`` / ``env_info['robot']['base_frame']``):
origin at the **table centre**, z up out of the surface, x/y in the plane. For this rig
+x points toward the FAR end (away from arm 1, whose base is at world x approx -0.62) and +y
toward one rail -- pick the axis directions to match the MJCF and keep them consistent when you
measure. Give at least 4 non-collinear points; 6-8 spread across the table is better.

Usage (measure each point first, then run)::

    python -m mini_platform.hw.calibrate_points \
        --intrinsics calibration.npz --camera-index 4 --resolution 640x480 \
        --puck-profile puck_profile.json \
        --points "0,0  0.3,0  -0.3,0  0,0.15  0,-0.15  0.25,0.1  -0.25,-0.1" \
        --out hw_calib.npz

It walks you through the points one at a time: place the puck, capture, it detects and
averages the pixel over a few frames, then solves and prints the reprojection + a world-xy
round-trip so you can confirm the signs BEFORE trusting it in play.

``--view`` (default on) opens the same live window as ``calibrate_robot``: the detector's own
green box + centroid on the puck (red border when nothing is detected), the mark to place it on,
and a magenta ring on every pixel already captured. SPACE/ENTER captures, ``s`` skips the mark,
``q`` stops early and solves with what it has. ``--no-view`` falls back to Enter prompts.
"""

import argparse

import numpy as np

from mini_platform.hw.real_camera import (CameraCalibration, RealPinholeCamera,
                                          PUCK_TOP_Z, load_intrinsics)


def parse_points(s):
    """Parse "x,y  x,y ..." (metres, table-centre frame) into an (N,2) array."""
    pts = []
    for tok in s.replace(",", " ").split():
        pts.append(float(tok))
    if len(pts) % 2 != 0:
        raise SystemExit("--points must be pairs 'x,y x,y ...' (even count of numbers)")
    arr = np.array(pts, float).reshape(-1, 2)
    if len(arr) < 4:
        raise SystemExit(f"need >= 4 points for PnP, got {len(arr)}")
    return arr


def detect_puck_pixel(cam, profile, n_frames, settle=5):
    """Average the detected puck centroid pixel over ``n_frames`` (drops non-detections)."""
    from mini_platform.vision.detector import detect_puck

    for _ in range(settle):          # let the threaded grab refresh to the current scene
        cam.read_rgb()
    us, vs = [], []
    tries = 0
    while len(us) < n_frames and tries < n_frames * 20:
        tries += 1
        frame = cam.read_rgb()
        if frame is None:
            continue
        det = detect_puck(frame, profile, backend="cv2")
        if det is None:
            continue
        us.append(det.centroid[0])
        vs.append(det.centroid[1])
    if len(us) < max(3, n_frames // 3):
        return None
    return np.array([np.median(us), np.median(vs)])


def wait_in_view(cv2, win, cam, profile, label, captured_px):
    """Live detection view until the operator acts. Returns 'capture', 'skip' or 'quit'."""
    from mini_platform.hw.calibrate_robot import _annotate     # the calibrate_robot overlay
    from mini_platform.vision.detector import detect_puck

    while True:
        frame = cam.read_rgb()
        if frame is not None:
            det = detect_puck(frame, profile, backend="cv2")
            bgr = _annotate(cv2, frame, det, [label, f"captured {len(captured_px)}    "
                                                     "SPACE/ENTER capture   s skip   q quit"])
            for u, v in captured_px:
                cv2.circle(bgr, (int(round(u)), int(round(v))), 7, (255, 0, 255), 2)
            cv2.imshow(win, bgr)
        key = cv2.waitKey(20) & 0xFF
        if key in (32, 13):
            return "capture"
        if key == ord("s"):
            return "skip"
        if key in (ord("q"), 27):
            return "quit"


def solve_extrinsic(K, dist, world_pts_3d, img_pts):
    """solvePnP world->camera, return (R_wc, C_world, reproj_err_px)."""
    import cv2

    ok, rvec, tvec = cv2.solvePnP(world_pts_3d.astype(np.float64),
                                  img_pts.astype(np.float64), K, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed on the puck-point correspondences")
    R_cw, _ = cv2.Rodrigues(rvec)
    t_cw = tvec.ravel()
    R_wc = R_cw.T
    C_world = -R_wc @ t_cw
    reproj, _ = cv2.projectPoints(world_pts_3d, rvec, tvec, K, dist)
    err = np.linalg.norm(reproj.reshape(-1, 2) - img_pts, axis=1)
    return R_wc, C_world, err


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--intrinsics", required=True, help="calibration.npz from cam_tests.py")
    ap.add_argument("--camera-index", type=int, default=4)
    ap.add_argument("--resolution", default="640x480", help="WxH; MUST match the intrinsics")
    ap.add_argument("--puck-profile", default=None,
                    help="tuned real-puck colour gate JSON (from mini_platform.hw.tune_puck)")
    ap.add_argument("--points", required=True,
                    help="measured world points 'x,y x,y ...' (metres, table-centre frame)")
    ap.add_argument("--z", type=float, default=PUCK_TOP_Z, help="puck top height above surface (m)")
    ap.add_argument("--frames", type=int, default=15, help="frames to median per point")
    ap.add_argument("--out", default="hw_calib.npz")
    ap.add_argument("--view", action=argparse.BooleanOptionalAction, default=True,
                    help="live annotated camera window while placing the puck (default ON; "
                         "--no-view for headless). Confirms the PUCK -- not a hand, a reflection "
                         "or the mallet -- is what gets detected before each capture.")
    args = ap.parse_args(argv)

    from mini_platform.hw.real_camera import V4L2Camera
    from mini_platform.vision.detector import PuckColorProfile

    w, h = (int(x) for x in args.resolution.lower().split("x"))
    K, dist = load_intrinsics(args.intrinsics)
    world_xy = parse_points(args.points)
    profile = PuckColorProfile.load(args.puck_profile) if args.puck_profile else None
    if profile is None:
        print("WARNING: no --puck-profile; the SIM red-puck gate will likely mis-detect a real "
              "puck. Tune one first (mini_platform.hw.tune_puck).")

    cam = V4L2Camera(index=args.camera_index, resolution=(w, h))
    cv2mod, win = None, None
    if args.view:
        try:
            import cv2 as cv2mod
            win = "calibrate points"
            cv2mod.namedWindow(win, cv2mod.WINDOW_NORMAL)
        except Exception as e:
            print(f"(live view unavailable: {e} -- continuing headless; --no-view to silence)")
            cv2mod = None

    obj, img, used = [], [], []
    try:
        for i, (x, y) in enumerate(world_xy):
            label = f"[{i + 1}/{len(world_xy)}] puck on world ({x:+.3f}, {y:+.3f}) m"
            if cv2mod is None:
                input(f"\n{label}, then press Enter... ")
            else:
                print(f"\n{label}: SPACE/ENTER capture, s skip, q quit (in the camera window)")
                action = wait_in_view(cv2mod, win, cam, profile, label, img)
                if action == "quit":
                    print("  stopping early -- solving with the points captured so far")
                    break
                if action == "skip":
                    print("  skipped")
                    continue
            px = detect_puck_pixel(cam, profile, args.frames)
            if px is None:
                print("  !! puck not detected reliably here -- SKIPPED (re-run if you need it)")
                continue
            print(f"  detected pixel ({px[0]:.1f}, {px[1]:.1f})")
            obj.append([x, y, args.z])
            img.append(px)
            used.append((x, y))
    finally:
        cam.close()
        if cv2mod is not None:
            cv2mod.destroyAllWindows()

    obj = np.array(obj, float)
    img = np.array(img, float)
    if len(obj) < 4:
        raise SystemExit(f"only {len(obj)} points detected; need >= 4 non-collinear. Aborting.")

    R_wc, C_world, err = solve_extrinsic(K, dist, obj, img)
    calib = CameraCalibration(K=K, dist=dist, R_wc=R_wc, C_world=C_world, resolution=(w, h))

    print("\n================ EXTRINSIC (point-based) ================")
    print(f"points used       : {len(obj)}")
    print(f"camera centre (m) : {np.round(C_world, 4)}")
    print(f"height above table: {C_world[2] - args.z:+.4f} m   "
          f"({'OK above' if C_world[2] > args.z else 'BAD: at/below surface'})")
    print(f"reprojection px   : mean={err.mean():.2f}  max={err.max():.2f}  "
          f"({'good' if err.mean() < 2 else 'HIGH -- check point measurements/detections'})")

    # World-xy round-trip: back-project each detected pixel and compare to what you measured.
    # This is the sign check that the board method failed -- x must come back with your sign.
    cam_model = RealPinholeCamera(K, dist, R_wc, C_world, (w, h))
    print("\nworld-xy round-trip (measured -> recovered; signs MUST match):")
    worst = 0.0
    for (x, y), (u, v) in zip(used, img):
        P = cam_model.pixel_to_plane(u, v, z_p=args.z)
        if P is None:
            print(f"  ({x:+.3f},{y:+.3f}) -> MISS (ray not hitting plane) -- bad geometry")
            continue
        d = np.hypot(P[0] - x, P[1] - y)
        worst = max(worst, d)
        print(f"  ({x:+.3f},{y:+.3f}) -> ({P[0]:+.3f},{P[1]:+.3f})   err {d*1000:.0f} mm")
    print(f"worst round-trip error: {worst*1000:.0f} mm")

    if C_world[2] <= args.z or err.mean() > 4:
        print("\nNOT saving: geometry looks unreliable (camera below surface or high reprojection). "
              "Re-measure the points and re-run.")
        return
    calib.save(args.out)
    print(f"\nsaved -> {args.out}\nVerify live next:  python -m mini_platform.hw.run_hw_match "
          f"--stage vision --view --calib {args.out} --camera-index {args.camera_index} "
          f"--puck-profile {args.puck_profile}")


if __name__ == "__main__":
    main()
