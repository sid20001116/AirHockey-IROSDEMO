"""One-shot sanity check for a camera extrinsic (``hw_calib.npz``) -- no camera needed.

If the puck is *detected* on-screen but the EKF estimate never moves, the usual cause is a
bad extrinsic: the back-projection ``pixel_to_plane`` returns None for every pixel (the ray
does not point down at the table plane), so ``RealPuckSensor.measure()`` returns None every
frame and the tracker stays frozen at its seed. This tool loads the calibration and reports
whether the geometry is physically sane -- camera above the surface, rays pointing down, the
image centre landing on the table within a sane extent.

    python -m mini_platform.hw.check_calib --calib hw_calib.npz
"""

import argparse

import numpy as np

from mini_platform.hw.real_camera import CameraCalibration, RealPinholeCamera, PUCK_TOP_Z


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", default="hw_calib.npz")
    ap.add_argument("--extent", type=float, default=0.6,
                    help="half-size (m) of a sane table region around the world origin")
    args = ap.parse_args(argv)

    calib = CameraCalibration.load(args.calib)
    cam = RealPinholeCamera(calib.K, calib.dist, calib.R_wc, calib.C_world, calib.resolution)
    w, h = calib.resolution
    C = cam.pos

    print("================ EXTRINSIC SANITY ================")
    print(f"resolution        : {w}x{h}")
    print(f"camera centre (m) : {np.round(C, 4)}")
    print(f"height above table: {C[2] - PUCK_TOP_Z:+.4f} m   "
          f"({'OK, camera is above' if C[2] > PUCK_TOP_Z else 'BAD: camera at/below the surface'})")

    # Back-project the image centre and the four corners onto the puck-top plane.
    pix = {"centre": (w / 2, h / 2), "TL": (0, 0), "TR": (w - 1, 0),
           "BL": (0, h - 1), "BR": (w - 1, h - 1)}
    print("\npixel            ray_z     plane hit (world x,y)      status")
    n_down = 0
    hits = []
    for name, (u, v) in pix.items():
        d = cam.pixel_to_ray(u, v)
        P = cam.intersect_z_plane(d)
        if d[2] < 0:
            n_down += 1
        if P is None:
            status = "MISS (ray not downward)"
            xy = "        --        "
        else:
            xy = f"({P[0]:+.3f}, {P[1]:+.3f})"
            hits.append(P[:2])
            status = "ok" if max(abs(P[0]), abs(P[1])) <= args.extent else "hit far from origin"
        print(f"{name:6s} {str((int(u), int(v))):10s} {d[2]:+.2f}   {xy}   {status}")

    print("\n---------------- VERDICT ----------------")
    if n_down == 0:
        print("BAD EXTRINSIC: every ray points UP (away from the table). R_wc is likely "
              "inverted/rotated. measure() will return None every frame -> EKF frozen.\n"
              "Re-run calibrate_extrinsics: board flat on the table at a KNOWN world pose, "
              "same resolution as intrinsics, and check the board-yaw/axis convention.")
    elif n_down < len(pix):
        print("PARTIAL: some rays miss the plane. The camera may be too oblique or the "
              "extrinsic is slightly off; the puck is only measurable where rays hit.")
    elif hits and max(np.max(np.abs(np.array(hits)), axis=0)) > 5 * args.extent:
        print("SUSPECT SCALE/POSE: rays hit the plane but far from the origin. The board "
              "world pose (--board-x/y/z/yaw) or the metric scale (square length) is likely "
              "wrong. World coords will be off, so velocities/approach direction will be too.")
    else:
        print("Geometry looks sane: camera above the table, rays hit the plane near the "
              "origin. If the estimate still freezes, the issue is downstream (marker/jacobian "
              "gate in measure(), or the frame ROTATION -- check vel direction in --view).")


if __name__ == "__main__":
    main()
