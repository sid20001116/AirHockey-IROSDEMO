"""Interactive colour tuner for the REAL puck (hardware camera).

The detector's default gate was tuned for the sim's dark-red puck; on the real table it
leaks (skin, wood, the table surface are all reddish). This tool captures the real camera,
lets you click the puck to sample its colour, and live-previews the detector's *own* output
so what you see is exactly what ``--stage vision``/``play`` will detect. It writes a
:class:`~mini_platform.vision.detector.PuckColorProfile` JSON that those stages load via
``--puck-profile``.

Run (cp310 env, camera on /dev/video4 -> index 4)::

    python -m mini_platform.hw.tune_puck --camera-index 4 --resolution 640x480 \
        --cam-auto-exposure 1 --cam-exposure 50 --cam-auto-priority 0 --out puck_profile.json

**Tune at the exposure you PLAY at.** ``sat_min``/``val_min`` are fixed thresholds, so a profile
fitted under one exposure mis-segments under another -- and V4L2 exposure settings persist on the
device between processes, so without the ``--cam-*`` flags this tool inherits whatever the last
program left set (auto, after a replug). The saved profile records the exposure it was fitted at,
and ``--stage play`` warns when it is asked to run that profile at a different one.

Window controls:
    LEFT-click  on the puck        -> add a positive colour sample
    RIGHT-click on hand/table/etc  -> add a negative sample (what to reject)
    a  auto-fit the hue/sat/val sliders from the samples
    s  save the profile to --out
    r  reset all samples
    c  save the current frame to a PNG (for records / offline tuning)
    q / ESC  quit

The sliders are the ground truth you save; auto-fit just moves them. Left panel = the live
frame with the detector's bbox/centroid overlay; right panel = the binary mask the detector
actually segments. Tune until the mask is the puck and nothing else.
"""

import argparse

import numpy as np

from mini_platform.vision.detector import (
    PuckColorProfile, detect_puck, puck_mask, rgb_to_hsv,
)

WIN = "tune puck  (L-click=puck  R-click=bg  a=autofit  s=save  r=reset  q=quit)"


def _profile_from_bars(cv2, marker_enabled):
    """Build an HSV PuckColorProfile from the current trackbar positions."""
    g = lambda n: cv2.getTrackbarPos(n, WIN)
    return PuckColorProfile(
        mode="hsv",
        hue_lo=float(g("hue_lo")), hue_hi=float(g("hue_hi")),
        sat_min=g("sat_min%") / 100.0, val_min=g("val_min%") / 100.0,
        min_red_area=int(g("min_area")),
        # Keep a wide bbox band while tuning (real resolution differs from the sim's 360);
        # the sanity band is not what we're tuning here.
        bbox_major_range=(3, 400),
        marker_enabled=marker_enabled,
    )


def _set_bars(cv2, p):
    cv2.setTrackbarPos("hue_lo", WIN, int(round(p.hue_lo)) % 360)
    cv2.setTrackbarPos("hue_hi", WIN, int(round(p.hue_hi)) % 360)
    cv2.setTrackbarPos("sat_min%", WIN, int(round(p.sat_min * 100)))
    cv2.setTrackbarPos("val_min%", WIN, int(round(p.val_min * 100)))
    cv2.setTrackbarPos("min_area", WIN, int(p.min_red_area))


def _sample_patch(frame_rgb, x, y, radius=3):
    """Return the (N,3) RGB pixels in a small square around (x, y)."""
    h, w = frame_rgb.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    return frame_rgb[y0:y1, x0:x1].reshape(-1, 3)


def _circ_stats(hues_deg):
    """Circular mean (deg) and per-sample signed offsets in (-180, 180]."""
    a = np.radians(hues_deg)
    mean = np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360.0
    off = (hues_deg - mean + 180.0) % 360.0 - 180.0
    return mean, off


def _autofit(pos_rgb, neg_rgb, pad_deg=6.0, sv_pad=0.06):
    """Fit an HSV profile that covers the positive samples; report negative leakage."""
    if len(pos_rgb) == 0:
        print("  no positive samples yet -- left-click the puck first.")
        return None
    h, s, v = rgb_to_hsv(np.asarray(pos_rgb, dtype=np.float32))
    mean, off = _circ_stats(h)
    lo_off = np.percentile(off, 2) - pad_deg
    hi_off = np.percentile(off, 98) + pad_deg
    # Clamp the band width so a wrapped band stays unambiguous (< ~340 deg total).
    hi_off = min(hi_off, lo_off + 340.0)
    hue_lo = (mean + lo_off) % 360.0
    hue_hi = (mean + hi_off) % 360.0
    sat_min = float(np.clip(np.percentile(s, 3) - sv_pad, 0.0, 1.0))
    val_min = float(np.clip(np.percentile(v, 3) - sv_pad, 0.0, 1.0))
    p = PuckColorProfile(mode="hsv", hue_lo=hue_lo, hue_hi=hue_hi,
                         sat_min=sat_min, val_min=val_min)

    print(f"  autofit: hue [{hue_lo:.0f},{hue_hi:.0f}] deg  sat>={sat_min:.2f}  val>={val_min:.2f}")
    print(f"           (puck hue mean {mean:.0f} deg, {len(pos_rgb)} px sampled)")
    if len(neg_rgb):
        # How many sampled background pixels would still pass this gate?
        hn, sn, vn = rgb_to_hsv(np.asarray(neg_rgb, dtype=np.float32))
        from mini_platform.vision.detector import _hue_band
        leak = (_hue_band(hn, hue_lo, hue_hi) & (sn >= sat_min) & (vn >= val_min)).mean()
        print(f"           background leakage: {leak * 100:.1f}% of {len(neg_rgb)} bg px pass "
              f"({'good' if leak < 0.02 else 'tighten sat/val or narrow hue'})")
    return p


def _mask_panel(cv2, mask):
    """3-channel uint8 view of the boolean detector mask."""
    m = (mask.astype(np.uint8) * 255)
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


def _overlay(cv2, frame_rgb, det, n_pos, n_neg):
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if det is not None:
        u0, v0, u1, v1 = det.bbox
        cv2.rectangle(bgr, (int(u0), int(v0)), (int(u1), int(v1)), (0, 255, 0), 1)
        cu, cv = int(round(det.centroid[0])), int(round(det.centroid[1]))
        cv2.drawMarker(bgr, (cu, cv), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
        if det.marker_centroid is not None:
            mu, mv = int(round(det.marker_centroid[0])), int(round(det.marker_centroid[1]))
            cv2.drawMarker(bgr, (mu, mv), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
        txt = f"puck: {det.area} px"
        col = (0, 255, 0)
    else:
        txt = "no detection"
        col = (0, 0, 255)
    cv2.putText(bgr, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    cv2.putText(bgr, f"samples  +{n_pos}px  -{n_neg}px", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return bgr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera-index", type=int, default=4)
    ap.add_argument("--resolution", default="640x480", help="WxH; match your calibration")
    ap.add_argument("--out", default="puck_profile.json")
    # The camera controls, with the same names and semantics as run_hw_match. A colour profile is
    # only valid at the EXPOSURE it was tuned at -- sat_min/val_min are fixed thresholds, so a
    # frame half as bright moves the gate out from under them. Without these the tuner ran at
    # whatever the device happened to hold (V4L2 exposure settings persist across processes), so
    # matching calibration to runtime was luck.
    ap.add_argument("--cam-fps", type=float, default=None,
                    help="frame rate to request from the camera (as --stage play)")
    ap.add_argument("--cam-exposure", type=int, default=None,
                    help="manual exposure, UVC exposure_absolute in 100 us units. USE THE VALUE "
                         "YOU PLAY AT (this rig: 50). Needs --cam-auto-exposure 1.")
    ap.add_argument("--cam-auto-exposure", type=int, default=None,
                    help="V4L2 exposure menu: 1 = manual, 3 = auto. Must be 1 or --cam-exposure "
                         "is silently ignored.")
    ap.add_argument("--cam-auto-priority", type=int, default=None,
                    help="0 = hold the frame rate, 1 = let the driver drop frames to buy "
                         "exposure. Pass 0.")
    ap.add_argument("--load", default=None, help="start the sliders from an existing profile")
    ap.add_argument("--snapshot", default=None,
                    help="tune on a saved image (BGR PNG/JPG) instead of the live camera")
    ap.add_argument("--marker", action="store_true",
                    help="also detect a blue yaw marker (default off: most real pucks have none)")
    args = ap.parse_args(argv)

    try:
        import cv2
    except Exception as e:
        raise SystemExit(f"OpenCV required for the tuner: {e}")

    w, h = (int(x) for x in args.resolution.lower().split("x"))

    # Frame source: live camera or a static snapshot.
    cam = None
    snap = None
    # Set only on the live-camera path; a --snapshot session has no device to read back, so the
    # saved profile carries no exposure stamp rather than a made-up one. Initialised here because
    # the save handler reads it either way.
    cam_props = None
    if args.snapshot:
        bgr = cv2.imread(args.snapshot)
        if bgr is None:
            raise SystemExit(f"could not read snapshot {args.snapshot}")
        snap = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        from mini_platform.hw.real_camera import V4L2Camera
        cam = V4L2Camera(index=args.camera_index, resolution=(w, h), fps=args.cam_fps,
                         exposure=args.cam_exposure, auto_exposure=args.cam_auto_exposure,
                         auto_priority=args.cam_auto_priority)
        # Read BACK, not what was asked for: a refused write is the failure worth catching (on
        # this rig --cam-auto-exposure 0.25 was rejected, the menu stayed on AUTO, and every
        # --cam-exposure after it was ignored).
        cam_props = dict(cam.props)
        print(f"camera {args.camera_index} ({cam.device}): "
              f"{cam_props.get('width')}x{cam_props.get('height')} {cam_props.get('fourcc')}  "
              f"exposure_auto={cam_props.get('exposure_auto')} "
              f"exposure_absolute={cam_props.get('exposure_absolute')} "
              f"auto_priority={cam_props.get('exposure_auto_priority')}")
        if cam_props.get("exposure_auto") == 3:
            print("  WARNING: exposure is on AUTO, so it drifts with the room light while you "
                  "tune.\n  The profile you save will be valid only for whatever brightness "
                  "happened to be\n  in frame -- sat_min/val_min are FIXED thresholds. Pin it to "
                  "the value you play at:\n  --cam-auto-exposure 1 --cam-exposure 50 "
                  "--cam-auto-priority 0")

    # Sample buffers (RGB pixels) and the latest frame the mouse callback samples from.
    pos, neg = [], []
    state = {"frame": None}

    def on_mouse(event, x, y, flags, _):
        if state["frame"] is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            px = _sample_patch(state["frame"], x, y)
            pos.extend(px.tolist())
            hh, ss, vv = rgb_to_hsv(px.astype(np.float32))
            print(f"  +puck @ ({x},{y})  hue~{np.median(hh):.0f} sat~{np.median(ss):.2f} "
                  f"val~{np.median(vv):.2f}   (+{len(pos)}px total)")
        elif event == cv2.EVENT_RBUTTONDOWN:
            neg.extend(_sample_patch(state["frame"], x, y).tolist())
            print(f"  -bg   @ ({x},{y})   (-{len(neg)}px total)")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    _noop = lambda *_: None
    cv2.createTrackbar("hue_lo", WIN, 345, 359, _noop)
    cv2.createTrackbar("hue_hi", WIN, 15, 359, _noop)
    cv2.createTrackbar("sat_min%", WIN, 45, 100, _noop)
    cv2.createTrackbar("val_min%", WIN, 25, 100, _noop)
    cv2.createTrackbar("min_area", WIN, 20, 500, _noop)
    cv2.setMouseCallback(WIN, on_mouse)

    if args.load:
        _set_bars(cv2, PuckColorProfile.load(args.load))
        print(f"loaded sliders from {args.load}")

    print(__doc__.split("Window controls:")[0].strip())
    print("Left-click the puck (a few spots), press 'a' to auto-fit, then 's' to save.")

    saved_frames = 0
    try:
        while True:
            if snap is not None:
                frame = snap.copy()
            else:
                frame = cam.read_rgb()
                if frame is None:
                    if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                        break
                    continue
            state["frame"] = frame

            profile = _profile_from_bars(cv2, marker_enabled=args.marker)
            # Segment through the SAME cv2 backend --stage vision/play use, so the mask panel
            # and bbox you tune against are exactly what the runtime detector will produce.
            mask = puck_mask(frame, profile, backend="cv2")
            det = detect_puck(frame, profile, backend="cv2")

            left = _overlay(cv2, frame, det, len(pos), len(neg))
            right = _mask_panel(cv2, mask)
            cv2.imshow(WIN, np.hstack([left, right]))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("a"):
                p = _autofit(pos, neg)
                if p is not None:
                    _set_bars(cv2, p)
            elif key == ord("r"):
                pos.clear(); neg.clear()
                print("  samples reset")
            elif key == ord("s"):
                out = _profile_from_bars(cv2, marker_enabled=args.marker)
                # Restore a real bbox band for runtime (we widened it while tuning).
                out.bbox_major_range = (3, 400)
                # Stamp the conditions this gate was fitted under, so `--stage play` can say so
                # when it is about to run the profile at a different exposure. Nothing reads
                # these as gates; they are provenance.
                out.tuned_exposure = cam_props.get("exposure_absolute") if cam_props else None
                out.tuned_exposure_auto = cam_props.get("exposure_auto") if cam_props else None
                out.save(args.out)
                print(f"  saved -> {args.out}  "
                      f"(hue [{out.hue_lo:.0f},{out.hue_hi:.0f}] sat>={out.sat_min:.2f} "
                      f"val>={out.val_min:.2f} min_area={out.min_red_area}"
                      f"{'' if out.tuned_exposure is None else f', exposure {out.tuned_exposure}'})")
            elif key == ord("c"):
                path = f"puck_frame_{saved_frames:03d}.png"
                cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                saved_frames += 1
                print(f"  wrote {path}")
    finally:
        if cam is not None:
            cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
