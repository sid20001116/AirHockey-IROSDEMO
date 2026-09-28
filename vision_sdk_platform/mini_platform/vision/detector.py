"""Colour segmentation -> subpixel centroids (vision assignment Part 2).

Two interchangeable backends, selected per call by ``detect_puck(..., backend=...)``:

* ``"numpy"`` (the DEFAULT) -- numpy + scipy.ndimage only, no OpenCV. This is the sim's
  reference path: its output is what the ``PuckVisionSensor`` bias/sigma constants were
  fitted against, so it is kept byte-for-byte and every sim test runs through it.
* ``"cv2"`` -- the same algorithm on OpenCV primitives (``cvtColor``/``inRange`` for the
  HSV gate, ``floodFill`` for hole-filling, ``connectedComponentsWithStats`` for the
  largest blob). On a cleanly-segmented 640x480 real frame -- the tuned operating case,
  where the gate passes only the puck -- this is ~1.7 ms vs ~55 ms for the pure-numpy path
  whose ``rgb_to_hsv`` dominates, an ~30x speedup (it narrows toward ~2-3x only on an
  UNtuned profile that leaks thousands of stray blobs, which is exactly what the tuner
  exists to prevent -- and numpy is slow there too). The HARDWARE sensor uses this. It is a
  drop-in equivalent: for the sim ``rgb_ratio`` gate it reproduces the
  numpy ``Detection`` to floating-point precision (floodFill == ``binary_fill_holes`` and
  4-connected components == ``ndimage.label``, both verified exactly); for the ``hsv``
  gate it matches within OpenCV's 2-degree hue quantization -- negligible for a real-puck
  band, and the yaw-marker sub-window is still segmented by the shared numpy gate so that
  half is identical too. ``backend="auto"`` uses cv2 when importable, else numpy.

The puck is found as the largest connected blob of a tunable colour gate; the yaw marker
is searched strictly inside the (hole-filled) puck component.

Two segmentation modes, selected by :class:`PuckColorProfile`:

* ``"rgb_ratio"`` (the SIM default) -- chromaticity RATIOS, never absolute thresholds. The
  sim puck material has specular=0.5, so highlights push pixels toward white (all channels
  rise together), which escapes absolute gates but cleanly fails a ratio test -- producing a
  hole in the mask (refilled below), never a false blob. These are the exact gates the sim
  was tuned with; the module-level defaults reproduce them byte-for-byte.
* ``"hsv"`` (for the REAL puck) -- a hue band plus saturation/value floors. Real lighting
  makes absolute-RGB or RGB-ratio gates leak (skin, wood and the table are all reddish with
  modest r/g, r/b), so on hardware we segment by hue, which is far more selective. Specular
  highlights blow out to low saturation and drop out of the mask -- exactly the same hole,
  refilled by ``binary_fill_holes`` the same way. Tune it with
  ``python -m mini_platform.hw.tune_puck`` against the real camera.

``detect_puck(rgb)`` with no profile uses the sim default, so all sim code and tests are
unchanged. The hardware path passes a profile loaded from the tuner's JSON.
"""

from dataclasses import dataclass, asdict

import numpy as np
from scipy import ndimage

#: Sim red-blob sanity gates: fewer pixels than this is noise, and a bounding box outside
#: this major-dimension band (px, AT 360-ROW RENDERS -- both bounds scale with the
#: render height) is either a speck or something far too close/large to be the puck
#: (the puck spans ~13-30 px across the table at 640x360, more near the rail).
MIN_RED_AREA = 20
BBOX_MAJOR_RANGE = (4, 90)
GATE_REF_HEIGHT = 360

#: Blue marker: evaluated in the red bbox expanded by this margin (a sub-window, for
#: cost only -- the strict component gate below does the actual confinement); below
#: the area floor the marker is considered unresolved and yaw is unavailable.
BBOX_MARGIN = 3
MIN_BLUE_AREA = 3


@dataclass
class PuckColorProfile:
    """Tunable colour gates for the puck blob and its yaw marker.

    The default is the sim ``rgb_ratio`` gate (identical to the historical hard-coded
    behaviour). The hardware tuner produces an ``hsv`` profile. Sanity gates (areas, bbox
    band, margins) are shared by both modes.

    HSV convention (matches :func:`rgb_to_hsv` below): hue in **degrees** ``[0, 360)``,
    saturation/value in ``[0, 1]``. A hue band may wrap through 0 (i.e. ``hue_lo > hue_hi``),
    which is the usual case for a red puck.
    """

    mode: str = "rgb_ratio"                 # "rgb_ratio" (sim) | "hsv" (real puck)

    # -- rgb_ratio puck gate (r > r_min & r > r_over_g*g & r > r_over_b*b) --
    r_min: float = 60.0
    r_over_g: float = 1.6
    r_over_b: float = 1.6

    # -- rgb_ratio marker gate (b > b_min & b > b_over_r*r & b > b_over_g*g) --
    b_min: float = 50.0
    b_over_r: float = 1.5
    b_over_g: float = 1.2

    # -- hsv puck gate --
    hue_lo: float = 345.0                   # wraps through 0 for red
    hue_hi: float = 15.0
    sat_min: float = 0.45
    val_min: float = 0.25

    # -- hsv marker gate (blue band); set marker_enabled False when the real puck has no
    #    physical marker (yaw then degrades to None, which the whole pipeline handles) --
    marker_enabled: bool = True
    marker_hue_lo: float = 200.0
    marker_hue_hi: float = 260.0
    marker_sat_min: float = 0.35
    marker_val_min: float = 0.20

    # -- shared sanity gates --
    min_red_area: int = MIN_RED_AREA
    bbox_major_range: tuple = BBOX_MAJOR_RANGE
    gate_ref_height: int = GATE_REF_HEIGHT
    bbox_margin: int = BBOX_MARGIN
    min_blue_area: int = MIN_BLUE_AREA

    # -- provenance, not a gate --
    #: The camera exposure this profile was FITTED at (UVC exposure_absolute), and the menu
    #: setting at the time (1 = manual, 3 = auto). `sat_min`/`val_min` are fixed thresholds, so a
    #: gate fitted under one exposure mis-segments under another; recording it lets `--stage play`
    #: say so instead of silently under-detecting. None on any profile tuned before this existed.
    tuned_exposure: float | None = None
    tuned_exposure_auto: int | None = None

    def to_dict(self):
        d = asdict(self)
        d["bbox_major_range"] = list(self.bbox_major_range)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        if "bbox_major_range" in d:
            d["bbox_major_range"] = tuple(d["bbox_major_range"])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path):
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        import json
        with open(path) as f:
            return cls.from_dict(json.load(f))


#: The sim gate, used whenever a caller passes no profile. Keeping this a module singleton
#: means ``detect_puck(rgb)`` behaves exactly as before this file gained profiles.
DEFAULT_PROFILE = PuckColorProfile()


@dataclass
class Detection:
    """Red puck blob (and optionally its blue yaw marker) in pixel coordinates."""

    centroid: tuple      # (u, v) subpixel centroid of the filled red blob
    area: int            # filled red blob area, px
    bbox: tuple          # (u_min, v_min, u_max, v_max), inclusive
    marker_centroid: tuple | None  # blue marker (u, v), or None when unresolved


def rgb_to_hsv(im):
    """Vectorised RGB->HSV for an array of 0..255 values (int or float, any shape (...,3)).

    Returns ``(h, s, v)`` with hue in degrees ``[0, 360)`` and s, v in ``[0, 1]``. Pure numpy
    so the detector stays OpenCV-free (the same reason the rest of this module does).
    """
    rgb = np.asarray(im, dtype=np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    diff = mx - mn
    safe = diff > 1e-12
    h = np.zeros_like(mx)
    is_r = safe & (mx == r)
    is_g = safe & (mx == g) & ~is_r
    is_b = safe & (mx == b) & ~is_r & ~is_g
    h[is_r] = (60.0 * (g[is_r] - b[is_r]) / diff[is_r]) % 360.0
    h[is_g] = 60.0 * (b[is_g] - r[is_g]) / diff[is_g] + 120.0
    h[is_b] = 60.0 * (r[is_b] - g[is_b]) / diff[is_b] + 240.0
    s = np.where(mx > 1e-12, diff / np.where(mx > 1e-12, mx, 1.0), 0.0)
    return h, s, mx  # value channel is the per-pixel max


def _hue_band(h, lo, hi):
    """Boolean mask of hues inside [lo, hi], handling a band that wraps through 0."""
    if lo <= hi:
        return (h >= lo) & (h <= hi)
    return (h >= lo) | (h <= hi)


def _puck_mask(im, profile):
    """Filled puck mask from an int16/float image (0..255), per the profile's mode."""
    if profile.mode == "hsv":
        h, s, v = rgb_to_hsv(im)
        m = _hue_band(h, profile.hue_lo, profile.hue_hi) & (s >= profile.sat_min) & (v >= profile.val_min)
    else:
        r, g, b = im[..., 0], im[..., 1], im[..., 2]
        m = (r > profile.r_min) & (r > profile.r_over_g * g) & (r > profile.r_over_b * b)
    return ndimage.binary_fill_holes(m)


def _red_mask(im, profile=None):
    """Filled puck mask (name kept for back-compat: vision_eval + tests call this).

    ``im`` is a (H,W,3) int16/float array of 0..255 values.
    """
    return _puck_mask(im, profile or DEFAULT_PROFILE)


def _marker_mask(sub, profile):
    """Boolean marker mask over a sub-window ``sub`` (int16/float 0..255, (h,w,3))."""
    if not profile.marker_enabled:
        return np.zeros(sub.shape[:2], dtype=bool)
    if profile.mode == "hsv":
        h, s, v = rgb_to_hsv(sub)
        return (_hue_band(h, profile.marker_hue_lo, profile.marker_hue_hi)
                & (s >= profile.marker_sat_min) & (v >= profile.marker_val_min))
    r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
    return (b > profile.b_min) & (b > profile.b_over_r * r) & (b > profile.b_over_g * g)


# --------------------------------------------------------------------------- cv2 fast path
# The three full-frame operations the numpy path spends its time in -- HSV conversion, hole
# filling, connected-component labelling -- have exact (or, for HSV, within-quantization)
# OpenCV equivalents that run ~10x faster on a real 640x480 frame. Everything below mirrors
# the numpy helpers above one-to-one so ``backend="cv2"`` yields the same ``Detection``.

def _fill_holes_cv2(cv2, mask8):
    """cv2 equivalent of ``ndimage.binary_fill_holes`` on a 0/255 uint8 mask.

    Flood the background inward from a guaranteed-exterior seed (a 1 px zero border makes the
    corner seed always background and border-connected, exactly like ndimage treating
    outside-the-image as background); whatever the flood cannot reach is an enclosed hole.
    4-connectivity matches ndimage's default background structure. Verified pixel-identical.
    """
    if not mask8.any():
        return mask8
    padded = cv2.copyMakeBorder(mask8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, ff, (0, 0), 255)           # 255 = fg OR exterior background
    exterior = padded[1:-1, 1:-1]
    holes = cv2.bitwise_not(exterior)                # 255 only at enclosed background
    return cv2.bitwise_or(mask8, holes)


def _puck_mask8_cv2(cv2, rgb, profile):
    """Filled puck mask as 0/255 uint8 from a uint8 RGB frame, per the profile's mode.

    HSV mode uses ``cvtColor``+``inRange`` (OpenCV hue is degrees/2 in [0,180), s/v in
    [0,255]); rgb_ratio keeps the exact numpy ratio test (a chromaticity ratio has no
    ``inRange`` form) -- cheap, and byte-identical to the numpy mask.
    """
    if profile.mode == "hsv":
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        m = _inrange_hsv_cv2(cv2, hsv, profile.hue_lo, profile.hue_hi,
                             profile.sat_min, profile.val_min)
    else:
        im = rgb.astype(np.int16)
        r, g, b = im[..., 0], im[..., 1], im[..., 2]
        m = ((r > profile.r_min) & (r > profile.r_over_g * g)
             & (r > profile.r_over_b * b)).astype(np.uint8) * 255
    return _fill_holes_cv2(cv2, m)


def _inrange_hsv_cv2(cv2, hsv, hue_lo, hue_hi, sat_min, val_min):
    """0/255 uint8 mask of an HSV band, thresholds given in the profile's units (hue deg,
    s/v in [0,1]); handles a band that wraps through 0 with two OR'd ``inRange`` calls."""
    h_lo, h_hi = int(round(hue_lo / 2.0)), int(round(hue_hi / 2.0))
    s_lo = int(round(sat_min * 255.0))
    v_lo = int(round(val_min * 255.0))
    if hue_lo <= hue_hi:
        return cv2.inRange(hsv, (h_lo, s_lo, v_lo), (h_hi, 255, 255))
    lo = cv2.inRange(hsv, (h_lo, s_lo, v_lo), (179, 255, 255))
    hi = cv2.inRange(hsv, (0, s_lo, v_lo), (h_hi, 255, 255))
    return cv2.bitwise_or(lo, hi)


def _detect_puck_cv2(cv2, rgb, profile):
    """cv2 implementation of :func:`detect_puck` (same gates, same sanity checks)."""
    filled = _puck_mask8_cv2(cv2, rgb, profile)
    # 4-connectivity to match ndimage.label's default structure.
    n, lbl, stats, cents = cv2.connectedComponentsWithStats(filled, connectivity=4)
    if n <= 1:                                        # background only
        return None
    largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    area = int(stats[largest, cv2.CC_STAT_AREA])
    if area < profile.min_red_area:
        return None
    x0 = int(stats[largest, cv2.CC_STAT_LEFT])
    y0 = int(stats[largest, cv2.CC_STAT_TOP])
    u0, u1 = x0, x0 + int(stats[largest, cv2.CC_STAT_WIDTH]) - 1
    v0, v1 = y0, y0 + int(stats[largest, cv2.CC_STAT_HEIGHT]) - 1
    h, w = filled.shape
    scale = h / profile.gate_ref_height
    major = max(u1 - u0 + 1, v1 - v0 + 1)
    lo, hi = profile.bbox_major_range
    if not lo * scale <= major <= hi * scale:
        return None

    centroid = (float(cents[largest, 0]), float(cents[largest, 1]))

    # Marker: same confinement as the numpy path -- the (numpy) marker gate over the bbox
    # sub-window, AND'd with the filled puck component. Reusing the shared numpy gate on a
    # tiny sub-window (negligible cost) keeps the marker byte-identical to the numpy path.
    m = profile.bbox_margin
    r0, r1 = max(v0 - m, 0), min(v1 + m, h - 1)
    c0, c1 = max(u0 - m, 0), min(u1 + m, w - 1)
    sub = rgb[r0:r1 + 1, c0:c1 + 1].astype(np.int16)
    blue = _marker_mask(sub, profile) & (lbl[r0:r1 + 1, c0:c1 + 1] == largest)
    marker = None
    if int(blue.sum()) >= profile.min_blue_area:
        bys, bxs = np.nonzero(blue)
        marker = (float(c0 + bxs.mean()), float(r0 + bys.mean()))
    return Detection(centroid, area, (u0, v0, u1, v1), marker)


def _import_cv2(required):
    """Return the cv2 module, or None. ``required`` -> raise an actionable error instead."""
    try:
        import cv2
        return cv2
    except Exception as e:                            # pragma: no cover - env-specific
        if required:
            raise RuntimeError(
                "detect_puck(backend='cv2') needs OpenCV, which failed to import "
                f"({e}). Install it (`pip install opencv-python`) or use backend='numpy'."
            ) from e
        return None


def detect_puck(rgb, profile=None, backend="numpy"):
    """Detect the puck in an (H, W, 3) uint8 frame; ``Detection`` or None.

    ``profile`` selects the colour gate (see :class:`PuckColorProfile`). Omit it for the
    sim's red-puck gate; pass a tuned ``hsv`` profile for the real camera.

    ``backend`` picks the implementation: ``"numpy"`` (default, the sim reference path),
    ``"cv2"`` (the ~10x-faster OpenCV path the hardware sensor uses -- see the module
    docstring), or ``"auto"`` (cv2 when importable, else numpy). All three return the same
    ``Detection`` type; cv2 matches numpy exactly for ``rgb_ratio`` and within hue
    quantization for ``hsv``.
    """
    profile = profile or DEFAULT_PROFILE
    if backend in ("cv2", "auto"):
        cv2 = _import_cv2(required=backend == "cv2")
        if cv2 is not None:
            return _detect_puck_cv2(cv2, rgb, profile)
    elif backend != "numpy":
        raise ValueError(f"unknown detect backend {backend!r} "
                         "(expected 'numpy', 'cv2', or 'auto')")
    im = rgb.astype(np.int16)
    red = _puck_mask(im, profile)
    lbl, n = ndimage.label(red)
    if n == 0:
        return None
    sizes = np.bincount(lbl.ravel())[1:]
    largest = int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(lbl == largest)
    area = int(xs.size)
    if area < profile.min_red_area:
        return None
    u0, u1 = int(xs.min()), int(xs.max())
    v0, v1 = int(ys.min()), int(ys.max())
    h, w = im.shape[:2]
    scale = h / profile.gate_ref_height
    major = max(u1 - u0 + 1, v1 - v0 + 1)
    lo, hi = profile.bbox_major_range
    if not lo * scale <= major <= hi * scale:
        return None

    centroid = (float(xs.mean()), float(ys.mean()))

    # The marker is gated STRICTLY to the (hole-filled) puck component: it is physically on
    # the puck, so its pixels are interior whenever its outline closes. This keeps blue-ish
    # background (the sim's blue-grey floor checker) from leaking into the marker centroid
    # when the puck sits near a rim; if the marker aliases open at the silhouette it falls
    # outside the component and yaw degrades honestly to None instead of using bad pixels.
    m = profile.bbox_margin
    r0, r1 = max(v0 - m, 0), min(v1 + m, h - 1)
    c0, c1 = max(u0 - m, 0), min(u1 + m, w - 1)
    sub = im[r0:r1 + 1, c0:c1 + 1]
    blue = _marker_mask(sub, profile)
    blue &= (lbl == largest)[r0:r1 + 1, c0:c1 + 1]

    marker = None
    if int(blue.sum()) >= profile.min_blue_area:
        bys, bxs = np.nonzero(blue)
        marker = (float(c0 + bxs.mean()), float(r0 + bys.mean()))
    return Detection(centroid, area, (u0, v0, u1, v1), marker)


def puck_mask(rgb, profile=None, backend="numpy"):
    """The filled boolean puck mask ``detect_puck`` segments, for previews/overlays.

    Takes a uint8 (H, W, 3) RGB frame. ``backend`` matches :func:`detect_puck` so a tuner
    or viewer can show exactly what the runtime cv2 path will segment. Returns a bool array.
    """
    profile = profile or DEFAULT_PROFILE
    if backend in ("cv2", "auto"):
        cv2 = _import_cv2(required=backend == "cv2")
        if cv2 is not None:
            return _puck_mask8_cv2(cv2, rgb, profile) > 0
    elif backend != "numpy":
        raise ValueError(f"unknown detect backend {backend!r} "
                         "(expected 'numpy', 'cv2', or 'auto')")
    return _puck_mask(rgb.astype(np.int16), profile)
