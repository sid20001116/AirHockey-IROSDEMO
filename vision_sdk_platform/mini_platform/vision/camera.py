"""Pinhole model of a named MuJoCo camera (vision assignment Part 1).

MuJoCo conventions: `fovy` is vertical and pixels are square, so the focal length comes
from the RENDER height -- never from the XML's declared `resolution`, which is the
hardware spec. The camera looks along -Z of its frame with +Y as image-up, and image v
grows downward.
"""

import mujoco
import numpy as np

# The puck's top face (Part 0: 50 x 7 mm cylinder on the z=0 playing surface).
# Detections are back-projected onto this exactly-known plane, which removes the
# parallax hazard (handoff H1) by construction.
PUCK_TOP_Z = 0.007


class PinholeCamera:
    """Intrinsics fixed from the render size; extrinsics read live from `data` on every
    call, so the same object serves a fixed camera and (later) a moving wrist camera."""

    def __init__(self, model, data, camera, resolution):
        self._data = data
        self._cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if self._cid < 0:
            raise ValueError(f"no camera named {camera!r}")
        w, h = resolution
        self.width, self.height = w, h
        self.f = (h / 2) / np.tan(np.radians(model.cam_fovy[self._cid]) / 2)
        self.cx, self.cy = (w - 1) / 2, (h - 1) / 2

    @property
    def pos(self):
        """Camera center, world frame."""
        return self._data.cam_xpos[self._cid]

    @property
    def rot(self):
        """World-from-camera rotation; columns: image-right, image-up, backward."""
        return self._data.cam_xmat[self._cid].reshape(3, 3)

    def pixel_to_ray(self, u, v):
        """Unit world-frame direction through pixel (u, v)."""
        d = self.rot @ np.array([(u - self.cx) / self.f, -(v - self.cy) / self.f, -1.0])
        return d / np.linalg.norm(d)

    def intersect_z_plane(self, d, z_p=PUCK_TOP_Z):
        """World point where ray `d` from the camera center meets the plane z = z_p, or
        None when the ray points at/above the horizon or the plane is behind the camera."""
        p = self.pos
        if d[2] >= -1e-9:
            return None
        t = (z_p - p[2]) / d[2]
        if t < 0:
            return None
        return p + t * d

    def pixel_to_plane(self, u, v, z_p=PUCK_TOP_Z):
        return self.intersect_z_plane(self.pixel_to_ray(u, v), z_p)

    def project(self, X):
        """World point -> pixel (u, v), or None when behind the camera."""
        Xc = self.rot.T @ (np.asarray(X, dtype=float) - self.pos)
        if Xc[2] >= 0:
            return None
        return self.cx + self.f * Xc[0] / -Xc[2], self.cy - self.f * Xc[1] / -Xc[2]

    def fd_jacobian(self, u, v, z_p=PUCK_TOP_Z, eps=0.5):
        """2x2 d(X_xy)/d(u, v) on the z_p plane, central differences at +-eps px, or None
        when any probe ray misses the plane."""
        pts = [self.pixel_to_plane(uu, vv, z_p)
               for uu, vv in ((u + eps, v), (u - eps, v), (u, v + eps), (u, v - eps))]
        if any(q is None for q in pts):
            return None
        return np.column_stack([(pts[0][:2] - pts[1][:2]) / (2 * eps),
                                (pts[2][:2] - pts[3][:2]) / (2 * eps)])
