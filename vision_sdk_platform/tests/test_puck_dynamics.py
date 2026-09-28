"""Pins the 50 x 7 mm puck's retuned damping (vision assignment Part 0).

The puck's joint damping models the air film, so the invariant is the deceleration
coefficient damping/mass (0.5 1/s linear, 0.4 1/s yaw), not the damping force -- see the
comment above the puck joints in assets/mini_table.xml. The scene is loaded raw (no env,
no controller, no rendering). The puck<->table contact pair is excluded in the XML, so on
a contact-free lane the slide/hinge damping is the only decelerating force and |v| decays
exponentially; the fitted slope of ln|v| is the deceleration coefficient.

Run with::

    cd mini_platform && python -m pytest tests/test_puck_dynamics.py -q
"""

import os

import mujoco
import numpy as np
import pytest

import mini_platform

# Start lane: the mallet free body defaults to the table origin in a raw load (the env
# normally seats it in the gripper), and arm 1 collapses uncontrolled near the -y rail.
# From (-0.45, +0.15) heading +x the puck passes 0.15 m from the mallet and, over 1 s at
# 0.5 1/s, travels 0.79 m of the 0.93 m runway to the away rim -- contact-free throughout,
# which the per-step assertion below verifies rather than assumes.
START_XY = (-0.45, 0.15)
DURATION = 1.0


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(
        os.path.join(mini_platform.ASSET_DIR, "mini_single.xml"))


def _fit_decel_slope(model, joint, v0):
    """Fitted slope of ln|qvel| for one puck dof over DURATION of raw stepping."""
    d = mujoco.MjData(model)
    d.qpos[model.joint("puck_x").qposadr[0]] = START_XY[0]
    d.qpos[model.joint("puck_y").qposadr[0]] = START_XY[1]
    dof = model.joint(joint).dofadr[0]
    d.qvel[dof] = v0
    mujoco.mj_forward(model, d)

    puck_gid = model.geom("puck").id
    n_steps = int(round(DURATION / model.opt.timestep))
    t = np.empty(n_steps)
    v = np.empty(n_steps)
    for i in range(n_steps):
        mujoco.mj_step(model, d)
        for c in range(d.ncon):
            assert puck_gid not in (d.contact[c].geom1, d.contact[c].geom2), \
                f"puck contact at t={d.time:.3f} s invalidates the damping fit"
        t[i] = d.time
        v[i] = abs(d.qvel[dof])
    assert v.min() > 0
    return np.polyfit(t, np.log(v), 1)[0]


# ------- constants: catches XML regressions with zero stepping

def test_constants_pinned(model):
    assert model.geom("puck").size[0] == pytest.approx(0.025)

    puck_bid = model.body("puck").id
    assert model.body_mass[puck_bid] == pytest.approx(0.0021838)
    # the compiler may reorder principal axes, so compare the sorted spectrum
    assert np.allclose(np.sort(model.body_inertia[puck_bid]),
                       [3.5014e-07, 3.5014e-07, 6.8244e-07], rtol=1e-4)

    damping = [model.dof_damping[model.joint(j).dofadr[0]]
               for j in ("puck_x", "puck_y", "puck_yaw")]
    assert np.allclose(damping, [1.0919e-3, 1.0919e-3, 2.7298e-7], rtol=1e-6)


# ------- deceleration coefficients: damping/mass = 0.5 1/s, damping/Izz = 0.4 1/s

def test_linear_damping_slope(model):
    slope = _fit_decel_slope(model, "puck_x", v0=1.0)
    assert -0.5 * 1.05 <= slope <= -0.5 * 0.95, f"linear decel {-slope:.4f} 1/s, want 0.5"


def test_yaw_damping_slope(model):
    slope = _fit_decel_slope(model, "puck_yaw", v0=10.0)
    assert -0.4 * 1.05 <= slope <= -0.4 * 0.95, f"yaw decel {-slope:.4f} 1/s, want 0.4"
