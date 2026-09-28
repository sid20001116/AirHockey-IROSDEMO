"""Position-controlled variants of the mini environments.

The `PositionControl` mixin itself is vendored in `mini_platform/core/position_control.py`;
only the metal arm's gains and the task compositions live here.
"""

import numpy as np

from mini_platform.core import sdk_params
from mini_platform.core.position_control import PositionControl
from mini_platform.core.sdk_control import SdkJointControl
from mini_platform.core import inverse_kinematics

from mini_platform.envs import env_base
from mini_platform.envs.env_defend import AirHockeyDefend
from mini_platform.envs.env_double import AirHockeyDouble
from mini_platform.envs.env_hitting import AirHockeyHit
from mini_platform.envs.env_prepare import AirHockeyPrepare

# The mini table's scale against the full-size one, for the quantities that genuinely scale
# with it (speeds, amplitudes). Positions along x do not -- see agents/baseline_mini.py.
TABLE_SCALE = 0.505226


def _lissajous_opponent_gen(env):
    """A lissajous wander around the opponent's home pose.

    This is NOT an agent -- it never tries to play the puck, exactly as in the organizers'
    `IiwaPositionHit`. The hit task scores you on scoring, not on beating anyone. Pass
    `opponent_agent=` to put a real agent on the other side.

    The amplitudes are scaled to the mini table, so the opponent traces roughly a
    5 cm x 16 cm box -- small enough that it reads as "parked and twitching" in the viewer.

    Module-level so both `MetalPositionHit` and `MetalSdkHit` share one copy. It only touches
    `env.dt`, `env.init_state` and `env.env_info`, so lifting it out of the class changed
    nothing about its behaviour.
    """
    vel = 3
    t = np.pi / 2
    cart_offset = np.array([env_base.HOME_POS[0], 0.])
    amplitude = np.array([0.1, 0.16]) * TABLE_SCALE
    prev_joint_pos = env.init_state

    while True:
        t += vel * env.dt
        cart_pos = amplitude * np.array([np.sin(t) * np.cos(t), np.cos(t)]) + cart_offset

        success, joint_pos = inverse_kinematics(
            env.env_info['robot']['robot_model'], env.env_info['robot']['robot_data'],
            np.concatenate([cart_pos, [env.env_info['robot']['universal_height']]]),
            initial_q=prev_joint_pos)
        assert success

        joint_vel = (joint_pos - prev_joint_pos) / env.dt

        prev_joint_pos = joint_pos

        yield np.vstack([joint_pos, joint_vel])


def _install_opponent(env, opponent_agent):
    """Wire `env._opponent_agent` and return the policy object (None for the scripted wander).

    `AirHockeyHit._preprocess_action` hands the opponent `np.split(obs, 2)[1]`, which
    `env_double._modify_observation` has already expressed in robot 2's own base frame. So a
    `MiniBaselineAgent(env_info, agent_id=2)` is a drop-in second player: it sees exactly the
    observation it would see if it were agent 1 on the other side.
    """
    if isinstance(opponent_agent, str):
        if opponent_agent != "baseline":
            raise ValueError(f"unknown opponent_agent {opponent_agent!r}; expected 'baseline', "
                             "None, or an object with .draw_action")
        # Lazy: agents/ imports envs/, so a module-level import here would be circular.
        from mini_platform.agents.baseline_mini import MiniBaselineAgent
        policy = MiniBaselineAgent(env.env_info, agent_id=2)
        env._opponent_agent = policy.draw_action
        return policy

    if opponent_agent is not None:
        env._opponent_agent = opponent_agent.draw_action
        return opponent_agent

    env._opponent_agent_gen = _lissajous_opponent_gen(env)
    env._opponent_agent = lambda obs: next(env._opponent_agent_gen)
    return None


def _reset_opponent(env):
    """Per-episode reset: a real agent needs `episode_start()`, the wander a fresh generator."""
    if getattr(env, "_opponent_policy", None) is not None:
        env._opponent_policy.episode_start()
    else:
        env._opponent_agent_gen = _lissajous_opponent_gen(env)


def _opponent_interp(opponent_agent, opponent_interp_order):
    """A real agent emits a proper (position, velocity) trajectory, so it wants the same cubic
    interpolation agent 1 gets. Order -1 (independent linear interp of pos and of vel) is only
    right for the scripted wander, whose "velocity" is a finite difference of its own path."""
    if isinstance(opponent_agent, str) and opponent_interp_order == -1:
        return 3
    return opponent_interp_order


class PositionControlMetal(PositionControl):
    """Position controller for the mini platform's 6-DOF metal arm.

    The gains are NOT `PositionControlIIWA`'s. Those are sized for a KUKA whose joints take
    320 N.m; this arm's actuators are 10-100 N.m, so a p_gain of 1500 would ask for a torque
    two orders of magnitude past saturation on joints 4-6 and the controller would spend the
    whole episode clipped. p_gain starts from the <position> servo kp values the MJCF used to
    carry (150/400/200/80/60/40) and d_gain is ~kp/16, i.e. near-critical damping at each
    joint's effective inertia.

    A 4x p / 2x d variant was measured over 40 episodes per task: it improves hit
    (0.38 -> 0.55) and costs prepare (0.82 -> 0.55), for the same total, and fixes nothing
    about the constraint violations. The less-tuned, better-justified setting is kept.
    """

    def __init__(self, *args, **kwargs):
        p_gain = [150., 400., 200., 80., 60., 40.]
        d_gain = [10., 20., 12., 5., 3., 2.]
        i_gain = [0., 0., 0., 0., 0., 0.]

        super(PositionControlMetal, self).__init__(p_gain=p_gain, d_gain=d_gain, i_gain=i_gain,
                                                   *args, **kwargs)


class MetalPositionHit(PositionControlMetal, AirHockeyHit):
    def __init__(self, interpolation_order, opponent_agent=None, opponent_interp_order=-1, *args, **kwargs):
        opponent_interp_order = _opponent_interp(opponent_agent, opponent_interp_order)
        # A string is resolved to a real agent below, once env_info exists; the base class
        # would try to call .draw_action on it.
        super().__init__(opponent_agent=None if isinstance(opponent_agent, str) else opponent_agent,
                         interpolation_order=(interpolation_order, opponent_interp_order), *args, **kwargs)

        self._opponent_policy = _install_opponent(self, opponent_agent)

    def setup(self, obs):
        super().setup(obs)
        _reset_opponent(self)

    def _default_opponent_action_gen(self):
        return _lissajous_opponent_gen(self)


class MetalPositionDefend(PositionControlMetal, AirHockeyDefend):
    pass


class MetalPositionPrepare(PositionControlMetal, AirHockeyPrepare):
    pass


class MetalPositionDouble(PositionControlMetal, AirHockeyDouble):
    """Both arms agent-controlled. Not registered as a task -- there is no `mini-tournament`
    yet, because the tournament env needs the scoring/faults logic that has not been ported."""
    pass


class SdkPositionControlMetal(SdkJointControl):
    """The METAL SDK's own low-level controller driving the mini platform's arm.

    Unlike `PositionControlMetal` there is nothing tuned here. Every gain, feedforward
    coefficient and limit is transcribed from `metal-python-ros/metal_sdk/native`; see
    `core/sdk_params.py` for the citations.

        kp = [300, 500, 500, 400, 25, 20]   against PositionControlMetal's
             [150, 400, 200,  80, 60, 40]

    -- 2-5x stiffer on J1-J4 and 2-2.4x SOFTER on J5/J6, with 2-4x less kd everywhere. That is
    only sane because it arrives with the SDK's feedforward, which removes gravity, Coriolis
    and viscous drag before the PD ever sees them.
    """

    def __init__(self, sdk_config=None, gain_set="nrt", **kwargs):
        if sdk_config is None:
            sdk_config = {"nrt": sdk_params.nrt_config,
                          "follow": sdk_params.follow_config}[gain_set]()
        super().__init__(sdk_config=sdk_config, **kwargs)


class MetalSdkHit(SdkPositionControlMetal, AirHockeyHit):
    def __init__(self, interpolation_order, opponent_agent=None, opponent_interp_order=-1,
                 **kwargs):
        opponent_interp_order = _opponent_interp(opponent_agent, opponent_interp_order)
        super().__init__(opponent_agent=None if isinstance(opponent_agent, str) else opponent_agent,
                         interpolation_order=(interpolation_order, opponent_interp_order),
                         **kwargs)

        self._opponent_policy = _install_opponent(self, opponent_agent)

    def setup(self, obs):
        super().setup(obs)
        _reset_opponent(self)


class MetalSdkDefend(SdkPositionControlMetal, AirHockeyDefend):
    pass


class MetalSdkPrepare(SdkPositionControlMetal, AirHockeyPrepare):
    pass


class MetalSdkDouble(SdkPositionControlMetal, AirHockeyDouble):
    """SDK-controlled counterpart of `MetalPositionDouble`. Not registered, same reason."""
    pass


class MetalPositionHit2P(MetalPositionHit):
    """`mini-hit` with a real second player instead of the scripted wander.

    The stock hit task follows the organizers' `IiwaPositionHit`: the opponent traces a
    lissajous and never plays the puck. That is fine for scoring one agent, but it makes the
    opponent an accidental goalkeeper -- measured on this rig it parks 5.9 cm off its own goal
    line and sweeps 1.33x the goal width, on a path that is deterministic in time. Here both
    arms run the baseline agent, so the opponent actually contests the puck.
    """

    def __init__(self, interpolation_order, opponent_agent="baseline", **kwargs):
        super().__init__(interpolation_order, opponent_agent=opponent_agent, **kwargs)


class MetalSdkHit2P(MetalSdkHit):
    """`MetalPositionHit2P` with the METAL SDK's low-level controller on both arms."""

    def __init__(self, interpolation_order, opponent_agent="baseline", **kwargs):
        super().__init__(interpolation_order, opponent_agent=opponent_agent, **kwargs)
