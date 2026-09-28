"""The mini air-hockey platform: a 40x20 in table and a 6-DOF "metal arm" gripping a mallet.

Self-contained: the challenge framework and the organizers' baseline control stack are
vendored under core/, estimation/, planning/ and fsm/ (see THIRD_PARTY.md). Nothing here
imports the parent repository.

Quick start::

    from mini_platform import make
    env = make("mini-hit")

or equivalently::

    from mini_platform.core.challenge_wrapper import AirHockeyChallengeWrapper
    env = AirHockeyChallengeWrapper("mini-hit")
"""

import os

#: Directory holding the MJCF scenes and meshes.
ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

#: Directory for measured constants, tables and reports produced by `mini_platform.tools`.
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

#: The `-sdk` variants differ from their siblings in exactly one thing: the low-level joint
#: controller is the METAL arm SDK's own law rather than the challenge stack's tuned PD.
TASKS = ("mini-hit", "mini-defend", "mini-prepare",
         "mini-hit-sdk", "mini-defend-sdk", "mini-prepare-sdk",
         "mini-hit-2p", "mini-hit-sdk-2p")


def make(task="mini-hit", interpolation_order=3, **kwargs):
    """Build a mini task wrapped in the challenge wrapper (constraints, success checks)."""
    from mini_platform.core.challenge_wrapper import AirHockeyChallengeWrapper

    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    return AirHockeyChallengeWrapper(task, interpolation_order=interpolation_order, **kwargs)


__all__ = ["ASSET_DIR", "ARTIFACT_DIR", "TASKS", "make"]
